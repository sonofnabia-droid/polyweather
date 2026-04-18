"""
polymarket_orders.py — VERSÃO FINAL LIMPA
==========================================
Execução de ordens no Polymarket CLOB com:
  - Credenciais API (derive + cache)
  - Allowance USDC (approve automático on-chain se < $100)
  - Saldo CLOB (sig_type=2 prioritário, fallback on-chain)
  - Ordens BUY/SELL com FOK makerAmount fix
  - Verificação de saldo pré-compra

Variáveis de ambiente:
    POLY_PRIVATE_KEY=0x...

Uso:
    from polymarket_orders import OrderExecutor
    ex = OrderExecutor(private_key)
    result = ex.buy(token_id, price=0.35, size_usdc=20.0)
    balance = ex.get_balance()
    positions = ex.get_open_orders()
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────

CLOB_HOST  = "https://clob.polymarket.com"
CHAIN_ID   = 137
CREDS_FILE = Path("live_bot_logs/poly_creds.json")

# Endereços na Polygon
USDC_ADDRESS     = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # CTF Exchange

# RPCs Polygon com fallback robusto
POLYGON_RPCS = [
    "https://rpc.ankr.com/polygon",
    "https://polygon-rpc.com",
    "https://1rpc.io/matic",
    "https://polygon.blockpi.network/v1/rpc/public",
    "https://polygon.llamarpc.com",
    "https://polygon-mainnet.g.alchemy.com/v2/demo",
]


# ══════════════════════════════════════════════════════════════════════
#  FUNÇÕES WEB3 AUXILIARES
# ══════════════════════════════════════════════════════════════════════

def _get_web3():
    """Obtém conexão Web3 funcional — tenta 6 RPCs até conectar."""
    from web3 import Web3
    for rpc in POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                logger.debug("Conectado a RPC: %s", rpc)
                return w3
        except Exception as e:
            logger.debug("Falha RPC %s: %s", rpc, e)
    logger.error("Não foi possível conectar a nenhum RPC Polygon")
    return None


def _get_wallet_usdc(private_key: str) -> float:
    """Lê saldo USDC.e on-chain da wallet (útil para debug manual)."""
    from web3 import Web3
    from eth_account import Account

    w3 = _get_web3()
    if not w3:
        return 0.0
    try:
        account = Account.from_key(private_key)
        token   = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=[{
                "name": "balanceOf", "type": "function",
                "inputs":  [{"name": "account", "type": "address"}],
                "outputs": [{"name": "", "type": "uint256"}],
            }],
        )
        return token.functions.balanceOf(account.address).call() / 1e6
    except Exception as e:
        logger.warning("Erro saldo wallet: %s", e)
        return 0.0


def _get_allowance(private_key: str) -> float:
    """Lê allowance actual do USDC para o Exchange contract."""
    from web3 import Web3
    from eth_account import Account

    w3 = _get_web3()
    if not w3:
        return 0.0
    try:
        account = Account.from_key(private_key)
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=[{
                "name": "allowance", "type": "function",
                "stateMutability": "view",
                "inputs":  [
                    {"name": "owner",   "type": "address"},
                    {"name": "spender", "type": "address"},
                ],
                "outputs": [{"name": "", "type": "uint256"}],
            }],
        )
        return contract.functions.allowance(
            account.address,
            Web3.to_checksum_address(EXCHANGE_ADDRESS),
        ).call() / 1e6
    except Exception as e:
        logger.warning("Erro allowance: %s", e)
        return 0.0


def _approve_usdc(private_key: str, amount_usd: float = 1_000_000) -> bool:
    """Aprova o Exchange contract a gastar USDC da wallet."""
    from web3 import Web3
    from eth_account import Account

    w3 = _get_web3()
    if not w3:
        return False

    ABI = [{
        "name": "approve", "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount",  "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    }]

    try:
        account  = Account.from_key(private_key)
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS), abi=ABI,
        )
        nonce = w3.eth.get_transaction_count(account.address)

        tx = contract.functions.approve(
            Web3.to_checksum_address(EXCHANGE_ADDRESS),
            int(amount_usd * 1e6),
        ).build_transaction({
            "from": account.address, "nonce": nonce,
            "gas": 100_000, "gasPrice": w3.eth.gas_price, "chainId": 137,
        })

        signed  = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Approve tx: %s", tx_hash.hex())

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
        if receipt.status == 1:
            logger.info("✅ Approve confirmado")
            return True
        logger.error("❌ Approve falhou (status=%s)", receipt.status)
        return False
    except Exception as e:
        logger.error("Approve erro: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now().isoformat()

def _today() -> str:
    return date.today().isoformat()


# ══════════════════════════════════════════════════════════════════════
#  OrderExecutor
# ══════════════════════════════════════════════════════════════════════

class OrderExecutor:
    """
    Executa ordens no Polymarket CLOB.
    
    Inicialização:
      1. Derivar/carregar credenciais API
      2. Verificar allowance → aprovar se < $100
      3. Logar saldo CLOB disponível (sig_type=2)
    """

    def __init__(self, private_key: str):
        if not private_key:
            raise ValueError("POLY_PRIVATE_KEY não definida")

        self._key = private_key
        logger.info("OrderExecutor: %s...%s", self._key[:10], self._key[-6:])
        self._client = self._init_client()

    # ── Inicialização ──────────────────────────────────────────────────

    def _init_client(self):
        from py_clob_client.client import ClobClient

        client = ClobClient(
            host=CLOB_HOST, key=self._key, chain_id=CHAIN_ID,
        )

        # 1. Credenciais API
        self._setup_creds(client)

        # 2. Allowance on-chain
        allowance = _get_allowance(self._key)
        logger.info("Allowance: $%.2f", allowance)
        if allowance < 100:
            logger.info("Allowance baixa — a aprovar $1M...")
            if _approve_usdc(self._key, 1_000_000):
                logger.info("✅ Allowance aprovada")
                time.sleep(5)
            else:
                logger.warning("⚠️ Falha no approve — ordens podem falhar")

        # 3. Logar saldo CLOB
        clob = self._read_clob_balance(client)
        logger.info("💰 Saldo CLOB disponível: $%.2f", clob)

        return client

    def _setup_creds(self, client) -> None:
        """Carrega credenciais guardadas ou deriva novas."""
        from py_clob_client.clob_types import ApiCreds, OpenOrderParams

        CREDS_FILE.parent.mkdir(exist_ok=True)

        if CREDS_FILE.exists():
            try:
                saved = json.loads(CREDS_FILE.read_text())
                creds = ApiCreds(
                    api_key=saved["api_key"],
                    api_secret=saved["api_secret"],
                    api_passphrase=saved["api_passphrase"],
                )
                client.set_api_creds(creds)
                client.get_orders(OpenOrderParams())  # testa autenticação
                logger.info("Credenciais carregadas de %s", CREDS_FILE)
                return
            except Exception as e:
                logger.warning("Creds inválidas (%s) — re-derivar", e)
                CREDS_FILE.unlink(missing_ok=True)

        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        CREDS_FILE.write_text(json.dumps({
            "api_key":        creds.api_key,
            "api_secret":     creds.api_secret,
            "api_passphrase": creds.api_passphrase,
        }, indent=2))
        logger.info("Credenciais derivadas e guardadas")

    @staticmethod
    def _read_clob_balance(client) -> float:
        """Lê saldo CLOB — sig_type=2 prioritário, fallback 0,1."""
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        best = 0.0
        for sig in [2, 0, 1]:  # 2 primeiro (saldo REAL para ordens)
            try:
                info = client.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL,
                        signature_type=sig,
                    )
                )
                bal = int(info.get("balance", 0)) / 1e6
                if bal > best:
                    best = bal
            except Exception:
                pass
        return best

    # ── Saldo ──────────────────────────────────────────────────────────

    def get_balance(self) -> float | None:
        """
        Saldo USDC disponível no CLOB.
        Prioridade: sig_type=2 → fallback on-chain.
        """
        try:
            bal = self._read_clob_balance(self._client)
            if bal > 0:
                return bal
        except Exception as e:
            logger.debug("CLOB balance falhou: %s", e)

        # Fallback on-chain (wallet, não CLOB — último recurso)
        wallet = _get_wallet_usdc(self._key)
        if wallet > 0:
            logger.info("CLOB=0, fallback wallet: $%.2f", wallet)
            return wallet
        return None

    def get_balance_raw(self, signature_type: int = 2) -> float:
        """Saldo para um signature_type específico (para debug/exactidão)."""
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        try:
            info = self._client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=signature_type,
                )
            )
            return int(info.get("balance", 0)) / 1e6
        except Exception as e:
            logger.error("get_balance_raw(%d): %s", signature_type, e)
            return 0.0

    # ── Order Book ─────────────────────────────────────────────────────

    def get_orderbook(self, token_id: str):
        try:
            return self._client.get_order_book(token_id)
        except Exception as e:
            if "404" not in str(e):
                logger.warning("get_orderbook %s: %s", token_id[:16], e)
            return None

    def get_best_prices(self, token_id: str) -> dict:
        book = self.get_orderbook(token_id)
        if not book:
            return {"bid": None, "ask": None, "spread": None}
        bids = sorted(book.bids or [], key=lambda x: -float(x.price))
        asks = sorted(book.asks or [], key=lambda x:  float(x.price))
        bid  = float(bids[0].price) if bids else None
        ask  = float(asks[0].price) if asks else None
        spr  = round(ask - bid, 4) if (bid and ask) else None
        return {"bid": bid, "ask": ask, "spread": spr}

    # ── Ordens abertas ─────────────────────────────────────────────────

    def get_open_orders(self) -> list[dict]:
        try:
            from py_clob_client.clob_types import OpenOrderParams
            orders = self._client.get_orders(OpenOrderParams())
            return orders if isinstance(orders, list) else []
        except Exception as e:
            logger.warning("get_open_orders: %s", e)
            return []

    # ── Comprar YES ────────────────────────────────────────────────────

    def buy(
        self,
        token_id:   str,
        price:      float,
        size_usdc:  float,
        label:      str  = "",
        order_type: str  = "FOK",
    ) -> dict:
        """
        Comprar YES.

        FOK (default):
            makerAmount (USDC) truncado a 2 casas decimais — requisito API.
            size_usdc_clean = floor(size_usdc * 100) / 100
            shares = round(size_usdc_clean / price, 4)

        GTC:
            shares = round(size_usdc / price, 4) → fica no book.
        """
        from py_clob_client.clob_types import OrderArgs, OrderType

        # ── Verificar saldo (sig_type=2 = saldo REAL para ordens) ─────
        current_bal = self.get_balance_raw(signature_type=2)
        logger.info(
            "💰 Saldo sig_type=2: $%.2f | Ordem: %s — $%.2f @ %.4f",
            current_bal, label, size_usdc, price,
        )

        if current_bal < size_usdc:
            err = f"Saldo CLOB insuficiente: ${current_bal:.2f} < ${size_usdc:.2f}"
            logger.error(err)
            result = self._make_result(
                success=False, token_id=token_id, price=price,
                size_usdc=size_usdc, shares=0, label=label,
                order_type=order_type, side="BUY", error=err,
            )
            self._log(result)
            return result

        # ── Calcular shares ───────────────────────────────────────────
        if order_type.upper() == "FOK":
            size_usdc_clean = math.floor(size_usdc * 100) / 100
            shares          = round(size_usdc_clean / price, 4)
            _otype          = OrderType.FOK
        else:
            size_usdc_clean = size_usdc
            shares          = round(size_usdc / price, 4)
            _otype          = OrderType.GTC

        if shares <= 0:
            err = f"Shares=0 (price={price}, size={size_usdc})"
            logger.error(err)
            result = self._make_result(
                success=False, token_id=token_id, price=price,
                size_usdc=size_usdc_clean, shares=0, label=label,
                order_type=order_type, side="BUY", error=err,
            )
            self._log(result)
            return result

        # ── Enviar ordem ──────────────────────────────────────────────
        try:
            logger.info("📝 BUY %s shares @ %.4f (%s)", shares, price, order_type)
            order_args = OrderArgs(
                price=round(price, 4), size=shares,
                side="BUY", token_id=token_id,
            )
            signed   = self._client.create_order(order_args)
            response = self._client.post_order(signed, _otype)

            order_id    = response.get("orderID") or response.get("id") or "?"
            status      = response.get("status", "unknown")
            api_success = response.get("success", False)
            success     = api_success and status in ("matched", "live", "delayed")

            if success:
                logger.info("✅ BUY %s: $%.2f @ %.4f — %s", label, size_usdc_clean, price, status)
            else:
                logger.warning("⚠️ BUY falhou: %s — %s", label, response)

            result = self._make_result(
                success=success, token_id=token_id, price=price,
                size_usdc=size_usdc_clean, shares=shares, label=label,
                order_type=order_type, side="BUY",
                order_id=order_id, status=status,
                error=None if success else f"status={status} response={response}",
            )

        except Exception as e:
            logger.error("❌ BUY erro (%s): %s", label, e)
            result = self._make_result(
                success=False, token_id=token_id, price=price,
                size_usdc=size_usdc_clean, shares=shares, label=label,
                order_type=order_type, side="BUY", error=str(e),
            )

        self._log(result)
        return result

    # ── Vender YES ─────────────────────────────────────────────────────

    def sell(
        self,
        token_id: str,
        price:    float,
        shares:   float,
        label:    str = "",
    ) -> dict:
        """Vender shares YES (GTC ao bid)."""
        from py_clob_client.clob_types import OrderArgs, OrderType

        try:
            order_args = OrderArgs(
                price=round(price, 4), size=shares,
                side="SELL", token_id=token_id,
            )
            signed   = self._client.create_order(order_args)
            response = self._client.post_order(signed, OrderType.GTC)

            order_id    = response.get("orderID") or response.get("id") or "?"
            status      = response.get("status", "unknown")
            api_success = response.get("success", False)
            success     = api_success and status in ("matched", "live", "delayed")

            if success:
                logger.info("✅ SELL %s: %.2f shares @ %.4f", label, shares, price)

            result = self._make_result(
                success=success, token_id=token_id, price=price,
                size_usdc=None, shares=shares, label=label,
                order_type="GTC", side="SELL",
                order_id=order_id, status=status,
                error=None if success else f"status={status}",
            )

        except Exception as e:
            logger.error("SELL erro (%s): %s", label, e)
            result = self._make_result(
                success=False, token_id=token_id, price=price,
                size_usdc=None, shares=shares, label=label,
                order_type="GTC", side="SELL", error=str(e),
            )

        self._log(result)
        return result

    # ── Cancelar ───────────────────────────────────────────────────────

    def cancel(self, order_id: str) -> bool:
        try:
            self._client.cancel(order_id)
            logger.info("Cancelado: %s", order_id)
            return True
        except Exception as e:
            logger.warning("cancel %s: %s", order_id, e)
            return False

    def cancel_all(self) -> bool:
        try:
            self._client.cancel_all()
            logger.info("Todas as ordens canceladas")
            return True
        except Exception as e:
            logger.warning("cancel_all: %s", e)
            return False

    # ── Internos ───────────────────────────────────────────────────────

    @staticmethod
    def _make_result(**kw) -> dict:
        """Constrói result dict consistente."""
        return {
            "success":    kw.get("success", False),
            "simulated":  False,
            "order_id":   kw.get("order_id"),
            "status":     kw.get("status", "error"),
            "order_type": kw.get("order_type", "").upper(),
            "side":       kw.get("side", ""),
            "token_id":   kw.get("token_id", ""),
            "price":      round(kw["price"], 4) if kw.get("price") else None,
            "size_usdc":  round(kw["size_usdc"], 2) if kw.get("size_usdc") is not None else None,
            "shares":     kw.get("shares", 0),
            "label":      kw.get("label", ""),
            "error":      kw.get("error"),
            "timestamp":  _now(),
        }

    def _log(self, result: dict) -> None:
        """Regista resultado em ficheiro JSON diário."""
        log_path = Path("live_bot_logs") / f"orders_{_today()}.json"
        try:
            log_path.parent.mkdir(exist_ok=True)
            existing = json.loads(log_path.read_text()) if log_path.exists() else []
            existing.append(result)
            log_path.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.warning("_log falhou: %s", e)


# ══════════════════════════════════════════════════════════════════════
#  PAPER TRADING
# ══════════════════════════════════════════════════════════════════════

def paper_buy(token_id: str, price: float, size_usdc: float,
              label: str = "") -> dict:
    """Simula compra sem enviar ao CLOB."""
    shares = round(size_usdc / price, 2) if price > 0 else 0
    result = {
        "success":    True,
        "simulated":  True,
        "order_id":   f"PAPER-{int(time.time())}",
        "status":     "SIMULATED",
        "order_type": "PAPER",
        "side":       "BUY",
        "token_id":   token_id,
        "price":      round(price, 4),
        "size_usdc":  round(size_usdc, 2),
        "shares":     shares,
        "label":      label,
        "error":      None,
        "timestamp":  _now(),
    }
    log_path = Path("live_bot_logs") / f"orders_{_today()}.json"
    try:
        log_path.parent.mkdir(exist_ok=True)
        existing = json.loads(log_path.read_text()) if log_path.exists() else []
        existing.append(result)
        log_path.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass
    return result
