"""
polymarket_orders.py
====================
Execução de ordens no Polymarket CLOB.

Variáveis de ambiente:
    POLY_PRIVATE_KEY=0x...
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CLOB_HOST  = "https://clob.polymarket.com"
CHAIN_ID   = 137
CREDS_FILE = Path("live_bot_logs/poly_creds.json")

# Contratos Polymarket na Polygon
USDC_ADDRESS     = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC (PoS)
# CTF Exchange — contrato que precisa de allowance para ordens CLOB
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
# Approve amount: 1 bilhão USDC (6 decimais) — aprovação permanente
APPROVE_AMOUNT   = 1_000_000_000 * 10**6

POLYGON_RPCS = [
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://1rpc.io/matic",
]


def _approve_usdc_onchain(private_key: str) -> bool:
    """
    Envia transação approve() onchain no contrato USDC da Polygon.
    Permite ao Exchange do Polymarket mover USDC da wallet.
    Só é necessário UMA VEZ por wallet.
    """
    from web3 import Web3
    from eth_account import Account

    ABI_APPROVE = [{
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount",  "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    }]

    account  = Account.from_key(private_key)
    spender  = Web3.to_checksum_address(EXCHANGE_ADDRESS)
    usdc     = Web3.to_checksum_address(USDC_ADDRESS)

    for rpc in POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 20}))
            if not w3.is_connected():
                logger.debug("RPC %s não conectado", rpc)
                continue

            contract = w3.eth.contract(address=usdc, abi=ABI_APPROVE)
            nonce    = w3.eth.get_transaction_count(account.address)

            tx = contract.functions.approve(spender, APPROVE_AMOUNT).build_transaction({
                "from":     account.address,
                "nonce":    nonce,
                "gas":      100_000,
                "gasPrice": w3.eth.gas_price,
                "chainId":  137,
            })

            signed  = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            logger.info("Approve tx enviada: %s", tx_hash.hex())

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
            if receipt.status == 1:
                logger.info("✓ Approve confirmado onchain (bloco %s)", receipt.blockNumber)
                return True
            else:
                logger.error("✗ Approve falhou onchain (status=0)")
                return False

        except Exception as e:
            logger.debug("RPC %s falhou: %s", rpc, e)
            continue

    logger.error("_approve_usdc_onchain: todos os RPCs falharam")
    return False


def _check_allowance_web3(private_key: str) -> float:
    """Lê a allowance actual onchain (sem depender do CLOB API)."""
    from web3 import Web3
    from eth_account import Account

    ABI_ALLOWANCE = [{
        "name": "allowance",
        "type": "function",
        "inputs": [
            {"name": "owner",   "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    }]

    account = Account.from_key(private_key)
    owner   = Web3.to_checksum_address(account.address)
    spender = Web3.to_checksum_address(EXCHANGE_ADDRESS)
    usdc    = Web3.to_checksum_address(USDC_ADDRESS)

    for rpc in POLYGON_RPCS:
        try:
            w3       = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
            contract = w3.eth.contract(address=usdc, abi=ABI_ALLOWANCE)
            raw      = contract.functions.allowance(owner, spender).call()
            return raw / 1e6
        except Exception as e:
            logger.debug("allowance RPC %s falhou: %s", rpc, e)
            continue

    return -1.0  # desconhecido


class OrderExecutor:
    """
    Executa ordens no Polymarket CLOB.
    Verifica e garante allowance onchain antes de qualquer ordem.
    Tenta depositar fundos automaticamente se o saldo no CLOB for 0.
    """

    def __init__(self, private_key: str):
        if not private_key:
            raise ValueError("POLY_PRIVATE_KEY não definida")
        self._key    = private_key
        self._client = self._init_client()

    # ── Inicialização ──────────────────────────────────

    def _init_client(self):
        from py_clob_client.client import ClobClient

        client = ClobClient(
            host     = CLOB_HOST,
            key      = self._key,
            chain_id = CHAIN_ID,
        )

        # Carregar credenciais guardadas ou derivar novas
        CREDS_FILE.parent.mkdir(exist_ok=True)
        if CREDS_FILE.exists():
            try:
                from py_clob_client.clob_types import ApiCreds, OpenOrderParams
                saved = json.loads(CREDS_FILE.read_text())
                creds = ApiCreds(
                    api_key        = saved["api_key"],
                    api_secret     = saved["api_secret"],
                    api_passphrase = saved["api_passphrase"],
                )
                client.set_api_creds(creds)
                client.get_orders(OpenOrderParams())
                logger.debug("Credenciais carregadas de %s", CREDS_FILE)
            except Exception as e:
                logger.debug("Creds inválidas (%s) — a re-derivar", e)
                CREDS_FILE.unlink(missing_ok=True)
                creds = client.create_or_derive_api_creds()
                client.set_api_creds(creds)
                CREDS_FILE.write_text(json.dumps({
                    "api_key":        creds.api_key,
                    "api_secret":     creds.api_secret,
                    "api_passphrase": creds.api_passphrase,
                }, indent=2))
                logger.info("Credenciais derivadas e guardadas em %s", CREDS_FILE)
        else:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            CREDS_FILE.write_text(json.dumps({
                "api_key":        creds.api_key,
                "api_secret":     creds.api_secret,
                "api_passphrase": creds.api_passphrase,
            }, indent=2))
            logger.info("Credenciais derivadas e guardadas em %s", CREDS_FILE)

        # ── Garantir allowance onchain ─────────────────────────────────────
        try:
            allowance = _check_allowance_web3(self._key)
            if allowance < 0:
                logger.warning("Não foi possível verificar allowance — a continuar na mesma")
            elif allowance < 1.0:
                logger.info(
                    "Allowance onchain insuficiente (%.2f USDC) — a enviar approve()...",
                    allowance,
                )
                ok = _approve_usdc_onchain(self._key)
                if ok:
                    time.sleep(5)  # aguardar propagação
                    new_allowance = _check_allowance_web3(self._key)
                    logger.info("Allowance após approve: %.2f USDC", new_allowance)
                else:
                    logger.error(
                        "APPROVE FALHOU — as ordens reais vão falhar com 'balance: 0'. "
                        "Verifica se tens MATIC na wallet para gas."
                    )
            else:
                logger.debug("Allowance onchain OK: %.2f USDC", allowance)
        except Exception as e:
            logger.warning("Verificação de allowance falhou: %s", e)

        # ── Verificar e Depositar Saldo no CLOB ─────────────────────────────
        # O erro "balance: 0" acontece quando tens USDC na carteira mas não no balanço do CLOB.
        time.sleep(2) 
        
        clob_balance = self.get_balance()
        
        # Se o saldo no CLOB for 0 ou muito baixo, tentamos depositar da Wallet Principal
        if clob_balance is not None and clob_balance < 1.0:
            logger.warning(
                "Saldo no CLOB baixo (%.2f USDC). A tentar depositar $20.00 da wallet...", 
                clob_balance
            )
            # Tenta depositar $20 para garantir margem para operar
            deposit_ok = self.deposit_usdc(20.0) 
            if deposit_ok:
                time.sleep(5) # Esperar confirmação na API
                new_balance = self.get_balance()
                logger.info("✓ Depósito realizado. Novo saldo CLOB: %.2f USDC", new_balance or 0)
            else:
                logger.error("✗ Falha no depósito. O bot não conseguirá negociar.")
        elif clob_balance is None:
            logger.warning("Não foi possível verificar saldo CLOB. A assumir que está OK.")

        return client

    # ── Depósito de Fundos (NOVO) ─────────────────────────────

    def deposit_usdc(self, amount_usdc: float) -> bool:
        """
        Move USDC da wallet principal para o endereço do Exchange CLOB.
        Necessário para resolver o erro "balance: 0" quando tens fundos na carteira
        mas a API do CLOB não os vê.
        """
        from web3 import Web3
        from eth_account import Account

        ABI_TRANSFER = [{
            "name": "transfer",
            "type": "function",
            "inputs": [
                {"name": "to",   "type": "address"},
                {"name": "value","type": "uint256"},
            ],
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "nonpayable",
        }]

        account = Account.from_key(self._key)
        to_addr = Web3.to_checksum_address(EXCHANGE_ADDRESS) # Enviar para o contrato do Exchange
        usdc    = Web3.to_checksum_address(USDC_ADDRESS)
        amount_raw = int(amount_usdc * 10**6)

        for rpc in POLYGON_RPCS:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 20}))
                if not w3.is_connected():
                    continue

                contract = w3.eth.contract(address=usdc, abi=ABI_TRANSFER)
                nonce    = w3.eth.get_transaction_count(account.address)
                gas_price = w3.eth.gas_price

                tx = contract.functions.transfer(to_addr, amount_raw).build_transaction({
                    "from":     account.address,
                    "nonce":    nonce,
                    "gas":      100_000, # Gás suficiente para transfer
                    "gasPrice": gas_price,
                    "chainId":  137,
                })

                signed  = w3.eth.account.sign_transaction(tx, self._key)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                logger.info("Depósito tx enviada: %s (%.2f USDC)", tx_hash.hex(), amount_usdc)

                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
                if receipt.status == 1:
                    logger.info("✓ Depósito confirmado no CLOB (bloco %s)", receipt.blockNumber)
                    return True
                else:
                    logger.error("✗ Depósito falhou onchain (status=0)")
                    return False

            except Exception as e:
                logger.debug("Depósito RPC %s falhou: %s", rpc, e)
                continue

        logger.error("deposit_usdc: todos os RPCs falharam")
        return False

    # ── Saldo ──────────────────────────────────────────

    def get_balance(self) -> float | None:
        """
        Saldo USDC disponivel — le sig_type 0,1,2 e devolve o maior.
        """
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            best = 0.0
            for sig in [0, 1, 2]:
                try:
                    info = self._client.get_balance_allowance(
                        params=BalanceAllowanceParams(
                            asset_type=AssetType.COLLATERAL,
                            signature_type=sig,
                        )
                    )
                    bal = int(info.get("balance", "0")) / 1e6
                    if bal > best:
                        best = bal
                except Exception:
                    pass
            return best if best > 0 else None
        except Exception as e:
            logger.debug("get_balance_allowance falhou: %s", e)

        # Fallback Web3 (apenas para verificar saldo na carteira, não no CLOB)
        for rpc in POLYGON_RPCS:
            try:
                from web3 import Web3
                from eth_account import Account
                ABI = [{"constant":True,"inputs":[{"name":"_owner","type":"address"}],
                        "name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],
                        "type":"function"}]
                w3       = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                addr     = Account.from_key(self._key).address
                contract = w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_ADDRESS), abi=ABI
                )
                bal = contract.functions.balanceOf(
                    Web3.to_checksum_address(addr)
                ).call()
                return round(bal / 1_000_000.0, 2)
            except Exception as e:
                logger.debug("Web3 RPC %s falhou: %s", rpc, e)
                continue

        logger.warning("get_balance: todos os métodos falharam")
        return None

    # ── Order Book ─────────────────────────────────────

    def get_orderbook(self, token_id: str) -> dict | None:
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

    # ── Ordens abertas ─────────────────────────────────

    def get_open_orders(self) -> list[dict]:
        try:
            from py_clob_client.clob_types import OpenOrderParams
            orders = self._client.get_orders(OpenOrderParams())
            return orders if isinstance(orders, list) else []
        except Exception as e:
            logger.warning("get_open_orders falhou: %s", e)
            return []

    # ── Comprar YES ────────────────────────────────────

    def buy(
        self,
        token_id:   str,
        price:      float,
        size_usdc:  float,
        label:      str = "",
        order_type: str = "FOK",
    ) -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType

        if order_type.upper() == "FOK":
            shares = math.floor(size_usdc / price)
            _otype = OrderType.FOK
        else:
            shares = round(size_usdc / price, 4)
            _otype = OrderType.GTC

        try:
            order_args = OrderArgs(
                price    = round(price, 4),
                size     = shares,
                side     = "BUY",
                token_id = token_id,
            )
            signed   = self._client.create_order(order_args)
            response = self._client.post_order(signed, _otype)

            order_id    = response.get("orderID") or response.get("id") or "?"
            status      = response.get("status", "unknown")
            api_success = response.get("success", False)
            success     = api_success and status in ("matched", "live", "delayed")

            result = {
                "success":    success,
                "simulated":  False,
                "order_id":   order_id,
                "status":     status,
                "order_type": order_type.upper(),
                "side":       "BUY",
                "token_id":   token_id,
                "price":      round(price, 4),
                "size_usdc":  round(size_usdc, 2),
                "shares":     shares,
                "label":      label,
                "error":      None if success else f"status={status} response={response}",
                "timestamp":  _now(),
            }

        except Exception as e:
            logger.error("buy falhou (%s): %s", label, e)
            result = {
                "success":    False,
                "simulated":  False,
                "order_id":   None,
                "status":     "error",
                "order_type": order_type.upper(),
                "side":       "BUY",
                "token_id":   token_id,
                "price":      round(price, 4),
                "size_usdc":  round(size_usdc, 2),
                "shares":     shares,
                "label":      label,
                "error":      str(e),
                "timestamp":  _now(),
            }

        self._log(result)
        return result

    # ── Vender YES ─────────────────────────────────────

    def sell(
        self,
        token_id: str,
        price:    float,
        shares:   float,
        label:    str = "",
    ) -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType

        try:
            order_args = OrderArgs(
                price    = round(price, 4),
                size     = shares,
                side     = "SELL",
                token_id = token_id,
            )
            signed   = self._client.create_order(order_args)
            response = self._client.post_order(signed, OrderType.GTC)

            order_id    = response.get("orderID") or response.get("id") or "?"
            status      = response.get("status", "unknown")
            api_success = response.get("success", False)
            success     = api_success and status in ("matched", "live", "delayed")

            result = {
                "success":   success,
                "simulated": False,
                "order_id":  order_id,
                "status":    status,
                "side":      "SELL",
                "token_id":  token_id,
                "price":     round(price, 4),
                "shares":    shares,
                "label":     label,
                "error":     None if success else f"status={status}",
                "timestamp": _now(),
            }

        except Exception as e:
            logger.error("sell falhou (%s): %s", label, e)
            result = {
                "success":   False,
                "simulated": False,
                "order_id":  None,
                "status":    "error",
                "side":      "SELL",
                "token_id":  token_id,
                "price":     price,
                "shares":    shares,
                "label":     label,
                "error":     str(e),
                "timestamp": _now(),
            }

        self._log(result)
        return result

    # ── Cancelar ───────────────────────────────────────

    def cancel(self, order_id: str) -> bool:
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            logger.warning("cancel %s falhou: %s", order_id, e)
            return False

    def cancel_all(self) -> bool:
        try:
            self._client.cancel_all()
            return True
        except Exception as e:
            logger.warning("cancel_all falhou: %s", e)
            return False

    # ── Logging ────────────────────────────────────────

    def _log(self, result: dict) -> None:
        log_path = Path("live_bot_logs") / f"orders_{_today()}.json"
        try:
            existing = json.loads(log_path.read_text()) if log_path.exists() else []
            existing.append(result)
            log_path.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.warning("_log falhou: %s", e)


# ── Helpers ────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat()

def _today() -> str:
    from datetime import date
    return date.today().isoformat()


# ── Paper simulation ──────────────────────────────────

def paper_buy(token_id: str, price: float, size_usdc: float,
              label: str = "") -> dict:
    shares = round(size_usdc / price, 2)
    result = {
        "success":   True,
        "simulated": True,
        "order_id":  f"PAPER-{int(time.time())}",
        "status":    "SIMULATED",
        "side":      "BUY",
        "token_id":  token_id,
        "price":     round(price, 4),
        "size_usdc": round(size_usdc, 2),
        "shares":    shares,
        "label":     label,
        "error":     None,
        "timestamp": _now(),
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
