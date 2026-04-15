"""
polymarket_orders.py BRANCH: INTEGRATION
====================
Execução de ordens no Polymarket CLOB.

Baseado no padrão testado e funcional:
    client = ClobClient(host=..., key=PRIVATE_KEY, chain_id=137)
    client.set_api_creds(client.create_or_derive_api_creds())
    signed = client.create_order(OrderArgs(...))
    resp   = client.post_order(signed, OrderType.GTC)

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
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CLOB_HOST  = "https://clob.polymarket.com"
CHAIN_ID   = 137
CREDS_FILE = Path("live_bot_logs/poly_creds.json")


class OrderExecutor:
    """
    Executa ordens no Polymarket CLOB.
    Padrão: create_or_derive_api_creds() sem signature_type nem funder.
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
                from py_clob_client.clob_types import ApiCreds
                saved = json.loads(CREDS_FILE.read_text())
                creds = ApiCreds(
                    api_key        = saved["api_key"],
                    api_secret     = saved["api_secret"],
                    api_passphrase = saved["api_passphrase"],
                )
                client.set_api_creds(creds)
                # get_ok() é público e não detecta 401 — usar endpoint autenticado
                from py_clob_client.clob_types import OpenOrderParams
                client.get_orders(OpenOrderParams())
                logger.debug("Credenciais carregadas de %s", CREDS_FILE)
                return client
            except Exception as e:
                logger.debug("Creds inválidas ou expiradas (%s) — a re-derivar", e)
                CREDS_FILE.unlink(missing_ok=True)

        # Derivar e guardar
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        CREDS_FILE.write_text(json.dumps({
            "api_key":        creds.api_key,
            "api_secret":     creds.api_secret,
            "api_passphrase": creds.api_passphrase,
        }, indent=2))
        logger.info("Credenciais derivadas e guardadas em %s", CREDS_FILE)
        return client

    # ── Saldo ──────────────────────────────────────────

    def get_balance(self) -> float | None:
        """
        Saldo USDC disponivel — le sig_type 0,1,2 e devolve o maior.
        O saldo util para ordens esta tipicamente em sig_type=2.
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

        # Fallback para Web3 se o método acima falhar
        RPC_LIST = [
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com",
            "https://1rpc.io/matic",
        ]
        for rpc in RPC_LIST:
            try:
                from web3 import Web3
                from eth_account import Account
                USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
                ABI  = [{"constant":True,"inputs":[{"name":"_owner","type":"address"}],
                         "name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],
                         "type":"function"}]
                w3       = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                addr     = Account.from_key(self._key).address
                contract = w3.eth.contract(address=Web3.to_checksum_address(USDC), abi=ABI)
                bal      = contract.functions.balanceOf(Web3.to_checksum_address(addr)).call()
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
        """Lista de ordens abertas (não preenchidas) no CLOB."""
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
        price:      float,      # preço limite 0–1, tipicamente o ask
        size_usdc:  float,      # USDC a gastar
        label:      str  = "",  # para logging
        order_type: str  = "FOK",  # "FOK" = market order, "GTC" = limit order
    ) -> dict:
        """
        Comprar YES num bracket.
        Devolve dict com success, order_id, status, error.

        order_type="FOK" (default): Fill-or-Kill ao ask.
            shares = floor(size/price) → makerAmount sempre com ≤ 2 casas decimais.
            Preenche imediatamente ou cancela — comportamento de market order.
        order_type="GTC": Good-Till-Cancelled.
            shares = round(size/price, 4) → fica no book se não encher.
        """
        from py_clob_client.clob_types import OrderArgs, OrderType

        if order_type.upper() == "FOK":
            # floor garante shares inteiro → shares×price nunca excede 2 casas decimais
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
        """Vender shares YES (fechar posição). Usa GTC ao bid."""
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

    # ── Cancelar ordem ─────────────────────────────────

    def cancel(self, order_id: str) -> bool:
        """Cancelar uma ordem aberta pelo ID."""
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            logger.warning("cancel %s falhou: %s", order_id, e)
            return False

    def cancel_all(self) -> bool:
        """Cancelar todas as ordens abertas de uma vez."""
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


# ── Paper simulation (sem rede) ───────────────────────

def paper_buy(token_id: str, price: float, size_usdc: float,
              label: str = "") -> dict:
    """Simula uma compra sem enviar nada ao CLOB."""
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
