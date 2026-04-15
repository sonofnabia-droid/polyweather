"""
polymarket_orders.py
====================
Versão "Safe Mode" - Não crasha se o depósito falhar, apenas regista o erro.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

# Configurar logging para vermos erros no console também
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CLOB_HOST  = "https://clob.polymarket.com"
CHAIN_ID   = 137
CREDS_FILE = Path("live_bot_logs/poly_creds.json")

USDC_ADDRESS     = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
APPROVE_AMOUNT   = 1_000_000_000 * 10**6

POLYGON_RPCS = [
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://1rpc.io/matic",
]

def _get_web3():
    from web3 import Web3
    for rpc in POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
            if w3.is_connected():
                return w3
        except Exception:
            continue
    return None

def _approve_usdc_onchain(private_key: str) -> bool:
    from web3 import Web3
    from eth_account import Account

    ABI_APPROVE = [{
        "name": "approve",
        "type": "function",
        "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    }]

    account = Account.from_key(private_key)
    spender = Web3.to_checksum_address(EXCHANGE_ADDRESS)
    usdc    = Web3.to_checksum_address(USDC_ADDRESS)
    
    w3 = _get_web3()
    if not w3: 
        logger.error("Web3 não conectou. Impossível aprovar.")
        return False

    try:
        contract = w3.eth.contract(address=usdc, abi=ABI_APPROVE)
        nonce    = w3.eth.get_transaction_count(account.address)
        
        # Estimar gas para evitar surpresas
        gas_estimate = contract.functions.approve(spender, APPROVE_AMOUNT).estimate_gas({"from": account.address})
        
        tx = contract.functions.approve(spender, APPROVE_AMOUNT).build_transaction({
            "from": account.address, "nonce": nonce,
            "gas": int(gas_estimate * 1.2), # 20% buffer
            "gasPrice": w3.eth.gas_price, "chainId": 137,
        })
        
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info(f"✓ Approve Tx enviada: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return receipt.status == 1
    except Exception as e:
        logger.error(f"✗ Erro ao fazer Approve: {e}")
        return False

class OrderExecutor:
    def __init__(self, private_key: str):
        if not private_key:
            raise ValueError("POLY_PRIVATE_KEY não definida")
        self._key = private_key
        # Inicializar cliente básico primeiro (sem on-chain)
        self._client = self._init_client_basic()
        
        # Tentar gerir fundos de forma segura (non-blocking)
        self._manage_funds_safe()

    def _init_client_basic(self):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, OpenOrderParams

        client = ClobClient(host=CLOB_HOST, key=self._key, chain_id=CHAIN_ID)
        CREDS_FILE.parent.mkdir(exist_ok=True)

        if CREDS_FILE.exists():
            try:
                saved = json.loads(CREDS_FILE.read_text())
                creds = ApiCreds(saved["api_key"], saved["api_secret"], saved["api_passphrase"])
                client.set_api_creds(creds)
                # Test connection silencioso
                try: client.get_orders(OpenOrderParams())
                except: pass 
            except Exception:
                CREDS_FILE.unlink(missing_ok=True)
                creds = client.create_or_derive_api_creds()
                client.set_api_creds(creds)
                self._save_creds(creds)
        else:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            self._save_creds(creds)
        
        return client

    def _manage_funds_safe(self):
        """
        Tenta aprovar e depositar. Se falhar, regista erro mas não crasha o bot.
        """
        logger.info("A verificar e gerir fundos (Auto-Deposit)...")
        from web3 import Web3
        from eth_account import Account
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        try:
            w3 = _get_web3()
            if not w3:
                logger.error("❌ RPC Web3 falhou. Não consigo verificar/depositar fundos automaticamente.")
                return

            account = Account.from_key(self._key)
            
            # 1. Verificar Saldo Wallet
            token = w3.eth.contract(
                address=Web3.to_checksum_address(USDC_ADDRESS),
                abi=[{"name":"balanceOf","type":"function","inputs":[{"name":"account","type":"address"}],"outputs":[{"name":"","type":"uint256"}]}]
            )
            wallet_balance = token.functions.balanceOf(account.address).call() / 1e6
            logger.info(f"💰 Saldo Wallet: ${wallet_balance:.2f}")

            # 2. Verificar Saldo CLOB
            clob_balance = 0.0
            try:
                for sig in [2, 1]:
                    info = self._client.get_balance_allowance(params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig))
                    bal = int(info.get("balance", 0)) / 1e6
                    if bal > clob_balance: clob_balance = bal
            except Exception as e:
                logger.warning(f"Aviso: Não consegui ler saldo CLOB: {e}")
            
            logger.info(f"💰 Saldo CLOB:  ${clob_balance:.2f}")

            # 3. Verificar Allowance
            try:
                usdc_contract = w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_ADDRESS),
                    abi=[{
                        "name":"allowance","type":"function","stateMutability":"view",
                        "inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],
                        "outputs":[{"name":"","type":"uint256"}]
                    }]
                )
                allowance = usdc_contract.functions.allowance(
                    account.address, Web3.to_checksum_address(EXCHANGE_ADDRESS)
                ).call() / 1e6
                
                if allowance < 100: 
                    logger.info(f"🔑 Allowance baixa ({allowance:.2f}). A aprovar...")
                    if _approve_usdc_onchain(self._key):
                        logger.info("✓ Approve OK!")
                    else:
                        logger.error("✗ Approve FALHOU. Verifica se tens MATIC.")
            except Exception as e:
                logger.error(f"Erro ao verificar allowance: {e}")

            # 4. Depositar se necessário
            if wallet_balance > 1.0 and clob_balance < 1.0:
                logger.warning(f"⚠  CLOB vazio! A tentar depositar $20.00...")
                logger.warning(f"⚠  NOTA: Isto requer MATIC para gas.")
                
                try:
                    deposit_ok = self.deposit_to_clob(20.0)
                    if deposit_ok:
                        logger.info("✅ DEPÓSITO REALIZADO COM SUCESSO!")
                        time.sleep(5)
                    else:
                        logger.error("❌ DEPÓSITO FALHOU. O bot vai continuar mas as ordens irão falhar.")
                except Exception as e:
                    logger.error(f"❌ Exceção no depósito: {e}")
            elif clob_balance >= 1.0:
                logger.info("✅ Fundos no CLOB suficientes.")

        except Exception as e:
            logger.error(f"❌ Erro geral na gestão de fundos (Bot vai iniciar mesmo assim): {e}")

    def _save_creds(self, creds):
        CREDS_FILE.write_text(json.dumps({
            "api_key": creds.api_key, "api_secret": creds.api_secret, "api_passphrase": creds.api_passphrase
        }, indent=2))

    def deposit_to_clob(self, amount: float) -> bool:
        from web3 import Web3
        from eth_account import Account
        
        w3 = _get_web3()
        if not w3: raise Exception("Sem conexão Web3")
        
        account = Account.from_key(self._key)
        usdc_addr = Web3.to_checksum_address(USDC_ADDRESS)
        exchange_addr = Web3.to_checksum_address(EXCHANGE_ADDRESS)
        
        abi_transfer = [{
            "name":"transfer","type":"function","inputs":[
                {"name":"to","type":"address"},{"name":"value","type":"uint256"}
            ],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"
        }]
        
        contract = w3.eth.contract(address=usdc_addr, abi=abi_transfer)
        amount_raw = int(amount * 10**6)
        
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Estimar gas
        try:
            gas_est = contract.functions.transfer(exchange_addr, amount_raw).estimate_gas({"from": account.address})
        except Exception as e:
            raise Exception(f"Falha ao estimar gas (talvez saldo insuficiente): {e}")

        tx = contract.functions.transfer(exchange_addr, amount_raw).build_transaction({
            "from": account.address, "nonce": nonce,
            "gas": int(gas_est * 1.2),
            "gasPrice": w3.eth.gas_price, "chainId": 137
        })
        
        signed = w3.eth.account.sign_transaction(tx, self._key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        
        logger.info(f"Tx Hash: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        
        if receipt.status == 1:
            return True
        else:
            raise Exception("Transação falhou na blockchain (status 0)")

    def get_balance(self) -> float | None:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        try:
            for sig in [2, 1, 0]:
                try:
                    info = self._client.get_balance_allowance(
                        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig)
                    )
                    bal = int(info.get("balance", 0)) / 1e6
                    if bal > 0: return bal
                except: pass
            return 0.0
        except: return None

    def get_orderbook(self, token_id: str):
        try: return self._client.get_order_book(token_id)
        except: return None

    def get_best_prices(self, token_id: str) -> dict:
        book = self.get_orderbook(token_id)
        if not book: return {"bid": None, "ask": None, "spread": None}
        bids = sorted(book.bids or [], key=lambda x: -float(x.price))
        asks = sorted(book.asks or [], key=lambda x:  float(x.price))
        bid  = float(bids[0].price) if bids else None
        ask  = float(asks[0].price) if asks else None
        spr  = round(ask - bid, 4) if (bid and ask) else None
        return {"bid": bid, "ask": ask, "spread": spr}

    def get_open_orders(self) -> list:
        try:
            from py_clob_client.clob_types import OpenOrderParams
            return self._client.get_orders(OpenOrderParams())
        except: return []

    def buy(self, token_id: str, price: float, size_usdc: float, label: str = "", order_type: str = "FOK") -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType
        
        if order_type.upper() == "FOK":
            shares = math.floor(size_usdc / price)
            _otype = OrderType.FOK
        else:
            shares = round(size_usdc / price, 4)
            _otype = OrderType.GTC

        try:
            order_args = OrderArgs(price=round(price, 4), size=shares, side="BUY", token_id=token_id)
            signed   = self._client.create_order(order_args)
            response = self._client.post_order(signed, _otype)

            order_id = response.get("orderID") or response.get("id") or "?"
            status   = response.get("status", "unknown")
            success  = response.get("success", False) and status in ("matched", "live", "delayed")

            return {
                "success": success, "order_id": order_id, "status": status,
                "price": price, "size_usdc": size_usdc, "shares": shares,
                "label": label, "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Buy falhou: {e}")
            return {"success": False, "error": str(e), "status": "error"}

    def sell(self, token_id: str, price: float, shares: float, label: str = "") -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType
        try:
            order_args = OrderArgs(price=round(price, 4), size=shares, side="SELL", token_id=token_id)
            signed   = self._client.create_order(order_args)
            response = self._client.post_order(signed, OrderType.GTC)
            return {"success": True, "status": response.get("status"), "order_id": response.get("id")}
        except Exception as e:
            return {"success": False, "error": str(e), "status": "error"}

    def cancel(self, order_id: str) -> bool:
        try: self._client.cancel(order_id); return True
        except: return False

def _now(): return datetime.now().isoformat()
def _today(): return datetime.now().date().isoformat()

def paper_buy(token_id: str, price: float, size_usdc: float, label: str = "") -> dict:
    return {
        "success": True, "simulated": True, "order_id": f"PAPER-{int(time.time())}",
        "status": "SIMULATED", "side": "BUY", "token_id": token_id,
        "price": round(price, 4), "size_usdc": round(size_usdc, 2),
        "shares": round(size_usdc / price, 2), "label": label, "timestamp": _now()
    }
