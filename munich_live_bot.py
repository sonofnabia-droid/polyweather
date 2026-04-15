"""
polymarket_orders.py
====================
Execução de ordens no Polymarket CLOB com gestão automática de depósitos.
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

# Endereços na Polygon (Baseado no teu script check_funds.py)
USDC_ADDRESS     = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (Bridged)
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # CTF Exchange
APPROVE_AMOUNT   = 1_000_000_000 * 10**6

POLYGON_RPCS = [
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://1rpc.io/matic",
    "https://polygon-rpc.com",
    "https://polygon-mainnet.g.alchemy.com/v2/demo",
]

def _get_web3():
    from web3 import Web3
    for rpc in POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                logger.info(f"Conectado a RPC: {rpc}")
                return w3
        except Exception:
            continue
    return None

def _ensure_0x_prefix(key: str) -> str:
    """Garante que a chave privada tem prefixo 0x"""
    if not key:
        return key
    key = key.strip()
    if not key.startswith('0x'):
        key = '0x' + key
    return key

def _approve_usdc_onchain(private_key: str) -> bool:
    from web3 import Web3
    from eth_account import Account
    
    private_key = _ensure_0x_prefix(private_key)

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
        logger.error("Não foi possível conectar a nenhum RPC")
        return False

    try:
        contract = w3.eth.contract(address=usdc, abi=ABI_APPROVE)
        nonce    = w3.eth.get_transaction_count(account.address)
        tx = contract.functions.approve(spender, APPROVE_AMOUNT).build_transaction({
            "from": account.address, "nonce": nonce,
            "gas": 100_000, "gasPrice": w3.eth.gas_price, "chainId": 137,
        })
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Approve tx enviada: %s", tx_hash.hex())
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
        return receipt.status == 1
    except Exception as e:
        logger.error("Approve falhou: %s", e)
        return False

class OrderExecutor:
    def __init__(self, private_key: str):
        # Garantir prefixo 0x na chave
        self._key = _ensure_0x_prefix(private_key)
        if not self._key or len(self._key) < 66:
            raise ValueError(f"POLY_PRIVATE_KEY inválida (comprimento: {len(self._key) if self._key else 0})")
        
        logger.info(f"Inicializando OrderExecutor com chave: {self._key[:10]}...{self._key[-6:]}")
        self._client = self._init_client()

    def _init_client(self):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, OpenOrderParams, BalanceAllowanceParams, AssetType

        client = ClobClient(host=CLOB_HOST, key=self._key, chain_id=CHAIN_ID)
        CREDS_FILE.parent.mkdir(exist_ok=True)

        # Creds
        if CREDS_FILE.exists():
            try:
                saved = json.loads(CREDS_FILE.read_text())
                creds = ApiCreds(saved["api_key"], saved["api_secret"], saved["api_passphrase"])
                client.set_api_creds(creds)
                client.get_orders(OpenOrderParams()) # Test connection
                logger.info("Credenciais carregadas do ficheiro")
            except Exception as e:
                logger.warning(f"Erro ao carregar credenciais: {e}. A criar novas...")
                CREDS_FILE.unlink(missing_ok=True)
                creds = client.create_or_derive_api_creds()
                client.set_api_creds(creds)
                self._save_creds(creds)
        else:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            self._save_creds(creds)

        # ── GESTÃO DE FUNDOS (SOLUÇÃO DO ERRO balance: 0) ───────────────────────
        
        # 1. Verificar Allowance
        from web3 import Web3
        from eth_account import Account
        w3 = _get_web3()
        if w3:
            try:
                account = Account.from_key(self._key)
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
                
                if allowance < 100: # Se allowance for pequena, aprovar mais
                    logger.info("Allowance baixa (%.2f). A aprovar...", allowance)
                    _approve_usdc_onchain(self._key)
                    time.sleep(5)
            except Exception as e:
                logger.warning("Erro ao verificar allowance: %s", e)

        # 2. Verificar Saldo Wallet vs CLOB
        wallet_balance = 0.0
        clob_balance = 0.0
        
        # Saldo na Carteira (On-chain)
        if w3:
            try:
                account = Account.from_key(self._key)
                token = w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_ADDRESS),
                    abi=[{"name":"balanceOf","type":"function","inputs":[{"name":"account","type":"address"}],"outputs":[{"name":"","type":"uint256"}]}]
                )
                wallet_balance = token.functions.balanceOf(account.address).call() / 1e6
                logger.info(f"Saldo Wallet (USDC.e): ${wallet_balance:.2f}")
            except Exception as e:
                logger.warning("Erro ao ler saldo wallet: %s", e)

        # Saldo no CLOB (API)
        try:
            # Tenta sig_type 2 (comum) e 1
            for sig in [2, 1]:
                try:
                    info = client.get_balance_allowance(params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig))
                    bal = int(info.get("balance", 0)) / 1e6
                    if bal > clob_balance:
                        clob_balance = bal
                except: pass
            logger.info(f"Saldo CLOB Disponível: ${clob_balance:.2f}")
        except Exception as e:
            logger.warning("Erro ao ler saldo CLOB: %s", e)

        # 3. Lógica de Depósito Automático
        # Se tens dinheiro na wallet mas não no CLOB, transfere.
        if wallet_balance > 1.0 and clob_balance < 1.0:
            logger.warning(f"⚠  CLOB vazio (${clob_balance:.2f}) mas Wallet tem fundos (${wallet_balance:.2f}).")
            logger.warning(f"⚠  A iniciar DEPÓSITO de $20.00 para o CLOB...")
            
            deposit_ok = self.deposit_to_clob(20.0)
            
            if deposit_ok:
                time.sleep(10) # Espera pela blockchain
                # Verificar novamente
                try:
                    info = client.get_balance_allowance(params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2))
                    clob_balance = int(info.get("balance", 0)) / 1e6
                    logger.info(f"✓ Depósito confirmado. Novo saldo CLOB: ${clob_balance:.2f}")
                except:
                    pass
            else:
                logger.error("✗ Falha no depósito. Verifica se tens MATIC para gas.")

        return client

    def _save_creds(self, creds):
        CREDS_FILE.write_text(json.dumps({
            "api_key": creds.api_key, "api_secret": creds.api_secret, "api_passphrase": creds.api_passphrase
        }, indent=2))

    def deposit_to_clob(self, amount: float) -> bool:
        """
        Transfere USDC da Wallet para o contrato Exchange (CLOB).
        """
        from web3 import Web3
        from eth_account import Account
        
        w3 = _get_web3()
        if not w3: return False
        
        account = Account.from_key(self._key)
        usdc_addr = Web3.to_checksum_address(USDC_ADDRESS)
        exchange_addr = Web3.to_checksum_address(EXCHANGE_ADDRESS)
        
        # ABI Transfer
        abi_transfer = [{
            "name":"transfer","type":"function","inputs":[
                {"name":"to","type":"address"},{"name":"value","type":"uint256"}
            ],"outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable"
        }]
        
        try:
            contract = w3.eth.contract(address=usdc_addr, abi=abi_transfer)
            amount_raw = int(amount * 10**6)
            
            nonce = w3.eth.get_transaction_count(account.address)
            tx = contract.functions.transfer(exchange_addr, amount_raw).build_transaction({
                "from": account.address, "nonce": nonce,
                "gas": 150_000, "gasPrice": w3.eth.gas_price, "chainId": 137
            })
            
            signed = w3.eth.account.sign_transaction(tx, self._key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            
            logger.info(f"Tx de Depósito enviada: {tx_hash.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                logger.info(f"✓ Depósito de ${amount:.2f} confirmado na blockchain.")
                return True
            else:
                logger.error("Tx de depósito falhou na blockchain.")
                return False
        except Exception as e:
            logger.error(f"Erro ao enviar depósito: {e}")
            return False

    def get_balance(self) -> float | None:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        try:
            # Prioriza sig_type 2
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
        
        # Tenta garantir que há saldo antes de comprar
        current_bal = self.get_balance()
        if current_bal is not None and current_bal < size_usdc:
            logger.error(f"Saldo insuficiente no CLOB (${current_bal:.2f}) para comprar ${size_usdc:.2f}")
            return {
                "success": False, "error": f"Saldo CLOB insuficiente: ${current_bal:.2f}", "status": "error"
            }

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

            logger.info(f"Ordem enviada: {label} - ${size_usdc:.2f} @ {price:.4f} - Status: {status}")
            
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
