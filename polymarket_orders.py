"""
polymarket_orders.py ??
====================
Execução de ordens no Polymarket CLOB com gestão automática de depósitos.
Versão corrigida: suporte a chaves com/sem 0x, sig_type=2 prioritário, allowance automático.
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

# Endereços na Polygon
USDC_ADDRESS     = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (Bridged)
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # CTF Exchange
APPROVE_AMOUNT   = 10_000_000 * 10**6  # $10M allowance (suficiente)

# RPCs alternativos (evitar problemas de DNS)
POLYGON_RPCS = [
    "https://rpc.ankr.com/polygon",
    "https://polygon-rpc.com",
    "https://1rpc.io/matic",
    "https://polygon.blockpi.network/v1/rpc/public",
    "https://polygon-mainnet.g.alchemy.com/v2/demo",
]

def _get_web3():
    """Obtém conexão Web3 funcional"""
    from web3 import Web3
    for rpc in POLYGON_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                logger.info(f"Conectado a RPC: {rpc}")
                return w3
        except Exception as e:
            logger.debug(f"Falha RPC {rpc}: {e}")
            continue
    logger.error("Não foi possível conectar a nenhum RPC Polygon")
    return None

def _ensure_0x_prefix(key: str) -> str:
    """Garante que a chave privada tem prefixo 0x"""
    if not key:
        return key
    key = str(key).strip()
    if not key.startswith('0x'):
        key = '0x' + key
    return key

def _approve_usdc_onchain(private_key: str, amount_usd: float = 1_000_000) -> bool:
    """
    Aprova o contrato da Polymarket a gastar USDC da wallet.
    amount_usd: montante em USD (default $1M - suficiente para todas as trades)
    """
    from web3 import Web3
    from eth_account import Account
    
    private_key = _ensure_0x_prefix(private_key)
    
    ABI_APPROVE = [{
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"}
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    }]

    account = Account.from_key(private_key)
    spender = Web3.to_checksum_address(EXCHANGE_ADDRESS)
    usdc    = Web3.to_checksum_address(USDC_ADDRESS)
    
    w3 = _get_web3()
    if not w3:
        logger.error("Não foi possível conectar a nenhum RPC para approve")
        return False

    try:
        contract = w3.eth.contract(address=usdc, abi=ABI_APPROVE)
        nonce = w3.eth.get_transaction_count(account.address)
        
        amount_raw = int(amount_usd * 10**6)
        
        tx = contract.functions.approve(spender, amount_raw).build_transaction({
            "from": account.address,
            "nonce": nonce,
            "gas": 100_000,
            "gasPrice": w3.eth.gas_price,
            "chainId": 137,
        })
        
        logger.info(f"Approve tx: spender={spender}, amount=${amount_usd:,.0f}")
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Approve tx enviada: %s", tx_hash.hex())
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
        if receipt.status == 1:
            logger.info("✅ Approve confirmado na blockchain")
            return True
        else:
            logger.error("❌ Approve falhou na blockchain")
            return False
            
    except Exception as e:
        logger.error("Approve falhou: %s", e)
        return False


class OrderExecutor:
    def __init__(self, private_key: str):
        """Inicializa o executor de ordens com a chave privada"""
        # Garantir prefixo 0x na chave
        self._key = _ensure_0x_prefix(private_key)
        if not self._key or len(self._key) < 66:
            raise ValueError(f"POLY_PRIVATE_KEY inválida (comprimento: {len(self._key) if self._key else 0})")
        
        logger.info(f"Inicializando OrderExecutor com chave: {self._key[:10]}...{self._key[-6:]}")
        self._client = self._init_client()

    def _init_client(self):
        """Inicializa o cliente CLOB com credenciais e gestão de fundos"""
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, OpenOrderParams, BalanceAllowanceParams, AssetType

        client = ClobClient(host=CLOB_HOST, key=self._key, chain_id=CHAIN_ID)
        CREDS_FILE.parent.mkdir(exist_ok=True)

        # ── Gestão de credenciais API ───────────────────────────────────────
        if CREDS_FILE.exists():
            try:
                saved = json.loads(CREDS_FILE.read_text())
                creds = ApiCreds(saved["api_key"], saved["api_secret"], saved["api_passphrase"])
                client.set_api_creds(creds)
                # Testar conexão
                client.get_orders(OpenOrderParams())
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

        # ── GESTÃO DE FUNDOS ────────────────────────────────────────────────
        
        # 1. Verificar Allowance on-chain
        from web3 import Web3
        from eth_account import Account
        
        w3 = _get_web3()
        if w3:
            try:
                account = Account.from_key(self._key)
                usdc_contract = w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_ADDRESS),
                    abi=[{
                        "name": "allowance",
                        "type": "function",
                        "stateMutability": "view",
                        "inputs": [
                            {"name": "owner", "type": "address"},
                            {"name": "spender", "type": "address"}
                        ],
                        "outputs": [{"name": "", "type": "uint256"}]
                    }]
                )
                allowance = usdc_contract.functions.allowance(
                    account.address,
                    Web3.to_checksum_address(EXCHANGE_ADDRESS)
                ).call() / 1e6
                
                logger.info(f"Allowance atual: ${allowance:,.2f}")
                
                # Se allowance for pequena (< $100), aprovar mais
                if allowance < 100:
                    logger.info(f"Allowance baixa (${allowance:,.2f}). A aprovar $1M...")
                    if _approve_usdc_onchain(self._key, amount_usd=1_000_000):
                        logger.info("✅ Allowance aprovada com sucesso")
                        time.sleep(5)  # Aguardar confirmação
                    else:
                        logger.warning("⚠️  Falha no approve. As ordens podem falhar.")
                        
            except Exception as e:
                logger.warning(f"Erro ao verificar allowance: {e}")

        # 2. Verificar Saldo Wallet vs CLOB
        wallet_balance = 0.0
        clob_balance = 0.0
        
        # Saldo na Carteira (On-chain)
        if w3:
            try:
                account = Account.from_key(self._key)
                token = w3.eth.contract(
                    address=Web3.to_checksum_address(USDC_ADDRESS),
                    abi=[{
                        "name": "balanceOf",
                        "type": "function",
                        "inputs": [{"name": "account", "type": "address"}],
                        "outputs": [{"name": "", "type": "uint256"}]
                    }]
                )
                wallet_balance = token.functions.balanceOf(account.address).call() / 1e6
                logger.info(f"Saldo Wallet (USDC.e): ${wallet_balance:.2f}")
            except Exception as e:
                logger.warning(f"Erro ao ler saldo wallet: {e}")

        # Saldo no CLOB (API) - PRIORIDADE sig_type=2
        try:
            # Tentar sig_type=2 primeiro (onde o saldo realmente está)
            info = client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=2
                )
            )
            clob_balance = int(info.get("balance", 0)) / 1e6
            logger.info(f"Saldo CLOB Disponível (sig_type=2): ${clob_balance:.2f}")
            
            # Se falhar, tentar outros tipos
            if clob_balance == 0:
                for sig in [1, 0]:
                    try:
                        info = client.get_balance_allowance(
                            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig)
                        )
                        bal = int(info.get("balance", 0)) / 1e6
                        if bal > clob_balance:
                            clob_balance = bal
                            logger.info(f"Saldo CLOB (sig_type={sig}): ${clob_balance:.2f}")
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Erro ao ler saldo CLOB: {e}")

        # 3. Lógica de Depósito Automático
        if wallet_balance > 5.0 and clob_balance < 5.0:
            logger.warning(f"⚠️  CLOB com saldo baixo (${clob_balance:.2f}) mas Wallet tem fundos (${wallet_balance:.2f})")
            logger.warning("⚠️  A iniciar DEPÓSITO de $25.00 para o CLOB...")
            
            deposit_ok = self.deposit_to_clob(25.0)
            
            if deposit_ok:
                time.sleep(10)  # Aguardar confirmação
                # Verificar novamente
                try:
                    info = client.get_balance_allowance(
                        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
                    )
                    clob_balance = int(info.get("balance", 0)) / 1e6
                    logger.info(f"✓ Depósito confirmado. Novo saldo CLOB: ${clob_balance:.2f}")
                except Exception as e:
                    logger.warning(f"Não foi possível verificar novo saldo: {e}")
            else:
                logger.error("✗ Falha no depósito. Verifica se tens MATIC para gas.")

        return client

    def _save_creds(self, creds):
        """Guarda as credenciais API em disco"""
        CREDS_FILE.write_text(json.dumps({
            "api_key": creds.api_key,
            "api_secret": creds.api_secret,
            "api_passphrase": creds.api_passphrase
        }, indent=2))

    def deposit_to_clob(self, amount: float) -> bool:
        """
        Transfere USDC da Wallet para o contrato Exchange (CLOB).
        Retorna True se sucesso.
        """
        from web3 import Web3
        from eth_account import Account
        
        w3 = _get_web3()
        if not w3:
            logger.error("Sem conexão RPC para depósito")
            return False
        
        account = Account.from_key(self._key)
        usdc_addr = Web3.to_checksum_address(USDC_ADDRESS)
        exchange_addr = Web3.to_checksum_address(EXCHANGE_ADDRESS)
        
        # ABI Transfer
        abi_transfer = [{
            "name": "transfer",
            "type": "function",
            "inputs": [
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"}
            ],
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "nonpayable"
        }]
        
        try:
            contract = w3.eth.contract(address=usdc_addr, abi=abi_transfer)
            amount_raw = int(amount * 10**6)
            
            # Verificar saldo antes
            balance_abi = [{
                "name": "balanceOf",
                "type": "function",
                "inputs": [{"name": "account", "type": "address"}],
                "outputs": [{"name": "", "type": "uint256"}]
            }]
            balance_contract = w3.eth.contract(address=usdc_addr, abi=balance_abi)
            balance = balance_contract.functions.balanceOf(account.address).call() / 1e6
            
            if balance < amount:
                logger.error(f"Saldo insuficiente: ${balance:.2f} < ${amount:.2f}")
                return False
            
            # Verificar MATIC para gas
            matic_balance = w3.eth.get_balance(account.address) / 1e18
            if matic_balance < 0.01:
                logger.error(f"MATIC insuficiente para gas: {matic_balance:.4f} (mínimo 0.01)")
                return False
            
            nonce = w3.eth.get_transaction_count(account.address)
            tx = contract.functions.transfer(exchange_addr, amount_raw).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 150_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137
            })
            
            logger.info(f"Enviando depósito de ${amount:.2f} USDC para CLOB...")
            signed = w3.eth.account.sign_transaction(tx, self._key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            
            logger.info(f"Tx de Depósito enviada: {tx_hash.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                logger.info(f"✓ Depósito de ${amount:.2f} confirmado na blockchain.")
                return True
            else:
                logger.error("❌ Tx de depósito falhou na blockchain.")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao enviar depósito: {e}")
            return False

    def get_balance(self) -> float | None:
        """Obtém o saldo USDC no CLOB (prioriza sig_type=2)"""
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        try:
            # PRIORIDADE: sig_type=2 primeiro (onde o saldo REAL está)
            for sig in [2, 1, 0]:
                try:
                    info = self._client.get_balance_allowance(
                        params=BalanceAllowanceParams(
                            asset_type=AssetType.COLLATERAL,
                            signature_type=sig
                        )
                    )
                    bal = int(info.get("balance", 0)) / 1e6
                    if bal > 0:
                        logger.debug(f"✅ Saldo encontrado em sig_type={sig}: ${bal:.2f}")
                        return bal
                except Exception as e:
                    logger.debug(f"sig_type={sig} falhou: {e}")
                    continue
            return 0.0
        except Exception as e:
            logger.error(f"Erro get_balance: {e}")
            return None

    def get_orderbook(self, token_id: str):
        """Obtém o order book para um token"""
        try:
            return self._client.get_order_book(token_id)
        except Exception as e:
            logger.error(f"Erro get_orderbook: {e}")
            return None

    def get_best_prices(self, token_id: str) -> dict:
        """Obtém o melhor bid/ask para um token"""
        book = self.get_orderbook(token_id)
        if not book:
            return {"bid": None, "ask": None, "spread": None}
        
        bids = sorted(book.bids or [], key=lambda x: -float(x.price))
        asks = sorted(book.asks or [], key=lambda x: float(x.price))
        bid = float(bids[0].price) if bids else None
        ask = float(asks[0].price) if asks else None
        spread = round(ask - bid, 4) if (bid and ask) else None
        return {"bid": bid, "ask": ask, "spread": spread}

    def get_open_orders(self) -> list:
        """Obtém todas as ordens abertas"""
        try:
            from py_clob_client.clob_types import OpenOrderParams
            return self._client.get_orders(OpenOrderParams())
        except Exception as e:
            logger.error(f"Erro get_open_orders: {e}")
            return []

    def buy(self, token_id: str, price: float, size_usdc: float, 
            label: str = "", order_type: str = "FOK") -> dict:
        """
        Compra shares YES de um token.
        
        Args:
            token_id: ID do token no Polymarket
            price: Preço por share (ex: 0.14 para 14¢)
            size_usdc: Montante em USDC a gastar
            label: Label para identificação
            order_type: "FOK" (Fill or Kill) ou "GTC" (Good till cancelled)
        
        Returns:
            dict com resultado da ordem
        """
        from py_clob_client.clob_types import OrderArgs, OrderType
        
        # Verificar saldo antes de comprar
        current_bal = self.get_balance()
        logger.info(f"💰 Saldo atual antes da compra: ${current_bal:.2f}")
        
        if current_bal is not None and current_bal < size_usdc:
            error_msg = f"Saldo CLOB insuficiente: ${current_bal:.2f} < ${size_usdc:.2f}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "status": "error"
            }

        # Calcular shares
        if order_type.upper() == "FOK":
            shares = math.floor(size_usdc / price)
            otype = OrderType.FOK
        else:
            shares = round(size_usdc / price, 4)
            otype = OrderType.GTC

        if shares <= 0:
            error_msg = f"Shares calculadas = 0 (price={price}, size={size_usdc})"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "status": "error"}

        try:
            order_args = OrderArgs(
                price=round(price, 4),
                size=shares,
                side="BUY",
                token_id=token_id
            )
            signed = self._client.create_order(order_args)
            response = self._client.post_order(signed, otype)

            order_id = response.get("orderID") or response.get("id") or "?"
            status = response.get("status", "unknown")
            success = response.get("success", False) and status in ("matched", "live", "delayed")

            if success:
                logger.info(f"✅ Ordem enviada: {label} - ${size_usdc:.2f} @ {price:.4f} - Status: {status}")
            else:
                logger.warning(f"⚠️ Ordem falhou: {label} - {response}")

            return {
                "success": success,
                "order_id": order_id,
                "status": status,
                "price": price,
                "size_usdc": size_usdc,
                "shares": shares,
                "label": label,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Buy falhou: {error_msg}")
            return {"success": False, "error": error_msg, "status": "error"}

    def sell(self, token_id: str, price: float, shares: float, label: str = "") -> dict:
        """
        Vende shares YES de um token.
        
        Args:
            token_id: ID do token no Polymarket
            price: Preço por share
            shares: Número de shares a vender
            label: Label para identificação
        
        Returns:
            dict com resultado da ordem
        """
        from py_clob_client.clob_types import OrderArgs, OrderType
        
        try:
            order_args = OrderArgs(
                price=round(price, 4),
                size=shares,
                side="SELL",
                token_id=token_id
            )
            signed = self._client.create_order(order_args)
            response = self._client.post_order(signed, OrderType.GTC)
            
            success = response.get("success", False)
            if success:
                logger.info(f"✅ Venda enviada: {label} - {shares:.2f} shares @ {price:.4f}")
            else:
                logger.warning(f"⚠️ Venda falhou: {label} - {response}")
                
            return {
                "success": success,
                "status": response.get("status"),
                "order_id": response.get("id"),
                "label": label
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sell falhou: {error_msg}")
            return {"success": False, "error": error_msg, "status": "error"}

    def cancel(self, order_id: str) -> bool:
        """Cancela uma ordem aberta"""
        try:
            self._client.cancel(order_id)
            logger.info(f"Ordem cancelada: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel falhou: {e}")
            return False


# ══════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES PARA PAPER TRADING
# ══════════════════════════════════════════════════════

def _now():
    return datetime.now().isoformat()

def _today():
    return datetime.now().date().isoformat()

def paper_buy(token_id: str, price: float, size_usdc: float, label: str = "") -> dict:
    """
    Simula uma compra (modo PAPER).
    """
    return {
        "success": True,
        "simulated": True,
        "order_id": f"PAPER-{int(time.time())}",
        "status": "SIMULATED",
        "side": "BUY",
        "token_id": token_id,
        "price": round(price, 4),
        "size_usdc": round(size_usdc, 2),
        "shares": round(size_usdc / price, 2) if price > 0 else 0,
        "label": label,
        "timestamp": _now()
    }
