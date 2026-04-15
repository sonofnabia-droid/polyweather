# diagnose.py
import os
import sys

print("=" * 50)
print("DIAGNÓSTICO - Modo REAL")
print("=" * 50)

# 1. Verificar POLY_PRIVATE_KEY
private_key = os.environ.get("POLY_PRIVATE_KEY")
if not private_key:
    print("❌ POLY_PRIVATE_KEY não encontrada nas variáveis de ambiente")
    print("   Execute: export POLY_PRIVATE_KEY='0x...'")
    sys.exit(1)
else:
    print(f"✅ POLY_PRIVATE_KEY encontrada: {private_key[:10]}...{private_key[-6:]}")
    print(f"   Comprimento: {len(private_key)} caracteres")

# 2. Verificar formato da chave
if not private_key.startswith("0x"):
    print("⚠️  A chave não começa com '0x' - pode ser problema")
else:
    print("✅ Formato OK (começa com 0x)")

# 3. Testar importações
try:
    from web3 import Web3
    print("✅ web3 importado")
except ImportError as e:
    print(f"❌ web3 não instalado: {e}")

try:
    from eth_account import Account
    print("✅ eth_account importado")
except ImportError as e:
    print(f"❌ eth_account não instalado: {e}")

try:
    from py_clob_client.client import ClobClient
    print("✅ py_clob_client importado")
except ImportError as e:
    print(f"❌ py_clob_client não instalado: {e}")

# 4. Testar conexão RPC
try:
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider("https://polygon.llamarpc.com", request_kwargs={"timeout": 10}))
    if w3.is_connected():
        print("✅ Conexão RPC Polygon OK")
        print(f"   Chain ID: {w3.eth.chain_id}")
    else:
        print("❌ Falha na conexão RPC")
except Exception as e:
    print(f"❌ Erro RPC: {e}")

# 5. Tentar criar OrderExecutor
try:
    from polymarket_orders import OrderExecutor
    print("\nTentando criar OrderExecutor...")
    executor = OrderExecutor(private_key)
    balance = executor.get_balance()
    print(f"✅ OrderExecutor criado com sucesso!")
    print(f"   Saldo USDC: ${balance:.2f}")
except Exception as e:
    print(f"❌ Falha ao criar OrderExecutor: {e}")
    import traceback
    traceback.print_exc()
