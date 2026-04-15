# fix_polymarket.py
import os
import time
from web3 import Web3
from eth_account import Account
from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

init(autoreset=True)
console = Console()

# Configuração
PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY", "")
if PRIVATE_KEY and not PRIVATE_KEY.startswith('0x'):
    PRIVATE_KEY = '0x' + PRIVATE_KEY
    console.print(f"[yellow]⚠️  Adicionado prefixo 0x à chave[/yellow]")

USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

console.print(Panel.fit("[bold cyan]🔧 Polymarket Fixer - Allowance & Balance[/bold cyan]", border_style="cyan"))

# ============================
# 1. VERIFICAR SALDO ON-CHAIN
# ============================
console.print("\n[bold]1. Verificando saldo on-chain (Polygon)[/bold]")

account = Account.from_key(PRIVATE_KEY)
wallet_addr = account.address
console.print(f"   Wallet: [green]{wallet_addr}[/green]")

# ABI para balanceOf
balance_abi = [{
    "name": "balanceOf",
    "type": "function",
    "inputs": [{"name": "account", "type": "address"}],
    "outputs": [{"name": "", "type": "uint256"}]
}]

wallet_balance = 0
matic_balance = 0

for rpc in ["https://rpc.ankr.com/polygon", "https://polygon-rpc.com", "https://1rpc.io/matic"]:
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        if w3.is_connected():
            # USDC balance
            usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=balance_abi)
            wallet_balance = usdc.functions.balanceOf(Web3.to_checksum_address(wallet_addr)).call() / 1e6
            
            # MATIC balance
            matic_balance = w3.eth.get_balance(Web3.to_checksum_address(wallet_addr)) / 1e18
            
            console.print(f"   ✅ Conectado a: {rpc}")
            break
    except Exception as e:
        continue

console.print(f"   💰 USDC.e Wallet: [bold green]${wallet_balance:.2f}[/bold green]")
console.print(f"   ⛽ MATIC: [bold yellow]{matic_balance:.4f}[/bold yellow]")

if matic_balance < 0.01:
    console.print(f"[red]❌ MATIC insuficiente para gas! Necessário pelo menos 0.01 MATIC[/red]")
    exit(1)

# ============================
# 2. VERIFICAR SALDO CLOB
# ============================
console.print("\n[bold]2. Verificando saldo no CLOB (sig_types)[/bold]")

client = ClobClient(host="https://clob.polymarket.com", key=PRIVATE_KEY, chain_id=137)
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)

table = Table(title="Estado Atual", show_lines=True)
table.add_column("Signature Type", justify="center", style="cyan")
table.add_column("Balance (USDC)", justify="center")
table.add_column("Allowance (USDC)", justify="center")
table.add_column("Status", justify="center")

clob_balance = 0
allowance = 0

for st in [0, 1, 2]:
    try:
        info = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=st)
        )
        bal = int(info.get("balance", "0")) / 1e6
        allow = int(info.get("allowance", "0")) / 1e6
        
        if bal > 0:
            clob_balance = bal
            allowance = allow
            table.add_row(f"[green]{st}[/green]", f"[bold green]${bal:.2f}[/bold green]", f"${allow:.2f}", "✅ Active")
        else:
            table.add_row(f"[yellow]{st}[/yellow]", f"${bal:.2f}", f"${allow:.2f}", "⚪ Empty")
            
    except Exception as e:
        table.add_row(f"[red]{st}[/red]", "[red]error[/red]", "[red]error[/red]", f"[red]{str(e)[:30]}[/red]")

console.print(table)

# ============================
# 3. FIX ALLOWANCE (se necessário)
# ============================
if allowance < 100 and wallet_balance > 10:
    console.print("\n[bold yellow]⚠️  Allowance baixa. A fazer approve na blockchain...[/bold yellow]")
    
    # ABI para approve
    approve_abi = [{
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"}
        ],
        "outputs": [{"name": "", "type": "bool"}]
    }]
    
    w3 = Web3(Web3.HTTPProvider("https://rpc.ankr.com/polygon"))
    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=approve_abi)
    
    amount_raw = 10_000_000 * 10**6  # $10M
    nonce = w3.eth.get_transaction_count(account.address)
    
    tx = usdc.functions.approve(
        Web3.to_checksum_address(EXCHANGE_ADDRESS),
        amount_raw
    ).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 100000,
        "gasPrice": w3.eth.gas_price,
        "chainId": 137
    })
    
    console.print(f"   📝 Approve amount: $10,000,000 USDC")
    console.print(f"   🔑 Nonce: {nonce}")
    
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    console.print(f"   ✅ Tx enviada: [green]{tx_hash.hex()}[/green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("   ⏳ Aguardando confirmação...", total=None)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        progress.remove_task(task)
    
    if receipt.status == 1:
        console.print(f"   [bold green]✅ Approve confirmado![/bold green]")
        time.sleep(3)
    else:
        console.print(f"   [red]❌ Approve falhou[/red]")
        exit(1)

elif allowance >= 100:
    console.print(f"\n[green]✅ Allowance OK: ${allowance:,.2f}[/green]")
else:
    console.print(f"\n[red]❌ Saldo wallet insuficiente para fazer approve[/red]")

# ============================
# 4. UPDATE BALANCE ALLOWANCE (forçar refresh)
# ============================
console.print("\n[bold]3. Forçando refresh do balance/allowance no CLOB[/bold]")

for asset in [AssetType.COLLATERAL, AssetType.CONDITIONAL]:
    for st in [0, 1, 2]:
        try:
            client.update_balance_allowance(
                params=BalanceAllowanceParams(asset_type=asset, signature_type=st)
            )
            console.print(f"   [green]✓[/green] {asset} signature_type={st}")
        except Exception as e:
            console.print(f"   [red]✗[/red] {asset} signature_type={st} - {str(e)[:50]}")

time.sleep(2)

# ============================
# 5. VERIFICAR NOVO ESTADO
# ============================
console.print("\n[bold]4. Estado após correções[/bold]")

for st in [0, 1, 2]:
    try:
        info = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=st)
        )
        bal = int(info.get("balance", "0")) / 1e6
        allow = int(info.get("allowance", "0")) / 1e6
        
        if bal > 0:
            console.print(f"   [green]✅ sig_type={st}:[/green] balance=${bal:.2f}, allowance=${allow:.2f}")
        else:
            console.print(f"   [yellow]⚪ sig_type={st}:[/yellow] balance=${bal:.2f}, allowance=${allow:.2f}")
            
    except Exception as e:
        console.print(f"   [red]❌ sig_type={st}: {e}[/red]")

# ============================
# 6. DEPÓSITO AUTOMÁTICO (se necessário)
# ============================
if wallet_balance > 10 and clob_balance < 5:
    console.print(f"\n[bold yellow]⚠️  CLOB com saldo baixo (${clob_balance:.2f}). A depositar $25...[/bold yellow]")
    
    w3 = Web3(Web3.HTTPProvider("https://rpc.ankr.com/polygon"))
    
    transfer_abi = [{
        "name": "transfer",
        "type": "function",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"}
        ],
        "outputs": [{"name": "", "type": "bool"}]
    }]
    
    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=transfer_abi)
    amount = 25.0
    amount_raw = int(amount * 10**6)
    
    nonce = w3.eth.get_transaction_count(account.address)
    tx = usdc.functions.transfer(
        Web3.to_checksum_address(EXCHANGE_ADDRESS),
        amount_raw
    ).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 150000,
        "gasPrice": w3.eth.gas_price,
        "chainId": 137
    })
    
    console.print(f"   📝 Depositando ${amount} USDC para CLOB...")
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    console.print(f"   ✅ Tx enviada: [green]{tx_hash.hex()}[/green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("   ⏳ Aguardando confirmação...", total=None)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        progress.remove_task(task)
    
    if receipt.status == 1:
        console.print(f"   [bold green]✅ Depósito confirmado![/bold green]")
        time.sleep(5)
        
        # Verificar novo saldo
        info = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
        )
        new_balance = int(info.get("balance", 0)) / 1e6
        console.print(f"   💰 Novo saldo CLOB: [bold green]${new_balance:.2f}[/bold green]")
    else:
        console.print(f"   [red]❌ Depósito falhou[/red]")

# ============================
# 7. TESTE FINAL
# ============================
console.print("\n[bold]5. Teste final - saldo disponível para trading[/bold]")

info = client.get_balance_allowance(
    params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
)
final_balance = int(info.get("balance", 0)) / 1e6
final_allowance = int(info.get("allowance", 0)) / 1e6

console.print(f"   💰 Saldo CLOB: [bold green]${final_balance:.2f}[/bold green]")
console.print(f"   🔓 Allowance: [bold green]${final_allowance:,.2f}[/bold green]")

if final_balance >= 5 and final_allowance >= 5:
    console.print(Panel.fit("[bold green]✅ Tudo pronto! O bot já pode fazer trades reais.[/bold green]", border_style="green"))
    console.print("\n[bold]Para executar o bot:[/bold]")
    console.print("   [cyan]echo \"R\" | python munich_live_bot.py --yes[/cyan]")
else:
    console.print(Panel.fit("[bold red]❌ Ainda há problemas. Verifique manualmente.[/bold red]", border_style="red"))

console.print("\n[dim]Feito![/dim]")
