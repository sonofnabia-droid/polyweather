"""
check_funds.py
==============
Diagnóstico completo de fundos na conta Polymarket.
Verifica: saldo CLOB, ordens abertas/históricas, posições, saldo on-chain.

Uso:
    python check_funds.py
"""

import json
import os
import sys
from pathlib import Path

PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY", "")
if not PRIVATE_KEY:
    print("❌  POLY_PRIVATE_KEY não definida.")
    sys.exit(1)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID  = 137

# ── Cores ─────────────────────────────────────────────
G  = "\033[92m"   # verde
Y  = "\033[93m"   # amarelo
R  = "\033[91m"   # vermelho
C  = "\033[96m"   # cyan
DIM = "\033[2m"
B  = "\033[1m"
RS = "\033[0m"

def sep(title=""):
    line = "─" * 56
    if title:
        print(f"\n{B}{C}{line}{RS}")
        print(f"{B}{C}  {title}{RS}")
        print(f"{B}{C}{line}{RS}")
    else:
        print(f"{DIM}{line}{RS}")

# ── 1. Inicializar cliente ────────────────────────────
sep("1. Inicializar cliente CLOB")

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    BalanceAllowanceParams, AssetType, OpenOrderParams
)

client = ClobClient(host=CLOB_HOST, key=PRIVATE_KEY, chain_id=CHAIN_ID)
try:
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    print(f"{G}✓  Credenciais L2 obtidas{RS}")
    print(f"    API key: {DIM}{creds.api_key[:20]}...{RS}")
except Exception as e:
    print(f"{R}✗  Falha nas credenciais: {e}{RS}")
    sys.exit(1)

# ── 2. Saldo CLOB (USDC no contrato) ─────────────────
sep("2. Saldo USDC no CLOB")

for sig_type in [0, 1, 2]:
    try:
        info = client.get_balance_allowance(
            params=BalanceAllowanceParams(
                asset_type     = AssetType.COLLATERAL,
                signature_type = sig_type,
            )
        )
        bal       = int(info.get("balance", "0"))
        allowance = int(info.get("allowance", "0"))
        bal_usdc  = bal / 1e6
        allow_usdc = allowance / 1e6
        color = G if bal_usdc > 0 else DIM
        print(f"  sig_type={sig_type}  balance={color}{B}${bal_usdc:.4f}{RS}  "
              f"allowance={allow_usdc:.2f}  raw={DIM}{info}{RS}")
    except Exception as e:
        print(f"  sig_type={sig_type}  {R}erro: {e}{RS}")

# ── 3. Saldo on-chain (Polygon) ───────────────────────
sep("3. Saldo USDC on-chain (Polygon)")

from eth_account import Account
from web3 import Web3

eoa = Account.from_key(PRIVATE_KEY).address
print(f"  EOA: {eoa}")

USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
USDC_E      = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
ERC20_ABI = [
    {"name":"balanceOf","type":"function","stateMutability":"view",
     "inputs":[{"name":"account","type":"address"}],
     "outputs":[{"name":"","type":"uint256"}]},
    {"name":"allowance","type":"function","stateMutability":"view",
     "inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],
     "outputs":[{"name":"","type":"uint256"}]},
]

for rpc in ["https://rpc.ankr.com/polygon", "https://polygon.llamarpc.com"]:
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        if not w3.is_connected():
            continue

        matic = w3.eth.get_balance(eoa)
        print(f"  MATIC:         {matic/1e18:.4f}  (gas)")

        for label, addr in [("USDC native", USDC_NATIVE), ("USDC.e", USDC_E)]:
            try:
                token = w3.eth.contract(
                    address=Web3.to_checksum_address(addr), abi=ERC20_ABI
                )
                bal = token.functions.balanceOf(Web3.to_checksum_address(eoa)).call()
                allow = token.functions.allowance(
                    Web3.to_checksum_address(eoa),
                    Web3.to_checksum_address(CTF_EXCHANGE)
                ).call()
                color = G if bal > 0 else DIM
                print(f"  {label:14s}: {color}{B}${bal/1e6:.4f}{RS}  "
                      f"allowance CTF={allow/1e6:.2f}")
            except Exception as e:
                print(f"  {label}: {R}{e}{RS}")
        break
    except Exception as e:
        print(f"  RPC {rpc}: {R}{e}{RS}")

# ── 4. Ordens abertas ─────────────────────────────────
sep("4. Ordens abertas (CLOB)")

try:
    open_orders = client.get_orders(OpenOrderParams())
    if not open_orders:
        print(f"  {DIM}Sem ordens abertas.{RS}")
    else:
        print(f"  {Y}{B}{len(open_orders)} ordem(ns) abertas:{RS}")
        for o in open_orders:
            oid    = (o.get("id") or o.get("orderID") or "?")[:20]
            price  = o.get("price", "?")
            size   = o.get("size") or o.get("originalSize") or "?"
            status = o.get("status", "?")
            asset  = (o.get("asset_id") or o.get("token_id") or "")[:16]
            print(f"    {oid}...  price={Y}{price}{RS}  size={size}  "
                  f"status={status}  asset=...{asset}")
except Exception as e:
    print(f"  {R}Erro: {e}{RS}")

# ── 5. Histórico de ordens (últimas 20) ───────────────
sep("5. Histórico de ordens (últimas 20)")

try:
    # Tenta get_orders sem filtro de status
    all_orders = client.get_orders()
    if isinstance(all_orders, dict):
        all_orders = all_orders.get("data", [])
    if not all_orders:
        print(f"  {DIM}Sem histórico de ordens acessível.{RS}")
    else:
        print(f"  {len(all_orders)} ordem(ns) no histórico:")
        for o in all_orders[-20:]:
            oid    = (o.get("id") or o.get("orderID") or "?")[:20]
            price  = o.get("price", "?")
            size   = o.get("size") or o.get("originalSize") or "?"
            status = o.get("status", "?")
            ts     = (o.get("createdAt") or o.get("timestamp") or "")[:19]
            color  = G if status == "matched" else (R if status in ("canceled","cancelled") else Y)
            print(f"    {DIM}{ts}{RS}  {oid}...  "
                  f"price={price}  size={size}  status={color}{status}{RS}")
except Exception as e:
    print(f"  {R}Erro: {e}{RS}")

# ── 6. Posições (via Gamma API) ───────────────────────
sep("6. Posições resolvidas (Gamma API)")

import requests

try:
    r = requests.get(
        "https://gamma-api.polymarket.com/positions",
        params={"user": eoa, "limit": 20},
        timeout=15,
    )
    r.raise_for_status()
    positions = r.json()
    if isinstance(positions, dict):
        positions = positions.get("positions") or positions.get("data") or []
    if not positions:
        print(f"  {DIM}Sem posições encontradas.{RS}")
    else:
        print(f"  {len(positions)} posição(ões):")
        total_value = 0.0
        for p in positions:
            title   = str(p.get("title") or p.get("market") or "?")[:40]
            outcome = p.get("outcome") or p.get("side") or "?"
            size    = float(p.get("size") or p.get("shares") or 0)
            price   = float(p.get("currentPrice") or p.get("price") or 0)
            value   = size * price
            total_value += value
            resolved = p.get("resolved") or p.get("isResolved")
            won      = p.get("won") or p.get("isWinner")
            if resolved:
                r_str = f"{G}WON{RS}" if won else f"{R}LOST{RS}"
            else:
                r_str = f"{Y}open{RS}"
            print(f"    {DIM}{title}{RS}")
            print(f"      outcome={outcome}  shares={size:.2f}  "
                  f"price={price:.4f}  value=${value:.2f}  {r_str}")
        print(f"\n  Total value estimado: {B}${total_value:.2f}{RS}")
except Exception as e:
    print(f"  {R}Erro Gamma API: {e}{RS}")

# ── 7. Posições via CLOB API ──────────────────────────
sep("7. Posições via CLOB API")

for endpoint in ["/positions", "/portfolio"]:
    try:
        import requests as _req
        resp = _req.get(
            f"https://clob.polymarket.com{endpoint}",
            headers={
                "POLY-API-KEY":        creds.api_key,
                "POLY-API-SECRET":     creds.api_secret,
                "POLY-API-PASSPHRASE": creds.api_passphrase,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  {endpoint}: {json.dumps(data, indent=2)[:500]}")
        else:
            print(f"  {endpoint}: {DIM}HTTP {resp.status_code}{RS}")
    except Exception as e:
        print(f"  {endpoint}: {R}{e}{RS}")

# ── 8. Logs locais ────────────────────────────────────
sep("8. Logs locais (live_bot_logs/)")

log_dir = Path("live_bot_logs")
if log_dir.exists():
    order_logs = sorted(log_dir.glob("orders_*.json"))
    for lf in order_logs[-3:]:   # últimos 3 dias
        try:
            orders = json.loads(lf.read_text())
            total_spent = sum(o.get("size_usdc", 0) for o in orders
                              if o.get("success"))
            print(f"  {lf.name}: {len(orders)} ordens  "
                  f"total gasto: {Y}${total_spent:.2f}{RS}")
            for o in orders:
                oid    = str(o.get("order_id") or "?")[:20]
                status = o.get("status", "?")
                size   = o.get("size_usdc", 0)
                label  = o.get("bracket_label") or o.get("label") or ""
                ts     = str(o.get("timestamp", ""))[:19]
                success = o.get("success", False)
                color = G if success else R
                print(f"    {DIM}{ts}{RS}  {oid}...  "
                      f"{label:12s}  ${size:.2f}  "
                      f"status={color}{status}{RS}")
        except Exception as e:
            print(f"  {lf.name}: {R}{e}{RS}")
else:
    print(f"  {DIM}live_bot_logs/ não encontrado{RS}")

sep()
print(f"\n{B}Diagnóstico completo.{RS}")
print(f"{DIM}Se os fundos não aparecem em nenhum lado, pode ser necessário")
print(f"contactar o suporte Polymarket com os order IDs dos logs acima.{RS}\n")
