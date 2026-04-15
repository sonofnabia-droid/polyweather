# check_matic.py
import os
from web3 import Web3
from eth_account import Account

private_key = "a802991f3acc76f930c673d913c099d97b2d2e0fbff5f1ce3b4e28f23fa442b5"
if not private_key.startswith('0x'):
    private_key = '0x' + private_key

account = Account.from_key(private_key)
print(f"Wallet: {account.address}")

for rpc in ["https://rpc.ankr.com/polygon", "https://polygon-rpc.com"]:
    try:
        w3 = Web3(Web3.HTTPProvider(rpc))
        if w3.is_connected():
            balance = w3.eth.get_balance(account.address) / 1e18
            print(f"MATIC balance: {balance:.6f} MATIC")
            break
    except:
        pass
