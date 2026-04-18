# Strategy Configurations

Este diretório contém as configurações da estratégia usadas pelo backtester e pelo live bot.

## Arquivos

- `default_config.json` - Configurações base (não deve ser modificada manualmente)
- `optimized_config.json` - Configurações otimizadas pelo Optuna (sobrescreve defaults)

## Estrutura

### Stop-Loss
- `temp_threshold` - Limite de temperatura em graus
- `prob_threshold` - Limite de probabilidade (0-1)
- `mode` - "temperature", "probability", "both", "most_restrictive"
- `min_pnl_to_exit` - PnL mínimo para sair (evita sair em perdas pequenas)

### Entry
- `mode` - "single" ou "phased"
- `single_threshold` - Probabilidade mínima para entrada em modo single
- `phased_parcel_size` - Tamanho de cada parcela em modo phased
- `p1/p2/p3_*` - Parâmetros para entrada em 3 fases

### Position
- `bet_size` - Tamanho da aposta em USDC
- `cooldown_minutes` - Intervalo mínimo entre compras
- `max_daily_loss` - Limite de perda diária
- `max_trades_per_day` - Número máximo de trades por dia

## Uso

```bash
# Ver config atual
python munich_strategy_config.py show

# Criar default config
python munich_strategy_config.py create-default

# Usar config específica no backtest
python munich_backtester_unified.py --config strategy_configs/default_config.json

# Usar config específica no live bot
python munich_live_bot_unified.py --config strategy_configs/default_config.json
```

## Otimização

O Optuna salva os melhores parâmetros em `optimized_config.json`. Estes serão usados automaticamente quando disponíveis.
