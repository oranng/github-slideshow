# SuperTrend V5 — торговая система

```
trading/
  pine/supertrend_v5.pine      оригинал V5 (как прислан)
  pine/supertrend_v5_1.pine    исправленная версия V5.1
  backtest/                    Python-порт + эмулятор брокера TradingView
  tests/                       тесты механики движка
  docs/REVIEW.md               разбор кода: ошибки и исправления
  docs/results_synthetic.md    прогон на синтетике (проверка движка, НЕ рынок)
```

## Бэктест

```bash
pip install -r trading/requirements.txt

# 1) CSV (экспорт TradingView или Binance klines; 1m или 5m — ресемплится в 5m/15m/30m)
python -m trading.backtest.run --csv trading/data/BTCUSDT.csv trading/data/ETHUSDT.csv

# 2) прямо с биржи через ccxt (нужен сетевой доступ)
python -m trading.backtest.run --ccxt BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT --days 365

# 3) синтетика — проверка, что движок работает
python -m trading.backtest.run --synthetic

python -m pytest -q trading/tests
```

Скрипт печатает таблицу V5 против V5.1 по каждому активу × TF (сделки, net %, winrate, PF, maxDD,
комиссии) и «воронку» фильтров: сколько разворотов SuperTrend отсекает ADX, EMA-bias и BB-фильтр.

Параметры обеих версий — `trading/backtest/strategy.py` (`Params()` = V5, `Params.v51()` = V5.1).

### Семантика движка (как в TradingView)
* сигнал на закрытии бара → рыночный вход по open следующего бара;
* `strategy.exit`, выставленный на закрытии бара, действует в следующем баре;
* внутри бара путь цены: open → ближний экстремум → дальний → close; гэп через стоп или тейк
  исполняется по open;
* комиссия считается от оборота на каждой стороне, проскальзывание — на рыночных и стоп-исполнениях.
