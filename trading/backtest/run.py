"""CLI: прогон V5 (оригинал) и V5.1 (исправленная) по активам и таймфреймам 5m/15m/30m.

Примеры:
  python -m trading.backtest.run --csv data/BTCUSDT_1m.csv data/ETHUSDT_1m.csv   # 1m CSV -> ресемплинг
  python -m trading.backtest.run --ccxt BTC/USDT ETH/USDT SOL/USDT --days 180
  python -m trading.backtest.run --synthetic                                       # проверка механики
"""
import argparse
from pathlib import Path

import pandas as pd

from . import data
from .engine import run
from .strategy import Params

TFS = ["5m", "15m", "30m"]


def flip_funnel(sig: pd.DataFrame) -> dict:
    """Сколько разворотов SuperTrend отсекает каждый фильтр (для оригинального входа)."""
    flips = sig["long_flip"] | sig["short_flip"]
    n = int(flips.sum())
    bias = (sig["long_flip"] & sig["f_bias_long"]) | (sig["short_flip"] & sig["f_bias_short"])
    return {
        "flips": n,
        "adx_ok_%": 100 * (flips & sig["f_adx"]).sum() / max(n, 1),
        "bias_ok_%": 100 * bias.sum() / max(n, 1),
        "vol_ok_%": 100 * (flips & sig["f_vol"]).sum() / max(n, 1),
        "all_ok_%": 100 * (sig["final_long"] | sig["final_short"]).sum() / max(n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", nargs="+", help="1m или 5m CSV по каждому активу")
    g.add_argument("--ccxt", nargs="+", help="символы ccxt, напр. BTC/USDT")
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--out", default="trading/docs/results.md")
    a = ap.parse_args()

    assets = {}
    if a.csv:
        for f in a.csv:
            assets[Path(f).stem] = ("csv", data.load_csv(f))
    elif a.ccxt:
        for s in a.ccxt:
            assets[s] = ("ccxt", None)
    else:
        for name, price, vol, seed in [("SYNTH-BTC", 60_000, 0.03, 1),
                                       ("SYNTH-ETH", 3_000, 0.04, 2),
                                       ("SYNTH-SOL", 150, 0.06, 3)]:
            assets[name] = ("syn", data.synthetic(days=a.days, start_price=price, daily_vol=vol, seed=seed))

    variants = {"V5 orig": Params(), "V5.1": Params.v51()}
    rows, funnels = [], []
    for name, (kind, base) in assets.items():
        for tf in TFS:
            if kind == "ccxt":
                df = data.fetch_ccxt(name, tf, a.days, a.exchange)
            else:
                df = data.resample(base, tf)
            for vname, p in variants.items():
                res = run(df, p)
                rows.append({"asset": name, "tf": tf, "version": vname, **res.stats()})
                if vname == "V5 orig":
                    funnels.append({"asset": name, "tf": tf, **flip_funnel(res.signals)})
            print(f"{name} {tf}: {len(df)} bars done")

    pd.set_option("display.width", 200)
    tbl = pd.DataFrame(rows).round(2)
    fun = pd.DataFrame(funnels).round(1)
    print(tbl.to_string(index=False))
    print(fun.to_string(index=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = "СИНТЕТИЧЕСКИЕ данные (проверка механики, НЕ рынок)" if a.synthetic else "рыночные данные"
    out.write_text(f"# Результаты бэктеста ({src})\n\n" + tbl.to_markdown(index=False)
                   + "\n\n## Воронка фильтров на разворотах SuperTrend (V5 orig)\n\n"
                   + fun.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
