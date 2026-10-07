import numpy as np
import pandas as pd

from trading.backtest import data, indicators as ta
from trading.backtest.engine import _intrabar_exit, run
from trading.backtest.strategy import Params


def test_intrabar_long_gap_fills_at_open():
    assert _intrabar_exit(1, 95, 96, 90, stop=97, limit=110) == (95, "SL/trail (gap)")
    assert _intrabar_exit(1, 112, 113, 111, stop=97, limit=110) == (112, "TP (gap)")


def test_intrabar_both_hit_uses_tv_path():
    # open ближе к high -> путь open->high->low: тейк лонга раньше стопа
    assert _intrabar_exit(1, 100, 101, 90, stop=95, limit=101)[1] == "TP"
    # open ближе к low -> open->low->high: стоп раньше
    assert _intrabar_exit(1, 100, 110, 99, stop=99, limit=105)[1] == "SL/trail"
    # шорт, open ближе к high -> стоп шорта (выше) раньше
    assert _intrabar_exit(-1, 100, 101, 90, stop=101, limit=95)[1] == "SL/trail"


def test_ema_rma_seeded_with_sma():
    x = np.arange(1, 11, dtype=float)
    e = ta.ema(x, 3)
    assert np.isnan(e[:2]).all() and e[2] == 2.0
    assert np.isclose(e[3], 0.5 * 4 + 0.5 * 2)
    r = ta.rma(x, 3)
    assert np.isclose(r[3], 4 / 3 + 2 * 2 / 3)


def test_percentrank():
    pr = ta.percentrank(np.array([1, 2, 3, 4, 2.5]), 4)
    assert pr[4] == 50.0


def test_run_is_deterministic_and_accounts_fees():
    df = data.resample(data.synthetic(days=20, seed=7), "15m")
    a, b = run(df, Params()), run(df, Params())
    assert a.stats() == b.stats()
    for t in a.trades:
        gross = t.side * t.qty * (t.exit_price - t.entry_price)
        assert np.isclose(t.pnl, gross - t.fees)
    assert np.isclose(a.equity.iloc[-1], 10_000 + sum(t.pnl for t in a.trades))


def test_fill_bar_protection(monkeypatch):
    """V5: на баре исполнения входа стопа нет; V5.1: стоп выставлен вместе со входом."""
    idx = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 100, 100], "high": [101, 101, 101, 101],
                       "low": [99, 99, 80, 99], "close": [100, 100, 95, 100]}, index=idx, dtype=float)
    sig = pd.DataFrame({"atr": [2.0] * 4, "final_long": [False, True, False, False],
                        "final_short": False, "chop": False}, index=idx)
    monkeypatch.setattr("trading.backtest.engine.compute_signals", lambda d, p: sig)
    v5 = run(df, Params(slippage_pct=0))
    v51 = run(df, Params.v51(slippage_pct=0))
    # вход на open бара 2 (=100), low бара 2 = 80
    assert v5.trades[0].reason == "End of data"           # обвал на баре входа не закрыл позицию
    assert v51.trades[0].reason == "SL/trail"
    assert np.isclose(v51.trades[0].exit_price, 100 - 1.8 * 2.0)
