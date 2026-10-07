"""SuperTrend V5 Adaptive Regime Engine — порт сигнальной части на Python.

`Params()` по умолчанию = оригинальный Pine-скрипт V5.
`Params.v51()` = исправленная версия (см. trading/docs/REVIEW.md и trading/pine/supertrend_v5_1.pine).
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import indicators as ta


@dataclass(frozen=True)
class Params:
    # Trend engine
    atr_len: int = 12
    atr_smooth_len: int = 21
    atr_mult_trend: float = 2.6
    # Regime
    adx_len: int = 14
    adx_threshold: float = 18.0
    # Volatility expansion
    bb_len: int = 20
    bb_mult: float = 2.0
    min_bb_width: float = 1.2          # % (режим "abs")
    vol_filter: str = "abs"            # "abs" | "pctrank"
    bb_rank_len: int = 200
    bb_rank_min: float = 50.0          # перцентиль ширины BB (режим "pctrank")
    require_bb_expanding: bool = True
    # Exits
    risk_atr: float = 1.8
    reward_atr: float = 3.5
    trail_atr: float = 1.4
    trail_mode: str = "original"       # "original" (не храповик, от close) | "ratchet"
    protect_fill_bar: bool = False     # ставить стоп сразу вместе со входом
    # Anti-chop / bias
    ema_filter_len: int = 200
    slope_mode: str = "pct"            # "pct" (как в V5) | "atr" (нормировано на ATR)
    min_slope: float = 0.02            # % за 10 баров (режим "pct")
    min_slope_atr: float = 0.10        # (EMA - EMA[10]) / ATR (режим "atr")
    chop_exit: bool = True
    # Entries
    entry_window: int = 0              # 0 = только бар разворота (V5); N = вход в течение N баров после разворота
    # Execution / costs
    commission_pct: float = 0.04       # % за сторону
    slippage_pct: float = 0.0          # % проскальзывания на рыночных/стоп исполнениях
    sizing: str = "pct_equity"         # "pct_equity" | "risk"
    qty_pct: float = 10.0              # % капитала на позицию (pct_equity)
    risk_pct: float = 0.5              # % капитала под риском до стопа (risk)
    max_leverage: float = 3.0

    @classmethod
    def v51(cls, **kw):
        base = cls(
            vol_filter="pctrank",
            slope_mode="atr",
            trail_mode="ratchet",
            protect_fill_bar=True,
            entry_window=6,
            commission_pct=0.05,
            slippage_pct=0.01,
        )
        return replace(base, **kw)


def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    hl2 = (h + l) / 2

    raw_atr = ta.atr(h, l, c, p.atr_len)
    s_atr = ta.ema(raw_atr, p.atr_smooth_len)

    up_base = hl2 - p.atr_mult_trend * s_atr
    dn_base = hl2 + p.atr_mult_trend * s_atr
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)
    for i in range(n):
        pu = up[i - 1] if i > 0 and not np.isnan(up[i - 1]) else 0.0   # nz(upTrend[1])
        pd_ = dn[i - 1] if i > 0 and not np.isnan(dn[i - 1]) else 0.0  # nz(dnTrend[1])
        pc = c[i - 1] if i > 0 else np.nan
        up[i] = max(up_base[i], pu) if pc > pu else up_base[i]
        dn[i] = min(dn_base[i], pd_) if pc < pd_ else dn_base[i]
        prev = trend[i - 1] if i > 0 else 1
        if prev == -1 and c[i] > dn[i]:
            trend[i] = 1
        elif prev == 1 and c[i] < up[i]:
            trend[i] = -1
        else:
            trend[i] = prev
    prev_trend = np.r_[1, trend[:-1]]
    long_flip = (trend == 1) & (prev_trend == -1)
    short_flip = (trend == -1) & (prev_trend == 1)

    _, _, adx = ta.dmi(h, l, c, p.adx_len, p.adx_len)
    trend_regime = adx > p.adx_threshold

    ema_bias = ta.ema(c, p.ema_filter_len)
    ema_lag = np.r_[np.full(10, np.nan), ema_bias[:-10]]
    if p.slope_mode == "pct":
        slope = (ema_bias - ema_lag) / ema_lag * 100
        thr = p.min_slope
    else:
        slope = (ema_bias - ema_lag) / s_atr
        thr = p.min_slope_atr
    bull_bias = (c > ema_bias) & (slope > thr)
    bear_bias = (c < ema_bias) & (slope < -thr)

    basis = ta.sma(c, p.bb_len)
    dev = ta.stdev(c, p.bb_len)
    bb_width = (2 * dev * p.bb_mult) / basis * 100
    if p.vol_filter == "abs":
        vol_ok = bb_width > p.min_bb_width
    else:
        vol_ok = ta.percentrank(bb_width, p.bb_rank_len) > p.bb_rank_min
    bb_expanding = bb_width > np.r_[np.nan, bb_width[:-1]]
    if p.require_bb_expanding:
        vol_ok = vol_ok & bb_expanding

    allow_long = trend_regime & bull_bias & vol_ok
    allow_short = trend_regime & bear_bias & vol_ok

    if p.entry_window <= 0:
        final_long = long_flip & allow_long
        final_short = short_flip & allow_short
    else:
        # бары с момента последнего разворота
        flip = long_flip | short_flip
        idx = np.where(flip, np.arange(n), -10**9)
        bars_since = np.arange(n) - np.maximum.accumulate(idx)
        in_window = bars_since <= p.entry_window
        final_long = (trend == 1) & in_window & allow_long
        final_short = (trend == -1) & in_window & allow_short

    weak_trend = adx < p.adx_threshold * 0.7
    if p.vol_filter == "abs":
        vol_collapse = bb_width < p.min_bb_width * 0.7
    else:
        vol_collapse = ta.percentrank(bb_width, p.bb_rank_len) < p.bb_rank_min * 0.5
    chop = weak_trend & vol_collapse

    out = pd.DataFrame(index=df.index)
    out["atr"] = s_atr
    out["trend"] = trend
    out["long_flip"] = long_flip
    out["short_flip"] = short_flip
    out["final_long"] = np.nan_to_num(final_long).astype(bool)
    out["final_short"] = np.nan_to_num(final_short).astype(bool)
    out["chop"] = np.nan_to_num(chop).astype(bool)
    # диагностика фильтров на барах разворота
    out["f_adx"] = trend_regime
    out["f_bias_long"] = bull_bias
    out["f_bias_short"] = bear_bias
    out["f_vol"] = np.nan_to_num(vol_ok).astype(bool)
    return out
