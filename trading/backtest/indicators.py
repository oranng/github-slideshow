"""Индикаторы с семантикой TradingView Pine v5 (ta.rma / ta.ema / ta.atr / ta.dmi / ta.stdev)."""
import numpy as np
import pandas as pd


def _seeded_recursive(src: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """Рекурсивное сглаживание, засеянное SMA первых `length` значений (как в Pine)."""
    out = np.full(len(src), np.nan)
    prev = np.nan
    for i in range(len(src)):
        x = src[i]
        if np.isnan(x):
            continue
        if np.isnan(prev):
            if i + 1 < length:
                continue
            window = src[i - length + 1:i + 1]
            if np.isnan(window).any():
                continue
            prev = window.mean()
        else:
            prev = alpha * x + (1 - alpha) * prev
        out[i] = prev
    return out


def ema(src, length):
    return _seeded_recursive(np.asarray(src, float), length, 2.0 / (length + 1))


def rma(src, length):
    return _seeded_recursive(np.asarray(src, float), length, 1.0 / length)


def sma(src, length):
    return pd.Series(src).rolling(length).mean().to_numpy()


def stdev(src, length):
    # Pine ta.stdev — смещённая (population) оценка
    return pd.Series(src).rolling(length).std(ddof=0).to_numpy()


def true_range(high, low, close):
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.nanmax(np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0)
    tr[0] = high[0] - low[0]
    return tr


def atr(high, low, close, length):
    return rma(true_range(high, low, close), length)


def dmi(high, low, close, di_len, adx_len):
    """Возвращает (+DI, -DI, ADX) как ta.dmi в Pine v5."""
    up = np.diff(high, prepend=np.nan)
    down = -np.diff(low, prepend=np.nan)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan
    tr_rma = rma(true_range(high, low, close), di_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = 100 * rma(plus_dm, di_len) / tr_rma
        minus = 100 * rma(minus_dm, di_len) / tr_rma
        s = plus + minus
        dx = np.abs(plus - minus) / np.where(s == 0, 1, s)
    adx = 100 * rma(dx, adx_len)
    return plus, minus, adx


def percentrank(src, length):
    """ta.percentrank: % из `length` предыдущих значений, которые <= текущего."""
    arr = np.asarray(src, float)
    out = np.full(len(arr), np.nan)
    if len(arr) <= length:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(arr[:-1], length)  # окно, заканчивающееся на i-1
    cur = arr[length:]
    out[length:] = 100.0 * (windows <= cur[:, None]).sum(axis=1) / length
    out[length:][np.isnan(windows).any(axis=1) | np.isnan(cur)] = np.nan
    return out
