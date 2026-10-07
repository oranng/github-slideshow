"""Загрузка OHLCV: CSV (экспорт TradingView / Binance), ccxt, синтетика для проверки движка."""
from pathlib import Path

import numpy as np
import pandas as pd

TF_RULE = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}


def load_csv(path) -> pd.DataFrame:
    """CSV с колонками time/open/high/low/close[/volume].

    Поддерживает экспорт TradingView (time в unix-секундах или ISO) и Binance klines
    (без заголовка, open_time в мс/мкс)."""
    path = Path(path)
    head = path.open().readline()
    if head[:1].isdigit():  # Binance data.binance.vision: без заголовка
        df = pd.read_csv(path, header=None, usecols=range(6),
                         names=["time", "open", "high", "low", "close", "volume"])
    else:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"timestamp": "time", "date": "time", "datetime": "time", "open_time": "time"})
    t = df["time"]
    if np.issubdtype(t.dtype, np.number):
        unit = "us" if t.iloc[0] > 1e14 else "ms" if t.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    df.index = idx
    return df[["open", "high", "low", "close"] + (["volume"] if "volume" in df else [])].astype(float).sort_index()


def fetch_ccxt(symbol="BTC/USDT", timeframe="5m", days=180, exchange="binanceusdm") -> pd.DataFrame:
    """Скачать историю через ccxt (нужен сетевой доступ к бирже)."""
    import ccxt  # pip install ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since = ex.milliseconds() - days * 86_400_000
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1500)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < 2:
            break
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]).drop_duplicates("time")
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    return df.iloc[:-1]  # последний бар не закрыт


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df:
        agg["volume"] = "sum"
    return df.resample(TF_RULE[tf], label="left", closed="left").agg(agg).dropna()


def synthetic(days=120, start_price=60_000.0, daily_vol=0.035, seed=0) -> pd.DataFrame:
    """1-минутные бары: смена режимов (тренд / флэт) + кластеризация волатильности.

    ТОЛЬКО для проверки механики движка — не является рыночными данными."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    sigma_1m = daily_vol / np.sqrt(1440)
    # режимы длиной 0.5–3 дня: 0 = флэт (mean-reversion), +1/-1 = тренд
    regime = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.uniform(0.5, 3) * 1440)
        regime[i:i + length] = rng.choice([0, 0, 1, -1])
        i += length
    # GARCH(1,1)-подобная волатильность на t-распределённых шоках
    shocks = rng.standard_t(4, n) / np.sqrt(2)
    garch = np.empty(n)
    v = 1.0
    for k in range(n):
        if k:
            v = 0.02 + 0.97 * v + 0.01 * shocks[k - 1] ** 2
        garch[k] = np.sqrt(v)
    garch /= garch.mean()
    drift = regime * sigma_1m * 0.06
    logp = np.empty(n)
    x = np.log(start_price)
    anchor = x
    for k in range(n):
        if regime[k] == 0:
            r = -0.002 * (x - anchor) + sigma_1m * garch[k] * shocks[k]
        else:
            r = drift[k] + sigma_1m * garch[k] * shocks[k]
            anchor = x
        x += r
        logp[k] = x
    close = np.exp(logp)
    open_ = np.r_[start_price, close[:-1]]
    wick = np.abs(rng.normal(0, sigma_1m * 0.6, (2, n))) * garch
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
