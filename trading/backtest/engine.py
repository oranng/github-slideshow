"""Бар-за-баром эмулятор брокера, повторяющий поведение TradingView strategy():

* сигнал считается на закрытии бара i, рыночный ордер исполняется по open бара i+1;
* strategy.exit(stop, limit), выставленный на закрытии бара i, активен в течение бара i+1;
* внутри бара путь цены: open -> ближний к open экстремум -> дальний -> close (как в TV);
* гэп через стоп/тейк исполняется по open;
* комиссия — процент от оборота на каждой стороне.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import Params, compute_signals


@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    exit_time: pd.Timestamp = None
    exit_price: float = np.nan
    reason: str = ""
    fees: float = 0.0
    pnl: float = 0.0  # net, после комиссий


@dataclass
class Result:
    trades: list
    equity: pd.Series
    initial: float
    signals: pd.DataFrame = field(repr=False, default=None)

    def stats(self) -> dict:
        t = self.trades
        eq = self.equity
        pnl = np.array([x.pnl for x in t]) if t else np.array([])
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        dd = (eq / eq.cummax() - 1).min() * 100 if len(eq) else 0.0
        gross_loss = -losses.sum()
        return {
            "trades": len(t),
            "net_%": (eq.iloc[-1] / self.initial - 1) * 100 if len(eq) else 0.0,
            "win_%": len(wins) / len(t) * 100 if t else np.nan,
            "PF": wins.sum() / gross_loss if gross_loss > 0 else np.nan,
            "avg_trade_%": np.mean([x.pnl / (x.entry_price * x.qty) * 100 for x in t]) if t else np.nan,
            "maxDD_%": dd,
            "fees_%_of_init": sum(x.fees for x in t) / self.initial * 100,
            "long/short": f"{sum(x.side > 0 for x in t)}/{sum(x.side < 0 for x in t)}",
        }


def _intrabar_exit(side, o, h, l, stop, limit):
    """Возвращает (цена, причина) если стоп/тейк сработал внутри бара."""
    if side > 0:
        if not np.isnan(stop) and o <= stop:
            return o, "SL/trail (gap)"
        if not np.isnan(limit) and o >= limit:
            return o, "TP (gap)"
        high_first = (h - o) <= (o - l)
        hit_stop = not np.isnan(stop) and l <= stop
        hit_tp = not np.isnan(limit) and h >= limit
    else:
        if not np.isnan(stop) and o >= stop:
            return o, "SL/trail (gap)"
        if not np.isnan(limit) and o <= limit:
            return o, "TP (gap)"
        high_first = (h - o) <= (o - l)
        hit_stop = not np.isnan(stop) and h >= stop
        hit_tp = not np.isnan(limit) and l <= limit
    if hit_stop and hit_tp:
        # для лонга: high раньше -> тейк раньше; для шорта: high раньше -> стоп раньше
        tp_first = high_first if side > 0 else not high_first
        return (limit, "TP") if tp_first else (stop, "SL/trail")
    if hit_stop:
        return stop, "SL/trail"
    if hit_tp:
        return limit, "TP"
    return None


def run(df: pd.DataFrame, p: Params = Params(), initial: float = 10_000.0) -> Result:
    sig = compute_signals(df, p)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = sig["atr"].to_numpy()
    f_long, f_short, chop = (sig[k].to_numpy() for k in ("final_long", "final_short", "chop"))
    times = df.index
    fee = p.commission_pct / 100
    slip = p.slippage_pct / 100

    cash = initial          # реализованный капитал
    pos = None              # открытая Trade
    trades = []
    equity = np.empty(len(c))

    pending_entry = None    # (side, qty, ref_close, ref_atr)
    pending_close = None    # причина
    stop = limit = np.nan   # активные уровни exit-ордера
    trail = np.nan          # храповик (trail_mode="ratchet")
    entry_atr = np.nan

    def close_pos(i, price, reason):
        nonlocal cash, pos, stop, limit, trail
        f = abs(pos.qty) * price * fee
        pos.fees += f
        gross = pos.side * pos.qty * (price - pos.entry_price)
        pos.pnl = gross - pos.fees
        pos.exit_time, pos.exit_price, pos.reason = times[i], price, reason
        cash += gross - f
        trades.append(pos)
        pos = None
        stop = limit = trail = np.nan

    for i in range(len(c)):
        # 1) рыночные ордера по open
        if pending_close and pos is not None:
            close_pos(i, o[i] * (1 - slip * pos.side), pending_close)
        pending_close = None
        if pending_entry is not None:
            side, qty, ref_close, ref_atr = pending_entry
            if pos is not None and pos.side != side:
                close_pos(i, o[i] * (1 - slip * pos.side), "Reverse")
            if pos is None:
                price = o[i] * (1 + slip * side)
                f = qty * price * fee
                cash -= f
                pos = Trade(side, times[i], price, qty, fees=f)
                entry_atr = ref_atr
                if p.protect_fill_bar:
                    # strategy.exit, выставленный вместе со входом (уровни от close сигнального бара)
                    stop = ref_close - side * p.risk_atr * ref_atr
                    limit = ref_close + side * p.reward_atr * ref_atr
                    trail = stop
                else:
                    stop = limit = np.nan
            pending_entry = None

        # 2) внутрибарное исполнение exit-ордера
        if pos is not None and not (np.isnan(stop) and np.isnan(limit)):
            hit = _intrabar_exit(pos.side, o[i], h[i], l[i], stop, limit)
            if hit:
                price, reason = hit
                if "SL" in reason:
                    price *= (1 - slip * pos.side)
                close_pos(i, price, reason)

        # 3) расчёт на закрытии бара
        eq_now = cash + (pos.side * pos.qty * (c[i] - pos.entry_price) if pos else 0.0)
        equity[i] = eq_now

        if np.isnan(atr[i]):
            continue

        want = 1 if f_long[i] else (-1 if f_short[i] else 0)
        cur_side = pos.side if pos else 0
        if want != 0 and cur_side != want:
            if p.sizing == "pct_equity":
                qty = eq_now * p.qty_pct / 100 / c[i]
            else:
                risk_cash = eq_now * p.risk_pct / 100
                qty = min(risk_cash / (p.risk_atr * atr[i]), eq_now * p.max_leverage / c[i])
            pending_entry = (want, qty, c[i], atr[i])

        if pos is not None:
            s = pos.side
            if p.trail_mode == "original":
                # V5: стоп = max(avg - 1.8ATR, close - 1.4ATR), TP = avg + 3.5ATR (ATR текущий)
                hard = pos.entry_price - s * atr[i] * p.risk_atr
                tr = c[i] - s * atr[i] * p.trail_atr
                stop = max(hard, tr) if s > 0 else min(hard, tr)
                limit = pos.entry_price + s * atr[i] * p.reward_atr
            else:
                hard = pos.entry_price - s * entry_atr * p.risk_atr
                tr = c[i] - s * atr[i] * p.trail_atr
                base = trail if not np.isnan(trail) else hard
                trail = max(base, tr, hard) if s > 0 else min(base, tr, hard)
                stop = trail
                limit = pos.entry_price + s * entry_atr * p.reward_atr
            if p.chop_exit and chop[i]:
                pending_close = "Chop Exit"

    if pos is not None:
        close_pos(len(c) - 1, c[-1], "End of data")
        equity[-1] = cash
    return Result(trades, pd.Series(equity, index=times), initial, sig)
