"""Snapshot-only liquidity detection.

What this does
- Detect buy-side (above highs) and sell-side (below lows) liquidity pools using pivot highs/lows.
- Identify liquidity sweeps (price runs a pool level) and classify *acceptance vs rejection*
  using only candles available up to the snapshot.

What this intentionally avoids
- No forward-looking data.
- No scores/confidence.
- No trade signals.

Outputs
- Returns a `LiquidityState` snapshot (see `chartbuddy.thesis.schemas`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from chartbuddy.thesis.schemas import Direction, LiquidityPool, LiquidityState, LiquiditySweep, Timeframe


class LookaheadBiasError(ValueError):
    """Raised when input candles would cause lookahead bias."""


@dataclass(frozen=True)
class LiquidityParams:
    """Parameters controlling pool and sweep detection."""

    # Pivot window sizes (see structure module).
    pivot_left: int = 2
    pivot_right: int = 2

    # Only analyze last N candles for pool detection.
    pool_lookback: int = 500

    # Identify equal-high/low pools if multiple pivots are within this tolerance.
    # e.g. 0.001 = 0.10%.
    equal_level_tol_pct: float = 0.001

    # Require at least this many pivot touches to form an equal-high/low pool.
    min_touches: int = 2

    # Sweep detection over the last N candles.
    sweep_lookback: int = 200

    # Number of candles after the sweep candle to classify acceptance vs rejection.
    # Note: "after" is still historical relative to the overall snapshot.
    post_sweep_bars: int = 3


def _to_frame(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    columns: Tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    if isinstance(candles, pd.DataFrame):
        df = candles.copy()
    else:
        df = pd.DataFrame(candles)

    if df.shape[1] >= len(columns) and not set(columns).issubset(df.columns):
        df = df.iloc[:, : len(columns)]
        df.columns = list(columns)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Candles missing required columns: {missing}")

    df = df[list(columns)].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("One or more candle timestamps could not be parsed")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("One or more OHLC values could not be parsed as numbers")

    return df


def _guard_no_lookahead(
    df: pd.DataFrame,
    *,
    as_of: Optional[pd.Timestamp],
    strict: bool,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No candles provided")

    ts = df["timestamp"]
    if not ts.is_monotonic_increasing:
        raise LookaheadBiasError(
            "Candle timestamps must be monotonic increasing (oldest -> newest). "
            "Refusing to sort automatically to avoid hiding lookahead issues."
        )

    if as_of is None:
        return df

    as_of = pd.to_datetime(as_of, utc=True)
    future_mask = ts > as_of

    if future_mask.any():
        if strict:
            max_future = ts[future_mask].max()
            raise LookaheadBiasError(
                f"Found candles after as_of={as_of.isoformat()} (max={max_future.isoformat()}). "
                "Refusing to proceed to prevent lookahead bias."
            )
        df = df.loc[~future_mask].copy()
        if df.empty:
            raise LookaheadBiasError("All candles are after as_of; nothing to analyze")

    return df


def _pivot_highs_lows(
    high: np.ndarray,
    low: np.ndarray,
    *,
    left: int,
    right: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = high.size
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)

    start = left
    end = n - right
    if n == 0 or start >= end:
        return ph, pl

    for i in range(start, end):
        hwin = high[i - left : i + right + 1]
        lwin = low[i - left : i + right + 1]

        # Strict uniqueness to avoid flat tops/bottoms.
        if high[i] == np.max(hwin) and np.sum(hwin == high[i]) == 1:
            ph[i] = True
        if low[i] == np.min(lwin) and np.sum(lwin == low[i]) == 1:
            pl[i] = True

    return ph, pl


def _cluster_levels(
    levels_with_ts: List[Tuple[float, pd.Timestamp]],
    *,
    tol_pct: float,
    min_touches: int,
) -> List[Tuple[float, pd.Timestamp, int]]:
    """Cluster levels by tolerance and return (cluster_level, latest_ts, touches)."""

    if not levels_with_ts:
        return []

    # Sort by level so clustering is stable.
    items = sorted(levels_with_ts, key=lambda x: x[0])

    clusters: List[List[Tuple[float, pd.Timestamp]]] = []
    current: List[Tuple[float, pd.Timestamp]] = [items[0]]

    def close_enough(a: float, b: float) -> bool:
        if a == 0.0:
            return abs(b) <= abs(tol_pct)
        return abs(a - b) / abs(a) <= tol_pct

    for level, ts in items[1:]:
        base_level = float(np.mean([x[0] for x in current]))
        if close_enough(base_level, level):
            current.append((level, ts))
        else:
            clusters.append(current)
            current = [(level, ts)]
    clusters.append(current)

    results: List[Tuple[float, pd.Timestamp, int]] = []
    for c in clusters:
        touches = len(c)
        if touches < min_touches:
            continue
        lvl = float(np.mean([x[0] for x in c]))
        latest = max([x[1] for x in c])
        results.append((lvl, latest, touches))

    # Prefer most recently observed clusters first.
    results.sort(key=lambda t: t[1], reverse=True)
    return results


def _classify_sweep(
    *,
    level: float,
    sweep_idx: int,
    df: pd.DataFrame,
    side: str,
    post_bars: int,
) -> str:
    """Classify a sweep as acceptance/rejection/unclear using post-sweep candles."""

    # Evaluate closes from the sweep candle through the next post_bars.
    end = min(len(df), sweep_idx + 1 + post_bars)
    closes = df["close"].iloc[sweep_idx:end].to_numpy(dtype=float)

    if closes.size == 0:
        return "unclear"

    last_close = float(closes[-1])

    if side == "buy":
        # Sweep above level: acceptance if market holds above level; rejection if back below.
        return "acceptance" if last_close >= level else "rejection"

    # side == "sell"
    return "acceptance" if last_close <= level else "rejection"


def detect_liquidity_state(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    *,
    symbol: str,
    timeframe: Timeframe,
    as_of: Optional[Union[str, int, float, datetime, pd.Timestamp]] = None,
    params: Optional[LiquidityParams] = None,
    strict_lookahead: bool = True,
) -> LiquidityState:
    """Detect liquidity pools and sweeps from historical candles."""

    if not symbol:
        raise ValueError("symbol is required")

    p = params or LiquidityParams()

    df = _to_frame(candles)
    df = _guard_no_lookahead(
        df,
        as_of=pd.to_datetime(as_of, utc=True) if as_of is not None else None,
        strict=strict_lookahead,
    )

    as_of_ts = pd.to_datetime(as_of, utc=True) if as_of is not None else df["timestamp"].iloc[-1]

    work = df.iloc[-p.pool_lookback :].copy() if len(df) > p.pool_lookback else df.copy()

    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)

    ph, pl = _pivot_highs_lows(high, low, left=p.pivot_left, right=p.pivot_right)

    piv_highs: List[Tuple[float, pd.Timestamp]] = [
        (float(high[i]), work["timestamp"].iloc[i]) for i in np.flatnonzero(ph)
    ]
    piv_lows: List[Tuple[float, pd.Timestamp]] = [
        (float(low[i]), work["timestamp"].iloc[i]) for i in np.flatnonzero(pl)
    ]

    # Equal-high/low pools (clusters).
    buy_clusters = _cluster_levels(piv_highs, tol_pct=p.equal_level_tol_pct, min_touches=p.min_touches)
    sell_clusters = _cluster_levels(piv_lows, tol_pct=p.equal_level_tol_pct, min_touches=p.min_touches)

    pools: List[LiquidityPool] = []

    for lvl, latest_ts, touches in buy_clusters:
        pools.append(
            LiquidityPool(
                kind="buy_side_equal_highs",
                level=lvl,
                observed_at=latest_ts.to_pydatetime(),
                notes=f"touches={touches}; tol_pct={p.equal_level_tol_pct}",
            )
        )

    for lvl, latest_ts, touches in sell_clusters:
        pools.append(
            LiquidityPool(
                kind="sell_side_equal_lows",
                level=lvl,
                observed_at=latest_ts.to_pydatetime(),
                notes=f"touches={touches}; tol_pct={p.equal_level_tol_pct}",
            )
        )

    # Also include the most recent single pivot high/low as simple pools.
    if piv_highs:
        lvl, ts = max(piv_highs, key=lambda t: t[1])
        pools.append(
            LiquidityPool(
                kind="buy_side_pivot_high",
                level=float(lvl),
                observed_at=ts.to_pydatetime(),
                notes="most_recent_pivot_high",
            )
        )

    if piv_lows:
        lvl, ts = max(piv_lows, key=lambda t: t[1])
        pools.append(
            LiquidityPool(
                kind="sell_side_pivot_low",
                level=float(lvl),
                observed_at=ts.to_pydatetime(),
                notes="most_recent_pivot_low",
            )
        )

    # Sweep detection is done on the full df but limited to the last sweep_lookback.
    sweep_df = df.iloc[-p.sweep_lookback :].copy() if len(df) > p.sweep_lookback else df.copy()

    sweeps: List[LiquiditySweep] = []

    # Build unique pool levels to scan. Use rounded key to avoid duplicates.
    level_items: List[Tuple[str, float]] = []
    for pool in pools:
        if pool.kind.startswith("buy_side"):
            level_items.append(("buy", float(pool.level)))
        elif pool.kind.startswith("sell_side"):
            level_items.append(("sell", float(pool.level)))

    seen: set = set()
    unique_levels: List[Tuple[str, float]] = []
    for side, lvl in level_items:
        key = (side, round(lvl, 8))
        if key in seen:
            continue
        seen.add(key)
        unique_levels.append((side, lvl))

    # Detect sweeps: crossing the level and then classifying by post-sweep closes.
    for side, lvl in unique_levels:
        if side == "buy":
            crossed = sweep_df["high"] >= lvl
            prev_below = sweep_df["high"].shift(1) < lvl
            sweep_mask = crossed & prev_below
        else:
            crossed = sweep_df["low"] <= lvl
            prev_above = sweep_df["low"].shift(1) > lvl
            sweep_mask = crossed & prev_above

        idxs = np.flatnonzero(sweep_mask.to_numpy())
        if idxs.size == 0:
            continue

        # Keep the most recent sweep for this level.
        sweep_i = int(idxs[-1])
        occurred_at = sweep_df["timestamp"].iloc[sweep_i]

        classification = _classify_sweep(level=lvl, sweep_idx=sweep_i, df=sweep_df, side=side, post_bars=p.post_sweep_bars)

        sweeps.append(
            LiquiditySweep(
                level=float(lvl),
                occurred_at=occurred_at.to_pydatetime(),
                direction=Direction.UP if side == "buy" else Direction.DOWN,
                notes=f"side={side}; classification={classification}; post_bars={p.post_sweep_bars}",
            )
        )

    # Keep sweeps sorted by recency.
    sweeps.sort(key=lambda s: s.occurred_at, reverse=True)

    return LiquidityState(
        as_of=as_of_ts.to_pydatetime(),
        symbol=symbol,
        timeframe=timeframe,
        pools=pools,
        recent_sweeps=sweeps,
        notes="snapshot-only; pivot-based pools; sweeps classified by post-sweep closes",
    )
