"""Snapshot-only market structure detection.

Goals
- Detect swing highs/lows and classify the latest relationships:
  - higher high (HH), lower high (LH)
  - higher low (HL), lower low (LL)
- Identify simple range high/low over a lookback window
- Return a `StructureState` snapshot (no direction inference)

Important
- This module uses *only past candles*.
- It includes explicit guardrails to prevent lookahead bias.
- It does NOT infer direction/trend; returned `StructureState.trend` is UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from chartbuddy.thesis.schemas import Direction, StructureState, Timeframe


class LookaheadBiasError(ValueError):
    """Raised when input candles would cause lookahead bias."""


SwingRelation = Literal["HH", "LH", "EQH", "HL", "LL", "EQL", "NA"]


@dataclass(frozen=True)
class StructureParams:
    """Parameters controlling swing detection and range calculation."""

    # Pivot window sizes. A pivot high at i means high[i] is the max over
    # [i-left, i+right] (similarly for pivot low).
    pivot_left: int = 2
    pivot_right: int = 2

    # Window for computing range_high / range_low (last N candles).
    range_lookback: int = 100


def _to_frame(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    columns: Tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    """Normalize candle inputs to a DataFrame with required columns."""

    if isinstance(candles, pd.DataFrame):
        df = candles.copy()
    else:
        df = pd.DataFrame(candles)

    # If the user passed a 2D list/array without column names, assume standard order.
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
    """Prevent lookahead bias by enforcing ordering and cutoff."""

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
    """Return boolean masks for pivot highs and pivot lows."""

    n = high.size
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)

    if n == 0:
        return ph, pl

    # Avoid edges where the full window doesn't exist.
    start = left
    end = n - right
    if start >= end:
        return ph, pl

    for i in range(start, end):
        hwin = high[i - left : i + right + 1]
        lwin = low[i - left : i + right + 1]

        # Require strict uniqueness to reduce flat-top noise.
        if high[i] == np.max(hwin) and np.sum(hwin == high[i]) == 1:
            ph[i] = True
        if low[i] == np.min(lwin) and np.sum(lwin == low[i]) == 1:
            pl[i] = True

    return ph, pl


def _classify_relation(prev: Optional[float], last: Optional[float], *, high_side: bool) -> SwingRelation:
    """Classify the relationship between the last two swings."""

    if prev is None or last is None:
        return "NA"

    if last > prev:
        return "HH" if high_side else "HL"
    if last < prev:
        return "LH" if high_side else "LL"
    return "EQH" if high_side else "EQL"


def detect_structure_state(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    *,
    symbol: str,
    timeframe: Timeframe,
    as_of: Optional[Union[str, int, float, datetime, pd.Timestamp]] = None,
    params: Optional[StructureParams] = None,
    strict_lookahead: bool = True,
) -> StructureState:
    """Detect snapshot-only market structure and return a `StructureState`.

    Notes
    - This function does NOT infer trend/direction.
    - HH/LH/HL/LL detection is included in the returned `notes` field.
    """

    if not symbol:
        raise ValueError("symbol is required")

    p = params or StructureParams()

    df = _to_frame(candles)
    df = _guard_no_lookahead(
        df,
        as_of=pd.to_datetime(as_of, utc=True) if as_of is not None else None,
        strict=strict_lookahead,
    )

    # Snapshot timestamp: prefer as_of if provided, otherwise last candle timestamp.
    as_of_ts = pd.to_datetime(as_of, utc=True) if as_of is not None else df["timestamp"].iloc[-1]

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    ph, pl = _pivot_highs_lows(high, low, left=p.pivot_left, right=p.pivot_right)

    high_idx = np.flatnonzero(ph)
    low_idx = np.flatnonzero(pl)

    last_high = float(high[high_idx[-1]]) if high_idx.size >= 1 else None
    prev_high = float(high[high_idx[-2]]) if high_idx.size >= 2 else None

    last_low = float(low[low_idx[-1]]) if low_idx.size >= 1 else None
    prev_low = float(low[low_idx[-2]]) if low_idx.size >= 2 else None

    high_rel: SwingRelation = _classify_relation(prev_high, last_high, high_side=True)
    low_rel: SwingRelation = _classify_relation(prev_low, last_low, high_side=False)

    # Range high/low as simple bounds over the last N candles.
    range_window = df.iloc[-p.range_lookback :] if len(df) > 0 else df
    range_high = float(range_window["high"].max()) if not range_window.empty else None
    range_low = float(range_window["low"].min()) if not range_window.empty else None

    # Notes are descriptive only; no actions, no predictions.
    notes_parts = [
        f"pivot(pivot_left={p.pivot_left}, pivot_right={p.pivot_right})",
        f"swing_high_relation={high_rel}",
        f"swing_low_relation={low_rel}",
        f"pivots_high={int(high_idx.size)}",
        f"pivots_low={int(low_idx.size)}",
        f"range_lookback={p.range_lookback}",
    ]

    return StructureState(
        as_of=as_of_ts.to_pydatetime(),
        symbol=symbol,
        timeframe=timeframe,
        swing_high=last_high,
        swing_low=last_low,
        trend=Direction.UNKNOWN,
        last_break_of_structure_at=None,
        last_break_of_structure_level=None,
        last_change_of_character_at=None,
        last_change_of_character_level=None,
        range_high=range_high,
        range_low=range_low,
        notes="; ".join(notes_parts),
    )
