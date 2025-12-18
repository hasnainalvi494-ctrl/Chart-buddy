"""Snapshot-only volume analysis.

What this does (direction-agnostic)
- Compares current volume to a recent baseline (median/mean) to label:
  - expansion
  - contraction
  - normal
- Detects lack of participation during "pushes":
  - a "push" is a large candle body (or range) relative to recent history
  - lack of participation means push candles occur on below-baseline volume

What this intentionally avoids
- No forward-looking data.
- No signals or recommendations.
- No directional inference.

Output
- Returns a `VolumeState` snapshot (see `chartbuddy.thesis.schemas`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from chartbuddy.thesis.schemas import VolumeState, Timeframe


class LookaheadBiasError(ValueError):
    """Raised when input candles would cause lookahead bias."""


@dataclass(frozen=True)
class VolumeParams:
    """Parameters controlling baseline, expansion/contraction, and push detection."""

    baseline_lookback: int = 50
    sum_lookback: int = 20

    # Expansion/contraction are based on last_volume vs baseline_median.
    expansion_mult: float = 1.50
    contraction_mult: float = 0.70

    # Push detection uses candle body (abs(close-open)) quantile threshold.
    push_lookback: int = 50
    push_body_quantile: float = 0.80

    # A push is considered low participation if volume < baseline * this.
    low_participation_mult: float = 0.85


def _to_frame(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    columns: Tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    """Normalize candle inputs to a DataFrame with required columns."""

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

    # Preserve extra columns (quote_volume, trade_count, etc.) if present.
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("One or more candle timestamps could not be parsed")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("One or more OHLC values could not be parsed as numbers")

    # volume can be missing for some venues; treat NaNs as 0 for sums, but keep last.
    if df["volume"].isna().any():
        df["volume"] = df["volume"].fillna(0.0)

    # Optional columns.
    if "quote_volume" in df.columns:
        df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    if "trade_count" in df.columns:
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce")

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


def analyze_volume_state(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    *,
    symbol: str,
    timeframe: Timeframe,
    as_of: Optional[Union[str, int, float, datetime, pd.Timestamp]] = None,
    params: Optional[VolumeParams] = None,
    strict_lookahead: bool = True,
) -> VolumeState:
    """Analyze volume in a snapshot-only, direction-agnostic way."""

    if not symbol:
        raise ValueError("symbol is required")

    p = params or VolumeParams()

    df = _to_frame(candles)
    df = _guard_no_lookahead(
        df,
        as_of=pd.to_datetime(as_of, utc=True) if as_of is not None else None,
        strict=strict_lookahead,
    )

    as_of_ts = pd.to_datetime(as_of, utc=True) if as_of is not None else df["timestamp"].iloc[-1]

    if len(df) < 5:
        return VolumeState(
            as_of=as_of_ts.to_pydatetime(),
            symbol=symbol,
            timeframe=timeframe,
            last_candle_volume=float(df["volume"].iloc[-1]),
            last_n_candles_volume_sum=float(df["volume"].sum()),
            last_n_candles=len(df),
            quote_volume=float(df["quote_volume"].iloc[-1]) if "quote_volume" in df.columns else None,
            trade_count=int(df["trade_count"].iloc[-1]) if "trade_count" in df.columns and pd.notna(df["trade_count"].iloc[-1]) else None,
            notes="insufficient_history",
        )

    baseline_window = df.iloc[-p.baseline_lookback :] if len(df) > p.baseline_lookback else df
    baseline_volumes = baseline_window["volume"].to_numpy(dtype=float)

    baseline_median = float(np.median(baseline_volumes))
    baseline_mean = float(np.mean(baseline_volumes))

    last_volume = float(df["volume"].iloc[-1])

    if baseline_median <= 0.0:
        vol_label = "unclear"
        vol_relation = "baseline_zero"
    else:
        ratio = last_volume / baseline_median
        if ratio >= p.expansion_mult:
            vol_label = "expansion"
        elif ratio <= p.contraction_mult:
            vol_label = "contraction"
        else:
            vol_label = "normal"
        vol_relation = "relative_to_baseline"

    # Detect low participation during pushes.
    push_window = df.iloc[-p.push_lookback :] if len(df) > p.push_lookback else df
    body = (push_window["close"] - push_window["open"]).abs().to_numpy(dtype=float)

    # Fallback to range if body is all zeros.
    if np.all(body == 0.0):
        body = (push_window["high"] - push_window["low"]).to_numpy(dtype=float)

    thresh = float(np.quantile(body, p.push_body_quantile)) if body.size > 0 else float("nan")

    if not np.isfinite(thresh) or thresh <= 0.0 or baseline_median <= 0.0:
        push_count = 0
        low_participation_count = 0
        participation_note = "push_eval_unavailable"
    else:
        is_push = body >= thresh
        push_count = int(np.sum(is_push))

        v = push_window["volume"].to_numpy(dtype=float)
        low_participation = is_push & (v < baseline_median * p.low_participation_mult)
        low_participation_count = int(np.sum(low_participation))

        if push_count == 0:
            participation_note = "no_pushes"
        elif low_participation_count == 0:
            participation_note = "pushes_participated"
        elif low_participation_count == push_count:
            participation_note = "pushes_low_participation"
        else:
            participation_note = "mixed_participation"

    sum_window = df.iloc[-p.sum_lookback :] if len(df) > p.sum_lookback else df
    sum_n = int(len(sum_window))
    sum_vol = float(sum_window["volume"].sum())

    # Keep notes descriptive; values are raw observations (not scores).
    notes = (
        f"volume={vol_label} ({vol_relation}); "
        f"baseline_lookback={int(len(baseline_window))}; "
        f"baseline_median={baseline_median:.6g}; baseline_mean={baseline_mean:.6g}; "
        f"last_volume={last_volume:.6g}; "
        f"push_body_quantile={p.push_body_quantile:.2f}; push_threshold={thresh:.6g}; "
        f"pushes={push_count}; low_participation_pushes={low_participation_count}; "
        f"participation={participation_note}"
    )

    return VolumeState(
        as_of=as_of_ts.to_pydatetime(),
        symbol=symbol,
        timeframe=timeframe,
        last_candle_volume=last_volume,
        last_n_candles_volume_sum=sum_vol,
        last_n_candles=sum_n,
        quote_volume=float(df["quote_volume"].iloc[-1]) if "quote_volume" in df.columns and pd.notna(df["quote_volume"].iloc[-1]) else None,
        trade_count=int(df["trade_count"].iloc[-1]) if "trade_count" in df.columns and pd.notna(df["trade_count"].iloc[-1]) else None,
        notes=notes,
    )
