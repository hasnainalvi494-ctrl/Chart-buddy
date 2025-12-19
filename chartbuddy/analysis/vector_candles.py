"""Snapshot-only vector candle / multi-candle cluster analysis.

Classification (descriptive only)
- impulse: expansion in range/body + expanded volume
- exhaustion: large range with wick-dominance + expanded volume
- compression: contracted range/body + contracted volume
- unclear: none of the above

Notes
- Uses only candles up to `as_of` (if provided) and enforces monotonic timestamps.
- Does NOT produce buy/sell signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from chartbuddy.thesis.schemas import Timeframe, VectorCandlePattern, VectorCandleState


class LookaheadBiasError(ValueError):
    """Raised when input candles would cause lookahead bias."""


@dataclass(frozen=True)
class VectorCandleParams:
    """Parameters for cluster-based candle classification."""

    lookback: int = 50

    # Impulse conditions
    impulse_range_mult: float = 1.60
    impulse_body_mult: float = 1.60
    impulse_volume_mult: float = 1.30

    # Compression conditions
    compression_range_mult: float = 0.65
    compression_body_mult: float = 0.65
    compression_volume_mult: float = 0.80

    # Exhaustion conditions
    exhaustion_range_mult: float = 1.50
    exhaustion_volume_mult: float = 1.30
    exhaustion_wick_ratio: float = 0.55  # (range-body)/range

    eps: float = 1e-12


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

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("One or more candle timestamps could not be parsed")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("One or more OHLC values could not be parsed as numbers")

    # Optional venue columns.
    for col in ("candle_open_time", "candle_close_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in (
        "quote_volume",
        "trades",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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


def analyze_vector_candle_state(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    *,
    symbol: str,
    timeframe: Timeframe,
    as_of: Optional[Union[str, int, float, datetime, pd.Timestamp]] = None,
    params: Optional[VectorCandleParams] = None,
    strict_lookahead: bool = True,
) -> VectorCandleState:
    """Analyze recent candles and classify the current cluster pattern."""

    if not symbol:
        raise ValueError("symbol is required")

    p = params or VectorCandleParams()

    df = _to_frame(candles)
    df = _guard_no_lookahead(
        df,
        as_of=pd.to_datetime(as_of, utc=True) if as_of is not None else None,
        strict=strict_lookahead,
    )

    # Snapshot timestamp: prefer as_of, else last candle timestamp.
    as_of_ts = pd.to_datetime(as_of, utc=True) if as_of is not None else df["timestamp"].iloc[-1]

    # Work window for baselines.
    win = df.iloc[-p.lookback :] if len(df) > p.lookback else df

    o = win["open"].to_numpy(dtype=float)
    h = win["high"].to_numpy(dtype=float)
    l = win["low"].to_numpy(dtype=float)
    c = win["close"].to_numpy(dtype=float)
    v = win["volume"].to_numpy(dtype=float) if "volume" in win.columns else np.zeros(len(win), dtype=float)

    body = np.abs(c - o)
    rng = np.maximum(h - l, p.eps)
    wick_ratio = (rng - body) / rng

    # Baselines
    base_range = float(np.median(rng))
    base_body = float(np.median(body))
    base_vol = float(np.median(v))

    # Last candle (most recent in win)
    last_range = float(rng[-1])
    last_body = float(body[-1])
    last_wick = float(wick_ratio[-1])
    last_vol = float(v[-1])

    def safe_ratio(x: float, base: float) -> float:
        return x / base if base > p.eps else float("nan")

    range_ratio = safe_ratio(last_range, base_range)
    body_ratio = safe_ratio(last_body, base_body)
    vol_ratio = safe_ratio(last_vol, base_vol)

    pattern = VectorCandlePattern.UNCLEAR

    if np.isfinite(range_ratio) and np.isfinite(body_ratio) and np.isfinite(vol_ratio):
        # Compression: contracted range/body and contracted volume.
        if (
            range_ratio <= p.compression_range_mult
            and body_ratio <= p.compression_body_mult
            and vol_ratio <= p.compression_volume_mult
        ):
            pattern = VectorCandlePattern.COMPRESSION

        # Exhaustion: large range + wick-dominance + expanded volume.
        elif (
            range_ratio >= p.exhaustion_range_mult
            and vol_ratio >= p.exhaustion_volume_mult
            and last_wick >= p.exhaustion_wick_ratio
        ):
            pattern = VectorCandlePattern.EXHAUSTION

        # Impulse: expanded range/body and expanded volume.
        elif (
            range_ratio >= p.impulse_range_mult
            and body_ratio >= p.impulse_body_mult
            and vol_ratio >= p.impulse_volume_mult
        ):
            pattern = VectorCandlePattern.IMPULSE

    # Last candle snapshot (from full df to preserve optional fields if present)
    last = df.iloc[-1]

    def _maybe_int(val) -> Optional[int]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return int(val)
        except Exception:
            try:
                return int(float(val))
            except Exception:
                return None

    trades_val: Optional[int] = None
    if "trades" in last.index and pd.notna(last["trades"]):
        trades_val = _maybe_int(last["trades"])
    elif "trade_count" in last.index and pd.notna(last["trade_count"]):
        trades_val = _maybe_int(last["trade_count"])

    notes = (
        f"pattern={pattern.value}; lookback={int(len(win))}; "
        f"base_range={base_range:.6g}; base_body={base_body:.6g}; base_vol={base_vol:.6g}; "
        f"last_range={last_range:.6g}; last_body={last_body:.6g}; last_wick_ratio={last_wick:.3f}; last_vol={last_vol:.6g}; "
        f"ratios(range={range_ratio:.3f}, body={body_ratio:.3f}, vol={vol_ratio:.3f})"
    )

    return VectorCandleState(
        as_of=as_of_ts.to_pydatetime(),
        symbol=symbol,
        timeframe=timeframe,
        open=float(last["open"]),
        high=float(last["high"]),
        low=float(last["low"]),
        close=float(last["close"]),
        volume=float(last["volume"]) if "volume" in last.index and pd.notna(last["volume"]) else None,
        quote_volume=float(last["quote_volume"]) if "quote_volume" in last.index and pd.notna(last["quote_volume"]) else None,
        trades=trades_val,
        taker_buy_base_volume=float(last["taker_buy_base_volume"]) if "taker_buy_base_volume" in last.index and pd.notna(last["taker_buy_base_volume"]) else None,
        taker_buy_quote_volume=float(last["taker_buy_quote_volume"]) if "taker_buy_quote_volume" in last.index and pd.notna(last["taker_buy_quote_volume"]) else None,
        candle_open_time=last["candle_open_time"].to_pydatetime() if "candle_open_time" in last.index and pd.notna(last["candle_open_time"]) else None,
        candle_close_time=last["candle_close_time"].to_pydatetime() if "candle_close_time" in last.index and pd.notna(last["candle_close_time"]) else None,
        source=str(last["source"]) if "source" in last.index and pd.notna(last["source"]) else None,
        pattern=pattern,
        notes=notes,
    )
