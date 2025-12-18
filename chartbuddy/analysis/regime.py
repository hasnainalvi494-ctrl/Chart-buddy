"""Snapshot-only market regime detection.

This module is intentionally conservative:
- uses only *past* candles (no forward-looking inputs)
- provides guardrails to prevent accidental lookahead bias

Output regimes are limited to:
- trending_up
- trending_down
- ranging
- unclear
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


Regime = Literal["trending_up", "trending_down", "ranging", "unclear"]


class LookaheadBiasError(ValueError):
    """Raised when input candles would cause lookahead bias."""


@dataclass(frozen=True)
class RegimeParams:
    """Tunable parameters for regime classification.

    Notes:
    - These are *not* scores/signals; they only control thresholds for the
      descriptive regime label.
    """

    lookback: int = 60

    # Kaufman Efficiency Ratio thresholds
    er_trend: float = 0.30
    er_range: float = 0.18

    # Minimum absolute slope of log(close) per bar to consider directional.
    # Using log-space makes slope comparable across price levels.
    min_abs_log_slope: float = 0.0003


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

    # Normalize timestamps to pandas datetime if possible.
    # Accept seconds/ms epoch ints, datetimes, or ISO strings.
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

    # Enforce strictly increasing timestamps. Sorting can hide accidental future data.
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
        # Non-strict mode truncates to <= as_of.
        df = df.loc[~future_mask].copy()
        if df.empty:
            raise LookaheadBiasError("All candles are after as_of; nothing to analyze")

    return df


def _efficiency_ratio(close: np.ndarray) -> float:
    """Kaufman Efficiency Ratio (0..1).

    ER = |close[-1] - close[0]| / sum(|diff(close)|)
    """

    if close.size < 3:
        return float("nan")

    net = float(abs(close[-1] - close[0]))
    denom = float(np.sum(np.abs(np.diff(close))))
    if denom == 0.0:
        return 0.0
    return net / denom


def _log_slope(close: np.ndarray) -> float:
    """Slope of log(close) over bars using least squares."""

    if close.size < 3:
        return float("nan")

    y = np.log(np.maximum(close, 1e-12))
    x = np.arange(close.size, dtype=float)
    # slope in log(price) per bar
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def detect_market_regime(
    candles: Union[pd.DataFrame, Sequence[dict], Sequence[Sequence[float]]],
    *,
    as_of: Optional[Union[str, int, float, pd.Timestamp]] = None,
    params: Optional[RegimeParams] = None,
    strict_lookahead: bool = True,
) -> Regime:
    """Detect a descriptive market regime from historical candles.

    Parameters
    - candles: DataFrame or records with columns: timestamp, open, high, low, close, volume
    - as_of: cutoff timestamp; any candle after this is future data
    - params: threshold configuration
    - strict_lookahead: if True, raises when candles include future data vs as_of;
      if False, truncates to <= as_of.

    Returns
    - One of: trending_up, trending_down, ranging, unclear
    """

    p = params or RegimeParams()

    df = _to_frame(candles)
    df = _guard_no_lookahead(df, as_of=pd.to_datetime(as_of, utc=True) if as_of is not None else None, strict=strict_lookahead)

    if len(df) < max(10, p.lookback // 2):
        return "unclear"

    window = df.iloc[-p.lookback :].copy()
    close = window["close"].to_numpy(dtype=float)

    er = _efficiency_ratio(close)
    slope = _log_slope(close)

    if not np.isfinite(er) or not np.isfinite(slope):
        return "unclear"

    # Determine direction from slope (descriptive only).
    if er >= p.er_trend and abs(slope) >= p.min_abs_log_slope:
        return "trending_up" if slope > 0 else "trending_down"

    if er <= p.er_range:
        return "ranging"

    return "unclear"


def _make_sample_ohlcv(
    *,
    kind: Regime,
    n: int = 120,
    start_price: float = 100.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Create small synthetic OHLCV samples for demo/testing."""

    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")

    if kind == "trending_up":
        drift = 0.12
        noise = rng.normal(0, 0.18, size=n)
        close = start_price + np.cumsum(drift + noise)
    elif kind == "trending_down":
        drift = -0.12
        noise = rng.normal(0, 0.18, size=n)
        close = start_price + np.cumsum(drift + noise)
    elif kind == "ranging":
        # mean-reverting-ish around start_price
        close = np.empty(n)
        close[0] = start_price
        for i in range(1, n):
            close[i] = close[i - 1] + rng.normal(0, 0.25) - 0.05 * (close[i - 1] - start_price)
    else:
        # unclear: mix of jumps and chop
        noise = rng.normal(0, 0.45, size=n)
        jumps = (rng.random(n) < 0.03) * rng.normal(0, 3.0, size=n)
        close = start_price + np.cumsum(noise + jumps)

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.0, 0.35, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.0, 0.35, size=n)
    volume = rng.uniform(10.0, 100.0, size=n)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def demo_print_sample_regimes() -> None:
    """Small demo that prints the detected regime for sample OHLCV data."""

    params = RegimeParams(lookback=60)

    for expected in ("trending_up", "trending_down", "ranging", "unclear"):
        df = _make_sample_ohlcv(kind=expected)  # type: ignore[arg-type]
        detected = detect_market_regime(df, params=params)
        print(f"expected={expected:12s} detected={detected}")

    # Demonstrate the lookahead guard: as_of earlier than the last candle.
    df = _make_sample_ohlcv(kind="trending_up")
    cutoff = df["timestamp"].iloc[-20]
    try:
        detect_market_regime(df, as_of=cutoff, strict_lookahead=True)
        print("lookahead guard: unexpected success")
    except LookaheadBiasError as e:
        print(f"lookahead guard: OK ({e})")

    # Non-strict mode: truncates to as_of.
    detected_truncated = detect_market_regime(df, as_of=cutoff, strict_lookahead=False)
    print(f"non-strict truncation detected={detected_truncated}")


if __name__ == "__main__":
    demo_print_sample_regimes()
