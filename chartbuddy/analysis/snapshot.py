"""Snapshot aggregation.

Build a single immutable snapshot object per symbol by aggregating:
- regime
- structure
- liquidity
- volume
- vector_candles

This module is snapshot-only:
- uses only candles up to `as_of` (if provided)
- refuses to auto-sort timestamps to avoid hiding lookahead bias
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import pandas as pd

from chartbuddy.analysis.liquidity import detect_liquidity_state
from chartbuddy.analysis.regime import detect_market_regime
from chartbuddy.analysis.structure import detect_structure_state
from chartbuddy.analysis.vector_candles import analyze_vector_candle_state
from chartbuddy.analysis.volume import analyze_volume_state
from chartbuddy.thesis.schemas import (
    MarketRegime,
    MarketRegimeKind,
    Timeframe,
    VectorCandleState,
    LiquidityState,
    StructureState,
    VolumeState,
)


@dataclass(frozen=True)
class SymbolSnapshot:
    """Immutable aggregated snapshot for a single symbol."""

    as_of: datetime
    symbol: str
    timeframe: Timeframe

    market_regime: MarketRegime
    structure: StructureState
    liquidity: LiquidityState
    volume: VolumeState
    vector_candle: VectorCandleState


def _map_regime_label_to_kind(label: str) -> MarketRegimeKind:
    if label == "trending_up":
        return MarketRegimeKind.TREND_UP
    if label == "trending_down":
        return MarketRegimeKind.TREND_DOWN
    if label == "ranging":
        return MarketRegimeKind.RANGE
    return MarketRegimeKind.UNKNOWN


def build_symbol_snapshot(
    candles: pd.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    as_of: Optional[Union[str, int, float, datetime, pd.Timestamp]] = None,
    strict_lookahead: bool = True,
) -> SymbolSnapshot:
    """Aggregate analysis modules into one immutable snapshot object."""

    if candles is None or len(candles) == 0:
        raise ValueError("candles must be a non-empty DataFrame")

    if not symbol:
        raise ValueError("symbol is required")

    # Enforce monotonic timestamps here too, so we fail early.
    if "timestamp" not in candles.columns:
        raise ValueError("candles DataFrame must include a 'timestamp' column")

    ts = pd.to_datetime(candles["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("One or more candle timestamps could not be parsed")

    if not ts.is_monotonic_increasing:
        raise ValueError(
            "Candle timestamps must be monotonic increasing (oldest -> newest). "
            "Refusing to sort automatically to avoid hiding lookahead issues."
        )

    as_of_ts = pd.to_datetime(as_of, utc=True) if as_of is not None else ts.iloc[-1]

    # Truncate to data that existed at `as_of_ts` to keep strict lookahead enabled
    # while ensuring each analyzer only sees past/current candles.
    mask = ts <= as_of_ts
    if not mask.any():
        raise ValueError("No candles at or before as_of")

    candles_at = candles.loc[mask].copy()
    candles_at["timestamp"] = ts.loc[mask]

    # Use the same candles input for all modules; each module applies its own
    # lookahead guard (including as_of cutoff behavior).
    regime_label = detect_market_regime(candles_at, as_of=as_of_ts, strict_lookahead=strict_lookahead)

    market_regime = MarketRegime(
        as_of=as_of_ts.to_pydatetime(),
        symbol=symbol,
        timeframe=timeframe,
        kind=_map_regime_label_to_kind(regime_label),
    )

    structure = detect_structure_state(
        candles_at,
        symbol=symbol,
        timeframe=timeframe,
        as_of=as_of_ts,
        strict_lookahead=strict_lookahead,
    )

    liquidity = detect_liquidity_state(
        candles_at,
        symbol=symbol,
        timeframe=timeframe,
        as_of=as_of_ts,
        strict_lookahead=strict_lookahead,
    )

    volume = analyze_volume_state(
        candles_at,
        symbol=symbol,
        timeframe=timeframe,
        as_of=as_of_ts,
        strict_lookahead=strict_lookahead,
    )

    vector_candle = analyze_vector_candle_state(
        candles_at,
        symbol=symbol,
        timeframe=timeframe,
        as_of=as_of_ts,
        strict_lookahead=strict_lookahead,
    )

    return SymbolSnapshot(
        as_of=as_of_ts.to_pydatetime(),
        symbol=symbol,
        timeframe=timeframe,
        market_regime=market_regime,
        structure=structure,
        liquidity=liquidity,
        volume=volume,
        vector_candle=vector_candle,
    )
