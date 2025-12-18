"""Snapshot-only schemas for thesis state.

These models intentionally avoid:
- forward-looking fields (no predictions/targets)
- scores (no numeric confidence/grading)
- signals (no entries/exits, no "should" actions)

They are meant to capture *what is observed now*.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Pydantic v2 uses ConfigDict; v1 uses inner Config. We support both.
try:  # pragma: no cover
    from pydantic import ConfigDict  # type: ignore

    _V2 = True
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore
    _V2 = False


class Timeframe(str, Enum):
    """Common timeframe labels (stringly-typed for interoperability)."""

    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H12 = "12h"
    D1 = "1d"


class MarketRegimeKind(str, Enum):
    """High-level regime classification (descriptive, not predictive)."""

    RANGE = "range"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    VOLATILE = "volatile"
    QUIET = "quiet"
    UNKNOWN = "unknown"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class VectorCandlePattern(str, Enum):
    """Descriptive multi-candle pattern classification (not a signal)."""

    IMPULSE = "impulse"
    EXHAUSTION = "exhaustion"
    COMPRESSION = "compression"
    UNCLEAR = "unclear"


class BaseSnapshotModel(BaseModel):
    """Base model with consistent serialization defaults."""

    if _V2:
        model_config = ConfigDict(extra="forbid", frozen=False)

    else:  # pragma: no cover

        class Config:
            extra = "forbid"


class MarketRegime(BaseSnapshotModel):
    """Observed market regime at a point in time."""

    as_of: datetime = Field(..., description="Timestamp the regime snapshot refers to.")
    symbol: str = Field(..., min_length=1, description="Market symbol, e.g. BTCUSDT.")
    timeframe: Timeframe = Field(..., description="Timeframe of the snapshot.")

    kind: MarketRegimeKind = Field(MarketRegimeKind.UNKNOWN, description="Descriptive regime kind.")
    direction: Direction = Field(Direction.UNKNOWN, description="Observed directional bias (descriptive).")

    session: Optional[str] = Field(
        default=None,
        description="Optional session label (e.g. 'asia', 'london', 'ny').",
    )
    notes: Optional[str] = Field(default=None, description="Free-form notes about current conditions.")


class StructureState(BaseSnapshotModel):
    """Observed market structure state (levels + last known events)."""

    as_of: datetime
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe

    swing_high: Optional[float] = Field(
        default=None, description="Most recent identified swing high price (if available)."
    )
    swing_low: Optional[float] = Field(
        default=None, description="Most recent identified swing low price (if available)."
    )

    trend: Direction = Field(Direction.UNKNOWN, description="Descriptive trend direction.")

    last_break_of_structure_at: Optional[datetime] = Field(
        default=None, description="Timestamp of last observed break of structure (if tracked)."
    )
    last_break_of_structure_level: Optional[float] = Field(
        default=None, description="Price level associated with last observed break of structure."
    )

    last_change_of_character_at: Optional[datetime] = Field(
        default=None, description="Timestamp of last observed change of character (if tracked)."
    )
    last_change_of_character_level: Optional[float] = Field(
        default=None, description="Price level associated with last observed change of character."
    )

    range_high: Optional[float] = Field(default=None, description="Observed range high, if ranging.")
    range_low: Optional[float] = Field(default=None, description="Observed range low, if ranging.")

    notes: Optional[str] = None


class LiquidityPool(BaseSnapshotModel):
    """A single observed liquidity pool (e.g. equal highs/lows, obvious stops)."""

    kind: str = Field(..., min_length=1, description="Descriptor, e.g. 'equal_highs', 'equal_lows'.")
    level: float = Field(..., description="Price level where liquidity is believed to sit.")
    observed_at: Optional[datetime] = Field(default=None, description="When this pool was observed.")
    notes: Optional[str] = None


class LiquiditySweep(BaseSnapshotModel):
    """Observed sweep/run on a prior level (purely historical, snapshot-anchored)."""

    level: float = Field(..., description="Swept level.")
    occurred_at: datetime = Field(..., description="When the sweep was observed to occur.")
    direction: Direction = Field(..., description="Direction of sweep relative to prior level.")
    notes: Optional[str] = None


class LiquidityState(BaseSnapshotModel):
    """Observed liquidity landscape (pools + recent sweeps)."""

    as_of: datetime
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe

    pools: List[LiquidityPool] = Field(default_factory=list, description="Observed liquidity pools.")
    recent_sweeps: List[LiquiditySweep] = Field(
        default_factory=list, description="Observed recent sweeps (historical events)."
    )

    notes: Optional[str] = None


class VolumeState(BaseSnapshotModel):
    """Observed volume-related information (raw values; no scoring)."""

    as_of: datetime
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe

    last_candle_volume: Optional[float] = Field(default=None, description="Volume of the last candle.")
    last_n_candles_volume_sum: Optional[float] = Field(
        default=None, description="Sum of volume over a recent window (if computed)."
    )
    last_n_candles: Optional[int] = Field(
        default=None, ge=1, description="Window size used for last_n_candles_volume_sum."
    )

    quote_volume: Optional[float] = Field(
        default=None, description="Quote volume if provided by venue/data source."
    )
    trade_count: Optional[int] = Field(default=None, ge=0, description="Trade count if available.")

    notes: Optional[str] = None


class VectorCandleState(BaseSnapshotModel):
    """Observed last candle state (OHLCV + optional venue fields).

    This is a snapshot of a *completed* candle (or last known candle), not a forecast.
    """

    as_of: datetime = Field(..., description="Timestamp the candle snapshot was taken.")
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe

    open: float
    high: float
    low: float
    close: float

    volume: Optional[float] = None
    quote_volume: Optional[float] = None
    trades: Optional[int] = Field(default=None, ge=0)

    taker_buy_base_volume: Optional[float] = None
    taker_buy_quote_volume: Optional[float] = None

    candle_open_time: Optional[datetime] = Field(
        default=None, description="Open time of the candle if known."
    )
    candle_close_time: Optional[datetime] = Field(
        default=None, description="Close time of the candle if known."
    )

    source: Optional[str] = Field(
        default=None, description="Data source label (e.g. 'binance_spot')."
    )

    pattern: Optional[VectorCandlePattern] = Field(
        default=None,
        description="Descriptive pattern classification from recent candles (not a signal).",
    )
    notes: Optional[str] = Field(default=None, description="Free-form snapshot notes.")


class ShortThesis(BaseSnapshotModel):
    """Snapshot-only thesis container.

    Holds observed context and component states. This is *not* a signal object.
    """

    as_of: datetime
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe

    title: Optional[str] = Field(default=None, description="Short human-readable title.")
    summary: Optional[str] = Field(default=None, description="Free-form snapshot summary.")

    market_regime: Optional[MarketRegime] = None
    structure: Optional[StructureState] = None
    liquidity: Optional[LiquidityState] = None
    volume: Optional[VolumeState] = None
    vector_candle: Optional[VectorCandleState] = None

    tags: List[str] = Field(default_factory=list, description="Optional categorical labels.")

    # For extensibility without locking in a schema too early.
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary snapshot metadata (venue, data version, etc.).",
    )
