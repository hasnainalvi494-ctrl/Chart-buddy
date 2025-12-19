"""Short thesis builder (snapshot-only).

Consumes a `SymbolSnapshot` (from `chartbuddy.analysis.snapshot`) and constructs a
`ShortThesis` *only* when all required descriptive conditions are met.

This module intentionally:
- uses only snapshot fields (no future data)
- does not output scores
- does not output buy/sell signals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from chartbuddy.analysis.snapshot import SymbolSnapshot
from chartbuddy.thesis.schemas import (
    Direction,
    MarketRegimeKind,
    ShortThesis,
    StructureState,
    VectorCandlePattern,
)


@dataclass(frozen=True)
class ThesisParams:
    """Heuristics for interpreting snapshot notes."""

    # If last swing high is within this % of range_high, treat as "at range high".
    range_high_tol_pct: float = 0.001


def _parse_kv_notes(notes: Optional[str]) -> dict:
    """Parse a semi-colon separated key=value notes string."""

    if not notes:
        return {}

    out = {}
    for part in notes.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _structure_has_lower_highs_or_range_highs(structure: StructureState, *, p: ThesisParams) -> bool:
    notes = _parse_kv_notes(structure.notes)

    # Preferred: explicit swing relationship annotation.
    if notes.get("swing_high_relation") == "LH":
        return True

    # Fallback: if the last swing high is essentially at the range high.
    if structure.swing_high is not None and structure.range_high is not None and structure.range_high != 0:
        diff_pct = abs(structure.swing_high - structure.range_high) / abs(structure.range_high)
        if diff_pct <= p.range_high_tol_pct:
            return True

    return False


def _buy_side_sweep_failed(snapshot: SymbolSnapshot) -> bool:
    """Buy-side sweep failed == most recent buy-side sweep classified as rejection."""

    for sweep in snapshot.liquidity.recent_sweeps:
        # In our liquidity module, buy-side sweeps use Direction.UP and notes include classification.
        if sweep.direction != Direction.UP:
            continue

        notes = sweep.notes or ""
        # The classifier writes: "classification=acceptance|rejection"
        return "classification=rejection" in notes

    return False


def _volume_did_not_expand(snapshot: SymbolSnapshot) -> bool:
    notes = snapshot.volume.notes or ""
    # Volume module writes: "volume=expansion|contraction|normal"
    return "volume=expansion" not in notes


def build_short_thesis(
    snapshot: SymbolSnapshot,
    *,
    params: Optional[ThesisParams] = None,
) -> Optional[ShortThesis]:
    """Construct a `ShortThesis` when all gating conditions are met.

    Conditions (all must be true):
    - Regime is ranging or bearish
    - Structure shows lower highs or range highs
    - Buy-side liquidity sweep failed
    - Volume did not expand
    - Vector candles show exhaustion

    Returns None if conditions are not met.
    """

    p = params or ThesisParams()

    # 1) Regime ranging or bearish
    regime_ok = snapshot.market_regime.kind in (MarketRegimeKind.RANGE, MarketRegimeKind.TREND_DOWN)

    # 2) Structure lower highs or range highs
    structure_ok = _structure_has_lower_highs_or_range_highs(snapshot.structure, p=p)

    # 3) Buy-side liquidity sweep failed
    sweep_ok = _buy_side_sweep_failed(snapshot)

    # 4) Volume did not expand
    volume_ok = _volume_did_not_expand(snapshot)

    # 5) Vector candle exhaustion
    vector_ok = snapshot.vector_candle.pattern == VectorCandlePattern.EXHAUSTION

    if not (regime_ok and structure_ok and sweep_ok and volume_ok and vector_ok):
        return None

    title = "Snapshot thesis: range/bear + LH/range-high + failed buy-side sweep + no vol expansion + exhaustion"
    summary = (
        "All gating conditions satisfied (descriptive only): "
        "regime in {ranging,bearish}, structure shows LH or at range high, "
        "buy-side sweep rejected, volume not expanded, and recent candles show exhaustion."
    )

    return ShortThesis(
        as_of=snapshot.as_of,
        symbol=snapshot.symbol,
        timeframe=snapshot.timeframe,
        title=title,
        summary=summary,
        market_regime=snapshot.market_regime,
        structure=snapshot.structure,
        liquidity=snapshot.liquidity,
        volume=snapshot.volume,
        vector_candle=snapshot.vector_candle,
        tags=["snapshot_only", "gated_thesis"],
        metadata={"builder": "thesis/builder.py"},
    )


def _make_demo_snapshot(*, symbol: str, meets: bool) -> SymbolSnapshot:
    """Create a small synthetic snapshot for demo printing.

    This constructs the snapshot object directly (no trading logic).
    """

    from datetime import datetime, timezone

    from chartbuddy.analysis.snapshot import SymbolSnapshot
    from chartbuddy.thesis.schemas import (
        LiquidityPool,
        LiquidityState,
        LiquiditySweep,
        MarketRegime,
        MarketRegimeKind,
        StructureState,
        Timeframe,
        VectorCandlePattern,
        VectorCandleState,
        VolumeState,
    )

    as_of = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    tf = Timeframe.M1

    market_regime = MarketRegime(
        as_of=as_of,
        symbol=symbol,
        timeframe=tf,
        kind=MarketRegimeKind.RANGE if meets else MarketRegimeKind.TREND_UP,
    )

    structure = StructureState(
        as_of=as_of,
        symbol=symbol,
        timeframe=tf,
        swing_high=105.0,
        swing_low=95.0,
        trend=Direction.UNKNOWN,
        range_high=105.0,
        range_low=95.0,
        notes="pivot(pivot_left=2, pivot_right=2); swing_high_relation=LH; swing_low_relation=HL; pivots_high=3; pivots_low=3; range_lookback=100",
    )

    liquidity = LiquidityState(
        as_of=as_of,
        symbol=symbol,
        timeframe=tf,
        pools=[LiquidityPool(kind="buy_side_equal_highs", level=105.0, observed_at=as_of)],
        recent_sweeps=(
            [
                LiquiditySweep(
                    level=105.0,
                    occurred_at=as_of,
                    direction=Direction.UP,
                    notes="side=buy; classification=rejection; post_bars=3",
                )
            ]
            if meets
            else []
        ),
        notes="demo",
    )

    volume = VolumeState(
        as_of=as_of,
        symbol=symbol,
        timeframe=tf,
        last_candle_volume=50.0,
        last_n_candles_volume_sum=1000.0,
        last_n_candles=20,
        notes="volume=normal (relative_to_baseline); baseline_lookback=50",
    )

    vector = VectorCandleState(
        as_of=as_of,
        symbol=symbol,
        timeframe=tf,
        open=100.0,
        high=106.0,
        low=94.0,
        close=99.0,
        volume=50.0,
        pattern=VectorCandlePattern.EXHAUSTION if meets else VectorCandlePattern.IMPULSE,
        notes="demo",
    )

    return SymbolSnapshot(
        as_of=as_of,
        symbol=symbol,
        timeframe=tf,
        market_regime=market_regime,
        structure=structure,
        liquidity=liquidity,
        volume=volume,
        vector_candle=vector,
    )


if __name__ == "__main__":
    # Simple printing test for sample symbols.
    for sym, meets in (("SAMPLE_OK", True), ("SAMPLE_NO", False)):
        snap = _make_demo_snapshot(symbol=sym, meets=meets)
        thesis = build_short_thesis(snap)
        print(sym, "->", "THESIS" if thesis is not None else "None")
        if thesis is not None:
            # Print a compact, descriptive view.
            print({
                "as_of": thesis.as_of.isoformat(),
                "symbol": thesis.symbol,
                "timeframe": thesis.timeframe.value,
                "regime": thesis.market_regime.kind.value if thesis.market_regime else None,
                "structure_notes": thesis.structure.notes if thesis.structure else None,
                "liquidity_sweeps": [s.notes for s in thesis.liquidity.recent_sweeps] if thesis.liquidity else [],
                "volume_notes": thesis.volume.notes if thesis.volume else None,
                "vector_pattern": thesis.vector_candle.pattern.value if thesis.vector_candle and thesis.vector_candle.pattern else None,
            })
