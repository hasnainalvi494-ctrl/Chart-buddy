"""Replay utilities.

Goal
- Evaluate whether a thesis would have existed at multiple historical offsets
  (e.g., t-10, t-20, t-50 bars) using only data available at each point.

This helps validate that the *evidence* for a thesis existed before subsequent
price moves (no lookahead).

Key behavior
- Offsets are measured in **bars before the last candle**.
  - offset=0   -> as_of = last candle timestamp
  - offset=10  -> as_of = timestamp at index (-1 - 10)

Output
- A JSON-serializable dict describing thesis presence at each offset and whether
  it persisted across all offsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from chartbuddy.analysis.snapshot import build_symbol_snapshot
from chartbuddy.thesis.builder import build_short_thesis
from chartbuddy.thesis.schemas import Timeframe


@dataclass(frozen=True)
class ReplayParams:
    offsets: Sequence[int] = (10, 20, 50)
    strict_lookahead: bool = True


def _ensure_monotonic_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("candles DataFrame must include a 'timestamp' column")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("One or more candle timestamps could not be parsed")

    if not ts.is_monotonic_increasing:
        raise ValueError(
            "Candle timestamps must be monotonic increasing (oldest -> newest). "
            "Refusing to sort automatically to avoid hiding lookahead issues."
        )

    out = df.copy()
    out["timestamp"] = ts
    return out


def replay_thesis_persistence(
    candles: pd.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    offsets: Sequence[int] = (10, 20, 50),
    strict_lookahead: bool = True,
) -> Dict[str, Any]:
    """Replay snapshot+thesis at multiple offsets and return JSON-friendly output."""

    if candles is None or len(candles) == 0:
        raise ValueError("candles must be a non-empty DataFrame")

    if not symbol:
        raise ValueError("symbol is required")

    df = _ensure_monotonic_timestamp(candles)

    results: List[Dict[str, Any]] = []
    all_present = True

    for offset in offsets:
        if offset < 0:
            raise ValueError(f"offsets must be >= 0 (got {offset})")

        idx = -1 - int(offset)

        if abs(idx) > len(df):
            results.append(
                {
                    "offset": int(offset),
                    "status": "skipped",
                    "reason": f"not_enough_bars (need >= {offset+1}, have {len(df)})",
                }
            )
            all_present = False
            continue

        as_of_ts = df["timestamp"].iloc[idx]

        # Truncate to the candles that existed at that time.
        # This allows strict lookahead mode to remain enabled while ensuring
        # each analysis run only sees past/current data.
        df_at = df.iloc[: idx + 1].copy()

        snap = build_symbol_snapshot(
            df_at,
            symbol=symbol,
            timeframe=timeframe,
            as_of=as_of_ts,
            strict_lookahead=strict_lookahead,
        )

        thesis = build_short_thesis(snap)
        present = thesis is not None

        results.append(
            {
                "offset": int(offset),
                "status": "ok",
                "as_of": as_of_ts.isoformat(),
                "thesis_present": bool(present),
                "thesis_title": thesis.title if thesis is not None else None,
            }
        )

        all_present = all_present and present

    out: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe.value,
        "offsets": [int(o) for o in offsets],
        "persisted_all_offsets": bool(all_present),
        "results": results,
    }

    return out


def replay_to_json(
    candles: pd.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    offsets: Sequence[int] = (10, 20, 50),
    strict_lookahead: bool = True,
    indent: int = 2,
) -> str:
    """Convenience wrapper returning a JSON string."""

    payload = replay_thesis_persistence(
        candles,
        symbol=symbol,
        timeframe=timeframe,
        offsets=offsets,
        strict_lookahead=strict_lookahead,
    )

    return json.dumps(payload, indent=indent, sort_keys=True)


if __name__ == "__main__":
    # Minimal example:
    # - expects a CSV with at least: timestamp, open, high, low, close, volume
    # - run: PYTHONPATH=/workspace python3 chartbuddy/replay/replayer.py sample_ohlcv.csv TEST 1m
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: replayer.py <sample_ohlcv.csv> <SYMBOL> <TIMEFRAME> [offsets...]\n"
            "Example: replayer.py sample_ohlcv.csv BTCUSDT 1m 10 20 50"
        )
        raise SystemExit(2)

    path = sys.argv[1]
    sym = sys.argv[2]
    tf = Timeframe(sys.argv[3])
    offs = tuple(int(x) for x in sys.argv[4:]) if len(sys.argv) > 4 else (10, 20, 50)

    candles_df = pd.read_csv(path)
    print(replay_to_json(candles_df, symbol=sym, timeframe=tf, offsets=offs))
