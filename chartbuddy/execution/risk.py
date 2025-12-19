"""Execution risk management.

Provides:
- Fixed-risk position sizing per trade (based on equity and stop distance)
- Max concurrent trades rule
- Max daily drawdown rule

This module is execution-focused and contains no analysis logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Optional


class RiskRuleViolation(RuntimeError):
    """Raised when a risk rule prevents an action."""


@dataclass(frozen=True)
class RiskConfig:
    """Risk parameters.

    All % values are expressed as decimals. Example:
    - risk_per_trade_pct=0.01 means 1% risk per trade.
    - max_daily_drawdown_pct=0.03 means stop trading after 3% drawdown.
    """

    risk_per_trade_pct: float = 0.01
    max_concurrent_trades: int = 3
    max_daily_drawdown_pct: float = 0.03

    # Optional rounding to match exchange lot size increments.
    quantity_step: Optional[float] = None
    min_quantity: Optional[float] = None


@dataclass
class RiskState:
    """Mutable state for applying risk rules."""

    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    day_start_equity: float = 0.0
    current_equity: float = 0.0

    open_trades: int = 0

    # Optional bookkeeping
    realized_pnl_today: float = 0.0
    peak_equity_today: float = 0.0


def _utc_day(ts: Optional[datetime] = None) -> date:
    ts = ts or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).date()


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    return (x // step) * step


def fixed_risk_position_size(
    *,
    equity: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_price: float,
    contract_multiplier: float = 1.0,
    quantity_step: Optional[float] = None,
    min_quantity: Optional[float] = None,
) -> float:
    """Compute position size (quantity) given fixed risk.

    Quantity is computed such that:
        worst_case_loss ~= |entry_price - stop_price| * quantity * contract_multiplier
        worst_case_loss == equity * risk_per_trade_pct

    Notes
    - Direction-agnostic: uses absolute stop distance.
    - Does not account for fees/slippage (execution layer can pad stop distance if desired).
    """

    if equity <= 0:
        raise ValueError("equity must be > 0")
    if not (0 < risk_per_trade_pct < 1):
        raise ValueError("risk_per_trade_pct must be between 0 and 1")
    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("entry_price and stop_price must be > 0")
    if contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be > 0")

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        raise ValueError("stop_distance must be > 0 (entry_price != stop_price)")

    risk_amount = equity * risk_per_trade_pct
    qty = risk_amount / (stop_distance * contract_multiplier)

    if quantity_step is not None:
        qty = _round_step(qty, quantity_step)

    if min_quantity is not None and qty < min_quantity:
        # Return 0 to indicate sizing not possible under min lot constraints.
        return 0.0

    return float(qty)


class RiskManager:
    """Applies risk sizing and trading halts based on configured rules."""

    def __init__(self, *, config: RiskConfig, starting_equity: float, now: Optional[datetime] = None):
        if starting_equity <= 0:
            raise ValueError("starting_equity must be > 0")

        self.config = config
        day = _utc_day(now)
        self.state = RiskState(
            day=day,
            day_start_equity=float(starting_equity),
            current_equity=float(starting_equity),
            peak_equity_today=float(starting_equity),
        )

    def _roll_day_if_needed(self, now: Optional[datetime] = None) -> None:
        day = _utc_day(now)
        if day != self.state.day:
            # Reset daily counters using current equity as the new day start.
            self.state.day = day
            self.state.day_start_equity = float(self.state.current_equity)
            self.state.peak_equity_today = float(self.state.current_equity)
            self.state.realized_pnl_today = 0.0

    def update_equity(self, *, equity: float, now: Optional[datetime] = None) -> None:
        """Update current equity. Caller can use this with mark-to-market equity."""

        if equity <= 0:
            raise ValueError("equity must be > 0")

        self._roll_day_if_needed(now)
        self.state.current_equity = float(equity)
        if equity > self.state.peak_equity_today:
            self.state.peak_equity_today = float(equity)

    def record_realized_pnl(self, *, pnl: float, now: Optional[datetime] = None) -> None:
        """Record realized PnL for the day (optional bookkeeping)."""

        self._roll_day_if_needed(now)
        self.state.realized_pnl_today += float(pnl)

    def daily_drawdown_pct(self, now: Optional[datetime] = None) -> float:
        self._roll_day_if_needed(now)
        start = self.state.day_start_equity
        if start <= 0:
            return 0.0
        dd = (start - self.state.current_equity) / start
        return float(max(0.0, dd))

    def can_open_new_trade(self, now: Optional[datetime] = None) -> None:
        """Raises RiskRuleViolation if a new trade is not allowed."""

        self._roll_day_if_needed(now)

        if self.state.open_trades >= self.config.max_concurrent_trades:
            raise RiskRuleViolation(
                f"max_concurrent_trades reached ({self.state.open_trades}/{self.config.max_concurrent_trades})"
            )

        dd = self.daily_drawdown_pct(now)
        if dd >= self.config.max_daily_drawdown_pct:
            raise RiskRuleViolation(
                f"max_daily_drawdown exceeded (dd={dd:.4f} >= {self.config.max_daily_drawdown_pct:.4f})"
            )

    def reserve_trade_slot(self, now: Optional[datetime] = None) -> None:
        """Increment open trade count if rules allow."""

        self.can_open_new_trade(now)
        self.state.open_trades += 1

    def release_trade_slot(self) -> None:
        """Decrement open trade count."""

        if self.state.open_trades <= 0:
            self.state.open_trades = 0
            return
        self.state.open_trades -= 1

    def size_for_trade(
        self,
        *,
        entry_price: float,
        stop_price: float,
        contract_multiplier: float = 1.0,
        equity: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> float:
        """Compute fixed-risk position size using current (or provided) equity.

        This does not change open trade counts; use reserve_trade_slot/release_trade_slot
        to apply concurrency limits.
        """

        self._roll_day_if_needed(now)
        use_equity = float(equity) if equity is not None else float(self.state.current_equity)

        return fixed_risk_position_size(
            equity=use_equity,
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            contract_multiplier=contract_multiplier,
            quantity_step=self.config.quantity_step,
            min_quantity=self.config.min_quantity,
        )
