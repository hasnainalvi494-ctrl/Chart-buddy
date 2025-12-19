"""Binance Futures execution client.

Execution-only responsibilities:
- Fetch OHLCV (futures klines)
- Place limit orders
- Place reduce-only stop orders

No analysis, scoring, regime detection, or thesis logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, Union

import pandas as pd

try:
    # python-binance
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except Exception as e:  # pragma: no cover
    Client = None  # type: ignore
    BinanceAPIException = Exception  # type: ignore
    BinanceRequestException = Exception  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


Side = Literal["BUY", "SELL"]
TimeInForce = Literal["GTC", "IOC", "FOK"]
StopOrderType = Literal["STOP", "STOP_MARKET"]


class BinanceClientImportError(RuntimeError):
    """Raised when python-binance is unavailable."""


@dataclass(frozen=True)
class BinanceFuturesConfig:
    api_key: str
    api_secret: str
    testnet: bool = False
    # Optional: set a request timeout in seconds.
    requests_params: Optional[Dict[str, Any]] = None


def _to_millis(ts: Union[int, float, datetime, pd.Timestamp, str, None]) -> Optional[int]:
    if ts is None:
        return None

    if isinstance(ts, (int, float)):
        # assume epoch ms if large, else seconds
        return int(ts if ts > 10_000_000_000 else ts * 1000)

    if isinstance(ts, str):
        ts = pd.to_datetime(ts, utc=True)

    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp() * 1000)


class BinanceFuturesClient:
    """Thin wrapper around Binance USDT-M Futures endpoints."""

    def __init__(self, config: BinanceFuturesConfig):
        if _IMPORT_ERROR is not None or Client is None:
            raise BinanceClientImportError(
                "python-binance is not installed/available in this environment"
            ) from _IMPORT_ERROR

        self._client = Client(
            api_key=config.api_key,
            api_secret=config.api_secret,
            requests_params=config.requests_params,
        )

        if config.testnet:
            # USDT-M Futures testnet base URL.
            # python-binance uses `FUTURES_URL` constant on the client instance.
            self._client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

    # -----------------
    # Market data
    # -----------------
    def fetch_ohlcv(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[Union[int, float, datetime, pd.Timestamp, str]] = None,
        end_time: Optional[Union[int, float, datetime, pd.Timestamp, str]] = None,
    ) -> pd.DataFrame:
        """Fetch futures klines and return as a DataFrame.

        Parameters
        - symbol: e.g. "BTCUSDT"
        - interval: Binance interval string, e.g. "1m", "5m", "1h"
        - limit: number of klines to request (max varies by endpoint; commonly 1500)
        - start_time/end_time: optional epoch(ms)/datetime/ISO string

        Returns
        DataFrame with standard Binance kline fields.
        """

        if not symbol:
            raise ValueError("symbol is required")

        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit),
        }

        st = _to_millis(start_time)
        et = _to_millis(end_time)
        if st is not None:
            params["startTime"] = st
        if et is not None:
            params["endTime"] = et

        try:
            rows = self._client.futures_klines(**params)
        except (BinanceAPIException, BinanceRequestException) as e:
            raise RuntimeError(f"Failed to fetch klines for {symbol} {interval}") from e

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )

        # Parse types.
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)

        for col in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype(int)

        return df

    # -----------------
    # Orders
    # -----------------
    def place_limit_order(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Union[int, float, str],
        price: Union[int, float, str],
        time_in_force: TimeInForce = "GTC",
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place a futures limit order."""

        if not symbol:
            raise ValueError("symbol is required")

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": str(quantity),
            "price": str(price),
            "reduceOnly": bool(reduce_only),
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        try:
            return self._client.futures_create_order(**params)
        except (BinanceAPIException, BinanceRequestException) as e:
            raise RuntimeError(f"Failed to place limit order for {symbol}") from e

    def place_reduce_only_stop_order(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Union[int, float, str],
        stop_price: Union[int, float, str],
        order_type: StopOrderType = "STOP_MARKET",
        price: Optional[Union[int, float, str]] = None,
        time_in_force: TimeInForce = "GTC",
        working_type: Literal["CONTRACT_PRICE", "MARK_PRICE"] = "CONTRACT_PRICE",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place a reduce-only stop order on Binance USDT-M Futures.

        Notes
        - STOP_MARKET: triggers a market order when stopPrice is hit.
        - STOP: triggers a limit order when stopPrice is hit (requires `price`).
        """

        if not symbol:
            raise ValueError("symbol is required")

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
            "stopPrice": str(stop_price),
            "reduceOnly": True,
            "workingType": working_type,
        }

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        if order_type == "STOP":
            if price is None:
                raise ValueError("price is required for STOP (limit) orders")
            params["price"] = str(price)
            params["timeInForce"] = time_in_force

        try:
            return self._client.futures_create_order(**params)
        except (BinanceAPIException, BinanceRequestException) as e:
            raise RuntimeError(f"Failed to place reduce-only stop order for {symbol}") from e
