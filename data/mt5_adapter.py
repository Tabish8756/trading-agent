
"""
MT5 market-data adapter for the XAUUSD recognition system.

Responsibilities:
- Connect to MetaTrader 5.
- Read symbol specifications.
- Retrieve historical OHLC candles.
- Retrieve tick data.
- Return normalized pandas DataFrames.
- Provide useful errors and clean shutdown.

This module does NOT place, modify, or close trades.
"""

from __future__ import annotations

import logging
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import pandas as pd


logger = logging.getLogger(__name__)


class MT5AdapterError(RuntimeError):
    """Raised when an MT5 data operation fails."""


class MT5Adapter:
    """Read-only market-data adapter for a MetaTrader 5 terminal."""

    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    def __init__(
        self,
        symbol: str = "XAUUSD",
        terminal_path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout_ms: int = 60_000,
        portable: bool = False,
    ) -> None:
        """
        Args:
            symbol: Broker-specific MT5 symbol name.
            terminal_path: Optional path to terminal64.exe.
            login: Optional MT5 account login.
            password: Optional account password.
            server: Optional broker server name.
            timeout_ms: Connection timeout in milliseconds.
            portable: Use MT5 portable mode if required.
        """
        self.symbol = symbol
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server
        self.timeout_ms = timeout_ms
        self.portable = portable
        self._connected = False

    # ---------------------------------------------------------
    # Connection management
    # ---------------------------------------------------------

    def connect(self) -> None:
        """Initialize MT5 and select the configured symbol."""

        if self._connected and mt5.terminal_info() is not None:
            self._ensure_symbol_selected()
            return

        kwargs: dict[str, Any] = {
            "timeout": self.timeout_ms,
            "portable": self.portable,
        }

        # Only pass explicitly provided account credentials.
        if self.login is not None:
            kwargs["login"] = self.login

        if self.password is not None:
            kwargs["password"] = self.password

        if self.server is not None:
            kwargs["server"] = self.server

        if self.terminal_path:
            initialized = mt5.initialize(
                self.terminal_path,
                **kwargs,
            )
        else:
            initialized = mt5.initialize(**kwargs)

        if not initialized:
            error = mt5.last_error()
            self._connected = False
            raise MT5AdapterError(
                f"MT5 initialization failed: {error}"
            )

        self._connected = True

        try:
            self._ensure_symbol_selected()
        except Exception:
            self.disconnect()
            raise

        logger.info("Connected to MT5. Symbol=%s", self.symbol)

    def disconnect(self) -> None:
        """Release the MT5 terminal connection."""

        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    def __enter__(self) -> "MT5Adapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def _require_connection(self) -> None:
        """Ensure MT5 is initialized before a data request."""

        if not self._connected:
            self.connect()

        if mt5.terminal_info() is None:
            self._connected = False
            raise MT5AdapterError(
                "MT5 terminal is unavailable. Reconnect and retry."
            )

    def _ensure_symbol_selected(self) -> None:
        """Make the requested symbol available in Market Watch."""

        info = mt5.symbol_info(self.symbol)

        if info is None:
            raise MT5AdapterError(
                f"Symbol '{self.symbol}' was not found. "
                "Check the exact broker symbol name."
            )

        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                raise MT5AdapterError(
                    f"Could not select '{self.symbol}': "
                    f"{mt5.last_error()}"
                )

    # ---------------------------------------------------------
    # Terminal and symbol information
    # ---------------------------------------------------------

    def get_terminal_info(self) -> dict[str, Any]:
        """Return useful terminal status information."""

        self._require_connection()

        info = mt5.terminal_info()
        if info is None:
            raise MT5AdapterError(
                f"Could not read terminal information: "
                f"{mt5.last_error()}"
            )

        return info._asdict()

    def get_account_info(self) -> dict[str, Any]:
        """
        Return account metadata for diagnostics.

        This method does not execute or modify trades.
        """

        self._require_connection()

        info = mt5.account_info()
        if info is None:
            raise MT5AdapterError(
                f"Could not read account information: "
                f"{mt5.last_error()}"
            )

        # Callers should avoid logging sensitive account details.
        return info._asdict()

    def get_symbol_info(self) -> dict[str, Any]:
        """Return symbol specifications and the latest available quote."""

        self._require_connection()

        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise MT5AdapterError(
                f"Could not read {self.symbol} specifications: "
                f"{mt5.last_error()}"
            )

        fields = [
            "name",
            "description",
            "currency_base",
            "currency_profit",
            "currency_margin",
            "digits",
            "point",
            "trade_tick_size",
            "trade_tick_value",
            "trade_contract_size",
            "volume_min",
            "volume_max",
            "volume_step",
            "trade_mode",
            "bid",
            "ask",
            "last",
            "spread",
            "visible",
            "trade_tick_value_profit",
            "trade_tick_value_loss",
        ]

        return {
            field: getattr(info, field, None)
            for field in fields
        }

    # ---------------------------------------------------------
    # Timeframe helpers
    # ---------------------------------------------------------

    def _get_timeframe(self, timeframe: str) -> int:
        """Convert a timeframe name such as M5 or H1 to MT5 enum."""

        normalized = timeframe.upper().strip()

        if normalized not in self.TIMEFRAMES:
            supported = ", ".join(self.TIMEFRAMES)
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported timeframes: {supported}"
            )

        return self.TIMEFRAMES[normalized]

    @staticmethod
    def _normalize_rates(rates) -> pd.DataFrame:
        """Convert MT5 rates to a DataFrame with UTC timestamps."""

        columns = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]

        if rates is None:
            raise MT5AdapterError(
                f"MT5 candle request failed: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)

        if df.empty:
            return pd.DataFrame(
                columns=columns + ["time_utc"]
            )

        df["time_utc"] = pd.to_datetime(
            df["time"], unit="s", utc=True
        )

        df = df.sort_values("time").reset_index(drop=True)

        return df

    # ---------------------------------------------------------
    # Historical candle retrieval
    # ---------------------------------------------------------

    def get_candles(
        self,
        timeframe: str,
        count: int = 1_000,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch the most recent candles.

        Args:
            timeframe: M1, M5, M15, M30, H1, H4 or D1.
            count: Number of candles requested.
            closed_only: If True, skip MT5 bar position 0,
                         which is normally the currently forming bar.

        Returns:
            Chronologically sorted DataFrame with a time_utc column.
        """

        if count < 1:
            raise ValueError("count must be at least 1")

        self._require_connection()
        tf = self._get_timeframe(timeframe)

        start_pos = 1 if closed_only else 0

        rates = mt5.copy_rates_from_pos(
            self.symbol,
            tf,
            start_pos,
            count,
        )

        df = self._normalize_rates(rates)

        if len(df) < count:
            logger.warning(
                "Requested %d %s candles for %s; received %d. "
                "Check terminal history and broker availability.",
                count,
                timeframe,
                self.symbol,
                len(df),
            )

        return df

    def get_candles_between(
        self,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch candles in a time range.

        Start/end must be timezone-aware datetimes. Candle timestamps
        returned by MT5 are interpreted as UTC.

        Note: A candle timestamp is its opening time. A candle that
        opened before 'end' may still be forming at the end boundary.
        Use closed-only filtering in the caller when necessary.
        """

        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")

        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("end must be timezone-aware")

        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)

        if start_utc >= end_utc:
            raise ValueError("start must be earlier than end")

        self._require_connection()
        tf = self._get_timeframe(timeframe)

        rates = mt5.copy_rates_range(
            self.symbol,
            tf,
            start_utc,
            end_utc,
        )

        df = self._normalize_rates(rates)

        # Keep candle opening timestamps inside [start, end).
        start_ts = pd.Timestamp(start_utc)
        end_ts = pd.Timestamp(end_utc)

        return df.loc[
            (df["time_utc"] >= start_ts)
            & (df["time_utc"] < end_ts)
        ].reset_index(drop=True)

    # ---------------------------------------------------------
    # Tick retrieval
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_ticks(ticks) -> pd.DataFrame:
        """Convert MT5 tick records to a normalized DataFrame."""

        columns = [
            "time",
            "time_msc",
            "bid",
            "ask",
            "last",
            "volume",
            "volume_real",
            "flags",
            "time_utc",
        ]

        if ticks is None:
            raise MT5AdapterError(
                f"MT5 tick request failed: {mt5.last_error()}"
            )

        df = pd.DataFrame(ticks)

        if df.empty:
            return pd.DataFrame(columns=columns)

        if "time_msc" in df.columns:
            df["time_utc"] = pd.to_datetime(
                df["time_msc"],
                unit="ms",
                utc=True,
            )
        else:
            df["time_utc"] = pd.to_datetime(
                df["time"],
                unit="s",
                utc=True,
            )

        df = df.sort_values(
            ["time_utc"]
        ).reset_index(drop=True)

        return df

    def get_ticks_between(
        self,
        start: datetime,
        end: datetime,
        flags: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch ticks between timezone-aware datetimes.

        By default, request all tick types. MT5 may provide only
        quote updates or incomplete trade-volume information,
        depending on the broker and instrument.
        """

        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")

        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("end must be timezone-aware")

        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)

        if start_utc >= end_utc:
            raise ValueError("start must be earlier than end")

        self._require_connection()

        tick_flags = (
            mt5.COPY_TICKS_ALL if flags is None else flags
        )

        ticks = mt5.copy_ticks_range(
            self.symbol,
            start_utc,
            end_utc,
            tick_flags,
        )

        df = self._normalize_ticks(ticks)

        # Enforce the requested half-open interval.
        start_ts = pd.Timestamp(start_utc)
        end_ts = pd.Timestamp(end_utc)

        return df.loc[
            (df["time_utc"] >= start_ts)
            & (df["time_utc"] < end_ts)
        ].reset_index(drop=True)

     
    def get_recent_ticks(
    self,
    lookback_seconds: int = 60,
    count: int = 10_000,
    flags: int | None = None,
) -> pd.DataFrame:
     """
    Fetch ticks in a recent UTC time window.

    Uses a time-range query and explicitly filters the result.
    The returned DataFrame may be empty if no ticks are available.
    """

     if lookback_seconds < 1:
        raise ValueError("lookback_seconds must be positive")

     if count < 1:
        raise ValueError("count must be positive")

     self._require_connection()

     end = datetime.now(timezone.utc)
     start = end - timedelta(seconds=lookback_seconds)

     tick_flags = (
        mt5.COPY_TICKS_ALL if flags is None else flags
     )

     ticks = mt5.copy_ticks_range(
        self.symbol,
        start,
        end,
        tick_flags,
    )

     df = self._normalize_ticks(ticks)

     if df.empty:
        return df

    # Explicitly enforce the requested time window.
     start_ts = pd.Timestamp(start)
     end_ts = pd.Timestamp(end)

     df = df.loc[
        (df["time_utc"] >= start_ts)
        & (df["time_utc"] <= end_ts)
    ]

    # If the response exceeds the requested limit, retain
    # the newest records rather than the oldest ones.
     return df.tail(count).reset_index(drop=True)


    def get_latest_closed_candle(
        self,
        timeframe: str,
    ) -> dict[str, Any]:
        """Return the most recent completed candle as a dictionary."""

        df = self.get_candles(
            timeframe=timeframe,
            count=1,
            closed_only=True,
        )

        if df.empty:
            raise MT5AdapterError(
                f"No closed {timeframe} candle available."
            )

        return df.iloc[-1].to_dict()

    def get_latest_quote(self) -> dict[str, Any]:
        """Return the latest broker-provided symbol quote."""

        self._require_connection()

        tick = mt5.symbol_info_tick(self.symbol)

        if tick is None:
            raise MT5AdapterError(
                f"Could not retrieve latest quote: "
                f"{mt5.last_error()}"
            )

        result = tick._asdict()

        timestamp_ms = result.get("time_msc")
        if timestamp_ms:
            result["time_utc"] = pd.to_datetime(
                timestamp_ms, unit="ms", utc=True
            )
        else:
            result["time_utc"] = pd.to_datetime(
                result["time"], unit="s", utc=True
            )

        return result
