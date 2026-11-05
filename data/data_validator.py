
"""
Data quality checks for the XAUUSD recognition system.

Validation only: this module does not silently alter market data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


class DataValidator:
    TIMEFRAME_SECONDS = {
        "M1": 60,
        "M5": 300,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H4": 14400,
        "D1": 86400,
    }

    REQUIRED_CANDLE_COLUMNS = [
        "time_utc",
        "open",
        "high",
        "low",
        "close",
    ]

    REQUIRED_TICK_COLUMNS = [
        "time_utc",
        "bid",
        "ask",
    ]

    @staticmethod
    def _now_utc(now: datetime | None = None) -> pd.Timestamp:
        if now is None:
            return pd.Timestamp.now(tz="UTC")

        timestamp = pd.Timestamp(now)

        if timestamp.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        return timestamp.tz_convert("UTC")

    def validate_candles(
        self,
        df: pd.DataFrame,
        timeframe: str,
        now: datetime | None = None,
        max_age_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Validate normalized MT5 candle data.

        Missing intervals are warnings rather than automatic errors:
        markets can close, and brokers may omit bars with no ticks.
        """

        timeframe = timeframe.upper()

        if timeframe not in self.TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        duration = self.TIMEFRAME_SECONDS[timeframe]
        errors: list[str] = []
        warnings: list[str] = []

        if df.empty:
            return {
                "valid": False,
                "timeframe": timeframe,
                "rows": 0,
                "errors": ["No candle data received."],
                "warnings": [],
                "stats": {},
            }

        missing_columns = [
            col for col in self.REQUIRED_CANDLE_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            return {
                "valid": False,
                "timeframe": timeframe,
                "rows": len(df),
                "errors": [
                    f"Missing columns: {missing_columns}"
                ],
                "warnings": [],
                "stats": {},
            }

        data = df.copy()

        # Validate timestamp parsing and timezone awareness.
        timestamps = pd.to_datetime(
            data["time_utc"], errors="coerce", utc=True
        )

        if timestamps.isna().any():
            errors.append(
                f"{int(timestamps.isna().sum())} invalid timestamps."
            )

        valid_timestamp_mask = timestamps.notna()
        valid_times = timestamps[valid_timestamp_mask]

        if valid_times.duplicated().any():
            errors.append("Duplicate candle timestamps found.")

        if not valid_times.is_monotonic_increasing:
            warnings.append(
                "Candle timestamps are not in ascending order."
            )

        # OHLC must be numeric and finite.
        price_columns = ["open", "high", "low", "close"]

        for col in price_columns:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

        prices = data[price_columns]

        if prices.isna().any().any():
            errors.append(
                "Missing or non-numeric OHLC values found."
            )

        if prices.isin([float("inf"), float("-inf")]).any().any():
            errors.append("Infinite OHLC values found.")

        finite_mask = (
            prices.notna().all(axis=1)
            & ~prices.isin([float("inf"), float("-inf")]).any(axis=1)
        )

        finite_prices = prices[finite_mask]

        if (finite_prices <= 0).any().any():
            errors.append("Non-positive OHLC prices found.")

        invalid_high = (
            finite_prices["high"]
            < finite_prices[["open", "close", "low"]].max(axis=1)
        )

        invalid_low = (
            finite_prices["low"]
            > finite_prices[["open", "close", "high"]].min(axis=1)
        )

        invalid_range = (
            finite_prices["high"] < finite_prices["low"]
        )

        invalid_ohlc_count = int(
            (invalid_high | invalid_low | invalid_range).sum()
        )

        if invalid_ohlc_count:
            errors.append(
                f"{invalid_ohlc_count} candles violate OHLC rules."
            )

        # Check candle opening timestamps align to timeframe boundaries.
        if valid_times.notna().any():
             epoch_seconds = valid_times.map(
            lambda timestamp: int(timestamp.timestamp())
            )
             misaligned = (epoch_seconds % duration) != 0

             if misaligned.any():
                warnings.append(
                    f"{int(misaligned.sum())} timestamps do not "
                    f"align to {timeframe} boundaries."
                )

        # Gaps are informative but not necessarily data corruption.
        sorted_times = valid_times.sort_values()
        differences = sorted_times.diff().dt.total_seconds()

        gaps = differences > duration

        if gaps.any():
            warnings.append(
                f"{int(gaps.sum())} candle interval gaps found. "
                "Inspect them against market hours and broker history."
            )

        # A candle's timestamp represents its opening time.
        now_utc = self._now_utc(now)
        last_timestamp = (
            valid_times.max() if not valid_times.empty else None
        )

        last_age_seconds = None

        if last_timestamp is not None:
            last_age_seconds = (
                now_utc - last_timestamp
            ).total_seconds()

            if last_age_seconds < 0:
                errors.append(
                    "Latest candle timestamp is in the future."
                )

            if max_age_seconds is not None:
                if last_age_seconds > max_age_seconds + duration:
                    warnings.append(
                        "Latest candle is older than expected. "
                        "Check market hours and data freshness."
                    )

        stats = {
            "rows": int(len(data)),
            "first_timestamp_utc": (
                valid_times.min().isoformat()
                if not valid_times.empty else None
            ),
            "last_timestamp_utc": (
                last_timestamp.isoformat()
                if last_timestamp is not None else None
            ),
            "last_bar_open_age_seconds": last_age_seconds,
            "missing_intervals": int(gaps.sum()),
            "invalid_ohlc_count": invalid_ohlc_count,
        }

        return {
            "valid": len(errors) == 0,
            "timeframe": timeframe,
            "rows": int(len(data)),
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
        }

    def validate_ticks(
        self,
        df: pd.DataFrame,
        now: datetime | None = None,
        expected_start: datetime | None = None,
        expected_end: datetime | None = None,
        max_age_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Validate normalized MT5 tick records.

        Optional start/end bounds are timezone-aware datetimes.
        """

        errors: list[str] = []
        warnings: list[str] = []

        if df.empty:
            return {
                "valid": False,
                "rows": 0,
                "errors": ["No tick data received."],
                "warnings": [],
                "stats": {},
            }

        missing_columns = [
            col for col in self.REQUIRED_TICK_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            return {
                "valid": False,
                "rows": len(df),
                "errors": [
                    f"Missing columns: {missing_columns}"
                ],
                "warnings": [],
                "stats": {},
            }

        data = df.copy()

        timestamps = pd.to_datetime(
            data["time_utc"], errors="coerce", utc=True
        )

        if timestamps.isna().any():
            errors.append(
                f"{int(timestamps.isna().sum())} invalid tick timestamps."
            )

        valid_times = timestamps.dropna()

        if valid_times.duplicated().any():
            warnings.append(
                "Duplicate tick timestamps found. "
                "Multiple ticks can legitimately share a timestamp."
            )

        if not valid_times.is_monotonic_increasing:
            warnings.append(
                "Tick timestamps are not in ascending order."
            )

        for col in ["bid", "ask"]:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

        bad_quotes = (
            data[["bid", "ask"]].isna().any(axis=1)
            | (data["bid"] <= 0)
            | (data["ask"] <= 0)
            | (data["ask"] < data["bid"])
        )

        if bad_quotes.any():
            warnings.append(
                f"{int(bad_quotes.sum())} ticks have invalid "
                "or incomplete bid/ask quotes."
            )

        now_utc = self._now_utc(now)

        def to_utc(value: datetime, name: str) -> pd.Timestamp:
            timestamp = pd.Timestamp(value)

            if timestamp.tzinfo is None:
                raise ValueError(
                    f"{name} must be timezone-aware"
                )

            return timestamp.tz_convert("UTC")

        start_ts = (
            to_utc(expected_start, "expected_start")
            if expected_start is not None else None
        )

        end_ts = (
            to_utc(expected_end, "expected_end")
            if expected_end is not None else None
        )

        if start_ts is not None and end_ts is not None:
            if start_ts >= end_ts:
                raise ValueError(
                    "expected_start must be earlier than expected_end"
                )

        outside_window_count = 0

        if start_ts is not None:
            outside_window_count += int(
                (valid_times < start_ts).sum()
            )

        if end_ts is not None:
            outside_window_count += int(
                (valid_times >= end_ts).sum()
            )

        if outside_window_count:
            warnings.append(
                f"{outside_window_count} ticks are outside "
                "the requested time window."
            )

        first_timestamp = (
            valid_times.min() if not valid_times.empty else None
        )

        last_timestamp = (
            valid_times.max() if not valid_times.empty else None
        )

        last_age_seconds = None

        if last_timestamp is not None:
            last_age_seconds = (
                now_utc - last_timestamp
            ).total_seconds()

            if last_age_seconds < 0:
                errors.append("Latest tick timestamp is in the future.")

            if max_age_seconds is not None:
                if last_age_seconds > max_age_seconds:
                    warnings.append(
                        "Latest tick is older than the configured "
                        "freshness limit."
                    )

        if start_ts is not None and first_timestamp is not None:
            if first_timestamp > start_ts:
                warnings.append(
                    "Tick data starts after the requested start time."
                )

        if end_ts is not None and last_timestamp is not None:
            if last_timestamp < end_ts:
                warnings.append(
                    "Tick data ends before the requested end time. "
                    "Check whether the market was active."
                )

        return {
            "valid": len(errors) == 0,
            "rows": int(len(data)),
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "rows": int(len(data)),
                "first_timestamp_utc": (
                    first_timestamp.isoformat()
                    if first_timestamp is not None else None
                ),
                "last_timestamp_utc": (
                    last_timestamp.isoformat()
                    if last_timestamp is not None else None
                ),
                "last_tick_age_seconds": last_age_seconds,
                "outside_window_count": outside_window_count,
            },
        }
