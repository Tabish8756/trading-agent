
from datetime import datetime, timedelta, timezone

from data.mt5_adapter import MT5Adapter
from data.data_validator import DataValidator


def print_report(title, report):
    print(f"\n=== {title} ===")
    print("Valid:", report["valid"])
    print("Rows:", report["rows"])

    for error in report["errors"]:
        print("[ERROR]", error)

    for warning in report["warnings"]:
        print("[WARNING]", warning)

    print("Stats:", report["stats"])


def main():
    validator = DataValidator()

    with MT5Adapter(symbol="XAUUSD") as client:

        # Validate completed candles for each timeframe.
        for timeframe in ("M1", "M5", "H1"):
            candles = client.get_candles(
                timeframe=timeframe,
                count=500,
                closed_only=True,
            )

            report = validator.validate_candles(
                candles,
                timeframe=timeframe,
            )

            print_report(f"{timeframe} CANDLES", report)

        # Request ticks for a precisely defined recent window.
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=60)

        ticks = client.get_ticks_between(
            start=start,
            end=end,
        )

        report = validator.validate_ticks(
            ticks,
            now=end,
        )

        print_report("RECENT TICKS", report)

        if not ticks.empty:
            print("\nTick window:")
            print("Requested start:", start.isoformat())
            print("Requested end:  ", end.isoformat())
            print("First tick:     ", ticks["time_utc"].min())
            print("Last tick:      ", ticks["time_utc"].max())

            # Fail loudly if the returned records are outside
            # the requested half-open interval [start, end).
            outside = (
                (ticks["time_utc"] < start)
                | (ticks["time_utc"] >= end)
            )

            print("Ticks outside window:", int(outside.sum()))


if __name__ == "__main__":
    main()
