
from datetime import datetime, timedelta, timezone

from data.mt5_adapter import MT5Adapter


def main():
    with MT5Adapter(symbol="XAUUSD") as mt5_client:

        # 1. Check broker symbol specifications
        print("\n=== SYMBOL INFORMATION ===")
        print(mt5_client.get_symbol_info())

        # 2. Retrieve recent completed M1 candles
        print("\n=== LAST 5 CLOSED M1 CANDLES ===")
        m1 = mt5_client.get_candles(
            timeframe="M1",
            count=5,
            closed_only=True,
        )
        print(m1.to_string(index=False))

        # 3. Retrieve recent completed M5 candles
        print("\n=== LAST 5 CLOSED M5 CANDLES ===")
        m5 = mt5_client.get_candles(
            timeframe="M5",
            count=5,
            closed_only=True,
        )
        print(m5.to_string(index=False))

        # 4. Retrieve recent completed H1 candles
        print("\n=== LAST 5 CLOSED H1 CANDLES ===")
        h1 = mt5_client.get_candles(
            timeframe="H1",
            count=5,
            closed_only=True,
        )
        print(h1.to_string(index=False))

        # 5. Check latest available quote
        print("\n=== LATEST QUOTE ===")
        print(mt5_client.get_latest_quote())

        # 6. Read ticks from the last 60 seconds
        print("\n=== RECENT TICKS ===")
        ticks = mt5_client.get_recent_ticks(
            lookback_seconds=60,
            count=1000,
        )
        print(f"Ticks retrieved: {len(ticks)}")
        print(ticks.tail(10).to_string(index=False))

        # 7. Example of historical tick retrieval
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=5)

        historical_ticks = mt5_client.get_ticks_between(
            start=start,
            end=end,
        )

        print(
            "\nTicks in requested five-minute window:",
            len(historical_ticks),
        )


if __name__ == "__main__":
    main()
