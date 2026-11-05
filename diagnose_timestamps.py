
from datetime import datetime, timezone

import pandas as pd

from data.mt5_adapter import MT5Adapter
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5


def main():
    print("=" * 65)
    print("TIMESTAMP DIAGNOSTIC")
    print("=" * 65)

    print("Computer local time:", datetime.now().astimezone().isoformat())
    print("Computer UTC time:  ", datetime.now(timezone.utc).isoformat())
    print("Pandas UTC time:    ", pd.Timestamp.now(tz="UTC").isoformat())

    with MT5Adapter(symbol="XAUUSD") as mt5_client:
        
        print("\n--- RAW MT5 API DIAGNOSTIC ---")
        print("System UTC:", datetime.now(timezone.utc).isoformat())

        terminal = mt5.terminal_info()
        account = mt5.account_info()

        print("Terminal connected:", getattr(terminal, "connected", None))
        print("Account server:", getattr(account, "server", None))

        # Compare raw rates with the adapter's normalized candle times.
        raw_rates = mt5.copy_rates_from_pos(
            "XAUUSD",
            mt5.TIMEFRAME_M1,
            0,
            3,
        )

        if raw_rates is not None and len(raw_rates):
            print("\nRaw M1 rate timestamps:")
            for rate in raw_rates:
                print({
                    "raw_time": int(rate["time"]),
                    "utc_interpretation": datetime.fromtimestamp(
                        int(rate["time"]), timezone.utc
                    ).isoformat(),
                })
        else:
            print("Raw rates unavailable:", mt5.last_error())

        # Request ticks using an explicitly UTC-aware time window.
        tick_start = datetime.now(timezone.utc) - timedelta(seconds=30)

        raw_ticks = mt5.copy_ticks_from(
            "XAUUSD",
            tick_start,
            20,
            mt5.COPY_TICKS_ALL,
        )

        if raw_ticks is not None and len(raw_ticks):
            print("\nRaw tick timestamps:")
            for tick in raw_ticks[-5:]:
                print({
                    "time": int(tick["time"]),
                    "time_msc": int(tick["time_msc"]),
                    "utc_from_milliseconds": datetime.fromtimestamp(
                        int(tick["time_msc"]) / 1000,
                        timezone.utc,
                    ).isoformat(),
                    "bid": float(tick["bid"]),
                    "ask": float(tick["ask"]),
                })
        else:
            print("Raw ticks unavailable:", mt5.last_error())

        print("\n--- END RAW MT5 DIAGNOSTIC ---")

        for timeframe in ("M1", "M5", "H1"):
            df = mt5_client.get_candles(
                timeframe=timeframe,
                count=5,
                closed_only=True,
            )

            print(f"\n--- {timeframe} CANDLES ---")

            if df.empty:
                print("No candles returned")
                continue

            print(
                df[
                    ["time", "time_utc", "open", "high", "low", "close"]
                ].to_string(index=False)
            )

            seconds = {
                "M1": 60,
                "M5": 300,
                "H1": 3600,
            }[timeframe]

            epoch_seconds = (
                df["time_utc"].astype("int64") // 1_000_000_000
            )

            print(
                "Alignment remainders:",
                (epoch_seconds % seconds).tolist(),
            )

            # NEW: Investigate raw and converted timestamp alignment.
            print("time dtype:", df["time"].dtype)
            print("time_utc dtype:", df["time_utc"].dtype)

            for _, row in df.head(2).iterrows():
                ts = pd.Timestamp(row["time_utc"])
                raw_epoch = int(row["time"])
                converted_epoch = int(ts.timestamp())

                print({
                    "raw_epoch": raw_epoch,
                    "raw_epoch_remainder": raw_epoch % seconds,
                    "timestamp_repr": repr(ts),
                    "timestamp_iso": ts.isoformat(),
                    "timestamp_epoch_seconds": ts.timestamp(),
                    "converted_remainder": converted_epoch % seconds,
                    "raw_vs_converted_seconds": (
                        converted_epoch - raw_epoch
                    ),
                })

        print("\n--- LATEST MT5 QUOTE ---")

        quote = mt5_client.get_latest_quote()

        print("Quote timestamp UTC:", quote.get("time_utc"))
        print("Quote raw time:      ", quote.get("time"))
        print("Quote raw time_msc:  ", quote.get("time_msc"))
        print("Bid:", quote.get("bid"), "Ask:", quote.get("ask"))

        print("\n--- RECENT TICKS ---")

        ticks = mt5_client.get_recent_ticks(lookback_seconds=60)

        if ticks.empty:
            print("No ticks returned in the last 60 seconds.")
        else:
            print("Tick rows:", len(ticks))
            print("First tick UTC:", ticks["time_utc"].min())
            print("Last tick UTC: ", ticks["time_utc"].max())
            print("Last tick raw time:", ticks["time"].iloc[-1])
            print("Last tick raw time_msc:", ticks["time_msc"].iloc[-1])


if __name__ == "__main__":
    main()
