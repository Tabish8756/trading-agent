import MetaTrader5 as mt5
import pandas as pd

def initialize():
    if not mt5.initialize():
        raise Exception("MT5 initialization failed")
    print("✅ MT5 Connected")

def shutdown():
    mt5.shutdown()

def get_rates(symbol, timeframe, n=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)

    if rates is None or len(rates) == 0:
        print(f"⚠️ No data for {symbol}")
        return None
    print('==================================================')
    print("data fetch", rates)

    df = pd.DataFrame(rates)

    if 'time' not in df.columns:
        print("⚠️ 'time' column missing")
        return None

    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_spread(symbol):
    tick = mt5.symbol_info_tick(symbol)
    return tick.ask - tick.bid


def get_candle_info(df):
    if df is None or len(df) < 2:
        return None

    # previous closed candle
    prev = df.iloc[-2]

    # current forming candle
    curr = df.iloc[-1]

    return {
        "prev": {
            "time": prev['time'],
            "open": prev['open'],
            "high": prev['high'],
            "low": prev['low'],
            "close": prev['close']
        },
        "curr": {
            "time": curr['time'],
            "open": curr['open'],
            "high": curr['high'],
            "low": curr['low'],
            "close": curr['close']
        }
    }