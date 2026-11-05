# import time
# import MetaTrader5 as mt5

# from mt5_client import get_candle_info, initialize, get_rates, get_spread

# def run():
#     initialize()

#     while True:
#         try:
#             print("\n--- Agent Running ---")
#             df_m1 = get_rates(config.SYMBOL, mt5.TIMEFRAME_M1, config.MAX_CANDLES)
#             df_m5 = get_rates(config.SYMBOL, mt5.TIMEFRAME_M5, config.MAX_CANDLES)
#             df_m15 = get_rates(config.SYMBOL, mt5.TIMEFRAME_M15, config.MAX_CANDLES)
#             df_h1 = get_rates(config.SYMBOL, mt5.TIMEFRAME_H1, config.MAX_CANDLES)
#             df_h4 = get_rates(config.SYMBOL, mt5.TIMEFRAME_H4, config.MAX_CANDLES)
            
#             m1 = get_candle_info(df_m1)
#             m5 = get_candle_info(df_m5)
#             m15 = get_candle_info(df_m15)
#             h1 = get_candle_info(df_h1)
#             h4 = get_candle_info(df_h4)
            
            
#             print('m1 previous close', m1['prev']['close'])
#             print('m1 previous time', m1['prev']['time'])
#             print('m1 current open', m1['curr']['open'])
#             print('m1 current time', m1['curr']['time'])
#             print('m1 current price', m1['curr']['close'])

#             spread = get_spread(config.SYMBOL)
#             # print('df-1', df_m1)
#             # print("Latest M1 Close:", df_m1['close'].iloc[-1])
#             # print("Latest M5 Close:", df_m5['close'].iloc[-1])
#             # print("Latest M15 Close:", df_m15['close'].iloc[-1])
#             # print("Latest H1 Candle:", df_h1['close'].iloc[-1])
#             # print("Latest H1 close Candele Time:", df_h1['close'].iloc[-2])
#             print("Spread:", spread)
#             analyze_result = analyze(df_h1=df_h1, df_m15=df_m15, df_m5=df_m5)
#             print("Analysis Result:", analyze_result)

#             # 🔜 Strategy will go here

#         except Exception as e:
#             print("Error:", e)

#         time.sleep(config.LOOP_INTERVAL)

# if __name__ == "__main__":
#     run()


