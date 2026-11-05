"""
STEP 2 - XAUUSD LEVEL MAP: H1 DIRECTION + M15 TIMING + KEY LEVELS (live mode)

No trading, no entries. One run prints:
    1. H1 trend (direction) and the level that breaks it.
    2. M15 trend (timing), and whether M15 is pulling back against H1 or aligned with it.
    3. A single LEVEL MAP sorted by price, with your live price marked inside it:
         - H1 swing high / low (the H1 break level is flagged)
         - M15 swing high / low
         - Asian session high / low
         - Today's high / low and previous day's high / low (by broker day, like the MT5 D1 candle)
         - Nearest round numbers (every 50, majors at every 100)
    4. LAYER 4 - PULLBACK ZONE: the 38.2%-61.8% retracement of the last M15 leg (the area where a pullback
       may end), with any map levels that fall inside it, and the status of price relative to the zone.
    5. LAYER 5 - M5 ENTRY TRIGGER: once price is in (or just touched) the pullback zone, has M5 started to
       turn? Gives the trigger level, the status (NO SETUP / WATCHING / TRIGGERED), the level where the
       idea fails, and the distance to the next key level.
    6. LAYER 6 - TRADE PLAN: entry, stop (beyond the pullback extreme + an ATR buffer), targets taken from the
       level map with their reward:risk, a verdict on how much room there is, and the position size for your
       risk per trade. Spread is counted against you.
    7. A plain-language reading of how H1 and M15 fit together.

Distances are shown in dollars and in M15 ATRs (the typical size of one M15 candle).

Usage:
    pip install MetaTrader5 pandas numpy
    python step2_xauusd_m15_levels.py
"""
import math
import sys
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ----------------------------- SETTINGS -----------------------------------
SYMBOL = "XAUUSD"
ALIASES = ["XAUUSD", "GOLD"]

H1_K = 5                      # swing size on H1 (same as step 1)
M15_K = 3                     # swing size on M15 (3 candles = 45 min to confirm). Keep fixed.
ATR_PERIOD = 14

H1_BARS = 150                 # ~1 week of H1
M15_BARS = 600                # ~6 trading days of M15 (needed for the previous day's range)

# Broker server time vs UTC. None = try to detect it from the live tick, else use the fallback.
SERVER_UTC_OFFSET = None
FALLBACK_UTC_OFFSET = 3       # only used if detection fails. Many brokers: GMT+3 in summer, GMT+2 in winter.

ASIAN_START_UTC = 0           # Asian session window in UTC hours [start, end)
ASIAN_END_UTC = 7

ROUND_STEP = 50               # round-number spacing
MAJOR_STEP = 100              # round numbers divisible by this are flagged "major"
LEVEL_WINDOW_H1_ATR = 6       # hide non-round levels farther than this many H1 ATRs from price
TURN_WINDOW = 8               # M15 aligned with H1 for <= this many candles = "fresh" signal
MIN_PREV_DAY_CANDLES = 60     # below this, the previous-day range is flagged as partial

FIB_TOP = 0.382               # pullback zone = retracement between these two ratios of the last leg.
FIB_BOTTOM = 0.618            # Keep fixed; do not tune to chase results.
MIN_LEG_ATR = 1.5             # ignore legs smaller than this many M15 ATRs (too small to be meaningful)
LEG_OLD_CANDLES = 200         # note if the leg ended more than this many M15 candles ago (~2 days)

M5_K = 2                      # swing size on M5 (2 candles = 10 min to confirm). Keep fixed.
M5_BARS = 400                 # ~1.5 days of M5
TOUCH_LOOKBACK = 12           # price must have touched the zone within this many M5 candles (1 hour)
LATE_M5_ATR = 2.0             # trigger is "late" if price is already this many M5 ATRs beyond the trigger level
TRIGGER_MAX_AGE = 12          # ...or if the trigger candle closed more than this many M5 candles ago

ACCOUNT_BALANCE = None        # None = read your balance from MT5; or type a number, e.g. 1000
RISK_PERCENT = 1.0            # % of balance you are willing to lose if the stop is hit
STOP_BUFFER_M5_ATR = 0.5      # stop sits this many M5 ATRs beyond the pullback extreme (room for noise)
MIN_RR = 1.5                  # the target you care about must offer at least this reward:risk
MIN_STOP_M5_ATR = 1.0         # a stop tighter than this is inside normal M5 noise
MAX_STOP_M15_ATR = 2.0        # a stop wider than this is a wide stop for an M5 entry
MAX_SPREAD_SHARE = 0.15       # warn if the spread is more than this share of the risk

NAMES = {1: "BULLISH", -1: "BEARISH", 0: "SIDEWAYS"}


# ----------------------------- DATA ---------------------------------------
def find_symbol():
    all_names = [s.name for s in (mt5.symbols_get() or [])]
    for alias in ALIASES:
        candidates = [alias] + sorted([n for n in all_names if n.upper().startswith(alias) and n != alias], key=len)
        for c in candidates:
            if mt5.symbol_select(c, True) and mt5.symbol_info(c) is not None:
                return c
    return None


def get_rates(symbol, tf, n, chunk_days):
    """Fetch up to n bars by walking backwards in chunks. Drops the still-forming last candle."""
    end = datetime.now(timezone.utc) + timedelta(days=2)
    frames, total, empty_streak = [], 0, 0
    while total < n and empty_streak < 3:
        start = end - timedelta(days=chunk_days)
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) == 0:
            empty_streak += 1
        else:
            empty_streak = 0
            frames.append(pd.DataFrame(rates))
            total += len(rates)
        end = start
    if not frames:
        raise RuntimeError(f"No data for {symbol}: {mt5.last_error()}")
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").tail(n).reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.iloc[:-1].reset_index(drop=True)


def load_all():
    if not mt5.initialize():
        sys.exit(f"MT5 init failed: {mt5.last_error()}")
    try:
        name = find_symbol()
        if name is None:
            sys.exit(f"{SYMBOL} not found at this broker. Check ALIASES.")
        h1 = get_rates(name, mt5.TIMEFRAME_H1, H1_BARS, 60)
        m15 = get_rates(name, mt5.TIMEFRAME_M15, M15_BARS, 15)
        m5 = get_rates(name, mt5.TIMEFRAME_M5, M5_BARS, 5)
        tick = mt5.symbol_info_tick(name)
        spec = get_spec(name)
    finally:
        mt5.shutdown()
    return name, h1, m15, m5, tick, spec


def get_spec(name):
    """Contract details needed for position sizing (with safe fallbacks for a typical 100 oz gold contract)."""
    info = mt5.symbol_info(name)
    acc = mt5.account_info()
    tv = getattr(info, "trade_tick_value", 0) or 0
    ts = getattr(info, "trade_tick_size", 0) or 0
    per_unit = tv / ts if tv and ts else 100.0                  # money per 1.00 price move, per 1 lot
    bal = ACCOUNT_BALANCE if ACCOUNT_BALANCE is not None else (acc.balance if acc is not None else None)
    return {"per_unit": per_unit, "per_unit_known": bool(tv and ts),
            "vmin": getattr(info, "volume_min", 0.01) or 0.01,
            "vstep": getattr(info, "volume_step", 0.01) or 0.01,
            "vmax": getattr(info, "volume_max", 100.0) or 100.0,
            "balance": bal, "currency": acc.currency if acc is not None else "USD"}


def server_offset(tick):
    """Hours that broker server time is ahead of UTC."""
    if SERVER_UTC_OFFSET is not None:
        return SERVER_UTC_OFFSET, "set manually"
    if tick is not None:
        diff = (tick.time - time.time()) / 3600
        off = round(diff)
        if abs(diff - off) < 0.17 and abs(off) <= 12:        # fresh tick -> offset is a clean whole hour
            return off, "auto-detected"
    return FALLBACK_UTC_OFFSET, "ASSUMED (could not detect; set SERVER_UTC_OFFSET if this is wrong)"


# ----------------------------- STRUCTURE ----------------------------------
def add_atr(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    return df


def add_structure_trend(df, k):
    """Market-structure label using only swings that are already CONFIRMED at each candle."""
    n = len(df)
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    win = 2 * k + 1
    hmax = pd.Series(h).rolling(win, center=True).max().values
    lmin = pd.Series(l).rolling(win, center=True).min().values
    is_sh = h == hmax
    is_sl = l == lmin

    lh = np.full(n, np.nan); ph = np.full(n, np.nan)
    ll = np.full(n, np.nan); pl = np.full(n, np.nan)
    cur_lh = prev_ph = cur_ll = prev_pl = np.nan
    cur_lh_i = cur_ll_i = -10 ** 9

    for t in range(n):
        i = t - k
        if i >= k:
            if is_sh[i] and not (h[i] == cur_lh and i - cur_lh_i <= 2 * k):
                prev_ph, cur_lh, cur_lh_i = cur_lh, h[i], i
            if is_sl[i] and not (l[i] == cur_ll and i - cur_ll_i <= 2 * k):
                prev_pl, cur_ll, cur_ll_i = cur_ll, l[i], i
        lh[t], ph[t], ll[t], pl[t] = cur_lh, prev_ph, cur_ll, prev_pl

    bull = (lh > ph) & (ll > pl) & (c > ll)
    bear = (lh < ph) & (ll < pl) & (c < lh)
    trend = np.where(bull, 1, np.where(bear, -1, 0)).astype(float)
    trend[np.isnan(lh) | np.isnan(ph) | np.isnan(ll) | np.isnan(pl)] = np.nan

    df["sh_last"], df["sh_prev"], df["sl_last"], df["sl_prev"] = lh, ph, ll, pl
    df["trend_ms"] = trend
    return df


def state_info(df, name):
    d = df.dropna(subset=["trend_ms", "atr"])
    if d.empty:
        sys.exit(f"Not enough {name} candles to confirm swings. Increase the bars setting.")
    r = d.iloc[-1]
    state = int(r["trend_ms"])
    run = 1
    for v in d["trend_ms"].values[-2::-1]:
        if v == state:
            run += 1
        else:
            break
    return r, state, run


# ----------------------------- LAYER 4: PULLBACK ZONE ---------------------
def swing_points(df, k):
    """Confirmed swings as an alternating zigzag: [{'i', 'kind' ('H'/'L'), 'price'}, ...]."""
    n = len(df)
    h, l = df["high"].values, df["low"].values
    win = 2 * k + 1
    hmax = pd.Series(h).rolling(win, center=True).max().values
    lmin = pd.Series(l).rolling(win, center=True).min().values
    pts = []
    for i in range(k, n - k):                       # only swings with k candles after them (confirmed)
        is_h, is_l = h[i] == hmax[i], l[i] == lmin[i]
        if is_h == is_l:                            # neither, or an ambiguous outside candle -> skip
            continue
        kind, price = ("H", h[i]) if is_h else ("L", l[i])
        if pts and pts[-1]["kind"] == kind:         # two highs (or lows) in a row: keep the more extreme
            if (kind == "H" and price >= pts[-1]["price"]) or (kind == "L" and price <= pts[-1]["price"]):
                pts[-1] = {"i": i, "kind": kind, "price": price}
        else:
            pts.append({"i": i, "kind": kind, "price": price})
    return pts


def find_leg(pts, direction, min_size):
    """Most recent completed leg of the given direction ('up' = low->high, 'down' = high->low)."""
    for j in range(len(pts) - 2, -1, -1):
        a, b = pts[j], pts[j + 1]
        if direction == "up" and a["kind"] == "L" and b["kind"] == "H" and b["price"] - a["price"] >= min_size:
            return a, b
        if direction == "down" and a["kind"] == "H" and b["kind"] == "L" and a["price"] - b["price"] >= min_size:
            return a, b
    return None


def pullback_zone(m15, pts, h1s, price, m15_atr, levels):
    """Fibonacci pullback zone of the last relevant M15 leg. Returns a dict (always has 'ok')."""
    if len(pts) < 2:
        return {"ok": False, "why": "Not enough confirmed M15 swings yet."}

    if h1s == 1:
        want, note = "up", None
    elif h1s == -1:
        want, note = "down", None
    else:
        want = "up" if pts[-1]["kind"] == "H" else "down"
        note = f"H1 has no bias, so the zone is for the most recent M15 leg ({want.upper()})."

    leg = find_leg(pts, want, MIN_LEG_ATR * m15_atr)
    if leg is None:
        return {"ok": False, "why": f"No {want.upper()} leg of at least {MIN_LEG_ATR} M15 ATR in the data."}
    a, b = leg
    start, end = a["price"], b["price"]
    size = abs(end - start)
    closes_after = m15["close"].values[b["i"] + 1:]

    if want == "up":                                # pullback DOWN from the leg high
        top, bot = end - FIB_TOP * size, end - FIB_BOTTOM * size
        mid = end - 0.5 * size
        broken = (len(closes_after) and closes_after.min() < start) or price < start
        if broken:
            status = "BROKEN - price closed beyond the leg start; the zone is no longer valid"
        elif price > end:
            status = "LEG EXTENDED - price is above the leg high; wait for a new swing to form"
        elif price > top:
            status = f"ABOVE the zone - pullback has not reached it ({(price - top) / m15_atr:.1f} M15 ATR above)"
        elif price >= bot:
            status = "INSIDE the zone - the area to watch"
        else:
            status = f"DEEPER than the zone - past 61.8% ({(bot - price) / m15_atr:.1f} M15 ATR below)"
        invalid = f"an M15 candle closes BELOW {start:.2f}"
        side = "pullback zone (buy-side area)"
    else:                                           # bounce UP from the leg low
        bot, top = end + FIB_TOP * size, end + FIB_BOTTOM * size
        mid = end + 0.5 * size
        broken = (len(closes_after) and closes_after.max() > start) or price > start
        if broken:
            status = "BROKEN - price closed beyond the leg start; the zone is no longer valid"
        elif price < end:
            status = "LEG EXTENDED - price is below the leg low; wait for a new swing to form"
        elif price < bot:
            status = f"BELOW the zone - bounce has not reached it ({(bot - price) / m15_atr:.1f} M15 ATR below)"
        elif price <= top:
            status = "INSIDE the zone - the area to watch"
        else:
            status = f"DEEPER than the zone - past 61.8% ({(price - top) / m15_atr:.1f} M15 ATR above)"
        invalid = f"an M15 candle closes ABOVE {start:.2f}"
        side = "bounce zone (sell-side area)"

    tol = 0.25 * m15_atr
    inside = [(p_, lab) for p_, lab in levels if bot - tol <= p_ <= top + tol]
    age = len(m15) - 1 - b["i"]
    return {"ok": True, "dir": want, "start": start, "end": end, "size": size, "top": top, "bot": bot,
            "mid": mid, "status": status, "invalid": invalid, "side": side, "inside": inside,
            "age": age, "note": note, "broken": bool(broken), "end_time": m15["time"].iloc[b["i"]]}


def print_zone(z, m15_atr):
    print(f"\n--- LAYER 4 : PULLBACK ZONE (M15) ---")
    if not z["ok"]:
        print(z["why"])
        return
    if z["note"]:
        print(f"[note] {z['note']}")
    arrow = "UP" if z["dir"] == "up" else "DOWN"
    print(f"Leg used           : {arrow}  {z['start']:.2f} -> {z['end']:.2f}  "
          f"(${z['size']:.2f} = {z['size'] / m15_atr:.1f} M15 ATR)")
    print(f"Leg ended          : {z['end_time']:%Y-%m-%d %H:%M}  ({z['age']} M15 candles ago)")
    print(f"Zone (38.2%-61.8%) : {min(z['top'], z['bot']):.2f} - {max(z['top'], z['bot']):.2f}   "
          f"(50% = {z['mid']:.2f})   [{z['side']}]")
    print(f"Status             : {z['status']}")
    if z["inside"]:
        print("Levels in the zone :")
        for p_, lab in z["inside"]:
            print(f"    {p_:9.2f}  {lab}")
        print(f"                     ({len(z['inside'])} level(s) overlap the zone = confluence)")
    else:
        print("Levels in the zone : none (zone has no other level inside it)")
    print(f"Zone is invalid if : {z['invalid']}")
    if z["age"] > LEG_OLD_CANDLES:
        print(f"[note] The leg ended {z['age']} candles ago, so this zone may be stale.")
    print("The zone shows where to WATCH, not an entry. The trigger is layer 5 below.")


# ----------------------------- LAYER 5: M5 ENTRY TRIGGER ------------------
def m5_trigger(m5, zone, price, levels, m15_atr, m5_atr):
    """Has M5 started to turn inside the layer-4 zone? Returns a dict that always has 'status'."""
    no = lambda why: {"status": "NO SETUP", "why": why}
    if not zone["ok"]:
        return no(zone["why"])
    if zone["broken"]:
        return no("The M15 zone is broken (price closed beyond the leg start).")
    if zone["status"].startswith("LEG EXTENDED"):
        return no("The M15 leg is still extending, so there is no pullback zone to work with yet.")

    buy = zone["dir"] == "up"
    side = "BUY" if buy else "SELL"
    zlo, zhi = min(zone["top"], zone["bot"]), max(zone["top"], zone["bot"])
    hi, lo, cl = m5["high"].values, m5["low"].values, m5["close"].values
    n = len(m5)
    last = n - 1

    touched = [i for i in range(max(0, n - TOUCH_LOOKBACK), n) if lo[i] <= zhi and hi[i] >= zlo]
    if not touched and not (zlo <= price <= zhi):
        dist = (zlo - price) if price < zlo else (price - zhi)
        return no(f"Price has not touched the zone in the last {TOUCH_LOOKBACK} M5 candles "
                  f"({dist / m15_atr:.1f} M15 ATR away).")
    entry_i = touched[0] if touched else last

    # the extreme of the pullback since it reached the zone: lowest low (buy) / highest high (sell)
    if buy:
        ext_i = entry_i + int(np.argmin(lo[entry_i:]))
        ext_price = lo[ext_i]
    else:
        ext_i = entry_i + int(np.argmax(hi[entry_i:]))
        ext_price = hi[ext_i]

    # trigger level = the last confirmed M5 swing high (buy) / low (sell) BEFORE that extreme
    leg5 = int(m5["time"].searchsorted(zone["end_time"]))
    pts = swing_points(m5, M5_K)
    kind = "H" if buy else "L"
    cand = [q for q in pts if q["kind"] == kind and leg5 <= q["i"] < ext_i]
    if cand:
        trig, src = cand[-1]["price"], f"last M5 swing {'high' if buy else 'low'} before the {'dip low' if buy else 'peak'}"
    else:
        a0 = min(leg5, ext_i)
        trig = hi[a0:ext_i + 1].max() if buy else lo[a0:ext_i + 1].min()
        src = f"no M5 swing found yet; using the {'highest high' if buy else 'lowest low'} since the pullback began"

    # did an M5 candle CLOSE beyond the trigger after the extreme?
    after = cl[ext_i + 1:]
    hit = np.where(after > trig)[0] if buy else np.where(after < trig)[0]
    j = ext_i + 1 + int(hit[0]) if len(hit) else None

    tol5 = 0.25 * m5_atr                            # next level BEYOND the trigger (not the trigger itself)
    ahead = [(p_, lab) for p_, lab in levels
             if (p_ > max(price, trig) + tol5 if buy else p_ < min(price, trig) - tol5)]
    nxt = (min(ahead) if buy else max(ahead)) if ahead else None

    out = {"side": side, "buy": buy, "trig": trig, "src": src, "ext_price": ext_price,
           "ext_time": m5["time"].iloc[ext_i], "nxt": nxt, "zone": (zlo, zhi), "j": j,
           "entry_time": m5["time"].iloc[entry_i]}
    if j is None:
        out["status"] = "WATCHING"
        out["gap"] = (trig - price) if buy else (price - trig)
    else:
        age = last - j
        beyond = (price - trig) if buy else (trig - price)
        late = age > TRIGGER_MAX_AGE or beyond > LATE_M5_ATR * m5_atr
        status = "TRIGGER FADED" if beyond <= 0 else ("TRIGGERED (LATE)" if late else "TRIGGERED")
        out.update({"status": status, "age": age,
                    "beyond": beyond, "trig_time": m5["time"].iloc[j]})
    out["risk"] = abs(price - ext_price)
    return out


def print_trigger(t, m5_state, m5_run, m5_atr, m15_atr, price):
    print(f"\n--- LAYER 5 : M5 ENTRY TRIGGER (K={M5_K}) ---")
    print(f"M5 structure       : {NAMES[m5_state]}  (for {m5_run} M5 candles = {m5_run * 5} min)   M5 ATR = {m5_atr:.2f}")
    if t["status"] == "NO SETUP":
        print(f"Status             : NO SETUP - {t['why']}")
        return
    word = "dip low" if t["buy"] else "peak"
    cmp_ = "ABOVE" if t["buy"] else "BELOW"
    print(f"Side watched       : {t['side']}  (zone {t['zone'][0]:.2f} - {t['zone'][1]:.2f}, "
          f"first touched {t['entry_time']:%H:%M})")
    print(f"Pullback {word:<8}  : {t['ext_price']:.2f}  (at {t['ext_time']:%H:%M})")
    print(f"Trigger level      : {t['trig']:.2f}  ({t['src']})")
    if t["status"] == "WATCHING":
        print(f"Status             : WATCHING - an M5 candle CLOSING {cmp_} {t['trig']:.2f} triggers "
              f"({t['gap']:.2f} away = {t['gap'] / m5_atr:.1f} M5 ATR)")
    elif t["status"] == "TRIGGER FADED":
        print(f"Status             : TRIGGER FADED - an M5 candle closed {cmp_.lower()} {t['trig']:.2f} at "
              f"{t['trig_time']:%H:%M}, but live price is back on the other side ({-t['beyond']:.2f} away). "
              f"Not an active trigger.")
    else:
        print(f"Status             : {t['status']} - M5 closed {cmp_.lower()} {t['trig']:.2f} at "
              f"{t['trig_time']:%H:%M} ({t['age']} candles ago); price is now {t['beyond']:.2f} "
              f"({t['beyond'] / m5_atr:.1f} M5 ATR) beyond it")
    print(f"Idea fails if      : an M5 candle closes {'BELOW' if t['buy'] else 'ABOVE'} {t['ext_price']:.2f}"
          f"  (reference only; stops come in layer 6)")
    print(f"Distance to that   : {t['risk']:.2f} = {t['risk'] / m5_atr:.1f} M5 ATR = {t['risk'] / m15_atr:.1f} M15 ATR")
    if t["nxt"]:
        d = abs(t["nxt"][0] - price)
        print(f"Next level beyond  : {t['nxt'][0]:.2f}  ({d:.2f} = {d / m15_atr:.1f} M15 ATR)  {t['nxt'][1]}")
    else:
        print("Next level beyond  : none on the map in that direction")


# ----------------------------- LAYER 6: TRADE PLAN ------------------------
def trade_plan(t, price, tick, levels, m5_atr, m15_atr, spec):
    """Entry / stop / targets / size for the layer-5 setup. Spread is added to the risk and taken off the reward."""
    if t["status"] not in ("WATCHING", "TRIGGERED"):
        why = t.get("why") or {"TRIGGERED (LATE)": "The trigger is late: the move is already underway.",
                               "TRIGGER FADED": "Price is back on the wrong side of the trigger."}.get(t["status"], t["status"])
        return {"ok": False, "why": f"No plan - {why}"}

    buy = t["buy"]
    spread = (tick.ask - tick.bid) if tick is not None else 0.0
    if t["status"] == "WATCHING":
        entry, how = t["trig"], f"planned: after an M5 close {'above' if buy else 'below'} {t['trig']:.2f}"
    else:
        entry, how = price, "at the current price (trigger already fired)"

    buffer_ = STOP_BUFFER_M5_ATR * m5_atr
    stop = t["ext_price"] - buffer_ if buy else t["ext_price"] + buffer_
    risk = abs(entry - stop) + spread
    if risk <= 0:
        return {"ok": False, "why": "No plan - the stop distance is zero."}

    tol = 0.25 * m5_atr
    beyond = [(p_, lab) for p_, lab in levels if (p_ > entry + tol if buy else p_ < entry - tol)]
    beyond.sort(key=lambda x: x[0] if buy else -x[0])
    targets = []
    for p_, lab in beyond[:4]:
        reward = abs(p_ - entry) - spread
        targets.append({"price": p_, "label": lab, "reward": reward, "rr": reward / risk})

    good = next((x for x in targets if x["rr"] >= MIN_RR), None)
    if targets and targets[0]["rr"] >= MIN_RR:
        verdict = f"OK - the first mapped level is already {targets[0]['rr']:.1f}R away."
    elif good:
        n_ob = targets.index(good)
        verdict = (f"CAUTION - the nearest level is only {targets[0]['rr']:.1f}R away; {MIN_RR}R is first reached at "
                   f"{good['price']:.2f} ({good['rr']:.1f}R), after passing {n_ob} level(s) that may stall price.")
    else:
        verdict = f"SKIP - no mapped level offers {MIN_RR}R. The reward is too small for this risk."

    warns = []
    if risk < MIN_STOP_M5_ATR * m5_atr:
        warns.append(f"Stop is tighter than {MIN_STOP_M5_ATR} M5 ATR: normal noise could hit it.")
    if risk > MAX_STOP_M15_ATR * m15_atr:
        warns.append(f"Stop is wider than {MAX_STOP_M15_ATR} M15 ATR: wide for an M5 entry.")
    if tick is None:
        warns.append("No live tick, so the spread was taken as 0.")
    elif spread / risk > MAX_SPREAD_SHARE:
        warns.append(f"Spread is {spread / risk * 100:.0f}% of the risk: costs eat a large part of each trade.")

    # position size
    size = {"known": spec["balance"] is not None}
    if size["known"]:
        allowed = spec["balance"] * RISK_PERCENT / 100
        per_lot = risk * spec["per_unit"]
        raw = allowed / per_lot
        lots = math.floor(raw / spec["vstep"] + 1e-9) * spec["vstep"]
        lots = min(lots, spec["vmax"])
        size.update({"allowed": allowed, "per_lot": per_lot, "lots": round(lots, 8),
                     "actual": lots * per_lot, "min_risk": spec["vmin"] * per_lot})
    return {"ok": True, "buy": buy, "entry": entry, "how": how, "stop": stop, "buffer": buffer_, "risk": risk,
            "spread": spread, "targets": targets, "verdict": verdict, "warns": warns, "size": size,
            "ext_price": t["ext_price"]}


def print_plan(pl, spec, m5_atr, m15_atr):
    print(f"\n--- LAYER 6 : TRADE PLAN ---")
    if not pl["ok"]:
        print(pl["why"])
        return
    side = "BUY" if pl["buy"] else "SELL"
    print(f"Side               : {side}")
    print(f"Entry              : {pl['entry']:.2f}  ({pl['how']})")
    print(f"Stop               : {pl['stop']:.2f}  (pullback {'low' if pl['buy'] else 'high'} {pl['ext_price']:.2f} "
          f"{'-' if pl['buy'] else '+'} buffer {pl['buffer']:.2f})")
    print(f"Risk (price move)  : {pl['risk']:.2f} = {pl['risk'] / m5_atr:.1f} M5 ATR = {pl['risk'] / m15_atr:.1f} M15 ATR"
          f"   (includes spread {pl['spread']:.2f})")
    print("Targets (map levels in the trade direction, reward net of spread):")
    if pl["targets"]:
        for i, x in enumerate(pl["targets"], 1):
            print(f"   T{i}  {x['price']:9.2f}   {x['reward']:+7.2f}  = {x['rr']:4.1f}R   {x['label']}")
    else:
        print("   none on the map beyond the entry")
    print(f"Verdict            : {pl['verdict']}")
    for w in pl["warns"]:
        print(f"[warning] {w}")
    sz = pl["size"]
    if not sz["known"]:
        print("Position size      : balance unknown. Set ACCOUNT_BALANCE at the top of the script.")
    else:
        cur = spec["currency"]
        print(f"Position size      : balance {spec['balance']:.2f} x {RISK_PERCENT}% = {sz['allowed']:.2f} {cur} allowed risk")
        if sz["lots"] >= spec["vmin"]:
            print(f"                     -> {sz['lots']:.2f} lots  (loses {sz['actual']:.2f} {cur} = "
                  f"{sz['actual'] / spec['balance'] * 100:.2f}% if the stop is hit)")
        else:
            print(f"                     -> NOT POSSIBLE: even the minimum lot ({spec['vmin']}) would risk "
                  f"{sz['min_risk']:.2f} {cur} = {sz['min_risk'] / spec['balance'] * 100:.2f}% of the balance")
        if not spec["per_unit_known"]:
            print("[note] Contract value not provided by the broker; assumed 100 oz per lot. Verify before using the size.")
    print("This is arithmetic on the map, not a recommendation. The setup itself is still untested (layer 7).")


# ----------------------------- LEVELS -------------------------------------
def build_levels(price, h1r, h1s, m15r, m15, offset, h1_atr, m15_atr):
    """Returns (levels, notes). levels = list of (price, label)."""
    lv, notes = [], []

    # H1 swings (the one that breaks the H1 trend is flagged)
    lv.append((h1r["sh_last"], "H1 swing high" + (" (BREAK LEVEL)" if h1s == -1 else "")))
    lv.append((h1r["sl_last"], "H1 swing low" + (" (BREAK LEVEL)" if h1s == 1 else "")))

    # M15 swings
    lv.append((m15r["sh_last"], "M15 swing high"))
    lv.append((m15r["sl_last"], "M15 swing low"))

    # Today / previous day (by broker day, like the MT5 D1 candle)
    m = m15.copy()
    m["sdate"] = m["time"].dt.date
    dates = sorted(m["sdate"].unique())
    t = m[m["sdate"] == dates[-1]]
    lv.append((t["high"].max(), "Today high"))
    lv.append((t["low"].min(), "Today low"))
    if len(dates) >= 2:
        p = m[m["sdate"] == dates[-2]]
        lv.append((p["high"].max(), "Prev day high"))
        lv.append((p["low"].min(), "Prev day low"))
        if len(p) < MIN_PREV_DAY_CANDLES:
            notes.append(f"Previous day has only {len(p)} M15 candles in the data: its high/low may be partial.")
    else:
        notes.append("No previous day in the data: increase M15_BARS.")

    # Asian session (window defined in UTC, converted using the broker offset)
    utc = m["time"] - pd.Timedelta(hours=offset)
    m["udate"], m["uhour"] = utc.dt.date, utc.dt.hour
    last_u = m["udate"].iloc[-1]
    a = m[(m["udate"] == last_u) & (m["uhour"] >= ASIAN_START_UTC) & (m["uhour"] < ASIAN_END_UTC)]
    if len(a) >= 4:
        forming = int(m["uhour"].iloc[-1]) < ASIAN_END_UTC
        tag = " (still forming)" if forming else ""
        lv.append((a["high"].max(), "Asian high" + tag))
        lv.append((a["low"].min(), "Asian low" + tag))
    else:
        notes.append("No Asian-session candles found for the latest day.")

    # keep non-round levels that are reasonably near price (always keep the H1 break level)
    near = []
    for p_, lab in lv:
        if "BREAK" in lab or abs(p_ - price) <= LEVEL_WINDOW_H1_ATR * h1_atr:
            near.append((p_, lab))

    # nearest round numbers: two below, two above
    base = math.floor(price / ROUND_STEP) * ROUND_STEP
    for r_ in [base, base - ROUND_STEP, base + ROUND_STEP, base + 2 * ROUND_STEP]:
        near.append((float(r_), f"Round {r_}" + (" (major)" if r_ % MAJOR_STEP == 0 else "")))

    # merge levels that sit almost on top of each other
    near.sort(key=lambda x: -x[0])
    tol = max(0.5, 0.25 * m15_atr)
    merged = []
    for p_, lab in near:
        if merged and merged[-1][0] - p_ < tol:
            merged[-1] = (merged[-1][0], merged[-1][1] + " + " + lab)
        else:
            merged.append((p_, lab))
    return merged, notes


def reading(h1s, m15s, m15_run, m15r):
    if h1s == 0:
        return ["H1 has no clear direction -> no bias. Skip, or be very selective."]
    side = "buy" if h1s == 1 else "sell"
    if m15s == h1s:
        mins = m15_run * 15
        if m15_run <= TURN_WINDOW:
            return [f"M15 turned back WITH H1 only {m15_run} candles ({mins} min) ago -> fresh timing signal, {side} side."]
        return [f"M15 has been aligned with H1 for {m15_run} candles ({mins // 60}h {mins % 60}m) -> the move is already "
                f"underway; entering now is late in the move, not at a pullback."]
    if m15s == -h1s:
        trigger = m15r["sh_last"] if h1s == 1 else m15r["sl_last"]
        word = "ABOVE" if h1s == 1 else "BELOW"
        return [f"M15 is moving AGAINST H1 (a pullback is in progress) -> wait.",
                f"The pullback is over when an M15 candle CLOSES {word} {trigger:.2f}."]
    return ["M15 is sideways -> no timing signal yet."]


# ----------------------------- MAIN ---------------------------------------
def main():
    name, h1_raw, m15_raw, m5_raw, tick, spec = load_all()
    h1 = add_structure_trend(add_atr(h1_raw.copy()), H1_K)
    m15 = add_structure_trend(add_atr(m15_raw.copy()), M15_K)
    m5 = add_structure_trend(add_atr(m5_raw.copy()), M5_K)

    h1r, h1s, h1_run = state_info(h1, "H1")
    m15r, m15s, m15_run = state_info(m15, "M15")
    m5r, m5s, m5_run = state_info(m5, "M5")
    h1_atr, m15_atr = h1r["atr"], m15r["atr"]

    offset, how = server_offset(tick)
    price = tick.bid if tick is not None else m15r["close"]
    price_src = "live bid" if tick is not None else "last M15 close (no live tick)"

    levels, notes = build_levels(price, h1r, h1s, m15r, m15, offset, h1_atr, m15_atr)

    line = "=" * 74
    print(f"\n{line}\n{name} - H1 + M15 LEVEL MAP\n{line}")
    print(f"Broker time        : UTC{offset:+d} ({how})")
    print(f"Price              : {price:.2f}  ({price_src})")
    print(f"Last closed candles: H1 {h1r['time']:%Y-%m-%d %H:%M} | M15 {m15r['time']:%Y-%m-%d %H:%M} | "
          f"M5 {m5r['time']:%H:%M}  (broker time)")

    print(f"\n--- H1 : DIRECTION (K={H1_K}) ---")
    print(f"Trend              : {NAMES[h1s]}  (for {h1_run} H1 candles)")
    print(f"H1 ATR             : {h1_atr:.2f}")
    if h1s == 1:
        print(f"Break level        : {h1r['sl_last']:.2f}  (H1 close below it ends the bullish structure; "
              f"{(price - h1r['sl_last']) / h1_atr:.1f} H1 ATR away)")
    elif h1s == -1:
        print(f"Break level        : {h1r['sh_last']:.2f}  (H1 close above it ends the bearish structure; "
              f"{(h1r['sh_last'] - price) / h1_atr:.1f} H1 ATR away)")
    else:
        print(f"Range              : {h1r['sl_last']:.2f} - {h1r['sh_last']:.2f}")

    print(f"\n--- M15 : TIMING (K={M15_K}) ---")
    print(f"Trend              : {NAMES[m15s]}  (for {m15_run} M15 candles = {m15_run * 15} min)")
    print(f"M15 ATR            : {m15_atr:.2f}")
    print(f"Last swing high/low: {m15r['sh_last']:.2f} / {m15r['sl_last']:.2f}")

    pts = swing_points(m15, M15_K)
    zone = pullback_zone(m15, pts, h1s, price, m15_atr, levels)
    print_zone(zone, m15_atr)

    m5_atr = m5r["atr"]
    trig = m5_trigger(m5, zone, price, levels, m15_atr, m5_atr)
    print_trigger(trig, m5s, m5_run, m5_atr, m15_atr, price)

    plan = trade_plan(trig, price, tick, levels, m5_atr, m15_atr, spec)
    print_plan(plan, spec, m5_atr, m15_atr)

    print(f"\n--- READING ---")
    for s in reading(h1s, m15s, m15_run, m15r):
        print(s)

    print(f"\n--- LEVEL MAP (high -> low) ---")
    rows = [(p_, lab) for p_, lab in levels] + [(price, ">>> PRICE <<<")]
    if zone["ok"] and not zone["broken"]:
        if zone["dir"] == "up":
            hi_lab, lo_lab = "-- Pullback zone TOP (38.2%) --", "-- Pullback zone BOTTOM (61.8%) --"
        else:
            hi_lab, lo_lab = "-- Bounce zone TOP (61.8%) --", "-- Bounce zone BOTTOM (38.2%) --"
        rows += [(max(zone["top"], zone["bot"]), hi_lab), (min(zone["top"], zone["bot"]), lo_lab)]
    rows.sort(key=lambda x: -x[0])
    print(f"{'price':>9}  {'diff $':>8}  {'diff M15 ATR':>12}   level")
    out_rows = []
    for p_, lab in rows:
        if lab.startswith(">>>"):
            print(f"{p_:9.2f}  {'':>8}  {'':>12}   {lab}")
            continue
        d = p_ - price
        print(f"{p_:9.2f}  {d:+8.2f}  {d / m15_atr:+12.1f}   {lab}")
        out_rows.append({"price": round(p_, 2), "diff_usd": round(d, 2),
                         "diff_m15_atr": round(d / m15_atr, 2), "level": lab})

    real = [r for r in out_rows if not r["level"].startswith("--")]     # leave the zone edges out
    above = [r for r in real if r["diff_usd"] > 0]
    below = [r for r in real if r["diff_usd"] <= 0]
    if above:
        r = above[-1]
        print(f"\nNearest level above: {r['price']:.2f}  ({r['diff_m15_atr']:+.1f} M15 ATR)  {r['level']}")
    if below:
        r = below[0]
        print(f"Nearest level below: {r['price']:.2f}  ({r['diff_m15_atr']:+.1f} M15 ATR)  {r['level']}")

    for n_ in notes:
        print(f"[note] {n_}")

    pd.DataFrame(out_rows).to_csv("m15_levels.csv", index=False)
    print("\nLevel map saved to m15_levels.csv")
    print("Times are broker server time. Levels are a map of where price may react, not trade signals.")


if __name__ == "__main__":
    main()