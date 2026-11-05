"""
STEP 1 - MARKET STRUCTURE TREND ON H1 FOR XAUUSD (GOLD)

No trading, no entries. The script has three modes:

    python step1_xauusd_structure.py live
        Reads the CURRENT H1 trend using about 1 week of data. Fast. Use this every day.

    python step1_xauusd_structure.py validate
        Uses about 1 year of H1 data to check whether the trend label has any edge.
        Splits the data into TRAIN / TEST. The HOLDOUT period is NOT touched here.

    python step1_xauusd_structure.py final
        Runs the HOLDOUT test. Do this ONCE, when you are finished choosing settings.
        A lock file stops you from running it again by accident.

Trend label for every closed H1 candle:
    BULLISH  : higher swing high AND higher swing low AND price still above the last swing low
    BEARISH  : lower swing high AND lower swing low AND price still below the last swing high
    SIDEWAYS : anything else

Moves are measured in ATRs (the typical size of one H1 candle), not dollars.

Setup:
    pip install MetaTrader5 pandas numpy
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ----------------------------- SETTINGS -----------------------------------
SYMBOL = "XAUUSD"
ALIASES = ["XAUUSD", "GOLD"]          # broker suffixes (XAUUSDm, GOLD.a ...) are detected automatically

SWING_K = 5                           # candles on each side that define a swing. Keep fixed; do not tune it.
ATR_PERIOD = 14

LIVE_BARS = 150                       # ~1 week of H1 gold + warm-up for swings and ATR
VALIDATE_BARS = 6000                  # ~1 year of H1 gold

HORIZONS = [2, 4, 6, 24]              # candles ahead to check (H1 candles = hours). Decide once, do not change.
MAIN_HORIZONS = [2, 4, 6]             # the ones that matter for intraday / scalping; 24 is a bonus

TRAIN_SHARE = 0.70                    # first 70% of the development data = TRAIN, rest = TEST
HOLDOUT_START = "2026-08-29"          # everything from this date on is locked away until 'final' mode.
                                      # (fixed date on purpose: it must not slide forward each time you run)
LOCK_FILE = "holdout_used.lock"

MIN_SAMPLES = 30                      # below this, a result is flagged as unreliable
NAMES = {1: "BULLISH", -1: "BEARISH", 0: "SIDEWAYS"}


# ----------------------------- DATA ---------------------------------------
def find_symbol():
    """Find the broker's name for gold (handles suffixes and the GOLD alias)."""
    all_names = [s.name for s in (mt5.symbols_get() or [])]
    for alias in ALIASES:
        candidates = [alias] + sorted([n for n in all_names if n.upper().startswith(alias) and n != alias], key=len)
        for c in candidates:
            if mt5.symbol_select(c, True) and mt5.symbol_info(c) is not None:
                return c
    return None


def get_rates(symbol, n):
    """Fetch up to n H1 bars by walking backwards in 60-day chunks (avoids the max-bars limit)."""
    end = datetime.now(timezone.utc) + timedelta(days=2)
    frames, total, empty_streak = [], 0, 0
    while total < n and empty_streak < 3:
        start = end - timedelta(days=60)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
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
    # The last row is the candle that is still forming. Drop it so every candle we use is CLOSED.
    return df.iloc[:-1].reset_index(drop=True)


def load(n_bars):
    if not mt5.initialize():
        sys.exit(f"MT5 init failed: {mt5.last_error()}")
    try:
        name = find_symbol()
        if name is None:
            sys.exit(f"{SYMBOL} not found at this broker. Check ALIASES.")
        raw = get_rates(name, n_bars)
    finally:
        mt5.shutdown()
    if len(raw) < n_bars:
        print(f"[warning] asked for {n_bars} candles but the broker only returned {len(raw)}.")
    df = add_structure_trend(add_atr(raw.copy()), SWING_K)
    print(f"{name}: {len(df)} closed H1 candles, {df.time.iloc[0]:%Y-%m-%d %H:%M} -> {df.time.iloc[-1]:%Y-%m-%d %H:%M}")
    return df


# ----------------------------- LABELS -------------------------------------
def add_atr(df):
    """ATR with Wilder smoothing (the same ATR that MT5 / TradingView style charts use)."""
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
        i = t - k                                   # swing candidate i is confirmed at candle t
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


# ----------------------------- LIVE MODE ----------------------------------
def live_report(df):
    d = df.dropna(subset=["trend_ms", "atr"])
    if d.empty:
        sys.exit("Not enough candles yet to confirm swings. Increase LIVE_BARS.")
    r = d.iloc[-1]
    state = int(r["trend_ms"])

    run = 1
    for v in d["trend_ms"].values[-2::-1]:
        if v == state:
            run += 1
        else:
            break

    print("\n" + "=" * 70)
    print(f"CURRENT H1 STRUCTURE - {SYMBOL} (swing K = {SWING_K})")
    print("=" * 70)
    print(f"Last closed candle : {r['time']:%Y-%m-%d %H:%M}   close = {r['close']:.2f}")
    print(f"Trend              : {NAMES[state]}   (for {run} H1 candles)")
    print(f"1 ATR (H1)         : {r['atr']:.2f}")
    print(f"Last swing high    : {r['sh_last']:.2f}   (previous: {r['sh_prev']:.2f})")
    print(f"Last swing low     : {r['sl_last']:.2f}   (previous: {r['sl_prev']:.2f})")

    if state == 1:
        lvl = r["sl_last"]
        print(f"\nBULLISH. The structure breaks if an H1 candle CLOSES BELOW {lvl:.2f}"
              f"  ({(r['close'] - lvl) / r['atr']:.1f} ATR away).")
        print("Bias: look for BUYS on pullbacks (entries on M5/M15). Avoid sells.")
    elif state == -1:
        lvl = r["sh_last"]
        print(f"\nBEARISH. The structure breaks if an H1 candle CLOSES ABOVE {lvl:.2f}"
              f"  ({(lvl - r['close']) / r['atr']:.1f} ATR away).")
        print("Bias: look for SELLS on pullbacks (entries on M5/M15). Avoid buys.")
    else:
        print(f"\nSIDEWAYS / unclear. Current range: {r['sl_last']:.2f} - {r['sh_last']:.2f}.")
        print("Bias: no clear direction on H1. Be selective or stay out.")

    print("\nLast 12 closed H1 candles:")
    tail = d.tail(12)[["time", "close", "trend_ms"]].copy()
    tail["trend"] = tail["trend_ms"].map(lambda v: NAMES[int(v)])
    print(tail[["time", "close", "trend"]].to_string(index=False))


# ----------------------------- VALIDATION HELPERS -------------------------
def tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def samples(a, h):
    """Forward move in ATRs after h candles, one sample every h candles (no overlap)."""
    fwd = (a["close"].shift(-h) - a["close"]) / a["atr"]
    s = a.assign(fwd=fwd, lab=a["trend_ms"]).dropna(subset=["fwd", "lab"])
    return s.iloc[::h]


def directional_edge(s):
    """label x forward move. Positive = bullish went up / bearish went down."""
    d = s[s["lab"] != 0]
    return d["lab"] * d["fwd"]


def report_period(a, title):
    print("\n" + "=" * 70)
    print(f"{title}: {a.time.iloc[0]:%Y-%m-%d} -> {a.time.iloc[-1]:%Y-%m-%d}  ({len(a)} candles)")
    print("%_up = how often price was HIGHER after N candles. A useful label: bull %_up clearly ABOVE all,")
    print("bear %_up clearly BELOW all. edge_ATR should be positive.")

    def pct_up(g):
        return (g["fwd"] > 0).mean() * 100 if len(g) else np.nan

    rows = []
    for h in HORIZONS:
        s = samples(a, h)
        e = directional_edge(s)
        rows.append({
            "horizon": f"{h}h", "n_bull": int((s.lab == 1).sum()), "n_bear": int((s.lab == -1).sum()),
            "n_side": int((s.lab == 0).sum()),
            "bull_%_up": pct_up(s[s.lab == 1]), "bear_%_up": pct_up(s[s.lab == -1]), "all_%_up": pct_up(s),
            "edge_ATR": e.mean(), "t": tstat(e),
            "enough_data": "ok" if len(e) >= MIN_SAMPLES else "LOW",
        })
    print(pd.DataFrame(rows).set_index("horizon").round(2).to_string())
    return rows


def consistency(rows_a, rows_b, name_a, name_b):
    both = [ra["horizon"] for ra, rb in zip(rows_a, rows_b)
            if ra["edge_ATR"] > 0 and rb["edge_ATR"] > 0]
    print(f"\nEdge positive in BOTH {name_a} and {name_b} at horizons: {', '.join(both) if both else 'none'}")
    print("(With 4 horizons, 1 hit can easily be luck. Look for the main horizons 2h/4h/6h agreeing.)")


def split_dev_holdout(df):
    d = df.dropna(subset=["trend_ms", "atr"]).reset_index(drop=True)
    cut = pd.Timestamp(HOLDOUT_START)
    return d[d.time < cut].reset_index(drop=True), d[d.time >= cut].reset_index(drop=True)


# ----------------------------- VALIDATE MODE ------------------------------
def run_validate(df):
    dev, hold = split_dev_holdout(df)
    if len(dev) < 500:
        sys.exit(f"Only {len(dev)} development candles before {HOLDOUT_START}. Need much more history.")

    split = dev.time.iloc[0] + (dev.time.iloc[-1] - dev.time.iloc[0]) * TRAIN_SHARE
    train, test = dev[dev.time < split].reset_index(drop=True), dev[dev.time >= split].reset_index(drop=True)

    print("\n" + "=" * 70)
    print(f"VALIDATION - {SYMBOL}, swing K = {SWING_K}, horizons = {HORIZONS}")
    print(f"HOLDOUT (locked, not used here): {HOLDOUT_START} onward = {len(hold)} candles")
    print(f"TRAIN up to {split:%Y-%m-%d} | TEST after that")

    r_tr = report_period(train, "TRAIN (older part)")
    r_te = report_period(test, "TEST (newer part)")
    consistency(r_tr, r_te, "TRAIN", "TEST")

    print("\nHOW TO READ THIS")
    print("- Rows marked LOW have fewer than", MIN_SAMPLES, "samples: ignore them.")
    print("- |t| below ~2 means the edge could easily be luck. Gold on H1 often mean-reverts, so a")
    print("  zero or negative edge is a valid (and useful) finding, not a bug.")
    print("- Do NOT change K or horizons to make this look better. When you are done, run 'final' once.")

    df.dropna(subset=["trend_ms"])[["time", "close", "atr", "sh_last", "sh_prev", "sl_last", "sl_prev",
                                    "trend_ms"]].to_csv("h1_structure_labels_xauusd.csv", index=False)
    print("\nLabels saved to h1_structure_labels_xauusd.csv")


# ----------------------------- FINAL (HOLDOUT) MODE -----------------------
def run_final(df):
    if os.path.exists(LOCK_FILE) and "--force" not in sys.argv:
        with open(LOCK_FILE) as f:
            print(f"The holdout was already used:\n{f.read()}")
        sys.exit("Running it again defeats its purpose. (Add --force only if you accept that.)")

    _, hold = split_dev_holdout(df)
    if len(hold) < 100:
        sys.exit(f"Only {len(hold)} holdout candles since {HOLDOUT_START}. Wait for more data.")

    print("\n" + "=" * 70)
    print(f"FINAL HOLDOUT TEST - {SYMBOL}, swing K = {SWING_K}, horizons = {HORIZONS}")
    report_period(hold, "HOLDOUT")
    print("\nThis is the one-time result. Whatever it shows, that is your honest answer.")

    with open(LOCK_FILE, "w") as f:
        f.write(f"used on {datetime.now():%Y-%m-%d %H:%M}, holdout from {HOLDOUT_START}, K={SWING_K}, "
                f"horizons={HORIZONS}\n")
    print(f"Lock file written: {LOCK_FILE}")


# ----------------------------- MAIN ---------------------------------------
def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "live":
        live_report(load(LIVE_BARS))
    elif mode == "validate":
        run_validate(load(VALIDATE_BARS))
    elif mode == "final":
        run_final(load(VALIDATE_BARS))
    else:
        sys.exit("Use: live | validate | final")


if __name__ == "__main__":
    main()