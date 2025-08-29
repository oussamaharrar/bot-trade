def macd_cross(row):
    return row.get("macd", 0) > row.get("macd_signal", 0)

def rsi_recovery(row):
    return row.get("rsi_14", 0) > 35

def low_volatility(volatility):
    return volatility < 1.0

def ema_vs_sma(row):
    return row.get("ema20", 0) > row.get("sma50", 0)

def obv_positive(row):
    return row.get("obv", 0) > 0

def adx_strong(row):
    return row.get("adx", 0) > 20

def roc_positive(row):
    return row.get("roc", 0) > 0

def stoch_bullish(row):
    return row.get("stoch_k", 0) > row.get("stoch_d", 0)

def mfi_ok(row):
    return row.get("mfi", 0) > 50

ACTIVE_SIGNALS = [
    macd_cross,
    rsi_recovery,
    low_volatility,
    ema_vs_sma,
    obv_positive,
    adx_strong,
    roc_positive,
    stoch_bullish,
    mfi_ok
]

def compute_recovery_signals(row, volatility):
    count = 0
    for func in ACTIVE_SIGNALS:
        try:
            if "volatility" in func.__code__.co_varnames:
                if func(volatility):
                    count += 1
            else:
                if func(row):
                    count += 1
        except:
            continue
    return count
