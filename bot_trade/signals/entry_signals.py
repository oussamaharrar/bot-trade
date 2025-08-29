def generate_entry_signals(row) -> list[str]:
    """
    Detects entry opportunities using common technical triggers.
    Returns list of entry signals.
    """
    signals = []

    # RSI recovery zone
    rsi = row.get("rsi_14", 50)
    if 45 < rsi < 60:
        signals.append("RSI recovery")

    # MACD bullish crossover
    macd = row.get("macd", 0)
    macd_signal = row.get("macd_signal", 0)
    if macd > macd_signal and macd > 0:
        signals.append("MACD bullish")

    # EMA above SMA
    ema = row.get("ema20", 0)
    sma = row.get("sma50", 0)
    if ema > sma:
        signals.append("EMA crossover")

    # Bollinger squeeze breakout
    upper = row.get("bollinger_upper", 0)
    lower = row.get("bollinger_lower", 0)
    if (upper - lower) < 0.01 * row.get("close", 1):
        signals.append("Bollinger squeeze")

    return signals
