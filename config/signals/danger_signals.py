def detect_danger_signals(row, volatility) -> list[str]:
    """
    Detects potential market danger signals from the current market row and volatility.
    Returns a list of textual notes indicating why the environment is considered dangerous.
    """
    notes = []

    # High volatility environment
    if volatility > 1.5:
        notes.append("High volatility")

    # RSI under 20 might indicate panic or crash zones
    rsi = row.get("rsi_14", 50)
    if rsi < 20:
        notes.append("RSI extremely oversold")

    # ADX high + trend reversal might mean violent movements
    adx = row.get("adx", 0)
    roc = row.get("roc", 0)
    if adx > 30 and roc < -3:
        notes.append("Strong trend with negative momentum")

    # MACD divergence
    macd = row.get("macd", 0)
    macd_signal = row.get("macd_signal", 0)
    if macd < macd_signal and macd < 0:
        notes.append("Bearish MACD crossover")

    # MFI extreme low (capital outflow)
    mfi = row.get("mfi", 50)
    if mfi < 20:
        notes.append("Money Flow Index under 20")

    return notes
