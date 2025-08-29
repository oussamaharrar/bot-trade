def check_freeze_conditions(row, volatility) -> list[str]:
    """
    Determines if trading should be frozen due to extreme conditions.
    Returns list of reasons for entering freeze mode.
    """
    reasons = []

    # Extreme volatility shock
    if volatility > 2.5:
        reasons.append("Volatility shock")

    # Sudden RSI drop below 10
    rsi = row.get("rsi_14", 50)
    if rsi < 10:
        reasons.append("RSI crash < 10")

    # Sharp OBV drop
    obv = row.get("obv", 0)
    if obv < -5000:
        reasons.append("OBV collapse")

    # Stochastic overreaction
    stoch_k = row.get("stoch_k", 50)
    stoch_d = row.get("stoch_d", 50)
    if stoch_k < 10 and stoch_d < 10:
        reasons.append("Stochastic crash")

    return reasons
