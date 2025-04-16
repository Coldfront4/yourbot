"""
risk_manager.py – Utility functions for risk management.
"""

from datetime import datetime
from logger import get_logger

logger = get_logger("risk_manager")

DAILY_DRAWNDOWN_LIMIT = -0.25

def check_trade_risk(symbol, price, portfolio_value, max_pct_per_trade=0.05):
    max_value = portfolio_value * max_pct_per_trade
    max_shares = max_value // price
    if max_shares <= 0:
        logger.warning(f"Trade rejected for {symbol}: exposure exceeds limit.")
        return False, 0
    return True, int(max_shares)

def daily_stop_loss(drawdown_frac):
    if drawdown_frac <= DAILY_DRAWNDOWN_LIMIT:
        logger.warning(f"Daily stop-loss triggered! Drawdown: {drawdown_frac:.2%}")
        return True
    return False

def log_risk(message):
    with open("risk_log.txt", "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} - {message}\n")
    logger.info(message)
