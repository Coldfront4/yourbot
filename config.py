import os
from dotenv import load_dotenv
from logger import get_logger

logger = get_logger("config")

load_dotenv()

ALPACA_CREDS = {
    "API_KEY": os.getenv("APCA_API_KEY_ID"),
    "API_SECRET": os.getenv("APCA_API_SECRET_KEY"),
    "PAPER": True
}

BASE_URL = os.getenv("BASE_URL", "https://paper-api.alpaca.markets")

TRADE_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", 
    "BTC/USD", "ETH/USD", "SPY", "QQQ"
]

logger.info("Configuration loaded successfully.")
