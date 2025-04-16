from alpaca_trade_api import REST
from dotenv import load_dotenv
import os
from logger import get_logger

logger = get_logger("Aplacatest")

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    logger.critical("Missing API_KEY, API_SECRET, or BASE_URL in your .env file")
    raise ValueError("Missing API_KEY, API_SECRET, or BASE_URL in your .env file")

try:
    api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL)
    account = api.get_account()
    logger.info(f"Account status: {account.status}")
except Exception as e:
    logger.error(f"Error fetching account information: {e}")
