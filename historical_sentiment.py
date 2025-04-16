# historical_sentiment.py

import os
import requests
from datetime import datetime
from typing import List, Tuple
from finbert_utils import estimate_sentiment
from logger import get_logger

logger = get_logger("historical_sentiment")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def fetch_newsapi_headlines(symbol: str, start_date: str, end_date: str) -> List[str]:
    """
    Fetch headlines from NewsAPI based on symbol (keyword) and date range.
    """
    assert NEWSAPI_KEY, "NEWSAPI_KEY must be set in environment variables"
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "from": start_date,
        "to": end_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 100,
        "apiKey": NEWSAPI_KEY
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        headlines = [article["title"] for article in data.get("articles", [])]
        logger.info(f"Fetched {len(headlines)} headlines for {symbol}")
        return headlines
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return []

def historical_sentiment(symbol: str, start: str, end: str) -> Tuple[float, str]:
    headlines = fetch_newsapi_headlines(symbol, start, end)
    if not headlines:
        return 0.0, "neutral"
    return estimate_sentiment(headlines)
