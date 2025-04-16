from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
from logger import get_logger
from config import ALPACA_CREDS, BASE_URL
from finbert_utils import estimate_sentiment

logger = get_logger("tradingbot")

class MLTrader(Strategy):
    """
    A simple sentiment-based trading strategy using FinBERT.
    Buys or sells based on extremely positive or negative sentiment.
    """
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        from alpaca_trade_api import REST
        try:
            self.api = REST(base_url=BASE_URL, key_id=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"])
        except Exception as e:
            logger.error(f"Error initializing Alpaca REST API: {e}")
            raise

    def position_sizing(self):
        cash = self.get_cash()
        price = self.get_last_price(self.symbol)
        if price is None or price <= 0:
            return cash, price, 0
        quantity = int(round(cash * self.cash_at_risk / price))
        return cash, price, quantity

    def get_dates(self):
        now = self.get_datetime()
        start_date = now - timedelta(days=3)
        return now.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, start_date = self.get_dates()
        try:
            news_items = self.api.get_news(symbol=self.symbol, start=start_date, end=today)
            headlines = [item.headline for item in news_items]
        except Exception as e:
            logger.error(f"Error fetching news for {self.symbol}: {e}")
            return 0.0, "neutral"
        prob, sentiment = estimate_sentiment(headlines)
        return prob, sentiment

    def on_trading_iteration(self):
        cash, price, quantity = self.position_sizing()
        if cash is None or price is None or cash < price:
            logger.info("Skipping trade: insufficient data or funds.")
            return
        prob, sentiment = self.get_sentiment()
        if sentiment == "positive" and prob > 0.9:
            if self.last_trade == "sell":
                self.sell_all()
            if quantity > 0:
                order = self.create_order(self.symbol, quantity, "buy", type="market")
                self.submit_order(order)
                self.last_trade = "buy"
                logger.info(f"Sentiment bullish – bought {quantity} shares of {self.symbol}")
        elif sentiment == "negative" and prob > 0.9:
            if self.last_trade == "buy":
                self.sell_all()
            if quantity > 0:
                order = self.create_order(self.symbol, quantity, "sell", type="market")
                self.submit_order(order)
                self.last_trade = "sell"
                logger.info(f"Sentiment bearish – sold {quantity} shares of {self.symbol}")

if __name__ == "__main__":
    try:
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        broker = Alpaca(ALPACA_CREDS)
        strategy = MLTrader(name="MLStrategy", broker=broker, parameters={"symbol": "SPY", "cash_at_risk": 0.5})
        results = strategy.backtest(YahooDataBacktesting, start_date, end_date)
        logger.info("Backtest completed successfully.")
    except Exception as e:
        logger.error(f"Error during backtest execution: {e}")
