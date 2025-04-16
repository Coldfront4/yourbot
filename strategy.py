import os, time, threading, asyncio, logging, re
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from stable_baselines3 import PPO

from lumibot.strategies.strategy import Strategy
from alpaca_trade_api import REST
from config import ALPACA_CREDS, BASE_URL, TRADE_UNIVERSE
from model import adjustment_model
from finbert_utils import estimate_sentiment
from risk_manager import check_trade_risk, daily_stop_loss
from logger import get_logger

logger = get_logger("strategy")

adjustment_model.eval()

class TradingEnv(gym.Env):
    def __init__(self, strategy):
        super(TradingEnv, self).__init__()
        num_symbols = len(strategy.dynamic.symbols)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 + num_symbols,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(num_symbols,), dtype=np.float32)
        self.strategy = strategy
        self.reset()

    def reset(self):
        self.current_step = 0
        return np.array(
            [self.strategy.get_total_portfolio_value(), getattr(self.strategy, "cash_reserve", 0.0)] +
            [0.0] * len(self.strategy.dynamic.symbols),
            dtype=np.float32
        )

    def step(self, action):
        self.current_step += 1
        allocation = action / np.sum(action)
        reward = np.random.rand()  # Placeholder reward; replace with actual logic
        done = self.current_step >= 20
        obs = self.reset()
        return obs, reward, done, {}

model_path = "ppo_rl_agent.zip"
if not os.path.exists(model_path):
    logger.critical(f"PPO model not found at {model_path}.")
    raise FileNotFoundError(f"PPO model not found at {model_path}. Make sure it exists.")
rl_agent = PPO.load(model_path)
logger.info("Loaded PPO agent successfully.")

def train_adjustment_model(training_data, epochs=5):
    global adjustment_model
    adjustment_model.train()
    optimizer = torch.optim.Adam(adjustment_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng()
    for epoch in range(epochs):
        losses = []
        for features, targets in training_data:
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)
            noise = torch.tensor(rng.normal(0, 0.01, size=X.shape), dtype=torch.float32)
            X = X + noise
            optimizer.zero_grad()
            outputs = adjustment_model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses) if losses else 0.0
        logger.info(f"Adjustment Model Training Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    torch.save(adjustment_model.state_dict(), "adjustment_model.pth")

def should_trigger_stop(drawdown, base_threshold=0.1, recent_vol=None):
    adj_thresh = base_threshold + (min(0.05, recent_vol * 2) if recent_vol else 0)
    return drawdown <= -adj_thresh

def should_resume_after_stop(last_stop_time, cooldown_minutes=30):
    if not last_stop_time:
        return False
    elapsed = datetime.utcnow() - last_stop_time
    return elapsed >= timedelta(minutes=cooldown_minutes)

def check_stop_or_resume(strategy):
    total_val = strategy.get_total_portfolio_value()
    drawdown_frac = total_val / strategy.initial_portfolio_value - 1
    recent_vals = [entry["total_value"] for entry in strategy.rebalance_history[-10:]]
    recent_vol = np.std(recent_vals) / np.mean(recent_vals) if len(recent_vals) >= 5 else None
    if not getattr(strategy, "stop_loss_triggered", False):
        if should_trigger_stop(drawdown_frac, base_threshold=0.1, recent_vol=recent_vol):
            strategy.daytrade_wallet['active'] = False
            strategy.rebal_wallet['active'] = False
            strategy.stop_loss_triggered = True
            strategy.last_stop_time = datetime.utcnow()
            log_trade("Daily stop loss triggered. All trading halted for the day.")
    else:
        if should_resume_after_stop(strategy.last_stop_time):
            strategy.daytrade_wallet['active'] = True
            strategy.rebal_wallet['active'] = True
            strategy.stop_loss_triggered = False
            strategy.last_stop_time = None
            log_trade("Stop-loss cooldown complete. Trading resumed.")

def log_pnl_snapshot(strategy):
    snapshot = {
        "time": datetime.utcnow().isoformat(),
        "value": strategy.get_total_portfolio_value(),
        "cash": strategy.dynamic.get_cash(),
        "positions": [(p.symbol, p.qty) for p in strategy.dynamic.get_positions()]
    }
    with open("pnl_snapshots.csv", "a") as f:
        if f.tell() == 0:
            f.write("time,value,cash,positions\n")
        f.write(f"{snapshot['time']},{snapshot['value']},{snapshot['cash']},\"{snapshot['positions']}\"\n")

def calculate_sma(prices, window=10):
    return np.mean(prices[-window:]) if len(prices) >= window else None

def calculate_volatility(prices, window=10):
    if len(prices) < window:
        return None
    returns = np.diff(prices[-window:]) / np.array(prices[-window:-1])
    return np.std(returns)

def compute_risk_parity_weights(price_history, symbols_list):
    volatilities = {}
    for symbol in symbols_list:
        prices = price_history.get(symbol, [])
        vol = calculate_volatility(prices, window=10)
        volatilities[symbol] = vol if (vol is not None and vol > 0) else 1.0
    inv_vol = {sym: 1.0 / vol for sym, vol in volatilities.items()}
    total_inv_vol = sum(inv_vol.values())
    return {sym: inv_vol[sym] / total_inv_vol for sym in inv_vol}

def update_threshold_by_volatility_with_tariff(price_history, symbol, base_threshold):
    vol = calculate_volatility(price_history.get(symbol, []), window=10)
    if vol is not None:
        tariff_prob = 0.3
        adjusted_threshold = base_threshold * (1 + vol) * (1 - 0.5 * tariff_prob)
        logger.info(f"{symbol}: Adjusted threshold = {adjusted_threshold:.3f}")
        return adjusted_threshold
    return base_threshold

def legitimize_order_size(symbol, proposed_shares, current_price, portfolio_value, max_trade_pct=0.05):
    max_shares = (portfolio_value * max_trade_pct) / current_price
    if proposed_shares > max_shares:
        logger.info(f"Capping order for {symbol}: {proposed_shares} -> {int(max_shares)} shares")
        return int(max_shares)
    return int(proposed_shares)

def log_trade(message: str):
    with open("trades_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n")
    logger.info(message)

COMMISSION_RATE = 0.001
SLIPPAGE_FACTOR = 0.0005

class TopTraderInspiredBot(Strategy):
    def __init__(self, api, symbols=None, **kwargs):
        self.simulate = kwargs.pop('simulate', True)
        super().__init__(broker=api, **kwargs)
        self.api = api
        self.symbols = symbols if symbols is not None else []
        self.sleeptime = "1H"
        self._cash = 100000.0
        self._positions = []

    def initialize(self, **kwargs):
        pass

    def get_cash(self):
        try:
            return float(super().get_cash())
        except Exception:
            return 100000.0

    def get_positions(self):
        if not self.simulate:
            try:
                return super().get_positions()
            except Exception:
                return []
        return self._positions

    def get_last_price(self, symbol):
        if not self.simulate:
            try:
                return super().get_last_price(symbol)
            except Exception as e:
                try:
                    client = REST(base_url=BASE_URL, key_id=ALPACA_CREDS['API_KEY'], secret_key=ALPACA_CREDS['API_SECRET'])
                    trade = client.get_latest_trade(symbol)
                    return float(trade.price)
                except Exception as err:
                    logger.error(f"Price fetch error for {symbol}: {err}")
                    return None
        return 100.0

    def create_order(self, symbol, quantity, side, **kwargs):
        if not self.simulate:
            return super().create_order(symbol, quantity, side, **kwargs)
        return {"symbol": symbol, "quantity": quantity, "side": side}

    def submit_order(self, order):
        if not self.simulate:
            return super().submit_order(order)
        price = self.get_last_price(order["symbol"])
        if price is None:
            logger.error("Order submission aborted: price is None.")
            return
        if order["side"] == "buy":
            cost = order["quantity"] * price * (1 + COMMISSION_RATE)
            self._cash -= cost
        elif order["side"] == "sell":
            proceeds = order["quantity"] * price * (1 - COMMISSION_RATE)
            self._cash += proceeds
        logger.info(f"Dynamic Order filled: {order}")

    def on_trading_iteration(self):
        if self.simulate:
            logger.info("Dynamic strategy iteration executed.")
        return 0.0

class SmartRebalanceModule:
    def __init__(self, symbols, cash):
        self.symbols = symbols
        self.cash = cash
        self.positions = {sym: 0 for sym in symbols}
        self.last_actions = []

    def get_last_price(self, symbol):
        return 100.0

    def total_value(self):
        total = self.cash
        for sym, qty in self.positions.items():
            total += qty * self.get_last_price(sym)
        return total

    def rebalance(self):
        self.last_actions = []
        total = self.total_value()
        if not self.symbols:
            return 0.0
        target_value = total / len(self.symbols)
        pnl = 0.0
        for sym in self.symbols:
            price = self.get_last_price(sym)
            current_value = self.positions[sym] * price
            diff = target_value - current_value
            if diff > price and self.cash >= diff:
                qty = int(diff // price)
                if qty > 0:
                    self.positions[sym] += qty
                    self.cash -= qty * price
                    logger.info(f"SmartRebalance: BUY {sym}: {qty} shares @ ${price}")
                    self.last_actions.append((sym, "buy", qty))
            elif diff < -price and self.positions[sym] > 0:
                qty = int((-diff) // price)
                qty = min(qty, self.positions[sym])
                if qty > 0:
                    self.positions[sym] -= qty
                    self.cash += qty * price
                    logger.info(f"SmartRebalance: SELL {sym}: {qty} shares @ ${price}")
                    self.last_actions.append((sym, "sell", qty))
        return pnl

class HybridStrategy(Strategy):
    def initialize(self, symbols: list = None, rebalance_universe: list = None,
                   total_cash: float = 1e7, allocation_ratio: float = 0.5,
                   profit_threshold: float = 0.05, calibration_interval: int = 5,
                   leverage: float = 2.0, liquidation_cooldown: int = 3600,
                   min_portfolio: float = 1e5, **kwargs):
        self.total_cash = total_cash
        self.allocation_ratio = allocation_ratio
        self.daytrade_wallet = {
            'initial': total_cash * allocation_ratio,
            'current': total_cash * allocation_ratio,
            'active': True
        }
        self.rebal_wallet = {
            'initial': total_cash * (1 - allocation_ratio),
            'current': total_cash * (1 - allocation_ratio),
            'active': True
        }
        self.drawdown_limit = 0.25
        self.profit_threshold = profit_threshold
        self.api = Alpaca(ALPACA_CREDS)
        simulate_subs = kwargs.pop('simulate', False)
        self.dynamic = TopTraderInspiredBot(api=self.api, symbols=symbols or [], simulate=simulate_subs, name="DynamicComponent")
        self.dynamic.initialize(symbols=symbols or [], **kwargs)
        self.dynamic._cash = self.daytrade_wallet['initial']
        rebal_symbols = rebalance_universe if rebalance_universe is not None else (symbols or [])
        self.rebalance = SmartRebalanceModule(rebal_symbols, self.rebal_wallet['current'])
        self.sleeptime = "1H"
        self.momentum_threshold = 1.005
        self.reversal_threshold = 0.995
        self.eq_profit_capture = 0.015
        self.eq_base_threshold = 0.002
        self.cr_base_threshold = 0.0015
        self.calibration_interval = calibration_interval
        self.rebalance_history = []
        self.price_history = {sym: [] for sym in (symbols or [])}
        self.cash_reserve = 0.0
        self.leverage = leverage
        self.target_cash = total_cash
        self.liquidation_cooldown = liquidation_cooldown
        self.last_liquidation_time = 0
        self.initial_portfolio_value = total_cash
        self.initial_value = self.dynamic.get_cash()
        self.min_portfolio = min_portfolio
        self.recent_capture_ratios = []
        self.adjustment_model = adjustment_model
        self.training_samples = []
        self.rl_agent = rl_agent
        self._stop_training = threading.Event()
        threading.Thread(target=self.continuous_training_loop, daemon=True).start()
        self._stop_monitoring = threading.Event()
        threading.Thread(target=self.monitor_performance, daemon=True).start()

    def update_wallet(self, wallet_name, pnl):
        wallet = self.daytrade_wallet if wallet_name == 'daytrade' else self.rebal_wallet
        wallet['current'] += pnl
        if wallet['current'] < wallet['initial'] * (1 - self.drawdown_limit):
            wallet['active'] = False
            log_trade(f"{wallet_name.capitalize()} strategy deactivated due to drawdown.")

    def can_execute_trade(self, wallet_name):
        wallet = self.daytrade_wallet if wallet_name == 'daytrade' else self.rebal_wallet
        return wallet['active']

    def execute_trade(self, wallet_name, trade_function, *args, **kwargs):
        if not self.can_execute_trade(wallet_name):
            log_trade(f"{wallet_name.capitalize()} trade blocked due to drawdown limit.")
            return 0.0
        pnl = trade_function(*args, **kwargs)
        self.update_wallet(wallet_name, pnl)
        return pnl

    def get_total_portfolio_value(self):
        total = self.dynamic.get_cash() + self.cash_reserve
        positions = self.dynamic.get_positions()
        for symbol in (self.dynamic.symbols or []):
            price = self.dynamic.get_last_price(symbol)
            if price is None:
                continue
            shares = 0.0
            for pos in positions:
                sym = getattr(pos, "symbol", getattr(pos, "asset", None))
                if sym == symbol:
                    shares = float(getattr(pos, "qty", getattr(pos, "quantity", 0)) or 0.0)
                    break
            total += shares * price
        return total

    def update_price_history(self):
        for symbol in (self.dynamic.symbols or []):
            price = self.dynamic.get_last_price(symbol)
            if price is None:
                continue
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > 50:
                self.price_history[symbol].pop(0)

    def calculate_momentum(self, symbol):
        prices = self.price_history.get(symbol, [])
        if len(prices) < 10:
            return None
        sma = calculate_sma(prices, window=10)
        return (prices[-1] / sma) if sma else None

    def continuous_training_loop(self):
        while not self._stop_training.is_set():
            if len(self.training_samples) >= 10:
                try:
                    logger.info("Continuous training: updating adjustment model with new samples...")
                    train_adjustment_model(self.training_samples, epochs=3)
                except Exception as e:
                    logger.error(f"Error during continuous training: {e}")
                self.training_samples = []
            time.sleep(30)

    def monitor_performance(self):
        while not self._stop_monitoring.is_set():
            total_value = self.get_total_portfolio_value()
            if total_value < 0.9 * self.initial_portfolio_value:
                logger.warning("Portfolio value below 90% of initial. Adopting conservative thresholds.")
                log_trade("Portfolio critically low; adjusting thresholds.")
                self.eq_profit_capture = max(self.eq_profit_capture * 0.9, 0.005)
            time.sleep(60)

    def rebalance_portfolio(self):
        self.update_price_history()
        total_value = self.get_total_portfolio_value()
        if total_value < self.min_portfolio:
            logger.warning("Portfolio value below minimum threshold. Halting trades.")
            log_trade("Portfolio critically low; halting all trades.")
            return
        net_profit = total_value - self.initial_value
        if net_profit >= self.target_cash:
            logger.info("Profit target reached. Liquidating all dynamic positions.")
            log_trade("Profit target reached; liquidating dynamic positions.")
            positions = self.dynamic.get_positions()
            for symbol in (self.dynamic.symbols or []):
                shares = 0.0
                for pos in positions:
                    sym = getattr(pos, "symbol", getattr(pos, "asset", None))
                    if sym == symbol:
                        shares = float(getattr(pos, "qty", getattr(pos, "quantity", 0)) or 0.0)
                        break
                if shares > 0:
                    order = self.dynamic.create_order(symbol, shares, "sell", type="market")
                    self.dynamic.submit_order(order)
                    msg = f"Liquidation ({symbol}): Sold {shares} shares @ ${self.dynamic.get_last_price(symbol):.2f}"
                    logger.info(msg)
                    log_trade(msg)
            return
        current_time = time.time()
        if net_profit >= self.target_cash and (current_time - self.last_liquidation_time) > self.liquidation_cooldown:
            logger.info("Target profit threshold exceeded. Partial liquidation of dynamic positions.")
            log_trade("Target threshold exceeded; liquidating some positions (cooldown).")
            positions = self.dynamic.get_positions()
            for symbol in (self.dynamic.symbols or []):
                shares = 0.0
                for pos in positions:
                    sym = getattr(pos, "symbol", getattr(pos, "asset", None))
                    if sym == symbol:
                        shares = float(getattr(pos, "qty", getattr(pos, "quantity", 0)) or 0.0)
                        break
                if shares > 0:
                    order = self.dynamic.create_order(symbol, shares, "sell", type="market")
                    self.dynamic.submit_order(order)
                    msg = f"Cooldown Liquidation ({symbol}): Sold {shares} shares @ ${self.dynamic.get_last_price(symbol):.2f}"
                    logger.info(msg)
                    log_trade(msg)
            self.last_liquidation_time = current_time

        target_allocations = {}
        for symbol in (self.dynamic.symbols or []):
            base_target = total_value * 0.10
            sentiment_score, sentiment = 0.0, None
            try:
                client = REST(base_url=BASE_URL, key_id=ALPACA_CREDS['API_KEY'], secret_key=ALPACA_CREDS['API_SECRET'])
                news_items = client.get_news(symbol=symbol, limit=5)
                headlines = [item.headline for item in news_items]
                sentiment_score, sentiment = estimate_sentiment(headlines)
            except Exception as e:
                logger.error(f"Sentiment fetch error for {symbol}: {e}")
            adjusted_target = base_target
            if sentiment == "negative":
                adjusted_target = base_target * (0.5 if sentiment_score > 0.9 else 0.8)
            elif sentiment == "positive":
                adjusted_target = base_target * (1.25 if sentiment_score > 0.9 else 1.1)
            if symbol in ["AAPL", "IBM", "CVX"]:
                adjusted_target *= 0.75
            target_allocations[symbol] = adjusted_target

        orders_executed = []
        positions = self.dynamic.get_positions()
        for symbol in (self.dynamic.symbols or []):
            current_price = self.dynamic.get_last_price(symbol)
            if current_price is None:
                logger.warning(f"No price data for {symbol}; skipping.")
                continue
            momentum = self.calculate_momentum(symbol)
            sma = calculate_sma(self.price_history.get(symbol, []), window=10)
            if sma is None or momentum is None:
                continue
            if current_price > sma * self.momentum_threshold:
                target_value = target_allocations.get(symbol, total_value * 0.10)
                shares_held = 0.0
                for pos in positions:
                    sym = getattr(pos, "symbol", getattr(pos, "asset", None))
                    if sym == symbol:
                        shares_held = float(getattr(pos, "qty", getattr(pos, "quantity", 0)) or 0.0)
                        break
                current_value = shares_held * current_price
                if current_value < target_value:
                    diff_value = target_value - current_value
                    shares_to_buy = int(diff_value // current_price)
                    shares_to_buy = legitimize_order_size(symbol, shares_to_buy, current_price, total_value, max_trade_pct=0.02)
                    if shares_to_buy >= 1:
                        if shares_to_buy * current_price > total_value * 0.05:
                            shares_to_buy = int((total_value * 0.05) // current_price)
                        if shares_to_buy >= 1:
                            order = self.dynamic.create_order(symbol, shares_to_buy, "buy", type="market")
                            self.dynamic.submit_order(order)
                            orders_executed.append({"symbol": symbol, "action": "buy", "shares": shares_to_buy, "reason": "momentum_entry"})
                            msg = f"Momentum Entry: Bought {shares_to_buy} {symbol} @ ${current_price:.2f}"
                            logger.info(msg)
                            log_trade(msg)
            elif current_price < sma * self.reversal_threshold:
                shares_held = 0.0
                for pos in positions:
                    sym = getattr(pos, "symbol", getattr(pos, "asset", None))
                    if sym == symbol:
                        shares_held = float(getattr(pos, "qty", getattr(pos, "quantity", 0)) or 0.0)
                        break
                if shares_held > 0:
                    order = self.dynamic.create_order(symbol, shares_held, "sell", type="market")
                    self.dynamic.submit_order(order)
                    orders_executed.append({"symbol": symbol, "action": "sell", "shares": shares_held, "reason": "momentum_exit"})
                    msg = f"Momentum Exit: Sold {shares_held} {symbol} @ ${current_price:.2f}"
                    logger.info(msg)
                    log_trade(msg)

        self.rebalance_history.append({
            "time": datetime.now().isoformat(),
            "total_value": self.get_total_portfolio_value(),
            "cash_reserve": self.cash_reserve,
            "orders": orders_executed
        })
        if len(self.rebalance_history) % self.calibration_interval == 0:
            self.calibrate_parameters()
        if self.can_execute_trade('rebal'):
            pnl = self.rebalance.rebalance()
            for (sym, action, qty) in self.rebalance.last_actions:
                if qty <= 0:
                    continue
                order = self.dynamic.create_order(sym, qty, action, type="market")
                self.dynamic.submit_order(order)
                log_trade(f"Rebalance: {action.upper()} {qty} {sym}")
            self.update_wallet('rebal', pnl)

    def calibrate_parameters(self):
        if len(self.rebalance_history) < self.calibration_interval:
            return
        recent = self.rebalance_history[-self.calibration_interval:]
        avg_value = np.mean([entry["total_value"] for entry in recent])
        current_value = self.get_total_portfolio_value()
        if current_value < avg_value:
            self.eq_base_threshold = min(0.03, self.eq_base_threshold + 0.0005)
            self.cr_base_threshold = min(0.015, self.cr_base_threshold + 0.00025)
            msg = f"Increasing thresholds: Equities={self.eq_base_threshold:.3f}, Others={self.cr_base_threshold:.3f}"
        else:
            self.eq_base_threshold = max(0.001, self.eq_base_threshold - 0.0005)
            self.cr_base_threshold = max(0.0005, self.cr_base_threshold - 0.00025)
            msg = f"Decreasing thresholds: Equities={self.eq_base_threshold:.3f}, Others={self.cr_base_threshold:.3f}"
        logger.info(msg)
        log_trade(msg)

    def on_trading_iteration(self):
        current_total = self.get_total_portfolio_value()
        drawdown_frac = current_total / self.initial_portfolio_value - 1
        if daily_stop_loss(drawdown_frac):
            self.daytrade_wallet['active'] = False
            self.rebal_wallet['active'] = False
            log_trade("Daily stop loss triggered. All trading halted for the day.")
            return

        if self.rl_agent is not None:
            try:
                state = np.array([self.get_total_portfolio_value(), self.cash_reserve, 0, 0], dtype=np.float32)
                action, _ = self.rl_agent.predict(state, deterministic=True)
                if action == 1:
                    logger.info("RL Agent signal: Adopt more aggressive stance.")
                elif action == 2:
                    logger.info("RL Agent signal: Adopt more conservative stance.")
            except Exception as e:
                logger.error(f"RL agent prediction error: {e}")

        logger.info("Running Hybrid Strategy iteration...")
        self.execute_trade('daytrade', self.dynamic.on_trading_iteration)
        self.execute_trade('rebal', self.rebalance.rebalance)

        for (sym, action, qty) in self.rebalance.last_actions:
            if qty > 0:
                order = self.dynamic.create_order(sym, qty, action, type="market")
                self.dynamic.submit_order(order)
                log_trade(f"Rebalance: {action.upper()} {qty} {sym}")

        self.rebalance_portfolio()
        total_val = self.get_total_portfolio_value()
        logger.info(f"Hybrid iteration complete. Combined Portfolio Value: {total_val:.2f}")

def log_trade(message: str):
    from datetime import datetime
    with open("trades_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n")
    logger.info(message)
