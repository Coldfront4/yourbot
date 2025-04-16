import re
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading, time
from datetime import datetime
import asyncio

from logger import get_logger
logger = get_logger("bot_gui")

# Import strategy and trading components
from strategy import HybridStrategy
from config import ALPACA_CREDS
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.backtesting import YahooDataBacktesting

class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot Control Panel")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.backtest_frame = ttk.Frame(self.notebook)
        self.live_frame = ttk.Frame(self.notebook)
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_frame, text="Backtest")
        self.notebook.add(self.live_frame, text="Live Trading")
        self.notebook.add(self.train_frame, text="Training")

        self._setup_backtest_tab()
        self._setup_live_tab()
        self._setup_training_tab()

        # Thread placeholders
        self.backtest_thread = None
        self.live_thread = None
        self.train_thread = None
        self.live_trader = None
        self.live_strategy = None

    def _setup_backtest_tab(self):
        ttk.Label(self.backtest_frame, text="Symbols (comma-separated):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.backtest_symbols_entry = ttk.Entry(self.backtest_frame, width=40)
        self.backtest_symbols_entry.insert(0, "AAPL,MSFT,GOOG")
        self.backtest_symbols_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(self.backtest_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.backtest_start_entry = ttk.Entry(self.backtest_frame, width=15)
        self.backtest_start_entry.insert(0, "2020-01-01")
        self.backtest_start_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(self.backtest_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.backtest_end_entry = ttk.Entry(self.backtest_frame, width=15)
        self.backtest_end_entry.insert(0, "2023-12-31")
        self.backtest_end_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        self.backtest_button = ttk.Button(self.backtest_frame, text="Run Backtest", command=self.start_backtest)
        self.backtest_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.backtest_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.backtest_ax = self.backtest_fig.add_subplot(111)
        self.backtest_canvas = FigureCanvasTkAgg(self.backtest_fig, master=self.backtest_frame)
        self.backtest_canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.backtest_log = tk.Text(self.backtest_frame, height=10, width=80)
        self.backtest_log.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.backtest_log.config(state='disabled')

    def _setup_live_tab(self):
        ttk.Label(self.live_frame, text="Symbols (comma-separated):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.live_symbols_entry = ttk.Entry(self.live_frame, width=40)
        self.live_symbols_entry.insert(0, "AAPL,MSFT,GOOG")
        self.live_symbols_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(self.live_frame, text="Initial Cash ($):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.live_cash_entry = ttk.Entry(self.live_frame, width=15)
        self.live_cash_entry.insert(0, "1000000")
        self.live_cash_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.live_start_button = ttk.Button(self.live_frame, text="Start Live Trading", command=self.start_live_trading)
        self.live_start_button.grid(row=2, column=0, padx=5, pady=10, sticky='w')
        self.live_stop_button = ttk.Button(self.live_frame, text="Stop Live Trading", command=self.stop_live_trading, state='disabled')
        self.live_stop_button.grid(row=2, column=1, padx=5, pady=10, sticky='w')

        self.live_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.live_ax = self.live_fig.add_subplot(111)
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=self.live_frame)
        self.live_canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.live_log = tk.Text(self.live_frame, height=10, width=80)
        self.live_log.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        self.live_log.config(state='disabled')

    def _setup_training_tab(self):
        self.train_button = ttk.Button(self.train_frame, text="Train PPO Agent", command=self.start_training)
        self.train_button.pack(padx=5, pady=5)
        self.train_log = tk.Text(self.train_frame, height=15, width=80)
        self.train_log.pack(padx=5, pady=5)
        self.train_log.config(state='disabled')

    def start_backtest(self):
        self.backtest_button.config(state='disabled')
        self.backtest_ax.clear()
        self.backtest_ax.set_title("Backtest Portfolio Value")
        self.backtest_ax.set_xlabel("Time Step")
        self.backtest_ax.set_ylabel("Portfolio Value")
        self.backtest_canvas.draw()
        self._clear_log(self.backtest_log)

        symbols = [s.strip() for s in self.backtest_symbols_entry.get().split(',') if s.strip()]
        if not symbols:
            symbols = ["SPY"]
        try:
            start_date = datetime.fromisoformat(self.backtest_start_entry.get())
            end_date = datetime.fromisoformat(self.backtest_end_entry.get())
        except Exception as e:
            self._append_log(self.backtest_log, f"Invalid date format: {e}")
            self.backtest_button.config(state='normal')
            return

        def run_backtest():
            try:
                from lumibot.brokers import Alpaca
                broker = Alpaca(ALPACA_CREDS)
                strategy = HybridStrategy(name="HybridBacktest", broker=broker, parameters={
                    "symbols": symbols,
                    "rebalance_universe": symbols,
                    "total_cash": 1000000.0,
                    "allocation_ratio": 0.5,
                    "simulate": True
                })
                open("trades_log.txt", "w").close()
                strategy.backtest(YahooDataBacktesting, start_date, end_date)
            except Exception as e:
                self._append_log(self.backtest_log, f"Backtest error: {e}")
                logger.error(f"Backtest error: {e}")
            finally:
                self.backtest_button.config(state='normal')

        self.backtest_thread = threading.Thread(target=run_backtest, daemon=True)
        self.backtest_thread.start()
        self._update_backtest_output()

    def _update_backtest_output(self):
        if self.backtest_thread and self.backtest_thread.is_alive():
            try:
                with open("trades_log.txt", "r") as f:
                    lines = f.readlines()
                values = []
                for line in lines:
                    if "Portfolio Value" in line or "total_value" in line:
                        try:
                            val = float(line.strip().split()[-1])
                        except Exception:
                            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                            if matches:
                                val = float(matches[-1])
                        values.append(val)
                if values:
                    self.backtest_ax.clear()
                    self.backtest_ax.plot(values, color='blue')
                    self.backtest_ax.set_title("Backtest Portfolio Value")
                    self.backtest_ax.set_xlabel("Iteration")
                    self.backtest_ax.set_ylabel("Portfolio Value")
                    self.backtest_canvas.draw()
            except Exception as e:
                logger.error(f"Error updating backtest output: {e}")
            self._refresh_log_widget(self.backtest_log)
            self.root.after(1000, self._update_backtest_output)
        else:
            self._refresh_log_widget(self.backtest_log)

    def start_live_trading(self):
        self.live_start_button.config(state='disabled')
        symbols = [s.strip() for s in self.live_symbols_entry.get().split(',') if s.strip()]
        if not symbols:
            symbols = ["SPY"]
        try:
            total_cash = float(self.live_cash_entry.get())
        except Exception as e:
            self._append_log(self.live_log, f"Invalid cash input, using default. {e}")
            total_cash = 1000000.0
        self.live_ax.clear()
        self.live_ax.set_title("Live Portfolio Value")
        self.live_ax.set_xlabel("Update Tick")
        self.live_ax.set_ylabel("Portfolio Value")
        self.live_canvas.draw()
        self._clear_log(self.live_log)

        try:
            from lumibot.brokers import Alpaca
            self.live_trader = Trader(broker=Alpaca(ALPACA_CREDS), debug=False)
            self.live_strategy = HybridStrategy(name="HybridLive", broker=self.live_trader.broker, parameters={
                "symbols": symbols,
                "rebalance_universe": symbols,
                "total_cash": total_cash,
                "allocation_ratio": 0.5
            })
            self.live_trader.add_strategy(self.live_strategy)
            open("trades_log.txt", "w").close()
        except Exception as e:
            self._append_log(self.live_log, f"Error initializing live trading: {e}")
            logger.error(f"Error initializing live trading: {e}")
            self.live_start_button.config(state='normal')
            return

        def run_live():
            try:
                import asyncio
                asyncio.set_event_loop(asyncio.new_event_loop())
                self.live_trader.run_all()
            except Exception as e:
                self._append_log(self.live_log, f"Live trading error: {e}")
                logger.error(f"Live trading error: {e}")
            finally:
                self.live_start_button.config(state='normal')
                self.live_stop_button.config(state='disabled')

        self.live_thread = threading.Thread(target=run_live, daemon=True)
        self.live_thread.start()
        self.live_stop_button.config(state='normal')
        self._update_live_output()

    def _update_live_output(self):
        if self.live_thread and self.live_thread.is_alive():
            try:
                total_val = self.live_strategy.get_total_portfolio_value()
                ydata = getattr(self, "_live_values", [])
                ydata.append(total_val)
                self._live_values = ydata[-100:]
                self.live_ax.clear()
                self.live_ax.plot(self._live_values, color='green')
                self.live_ax.set_title("Live Portfolio Value")
                self.live_ax.set_xlabel("Time Step")
                self.live_ax.set_ylabel("Portfolio Value")
                self.live_canvas.draw()
            except Exception as e:
                logger.error(f"Error updating live chart: {e}")
            self._refresh_log_widget(self.live_log)
            self.root.after(2000, self._update_live_output)
        else:
            self._refresh_log_widget(self.live_log)

    def stop_live_trading(self):
        if self.live_trader:
            try:
                self.live_trader.stop_all()
            except Exception as e:
                self._append_log(self.live_log, f"Error stopping live trader: {e}")
                logger.error(f"Error stopping live trader: {e}")

    def start_training(self):
        self.train_button.config(state='disabled')
        self._clear_log(self.train_log)

        def run_training():
            try:
                import train_ppo_agent
                from stable_baselines3 import PPO
                env = train_ppo_agent.RealisticTradingEnv()
                model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, n_steps=256, batch_size=64)
                model.learn(total_timesteps=20000)
                model.save("ppo_rl_agent.zip")
                self._append_log(self.train_log, "Training complete. Saved model as ppo_rl_agent.zip")
                logger.info("Training complete. Model saved as ppo_rl_agent.zip")
            except Exception as e:
                self._append_log(self.train_log, f"Training error: {e}")
                logger.error(f"Training error: {e}")
            finally:
                self.train_button.config(state='normal')

        self.train_thread = threading.Thread(target=run_training, daemon=True)
        self.train_thread.start()

    def _append_log(self, widget, message):
        widget.config(state='normal')
        widget.insert(tk.END, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        widget.config(state='disabled')
        widget.see(tk.END)

    def _refresh_log_widget(self, widget):
        widget.config(state='normal')
        try:
            with open("trades_log.txt", "r") as f:
                content = f.read()
        except Exception as e:
            content = ""
            logger.error(f"Error reading trades log: {e}")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state='disabled')
        widget.see(tk.END)

    def _clear_log(self, widget):
        widget.config(state='normal')
        widget.delete("1.0", tk.END)
        widget.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()
