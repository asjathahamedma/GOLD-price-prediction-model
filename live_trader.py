import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import time
import os

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
    def _calculate_rsi(self, window=14):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
    def _calculate_support_resistance(self, window=50):
        self.data['rolling_min'] = self.data['low'].rolling(window=window).min()
        self.data['rolling_max'] = self.data['high'].rolling(window=window).max()
        self.data['dist_to_support'] = self.data['close'] - self.data['rolling_min']
        self.data['dist_to_resistance'] = self.data['rolling_max'] - self.data['close']
        self.data.drop(['rolling_min', 'rolling_max'], axis=1, inplace=True)
    def _calculate_volume_spikes(self, window=50):
        volume_col = 'real_volume' if 'real_volume' in self.data.columns and self.data['real_volume'].sum() > 0 else 'tick_volume'
        self.data['volume_mavg'] = self.data[volume_col].rolling(window=window).mean()
        self.data['volume_spike_ratio'] = self.data[volume_col] / self.data['volume_mavg']
        self.data.drop('volume_mavg', axis=1, inplace=True)
    def _add_candlestick_patterns(self):
        bullish_engulfing = (self.data['close'].shift(1) < self.data['open'].shift(1)) & (self.data['close'] > self.data['open']) & (self.data['open'] < self.data['close'].shift(1)) & (self.data['close'] > self.data['open'].shift(1))
        self.data['bullish_engulfing'] = bullish_engulfing.astype(int)
        bearish_engulfing = (self.data['close'].shift(1) > self.data['open'].shift(1)) & (self.data['close'] < self.data['open']) & (self.data['open'] > self.data['close'].shift(1)) & (self.data['close'] < self.data['open'].shift(1))
        self.data['bearish_engulfing'] = bearish_engulfing.astype(int)
    def create_features(self):
        self._calculate_rsi()
        self._calculate_support_resistance()
        self._calculate_volume_spikes()
        self._add_candlestick_patterns()
        self.data.dropna(inplace=True)
        return self.data

class LiveTrader:
    def __init__(self, symbol, model_path, lot_size=0.1, sl_percent=1.0, tp_percent=2.0):
        self.symbol = symbol
        self.model = PPO.load(model_path)
        self.lot_size = lot_size
        self.sl_percent = sl_percent
        self.tp_percent = tp_percent

    def _connect_to_mt5(self):
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return False
        account_info = mt5.account_info()
        if account_info is not None:
            print(f"Connected to account #{account_info.login} on {account_info.server}")
        return True

    def _get_latest_data(self, num_bars=300):
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, num_bars)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def _get_open_position(self):
        """Checks for an open position for the symbol."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            return None
        return positions[0] # Return the first open position

    def _execute_trade(self, action, position):
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info.visible:
            print("[!] Market is closed or symbol not visible.")
            return

        lot = self.lot_size
        ask_price = mt5.symbol_info_tick(self.symbol).ask
        bid_price = mt5.symbol_info_tick(self.symbol).bid

        # --- AI wants to be LONG ---
        if action == 1 and not position:
            print("[*] AI signals BUY, preparing to open new LONG position...")
            sl = ask_price - (self.sl_percent / 100 * ask_price)
            tp = ask_price + (self.tp_percent / 100 * ask_price)
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lot,
                "type": mt5.ORDER_TYPE_BUY, "price": ask_price, "sl": sl, "tp": tp,
                "deviation": 20, "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC, # <-- THE FIX
                "comment": "AI_BOT_BUY",
            }
            result = mt5.order_send(request)
            print(f"[*] Broker response: {result}")
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"[!] Failed to open LONG position.")
        
        # --- AI wants to be SHORT ---
        elif action == 2 and not position:
            print("[*] AI signals SELL, preparing to open new SHORT position...")
            sl = bid_price + (self.sl_percent / 100 * bid_price)
            tp = bid_price - (self.tp_percent / 100 * bid_price)
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lot,
                "type": mt5.ORDER_TYPE_SELL, "price": bid_price, "sl": sl, "tp": tp,
                "deviation": 20, "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC, # <-- THE FIX
                "comment": "AI_BOT_SELL",
            }
            result = mt5.order_send(request)
            print(f"[*] Broker response: {result}")
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"[!] Failed to open SHORT position.")

        # --- Logic for closing positions ---
        elif (action == 2 and position and position.type == 0): # Close LONG
            print(f"[*] AI signals SELL, closing existing LONG position...")
            request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lot, "type": mt5.ORDER_TYPE_SELL, "price": bid_price, "deviation": 20, "position": position.ticket}
            mt5.order_send(request)
        elif (action == 1 and position and position.type == 1): # Close SHORT
            print(f"[*] AI signals BUY, closing existing SHORT position...")
            request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lot, "type": mt5.ORDER_TYPE_BUY, "price": ask_price, "deviation": 20, "position": position.ticket}
            mt5.order_send(request)

    def run(self):
        if not self._connect_to_mt5():
            return
            
        print(f"[*] AI Trader started for {self.symbol}. Waiting for next candle...")

        while True:
            now = time.time()
            next_run = (now // 900 + 1) * 900
            time.sleep(next_run - now)

            latest_data = self._get_latest_data()
            fe = FeatureEngineer(latest_data)
            data_with_features = fe.create_features()
            
            if data_with_features.empty:
                print("Could not generate features. Skipping.")
                continue

            latest_observation = data_with_features.iloc[-1].values.astype(np.float32)
            action, _ = self.model.predict(latest_observation, deterministic=True)
            
            open_position = self._get_open_position()
            
            print(f"\n{pd.Timestamp.now()} - Current Position: {'LONG' if open_position and open_position.type == 0 else 'SHORT' if open_position else 'None'} | AI Action: {'HOLD' if action == 0 else 'BUY' if action == 1 else 'SELL'}")

            self._execute_trade(action, open_position)

if __name__ == "__main__":
    trader = LiveTrader(symbol="GOLD", model_path="ppo_trading_agent_gold_advanced.zip")
    trader.run()