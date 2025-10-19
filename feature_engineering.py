import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

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
        bullish_engulfing = (self.data['close'].shift(1) < self.data['open'].shift(1)) & \
                              (self.data['close'] > self.data['open']) & \
                              (self.data['open'] < self.data['close'].shift(1)) & \
                              (self.data['close'] > self.data['open'].shift(1))
        self.data['bullish_engulfing'] = bullish_engulfing.astype(int)
        bearish_engulfing = (self.data['close'].shift(1) > self.data['open'].shift(1)) & \
                             (self.data['close'] < self.data['open']) & \
                             (self.data['open'] > self.data['close'].shift(1)) & \
                             (self.data['close'] < self.data['open'].shift(1))
        self.data['bearish_engulfing'] = bearish_engulfing.astype(int)
        
    def _add_trend_direction(self, fast_ema=50, slow_ema=200):
        print("[*] Adding Trend Direction feature...")
        self.data['ema_fast'] = self.data['close'].ewm(span=fast_ema, adjust=False).mean()
        self.data['ema_slow'] = self.data['close'].ewm(span=slow_ema, adjust=False).mean()
        self.data['trend_direction'] = np.where(self.data['ema_fast'] > self.data['ema_slow'], 1, -1)
        self.data.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True)
        
    def _add_stochastic_oscillator(self, window=14):
        print("[*] Adding Stochastic Oscillator...")
        low_min = self.data['low'].rolling(window=window).min()
        high_max = self.data['high'].rolling(window=window).max()
        self.data['stoch_k'] = 100 * ((self.data['close'] - low_min) / (high_max - low_min))
        self.data['stoch_d'] = self.data['stoch_k'].rolling(window=3).mean()


    def create_master_features(self):
        print("[*] Loading data for master feature engineering...")
        self.data = pd.read_csv(self.filepath, index_col='time', parse_dates=True)
        self.data.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')
        
        self._calculate_rsi()
        self._calculate_support_resistance()
        self._calculate_volume_spikes()
        self._add_candlestick_patterns()
        self._add_trend_direction()
        self._add_stochastic_oscillator()
        
        self.data.dropna(inplace=True)
        
        print("[*] Scaling all features...")
        ohlcv_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        feature_cols = [col for col in self.data.columns if col not in ohlcv_cols]
        
        scaler = StandardScaler()
        self.data[feature_cols] = scaler.fit_transform(self.data[feature_cols])
        self.data.dropna(inplace=True)
        
        output_filepath = 'gold_15m_data_final_features.csv'
        self.data.to_csv(output_filepath)
        
        print("\n[*] Final feature engineering complete.")
        print(f"[*] New dataset saved to '{output_filepath}'")
        print(f"[*] Total features created: {len(feature_cols)}")
        print(self.data.tail())

if __name__ == "__main__":
    fe = FeatureEngineer(filepath='GOLD_15_data_mt5.csv')
    fe.create_master_features()