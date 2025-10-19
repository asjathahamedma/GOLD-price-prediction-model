import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, filepath, initial_capital=10000, fee_percent=0.001):
        self.filepath = filepath
        self.initial_capital = initial_capital
        self.fee_percent = fee_percent
        self.data = None
        self.results = None

    def _calculate_rsi(self, data, window=14):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _is_bullish_engulfing(self, df, i):
        if i == 0: return False
        current = df.iloc[i]
        previous = df.iloc[i-1]
        if previous['close'] < previous['open'] and \
           current['close'] > current['open'] and \
           current['open'] < previous['close'] and \
           current['close'] > previous['open']:
            return True
        return False

    def _is_bearish_engulfing(self, df, i):
        if i == 0: return False
        current = df.iloc[i]
        previous = df.iloc[i-1]
        if previous['close'] > previous['open'] and \
           current['close'] < current['open'] and \
           current['open'] > previous['close'] and \
           current['close'] < previous['open']:
            return True
        return False

    def _load_and_prepare_data(self):
        print("[*] Loading data and preparing features...")
        self.data = pd.read_csv(self.filepath, index_col='time', parse_dates=True)
        self.data.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')
        
        print("[*] Calculating S/R, Price Action, and Indicator features...")
        self.data['rolling_min'] = self.data['low'].rolling(window=50).min()
        self.data['rolling_max'] = self.data['high'].rolling(window=50).max()
        self.data['dist_to_support'] = self.data['close'] - self.data['rolling_min']
        self.data['dist_to_resistance'] = self.data['rolling_max'] - self.data['close']
        
        self.data['RSI_14'] = self._calculate_rsi(self.data)
        
        self.data.dropna(inplace=True)
        print("[*] Data and features ready.")
        
    def _calculate_fees(self, trade_value):
        return trade_value * self.fee_percent

    def run_combined_strategy(self):
        self._load_and_prepare_data()
        print("[*] Running combined strategy backtest (S/R + Price Action + Indicators)...")

        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = [self.initial_capital]

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            
            # --- Buy Signal Conditions ---
            is_near_support = row['dist_to_support'] < (row['close'] * 0.005)
            is_bullish_pattern = self._is_bullish_engulfing(self.data, i)
            is_not_overbought = row['RSI_14'] < 70
            
            if is_near_support and is_bullish_pattern and is_not_overbought and position == 0:
                buy_price = row['close']
                trade_value = capital
                fee = self._calculate_fees(trade_value)
                capital -= fee
                position = capital / buy_price
                trades.append({'date': row.name, 'type': 'BUY', 'price': buy_price})

            # --- Sell Signal Conditions ---
            is_near_resistance = row['dist_to_resistance'] < (row['close'] * 0.005)
            is_bearish_pattern = self._is_bearish_engulfing(self.data, i)
            is_not_oversold = row['RSI_14'] > 30
            
            if is_near_resistance and is_bearish_pattern and is_not_oversold and position > 0:
                sell_price = row['close']
                trade_value = position * sell_price
                fee = self._calculate_fees(trade_value)
                capital = trade_value - fee
                trades.append({'date': row.name, 'type': 'SELL', 'price': sell_price})
                position = 0

            current_value = capital if position == 0 else position * row['close']
            portfolio_values.append(current_value)
        
        self.results = self._analyze_trades(trades)
        self._print_results(portfolio_values)

    def _analyze_trades(self, trades):
        trade_log = pd.DataFrame(trades)
        profits = []
        num_pairs = len(trade_log) // 2
        for i in range(num_pairs):
            buy_trade = trade_log.iloc[i*2]
            sell_trade = trade_log.iloc[i*2 + 1]
            if buy_trade['type'] == 'BUY' and sell_trade['type'] == 'SELL':
                profit = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                profits.append(profit)
        return pd.Series(profits)

    def _print_results(self, portfolio_values):
        print("\n--- Backtest Results ---")
        portfolio = pd.Series(portfolio_values)
        final_capital = portfolio.iloc[-1]
        
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital:   ${final_capital:,.2f}")
        
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        print(f"Total Return:    {total_return:.2f}%")
        
        if not self.results.empty:
            win_rate = (self.results > 0).mean() * 100
            gross_profit = self.results[self.results > 0].sum()
            gross_loss = abs(self.results[self.results < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

            print(f"Win Rate:        {win_rate:.2f}%")
            print(f"Profit Factor:   {profit_factor:.2f}")

            daily_returns = portfolio.pct_change().dropna()
            if not daily_returns.empty and daily_returns.std() != 0:
                sharpe_ratio = np.sqrt(252 * (15*4)) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0.0
            print(f"Sharpe Ratio:    {sharpe_ratio:.2f}")

            cumulative_max = portfolio.cummax()
            drawdown = (portfolio - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min() * 100
            print(f"Max Drawdown:    {max_drawdown:.2f}%")
        
        print(f"Total Trades:    {len(self.results)}")

# --- Main execution block ---
if __name__ == "__main__":
    bt = Backtester(filepath='GOLD_15_data_mt5.csv')
    bt.run_combined_strategy()