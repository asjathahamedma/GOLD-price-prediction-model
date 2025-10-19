import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df, initial_capital=10000, fee_percent=0.001):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_capital = initial_capital
        self.fee_percent = fee_percent
        
        self.action_space = gym.spaces.Discrete(3) # Hold, Buy, Sell
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32
        )
        
        self.current_step = 0
        self.buy_price = 0 # Price at which the last buy was made
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.capital = self.initial_capital
        self.shares_held = 0
        self.net_worth = self.initial_capital
        self.buy_price = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0

        # Action: Buy
        if action == 1 and self.capital > 0:
            self.buy_price = current_price
            trade_value = self.capital
            fee = trade_value * self.fee_percent
            self.capital -= fee
            self.shares_held = self.capital / self.buy_price
            self.capital = 0

        # Action: Sell
        elif action == 2 and self.shares_held > 0:
            sell_price = current_price
            trade_value = self.shares_held * sell_price
            fee = self._calculate_fees(trade_value)
            self.capital = trade_value - fee
            self.shares_held = 0
            
            # --- REWARD SHAPING LOGIC ---
            profit = sell_price - self.buy_price
            if profit > 0:
                reward = 10 # Large reward for a profitable trade
            else:
                reward = -10 # Large penalty for a losing trade
            
            self.buy_price = 0 # Reset buy price

        # Calculate net worth for this step
        self.net_worth = self.capital + (self.shares_held * current_price)
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        
        observation = self._next_observation()
        
        return observation, reward, terminated, False, {}

    # Helper function for calculating fees
    def _calculate_fees(self, trade_value):
        return trade_value * self.fee_percent

# --- Example of how to use the environment ---
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    df = pd.read_csv('gold_15m_data_with_features.csv', index_col='time', parse_dates=True)
    env = TradingEnv(df)
    
    check_env(env)
    print("\n[*] Environment check complete. No errors found.")