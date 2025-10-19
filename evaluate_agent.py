import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from trading_env import TradingEnv

# --- Configuration ---
DATA_FILEPATH = 'gold_15m_data_master_features.csv'
MODEL_PATH = 'ppo_trading_agent_gold_advanced.zip'

def evaluate_agent():
    print("[*] Loading data and trained model for evaluation...")
    df = pd.read_csv(DATA_FILEPATH, index_col='time', parse_dates=True)
    
    env = TradingEnv(df)
    model = PPO.load(MODEL_PATH, env=env)
    
    print("[*] Starting evaluation...")
    obs, info = env.reset()
    done = False
    net_worth_history = [env.initial_capital]

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        net_worth_history.append(env.net_worth)

    print("[*] Evaluation complete.")
    
    # --- UPDATED PLOTTING SECTION ---
    print("[*] Plotting net worth over time...")
    plt.figure(figsize=(14, 7))
    plt.plot(net_worth_history)
    plt.title("AI Agent Performance on Historical Data")
    plt.xlabel(f"Timesteps ({len(df)} total)")
    plt.ylabel("Net Worth (USD)")
    plt.grid(True)
    
    # Save the figure as an image file
    plt.savefig('evaluation_results.png')
    print("[*] Chart saved to 'evaluation_results.png'")
    
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    evaluate_agent()