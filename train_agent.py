import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from trading_env import TradingEnv

# --- Configuration ---
DATA_FILEPATH = 'gold_15m_data_master_features.csv'
TOTAL_TIMESTEPS = 500000  # Increased for a more thorough training session
MODEL_SAVE_PATH = 'ppo_trading_agent_gold_advanced'
TRAIN_SPLIT_PERCENT = 0.8 # Use the first 80% of data for training

def train_and_validate_agent():
    """
    Loads data, splits it, trains the agent on the training set,
    and validates its performance on the unseen validation set.
    """
    print("[*] Loading master feature data...")
    df = pd.read_csv(DATA_FILEPATH, index_col='time', parse_dates=True)
    
    # --- 1. Split the data ---
    split_index = int(len(df) * TRAIN_SPLIT_PERCENT)
    train_df = df.iloc[:split_index]
    validation_df = df.iloc[split_index:]
    
    print(f"[*] Data split into {len(train_df)} training steps and {len(validation_df)} validation steps.")

    # --- 2. Train the agent on the training set ---
    print("\n--- Phase 1: Training ---")
    train_env = TradingEnv(train_df)
    model = PPO("MlpPolicy", train_env, verbose=1)
    
    print(f"[*] Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("[*] Training complete.")
    
    model.save(MODEL_SAVE_PATH)
    print(f"[*] Model saved to '{MODEL_SAVE_PATH}.zip'.")

    # --- 3. Validate the agent on the unseen validation set ---
    print("\n--- Phase 2: Validation ---")
    validation_env = TradingEnv(validation_df)
    obs, info = validation_env.reset()
    done = False
    net_worth_history = [validation_env.initial_capital]

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = validation_env.step(action)
        done = terminated or truncated
        net_worth_history.append(validation_env.net_worth)

    print("[*] Validation complete.")
    
    # Plot the validation results
    plt.figure(figsize=(14, 7))
    plt.plot(net_worth_history)
    plt.title("AI Agent Performance on Unseen Validation Data")
    plt.xlabel(f"Timesteps ({len(validation_df)} total)")
    plt.ylabel("Net Worth (USD)")
    plt.grid(True)
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    train_and_validate_agent()