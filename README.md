GOLD Price Prediction and Algorithmic Trading Bot

This project is a comprehensive Reinforcement Learning (RL) system designed to develop and deploy an autonomous algorithmic trading agent for the GOLD (XAUUSD) financial instrument. It leverages the Proximal Policy Optimization (PPO) algorithm to learn a profitable trading policy directly from historical price and technical indicator data.

The system is fully modular, covering the entire workflow from data acquisition and feature engineering to training, backtesting, and live trading execution.

🚀 Key Features and Technology

Category	Technology / Concept	Details
Core AI Model	Proximal Policy Optimization (PPO)	A state-of-the-art policy-gradient Reinforcement Learning algorithm.
RL Framework	Stable Baselines3 (PPO, MlpPolicy)	Used for efficient implementation and training of the PPO agent.
Trading Environment	Gymnasium (gym.Env)	A custom TradingEnv defines the agent's actions (Hold, Buy, Sell), state space, and reward mechanism.
Data Source/Deployment	MetaTrader 5 (MT5)	Used via the MetaTrader5 Python package for fetching historical data and executing live trades.
Data Processing	Pandas, NumPy, Scikit-learn	Used for data cleaning, creating advanced technical features, and normalizing the data with StandardScaler.
Backtesting & Eval	Custom Backtester & Matplotlib	Includes a professional backtester for traditional strategies and a separate evaluation script for the AI agent, complete with performance plotting.

📁 Project Structure

.
├── GOLD_15_data_mt5.csv         # Raw historical 15-minute GOLD data (example)
├── gold_15m_data_final_features.csv # Cleaned and feature-engineered dataset
├── ppo_trading_agent_gold_advanced.zip # Trained PPO model artifact
├── main.py                      # Data acquisition script (Fetches data from MT5)
├── feature_engineering.py       # Creates all technical indicators and scales features
├── trading_env.py               # Defines the custom RL environment (Gymnasium)
├── train_agent.py               # Loads data, trains the PPO agent, and validates performance
├── evaluate_agent.py            # Evaluates the trained agent on unseen data
├── backtester.py                # A traditional backtester for non-RL strategy comparison
├── live_trader.py               # Connects to MT5 to execute trades in real-time
└── run_project.py               # Orchestrates the data -> train -> evaluate workflow

⚙️ How to Run the Project

This project uses run_project.py to automate the workflow.

Prerequisites

    Python 3.8+

    MetaTrader 5 Terminal: Ensure you have an MT5 terminal installed and running with a valid account (demo or live).

Installation

    Clone the repository:
    Bash

git clone https://github.com/asjathahamedma/GOLD-price-prediction-model.git
cd GOLD-price-prediction-model

Create a virtual environment and install dependencies:
Bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install pandas numpy stable-baselines3 gymnasium scikit-learn MetaTrader5 matplotlib

Step-by-Step Execution

The primary workflow is handled by run_project.py:
Step	Script	Description
1 (Data)	main.py	Connects to MT5 and fetches up to 50,000 bars of 15-minute GOLD data, saving it as GOLD_15_data_mt5.csv.
2 (Features)	feature_engineering.py	Reads the raw data, calculates features like RSI, S/R, volume spikes, and candlestick patterns, then saves the final scaled dataset.
3 (Train)	train_agent.py	Splits the feature data into training (80%) and validation (20%) sets. Trains the PPO agent for 500,000 timesteps and saves the model.
4 (Evaluate)	evaluate_agent.py	Loads the trained model and runs it on the historical validation data, plotting the resulting net worth curve.

To run the entire workflow:

Bash

# Ensure your virtual environment is active
python run_project.py

Live Trading

To deploy the trained model for real-time execution, run the live_trader.py script. Use caution and understand the risks before running this on a live account.
Bash

python live_trader.py

The live trader connects to MT5, waits for the start of the next 15-minute candle, fetches the latest data, executes feature engineering on the fly, and uses the saved PPO model to decide on a BUY, SELL, or HOLD action.

📊 Feature Engineering Details

The feature_engineering.py script is robust, generating the following types of indicators, all of which are standardized (StandardScaler) before training:

    Momentum/Oscillators: Relative Strength Index (RSI), Stochastic Oscillator (stoch_k, stoch_d).

    Support/Resistance: Distance to rolling minimum/maximum over a 50-bar window.

    Volume: Volume moving average and volume spike detection.

    Price Action: Boolean flags for various candlestick patterns (e.g., Bullish/Bearish Engulfing).

    Trend: Simple price change over a lookback window to identify local trend direction.
