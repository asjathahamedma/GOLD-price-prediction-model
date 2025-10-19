import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def get_mt5_data(symbol='GOLD', timeframe=mt5.TIMEFRAME_M15, num_bars=50000):
    """
    Connects to a running MT5 terminal and fetches historical data.
    """
    # Establish connection to the MetaTrader 5 terminal
    if not mt5.initialize():
        print(f"[!] initialize() failed, error code = {mt5.last_error()}")
        return None

    print(f"[*] Connected to MetaTrader 5, version: {mt5.version()}")

    # Request historical data
    print(f"[*] Fetching {num_bars} bars of {symbol} at {timeframe} timeframe...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)

    # Shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()

    if rates is None:
        print(f"[!] No data received for {symbol}. Check the symbol name.")
        return None

    # Convert the data to a pandas DataFrame
    data = pd.DataFrame(rates)
    # Convert timestamp to a readable datetime format
    data['time'] = pd.to_datetime(data['time'], unit='s')
    
    file_path = f'{symbol}_{timeframe}_data_mt5.csv'
    data.to_csv(file_path)

    print(f"[*] Data fetched successfully. Shape: {data.shape}")
    print(f"[*] Saved to '{file_path}'")
    print(data.tail())
    
    return data

# --- Main execution block ---
if __name__ == "__main__":
    get_mt5_data()