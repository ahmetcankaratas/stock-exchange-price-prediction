import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_stock_data(symbol="ASELS.IS", start_date="2018-01-01", end_date=None):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (ASELS.IS for ASELSAN)
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format (default: today)
    
    Returns:
    --------
    pandas.DataFrame
        Historical stock data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    
    return df

def save_data(df, filename="aselsan_stock_data.csv"):
    """
    Save the DataFrame to CSV file
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data to save
    filename : str
        Name of the output file
    """
    output_path = os.path.join("data", filename)
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")
    print(f"Shape of the data: {df.shape}")
    print("\nFirst few rows of the data:")
    print(df.head())
    
    # Basic statistics
    print("\nBasic statistics of the data:")
    print(df.describe())

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Fetch and save data
    df = fetch_stock_data()
    save_data(df) 