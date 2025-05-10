import http.client
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def fetch_stock_data():
    """
    Fetch stock data using Yahoo Finance API via RapidAPI
    following the exact example implementation
    """
    try:
        # Get API key from environment
        api_key = os.getenv('RAPIDAPI_KEY')
        if not api_key:
            raise ValueError("RAPIDAPI_KEY not found in .env file")
        
        # Create connection
        conn = http.client.HTTPSConnection("yahoo-finance15.p.rapidapi.com")
        
        # Set up headers exactly as in the example
        headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': "yahoo-finance15.p.rapidapi.com"
        }
        
        # Make the request exactly as shown in the example
        print("Fetching stock data...")
        conn.request("GET", "/api/v2/markets/tickers?page=1&type=STOCKS", headers=headers)
        
        # Get response
        res = conn.getresponse()
        data = res.read()
        
        # Decode and parse JSON response
        response_data = json.loads(data.decode("utf-8"))
        
        # Close connection
        conn.close()
        
        # Convert to DataFrame if data is available
        if response_data and 'body' in response_data:
            print("\nProcessing response data...")
            
            # Create DataFrame from the body list
            df = pd.DataFrame(response_data['body'])
            
            # Clean up numeric columns
            if 'lastsale' in df.columns:
                df['lastsale'] = df['lastsale'].str.replace('$', '').astype(float)
            if 'netchange' in df.columns:
                df['netchange'] = pd.to_numeric(df['netchange'], errors='coerce')
            if 'pctchange' in df.columns:
                df['pctchange'] = df['pctchange'].str.rstrip('%').astype(float) / 100
            
            print("\nSuccessfully created DataFrame")
            print(f"Total records in response: {response_data['meta']['totalrecords']}")
            return df
        else:
            print("No data found in response")
            return None
            
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return None

def save_data(df, filename="stock_data.csv"):
    """
    Save the DataFrame to CSV file
    """
    try:
        if df is None:
            print("No data to save")
            return
            
        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("data", filename)
        
        df.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")
        print(f"Number of records: {len(df)}")
        print("\nColumns in the data:")
        print(df.columns.tolist())
        print("\nSample of the data:")
        print(df.head().to_string())
        
        # Print some basic statistics
        if 'lastsale' in df.columns:
            print("\nPrice Statistics:")
            print(df['lastsale'].describe())
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def fetch_thy_historical_data(interval="5m"):
    """
    Fetch historical data for Turkish Airlines (THYAO.IS) stock
    Args:
        interval (str): Time interval for data points (e.g., "5m" for 5 minutes)
    """
    try:
        # Get API key from environment
        api_key = os.getenv('RAPIDAPI_KEY')
        if not api_key:
            raise ValueError("RAPIDAPI_KEY not found in .env file")
        
        # Create connection
        conn = http.client.HTTPSConnection("yahoo-finance15.p.rapidapi.com")
        
        # Set up headers
        headers = {
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': "yahoo-finance15.p.rapidapi.com"
        }
        
        # Make the request for THY historical data
        endpoint = f"/api/v1/markets/stock/history?symbol=THYAO.IS&interval={interval}&diffandsplits=false"
        print(f"Fetching THY historical data with interval {interval}...")
        conn.request("GET", endpoint, headers=headers)
        
        # Get response
        res = conn.getresponse()
        data = res.read()
        
        # Decode and parse JSON response
        response_data = json.loads(data.decode("utf-8"))
        
        # Close connection
        conn.close()
        
        # Convert to DataFrame if data is available
        if response_data and isinstance(response_data, dict):
            print("\nProcessing THY historical data...")
            
            # Create DataFrame from the response
            df = pd.DataFrame(response_data)
            
            # Save the historical data
            output_path = os.path.join("data", f"thy_historical_{interval}.csv")
            df.to_csv(output_path, index=False)
            print(f"\nTHY historical data saved to {output_path}")
            print(f"Number of records: {len(df)}")
            print("\nSample of the data:")
            print(df.head().to_string())
            
            return df
        else:
            print("No historical data found in response")
            return None
            
    except Exception as e:
        print(f"Error fetching THY historical data: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return None

if __name__ == "__main__":
    # Fetch regular stock data
    df = fetch_stock_data()
    
    # Save the data if successful
    if df is not None:
        save_data(df, "stocks.csv")
    
    # Fetch THY historical data
    thy_df = fetch_thy_historical_data() 