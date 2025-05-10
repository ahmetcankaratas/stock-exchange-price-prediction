import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(filename="thy_stock_data.csv"):
    """
    Load the stock data from CSV file
    
    Parameters:
    -----------
    filename : str
        Name of the input file
    
    Returns:
    --------
    pandas.DataFrame
        Loaded stock data
    """
    input_path = os.path.join("data", filename)
    df = pd.read_csv(input_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input stock data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional technical indicators
    """
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Trading Volume features
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    return df

def prepare_sequences(df, sequence_length=60, target_column='Close', train_split=0.8):
    """
    Prepare sequences for time series prediction
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input stock data
    sequence_length : int
        Number of time steps to look back
    target_column : str
        Column to predict
    train_split : float
        Ratio of training data
    
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, scaler)
    """
    # Select features for prediction
    features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line',
                'BB_middle', 'BB_upper', 'BB_lower', 'Volume_MA5']
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 0])  # 0 index corresponds to Close price
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def process_data(input_file="thy_stock_data.csv", sequence_length=60):
    """
    Main function to process the data
    
    Parameters:
    -----------
    input_file : str
        Name of the input file
    sequence_length : int
        Number of time steps to look back
    
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, scaler)
    """
    # Load data
    df = load_data(input_file)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Prepare sequences
    return prepare_sequences(df, sequence_length)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler = process_data()
    print("Data preprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}") 