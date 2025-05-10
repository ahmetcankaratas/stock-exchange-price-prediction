import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_model(sequence_length, n_features):
    """
    Create LSTM model for stock price prediction
    
    Parameters:
    -----------
    sequence_length : int
        Number of time steps in each sequence
    n_features : int
        Number of features in the input data
    
    Returns:
    --------
    tensorflow.keras.Model
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Train the LSTM model
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        The LSTM model to train
    X_train, y_train : numpy.ndarray
        Training data
    X_test, y_test : numpy.ndarray
        Test data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    
    Returns:
    --------
    tuple
        (trained model, training history)
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance and make predictions
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained LSTM model
    X_test, y_test : numpy.ndarray
        Test data
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used for data normalization
    
    Returns:
    --------
    tuple
        (predictions, true values)
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    # Create dummy array for inverse transform
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 0] = predictions.flatten()
    predictions_unscaled = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = y_test
    true_values_unscaled = scaler.inverse_transform(dummy)[:, 0]
    
    # Calculate metrics
    mse = np.mean((predictions_unscaled - true_values_unscaled) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_unscaled - true_values_unscaled))
    
    print(f"\nModel Performance Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return predictions_unscaled, true_values_unscaled

def plot_results(history, predictions, true_values, symbol="THY"):
    """
    Plot training history and prediction results
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    predictions : numpy.ndarray
        Predicted values
    true_values : numpy.ndarray
        Actual values
    symbol : str
        Stock symbol for plot titles
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{symbol} Stock Price Prediction - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("output", f"{symbol.lower()}_training_history.png"))
    plt.close()
    
    # Plot predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title(f'{symbol} Stock Price - Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("output", f"{symbol.lower()}_predictions.png"))
    plt.close()
    
    # Calculate and print metrics
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    
    print("\nModel Performance Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Save metrics to file
    with open(os.path.join("output", f"{symbol.lower()}_metrics.txt"), "w") as f:
        f.write(f"Model Performance Metrics for {symbol}:\n")
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")

if __name__ == "__main__":
    from data_preprocessing import process_data
    
    # Get processed data
    X_train, y_train, X_test, y_test, scaler = process_data()
    
    # Create and train model
    model = create_model(X_train.shape[1], X_train.shape[2])
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model and plot results
    predictions, true_values = evaluate_model(model, X_test, y_test, scaler)
    plot_results(history, predictions, true_values) 