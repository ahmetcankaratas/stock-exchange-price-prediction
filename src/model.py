import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras_tuner as kt
from datetime import datetime, timedelta
import pandas as pd

def build_model(hp, sequence_length, n_features):
    """
    Build LSTM model with hyperparameters to tune
    
    Parameters:
    -----------
    hp : keras_tuner.HyperParameters
        Hyperparameters object
    sequence_length : int
        Number of time steps in each sequence
    n_features : int
        Number of features in the input data
    
    Returns:
    --------
    tensorflow.keras.Model
        Compiled LSTM model
    """
    model = Sequential()
    
    # Tune number of LSTM layers
    n_lstm_layers = hp.Int('n_lstm_layers', 1, 3)
    
    # First LSTM layer
    model.add(LSTM(
        units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
        return_sequences=(n_lstm_layers > 1),
        input_shape=(sequence_length, n_features)
    ))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Additional LSTM layers
    for i in range(1, n_lstm_layers):
        model.add(LSTM(
            units=hp.Int(f'lstm_units_{i+1}', min_value=32, max_value=128, step=32),
            return_sequences=(i < n_lstm_layers - 1)
        ))
        model.add(Dropout(hp.Float(f'dropout_{i+1}', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Tune learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
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

def predict_future(model, last_sequence, scaler, n_steps=60):
    """
    Predict future values
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    last_sequence : numpy.ndarray
        Last sequence of data
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used for data normalization
    n_steps : int
        Number of future steps to predict
    
    Returns:
    --------
    tuple
        (predictions, dates)
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    # Get the last date from the data
    last_date = pd.date_range(end=pd.Timestamp.now(), periods=len(current_sequence))[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_steps)
    
    for _ in range(n_steps):
        # Predict next value
        next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = next_pred[0, 0]
    
    # Inverse transform predictions
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 0] = predictions
    predictions_unscaled = scaler.inverse_transform(dummy)[:, 0]
    
    return predictions_unscaled, future_dates

def plot_tuning_results(tuner, symbol="THY"):
    """
    Plot hyperparameter tuning results
    
    Parameters:
    -----------
    tuner : keras_tuner.Tuner
        The tuner object containing tuning results
    symbol : str
        Stock symbol for plot titles
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Get all trials
    trials = tuner.oracle.trials
    
    # Extract hyperparameters and scores
    results = []
    for trial in trials.values():
        if trial.status == 'COMPLETED':
            results.append({
                'score': trial.score,
                'hyperparameters': trial.hyperparameters.values
            })
    
    # Plot validation loss for each trial
    plt.figure(figsize=(12, 6))
    scores = [r['score'] for r in results]
    plt.plot(range(len(scores)), scores, 'bo-')
    plt.title(f'{symbol} Stock Price Prediction - Hyperparameter Tuning Results')
    plt.xlabel('Trial')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join("output", f"{symbol.lower()}_tuning_results.png"))
    plt.close()
    
    # Save tuning results to file
    with open(os.path.join("output", f"{symbol.lower()}_tuning_results.txt"), "w") as f:
        f.write(f"Hyperparameter Tuning Results for {symbol}:\n\n")
        for i, result in enumerate(results):
            f.write(f"Trial {i+1}:\n")
            f.write(f"Validation Loss: {result['score']:.4f}\n")
            f.write("Hyperparameters:\n")
            for param, value in result['hyperparameters'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")

def plot_results(history, predictions, true_values, future_predictions=None, future_dates=None, symbol="THY"):
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
    future_predictions : numpy.ndarray, optional
        Future predictions
    future_dates : numpy.ndarray, optional
        Future dates
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
    plt.figure(figsize=(15, 7))
    
    # Plot historical data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=len(true_values))
    plt.plot(dates, true_values, label='Actual Price', color='blue')
    plt.plot(dates, predictions, label='Predicted Price', color='red')
    
    # Plot future predictions if available
    if future_predictions is not None and future_dates is not None:
        plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', linestyle='--')
    
    plt.title(f'{symbol} Stock Price - Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m-%Y'))
    
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
        
        if future_predictions is not None:
            f.write("\nFuture Predictions:\n")
            for date, pred in zip(future_dates, future_predictions):
                f.write(f"{date.strftime('%d-%m-%Y')}: {pred:.2f}\n")

def hyperparameter_tuning(X_train, y_train, X_test, y_test, sequence_length, n_features, max_trials=10):
    """
    Perform hyperparameter tuning using Keras Tuner
    
    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training data
    X_test, y_test : numpy.ndarray
        Test data
    sequence_length : int
        Number of time steps in each sequence
    n_features : int
        Number of features in the input data
    max_trials : int
        Maximum number of trials for hyperparameter tuning
    
    Returns:
    --------
    tuple
        (best model, tuner)
    """
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, sequence_length, n_features),
        objective='val_loss',
        max_trials=max_trials,
        directory='tuner_results',
        project_name=f'stock_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    tuner.search(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    
    # Build the model with the best hyperparameters
    best_model = build_model(best_hps, sequence_length, n_features)
    
    return best_model, tuner

if __name__ == "__main__":
    from data_preprocessing import process_data
    
    # Get processed data
    X_train, y_train, X_test, y_test, scaler = process_data()
    
    # Perform hyperparameter tuning
    best_model, tuner = hyperparameter_tuning(
        X_train, y_train, X_test, y_test,
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2]
    )
    
    # Plot tuning results
    plot_tuning_results(tuner)
    
    # Train the best model
    best_model, history = train_model(best_model, X_train, y_train, X_test, y_test, epochs=30)
    
    # Evaluate model and plot results
    predictions, true_values = evaluate_model(best_model, X_test, y_test, scaler)
    
    # Make future predictions
    last_sequence = X_test[-1]  # Get the last sequence from test data
    future_predictions, future_dates = predict_future(best_model, last_sequence, scaler, n_steps=60)
    
    # Plot results including future predictions
    plot_results(history, predictions, true_values, future_predictions, future_dates) 