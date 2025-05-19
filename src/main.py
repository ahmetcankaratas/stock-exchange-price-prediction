import os
from data_collection import fetch_thy_historical_data, save_data
from data_preprocessing import process_data
from model import build_model, train_model, evaluate_model, plot_results, hyperparameter_tuning, predict_future

def main():
    """
    Main function to run the entire Turkish Airlines (THY) stock price prediction pipeline
    """
    print("=== Turkish Airlines (THY) Stock Price Prediction Pipeline ===")
    
    # Step 1: Data Collection
    print("\n1. Collecting THY historical stock data...")
    # Fetch daily interval data
    df = fetch_thy_historical_data(interval="1d")
    if df is not None:
        save_data(df, "thy_historical_1d.csv")
    
        # Step 2: Data Preprocessing
        print("\n2. Preprocessing THY data...")
        X_train, y_train, X_test, y_test, scaler = process_data("thy_historical_1d.csv")
        if X_train is None:
            print("Data preprocessing failed!")
            return
            
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Step 3: Model Creation and Training
        print("\n3. Creating and training the model...")
        # Perform hyperparameter tuning
        best_model, tuner = hyperparameter_tuning(
            X_train, y_train, X_test, y_test,
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2]
        )
        
        # Train the best model
        best_model, history = train_model(best_model, X_train, y_train, X_test, y_test)
        
        # Step 4: Model Evaluation and Visualization
        print("\n4. Evaluating model and generating visualizations...")
        predictions, true_values = evaluate_model(best_model, X_test, y_test, scaler)
        
        # Make future predictions
        last_sequence = X_test[-1]  # Get the last sequence from test data
        future_predictions, future_dates = predict_future(best_model, last_sequence, scaler, n_steps=60)
        
        # Plot results including future predictions
        plot_results(history, predictions, true_values, future_predictions, future_dates)
        
        print("\nPipeline completed successfully!")
        print("Check the 'output' directory for visualization results.")
    else:
        print("\nFailed to fetch THY data. Please check your API key and connection.")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    main() 