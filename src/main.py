import os
from data_collection import fetch_stock_data, save_data
from data_preprocessing import process_data
from model import create_model, train_model, evaluate_model, plot_results

def main():
    """
    Main function to run the entire stock price prediction pipeline
    """
    print("=== ASELSAN Stock Price Prediction Pipeline ===")
    
    # Step 1: Data Collection
    print("\n1. Collecting ASELSAN stock data...")
    df = fetch_stock_data()
    save_data(df)
    
    # Step 2: Data Preprocessing
    print("\n2. Preprocessing data...")
    X_train, y_train, X_test, y_test, scaler = process_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 3: Model Creation and Training
    print("\n3. Creating and training the model...")
    model = create_model(X_train.shape[1], X_train.shape[2])
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Step 4: Model Evaluation and Visualization
    print("\n4. Evaluating model and generating visualizations...")
    predictions, true_values = evaluate_model(model, X_test, y_test, scaler)
    plot_results(history, predictions, true_values)
    
    print("\nPipeline completed successfully!")
    print("Check the 'output' directory for visualization results.")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    main() 