# Stock Price Prediction Project

This project implements stock price prediction for BIST 50 stocks using Prophet and LSTM models, with a special focus on Turkish Airlines (THYAO).

## Project Structure
- `data/`: Contains raw and processed data files
  - `stocks.csv`: General stock market data
  - `thy_historical_{interval}.csv`: Historical THY stock data at different intervals
- `notebooks/`: Jupyter notebooks with analysis and visualization
- `src/`: Source code for the project
  - `data_collection.py`: Scripts for collecting stock data using RapidAPI
  - `data_preprocessing.py`: Data cleaning and feature engineering
  - `model.py`: Machine learning model implementation
  - `utils.py`: Utility functions
  - `visualization.py`: Data visualization functions

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your RapidAPI key:
```
RAPIDAPI_KEY=your_api_key_here
```

## Usage
1. Run data collection:
```bash
python src/data_collection.py
```
This will:
- Fetch general stock market data
- Collect THY historical data with customizable intervals
- Save data to CSV files in the `data/` directory


streamlit run src/xu050-stock-price-predictor-main/stock_price_forecasting.py

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Features

- BIST 50 stock price prediction using Prophet
- THYAO detailed analysis using both Prophet and LSTM
- Interactive visualizations
- Performance metrics
- Technical indicators

## Project Structure

```
stock-exchange-price-prediction/
├── src/
│   ├── xu050-stock-price-predictor-main/
│   │   └── stock_price_forecasting.py
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   └── main.py
├── data/
├── output/
├── requirements.txt
├── README.md
└── .env
```

## Data Collection
The project uses the Yahoo Finance API through RapidAPI to collect stock data:
- General market data endpoint: `/api/v2/markets/tickers`
- THY historical data endpoint: `/api/v1/markets/stock/history`
- Customizable intervals for historical data (5m, 1h, 1d)
- Automatic data cleaning and formatting

## Project Components
1. Data Collection: Using Yahoo Finance API via RapidAPI
2. Data Preprocessing: Cleaning, feature engineering, and data preparation
3. Exploratory Data Analysis: Understanding patterns and relationships
4. Model Development: Implementation of various ML models
5. Model Evaluation: Performance metrics and visualization
6. Prediction: Making future price predictions

## Technical Indicators Used
- Moving Averages (5-day and 20-day)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume Indicators

## Model Architecture
- LSTM (Long Short-Term Memory) neural network
- Multiple LSTM layers with dropout for regularization
- Sequence-based prediction (60-day lookback period)
- Early stopping to prevent overfitting

## API Rate Limits
- Please note that the RapidAPI Yahoo Finance endpoint has rate limits
- Default rate limits apply based on your RapidAPI subscription
- Implement appropriate delays between requests if needed 