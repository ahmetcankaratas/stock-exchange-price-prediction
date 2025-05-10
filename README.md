# ASELSAN Stock Price Prediction Project

This data science project focuses on predicting ASELSAN stock prices using various machine learning techniques. The project demonstrates the complete data science pipeline from data collection to model evaluation.

## Project Structure
- `data/`: Contains raw and processed data files
- `notebooks/`: Jupyter notebooks with analysis and visualization
- `src/`: Source code for the project
  - `data_collection.py`: Scripts for collecting stock data
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

## Usage
1. Run data collection:
```bash
python src/data_collection.py
```

2. Execute the complete pipeline:
```bash
python src/main.py
```

## Project Components
1. Data Collection: Using yfinance to fetch ASELSAN stock data
2. Data Preprocessing: Cleaning, feature engineering, and data preparation
3. Exploratory Data Analysis: Understanding patterns and relationships
4. Model Development: Implementation of various ML models
5. Model Evaluation: Performance metrics and visualization
6. Prediction: Making future price predictions 