import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load data from a CSV file"""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None

def basic_analysis(data):
    """Perform basic data analysis"""
    logging.info("Starting basic analysis")
    print("Data Summary:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
    logging.info("Basic analysis completed")

def visualize_data(data):
    """Visualize data using seaborn and matplotlib"""
    logging.info("Starting data visualization")
    sns.pairplot(data)
    plt.show()
    logging.info("Data visualization completed")

def preprocess_data(data, target_column):
    """Preprocess the data: Handle missing values and split into train/test"""
    logging.info("Starting data preprocessing")
    data = data.dropna()
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data preprocessing completed")
    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    """Build and train a simple Linear Regression model"""
    logging.info("Starting model training")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    logging.info("Starting model evaluation")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    logging.info("Model evaluation completed")

def main():
    # Load the dataset
    data = load_data('your_dataset.csv')
    
    if data is not None:
        # Perform basic analysis
        basic_analysis(data)
        
        # Visualize the data
        visualize_data(data)
        
        # Preprocess the data
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column='YourTargetColumn')
        
        # Build and train the model
        model = build_model(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
