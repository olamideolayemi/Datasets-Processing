import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.impute import SimpleImputer

def download_kaggle_dataset(dataset: str, output_dir: str):
    """Downloads a dataset from Kaggle."""
    try:
        # Specify the path to kaggle.json
        os.environ['KAGGLE_CONFIG_DIR'] = "c:/User/IRISH/.kaggle"
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=output_dir, unzip=True)
        print(f"Dataset downloaded successfully to {output_dir}.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

def find_csv_file(directory: str):
    """Finds the first CSV file in a directory."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if csv_files:
        print(f"Found CSV file: {csv_files[0]}")
        return csv_files[0]
    else:
        print("No CSV file found in the directory.")
        return None

def process_dataset(input_file, output_file, row_limit=200):
    """Reads a dataset, limits rows, and saves to another file."""
    try:
        df = pd.read_csv(input_file, on_bad_lines='skip')  # Skip bad rows
        df.head(row_limit).to_csv(output_file, index=False)
        print(f"Processed file saved as {output_file}.")
    except Exception as e:
        print(f"Error processing the dataset: {e}")

def analyze_dataset(file):
    """Prints rows, columns, and unique classes in the last column."""
    try:
        df = pd.read_csv(file)
        rows, cols = df.shape
        unique_classes = df.iloc[:, -1].nunique()
        print(f"Rows: {rows}, Columns: {cols}, Unique Classes in last column: {unique_classes}")
    except Exception as e:
        print(f"Error analyzing the dataset: {e}")

def perform_linear_regression(file):
    """Applies linear regression for predictions."""
    try:
        df = pd.read_csv(file)

        # Separate features (X) and target (y)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Handle non-numeric data in features
        X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding for categorical variables

        # Impute missing values in features
        imputer = SimpleImputer(strategy="mean")  # Replace NaN with the mean
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Ensure the target variable is numeric
        if y.dtype == 'object':
            y = pd.factorize(y)[0]  # Convert target to numeric using factorization

        # Drop rows where target variable (y) is NaN
        if np.isnan(y).any():
            valid_indices = ~np.isnan(y)  # Get indices of non-NaN values
            X = X[valid_indices]
            y = y[valid_indices]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")

        # For classification, calculate accuracy if target is integer-like
        if np.issubdtype(y.dtype, np.integer):
            y_pred_class = [round(pred) for pred in predictions]
            accuracy = accuracy_score(y_test, y_pred_class)
            print(f"Accuracy for classification: {accuracy}")

    except Exception as e:
        print(f"Error performing linear regression: {e}")

# Main Execution
if __name__ == "__main__":
    # This is where you input the Kaggle dataset identifier
    kaggle_dataset = "syedanwarafridi/vehicle-sales-data"
    download_dir = "./"
    prediction_file = os.path.join(download_dir, "For_Prediction.csv")

    # Step 1: Download the dataset
    os.makedirs(download_dir, exist_ok=True)
    download_kaggle_dataset(kaggle_dataset, download_dir)

    # Step 2: Find the CSV file
    raw_file = find_csv_file(download_dir)
    if not raw_file:
        print("Exiting script as no CSV file was found.")
        exit()

    # Step 3: Process the dataset
    process_dataset(raw_file, prediction_file)

    # Step 4: Analyze the dataset
    analyze_dataset(prediction_file)

    # Step 5: Perform Linear Regression
    perform_linear_regression(prediction_file)
