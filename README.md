Problem Statement
Develop a Python program that performs the following tasks using the Sales Transactions Dataset:
Downloads a dataset from the web.
Reads up to 200 rows and saves it into another file.
Analyzes the data to report the number of rows, columns, and unique classes in the last column.
Uses linear regression to predict dependent variables.

Problem Specification
Requirements:
Download a CSV dataset from Kaggle.
Handle all exceptions during downloading and processing.
Save only the first 200 rows into a new file.
Analyze the dataset for rows, columns, and unique classes in the last column.
Implement linear regression for predictions.
Datasets:
Sales Transactions Dataset


Analysis
The solution involves:

Downloading the dataset: Using requests library.
Processing the dataset: Using pandas to handle CSV files, including cleansing bad rows and limiting row count.
Analyzing data: Reporting basic statistics (rows, columns, classes).
Building a model: Using scikit-learn for linear regression.


Results
When executed:
- The dataset is downloaded and saved to a file named 'car_prices.csv (named after the kaggle dataset)'.
- The first 200 rows are saved to a new file named 'For_Prediction.csv'.
- The dataset is analyzed, showing 200 rows, 16 columns, and 39 unique classes
- The linear regression model is trained and tested, with a mean squared error (MSE) of
84.50261645189454 and a classification accuracy of 0.075 (since this is a regression problem, the accuracy is not applicable and is reported as 0.0).

Output:
File download status.
Processed dataset saved message.
Dataset analysis results (rows, columns, unique classes).
Linear regression performance (MSE and classification accuracy).
Generated Files:
car_prices.csv: Original dataset.
For_Prediction.csv: Processed dataset with 200 rows.