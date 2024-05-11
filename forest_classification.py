import pandas as pd
from scipy.stats import zscore
from joblib import load
import numpy as np
# Load the data

def classify():
    data_path = 'final_dupe.csv' 
    data = pd.read_csv(data_path)

    # Load the models and scaler
    rf_model = load('random_forest_model.joblib')
    scaler = load('scaler.joblib')
    le = load('label_encoder.joblib')

    # Calculating monthly statistics and features
    monthly_columns = [col for col in data.columns if 'Turnover' in col or 'Profit' in col or 'Employee' in col]
    monthly_data = data[monthly_columns]

    turnover_cols = [col for col in monthly_columns if 'Turnover' in col]
    profit_cols = [col for col in monthly_columns if 'Profit' in col]
    employee_cols = [col for col in monthly_columns if 'Employee' in col]

    data['Average Turnover'] = monthly_data[turnover_cols].mean(axis=1)
    data['Average Profit'] = monthly_data[profit_cols].mean(axis=1)
    data['Average Employees'] = monthly_data[employee_cols].mean(axis=1)
    data['Std Employees'] = monthly_data[employee_cols].std(axis=1)

    data['Risk Score'] = (1 / (data['Average Turnover'] + data['Average Profit'])) + data['Std Employees']


    # Transform the 'Sector' column (assuming le has been fitted previously)
    data['Sector'] = le.transform(data['Sector'])

    # Prepare features
    X = data[['Sector'] + monthly_columns + ['Average Turnover', 'Average Profit', 'Average Employees', 'Std Employees', 'Risk Score']]
    data.fillna({'Risk Score': data['Risk Score'].median()}, inplace=True)

    if data['Risk Score'].nunique() > 1:
        y = pd.qcut(data['Risk Score'], 5, labels=['Most Secure', 'Secure', 'Moderate Risky', 'Less Risky', 'Most Risky'])

    # Scaling the features
    X_scaled = scaler.fit_transform(X)  # No need to split if predicting on all data

    # Predictions
    rf_predictions = rf_model.predict(X_scaled)
    return (rf_predictions)

print(classify())
    
# data['Predicted Risk Category'] = rf_predictions

# # Save the data with predictions
# output_path = 'new_test.csv'
# data.to_csv(output_path, index=False)

# print(f"Predictions saved to {output_path}")
