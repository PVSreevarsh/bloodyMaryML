import pandas as pd
from joblib import load

model = load('best_random_forest_model.joblib')


data = pd.read_csv('final_dupe.csv')

monthly_columns = [col for col in data.columns if 'Turnover' in col or 'Profit' in col or 'Employee' in col]
monthly_data = data.loc[:, monthly_columns]

turnover_cols = [col for col in monthly_columns if 'Turnover' in col]
profit_cols = [col for col in monthly_columns if 'Profit' in col]
employee_cols = [col for col in monthly_columns if 'Employee' in col]

data['Average Turnover'] = monthly_data[turnover_cols].mean(axis=1)
data['Average Profit'] = monthly_data[profit_cols].mean(axis=1)
data['Average Employees'] = monthly_data[employee_cols].mean(axis=1)
data['Std Employees'] = monthly_data[employee_cols].std(axis=1)
data['Risk Score'] = (1 / (data['Average Turnover'] + data['Average Profit'])) + data['Std Employees']
data['Interaction'] = data['Average Turnover'] * data['Average Profit']

# Prediction
predictions = model.predict(data)

labels = ['Most Secure', 'Secure', 'Moderate Risky', 'Less Risky', 'Most Risky']
categorical_predictions = [labels[pred] for pred in predictions]

print(categorical_predictions)
