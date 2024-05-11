import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


# Load the data
data_path = 'final.csv'  # Update this path
data = pd.read_csv(data_path)

# Calculating monthly statistics and features
monthly_columns = [col for col in data.columns if 'Turnover' in col or 'Profit' in col or 'Employee' in col]
monthly_data = data.loc[:, monthly_columns]

turnover_cols = [col for col in monthly_columns if 'Turnover' in col]
profit_cols = [col for col in monthly_columns if 'Profit' in col]
employee_cols = [col for col in monthly_columns if 'Employee' in col]

data['Average Turnover'] = monthly_data[turnover_cols].mean(axis=1)
data['Average Profit'] = monthly_data[profit_cols].mean(axis=1)
data['Average Employees'] = monthly_data[employee_cols].mean(axis=1)
data['Std Employees'] = monthly_data[employee_cols].std(axis=1)
data['Norm Turnover'] = zscore(data['Average Turnover'])
data['Norm Profit'] = zscore(data['Average Profit'])
data['Norm Std Employees'] = zscore(data['Std Employees'])
data['Risk Score'] = (1 / (data['Norm Turnover'] + data['Norm Profit'])) + data['Norm Std Employees']

# Encoding the Sector column which is categorical
le = LabelEncoder()
data['Sector'] = le.fit_transform(data['Sector'])

# Prepare features and labels
X = data[['Sector'] + list(monthly_columns) + ['Average Turnover', 'Average Profit', 'Average Employees', 'Std Employees', 'Norm Turnover', 'Norm Profit', 'Norm Std Employees', 'Risk Score']]
y = pd.qcut(data['Risk Score'], 5, labels=['Most Secure', 'Secure', 'Moderate Risky', 'Less Risky', 'Most Risky'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test_scaled)

# Saving the Random Forest model to a joblib file
model_filename = 'random_forest_model.joblib'
dump(rf_model, model_filename)
scaler_filename = 'scaler.joblib'
dump(scaler, scaler_filename)
labelencoder_filename = 'label_encoder.joblib'
dump(le, labelencoder_filename)

# Evaluations
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_report = classification_report(y_test, rf_predictions)


# Print RF results
print(f"Random Forest Test Accuracy: {rf_accuracy}")
print(f"RF Classification Report:\n{rf_report}")


rf_conf_matrix = confusion_matrix(y_test, rf_predictions, labels=['Less Risky', 'Moderate Risky', 'Most Secure', 'Secure', 'Most Risky'])
rf_conf_matrix_df = pd.DataFrame(rf_conf_matrix, index=['Actual LR', 'Actual MR', 'Actual MS', 'Actual S', 'Actual VR'], columns=['Predicted LR', 'Predicted MR', 'Predicted MS', 'Predicted S', 'Predicted VR'])
print("RF Confusion Matrix on Testing dataset:")
print(rf_conf_matrix_df)
