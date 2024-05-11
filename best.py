import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from joblib import dump

# Load the data
data = pd.read_csv('final.csv')

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

features = ['Sector', 'Average Turnover', 'Average Profit', 'Average Employees', 'Std Employees', 'Interaction']
X = data[features]
y = pd.qcut(data['Risk Score'], 5, labels=False)  # Encoding as ordinal

numeric_features = ['Average Turnover', 'Average Profit', 'Average Employees', 'Std Employees', 'Interaction']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Sector']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Random Forest pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

# Grid search for hyperparameters
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5)
grid_search.fit(X, y)

# Best model
best_rf = grid_search.best_estimator_

# Save model
dump(best_rf, 'best_random_forest_model.joblib')

# Print best model performance
print("Best Model Parameters:", grid_search.best_params_)
print("Best Model Score:", grid_search.best_score_)
