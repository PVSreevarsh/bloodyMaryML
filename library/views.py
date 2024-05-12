from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
from joblib import load


class SMELabel(APIView):
    permission_classes = []
    authentication_classes = []
    
    def post(self, request):
        data = request.data
        data=pd.DataFrame(data)
        model = load('./LMS/best_random_forest_model.joblib')  
        try:
            monthly_columns = [col for col in data.columns if 'Turnover' in col or 'Profit' in col or 'Employee' in col]
            monthly_data = data.loc[:, monthly_columns]

            # Calculate statistics
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

            # Convert predictions to labels
            labels = ['Most Secure', 'Secure', 'Moderate Risky', 'Less Risky', 'Most Risky']
            categorical_predictions = [labels[pred] for pred in predictions]

            # Output predictions
            print(categorical_predictions)
            return Response({'prediction': categorical_predictions}, status=200)
        
        except Exception as e:
            return Response({'error': str(e)}, status=500)

                
                
