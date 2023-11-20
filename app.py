
import torch
import json
import joblib
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from model import LinearRegressionModel

app = Flask(__name__)




def feature_engineering(data):
    data['Date'] = pd.to_datetime(data['Date'])
    # Basic time features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    # data['DayOfWeek'] = data['Date'].dt.dayofweek

    data['Lag_1'] = data['Receipt_Count'].shift(1)

    # Rolling window features (e.g., rolling average of the past 7 days)
    data['Rolling_Mean_7'] = data['Receipt_Count'].rolling(window=7).mean()

    # Drop rows with NaN values which are a result of lagged features
    data = data.dropna()

    return data




def feature_engineering_for_prediction(data, prediction_date):
    # Basic time features
    prediction_date = pd.to_datetime(prediction_date)

    prediction_data = pd.DataFrame({
        'Month': [prediction_date.month],
        'Day': [prediction_date.day],
        'Year': [prediction_date.year]
    })

    # Calculating lagged feature for prediction date
    last_row = data.iloc[-1]  # Considering the last available data point

    prediction_data['Lag_1'] = last_row['Receipt_Count']

    # Calculate rolling mean for the prediction date
    last_7_days = data.tail(7)['Receipt_Count']  # Last 7 days before prediction

    prediction_data['Rolling_Mean_7'] = last_7_days.mean()

    return data, prediction_data



# Load your model the input size changes according to the features considered during training
model = LinearRegressionModel(4)
model.load_state_dict(torch.load('model_weights/linear_model.pt'))
model.eval()

x_scaler = joblib.load('assets/x_scaler.pkl')
y_scaler = joblib.load('assets/y_scaler.pkl')

# Load your 2021 data for visualization
data_2021 = pd.read_csv('data_daily.csv')

@app.route('/',methods=['GET','POST'])
def index():
    # try:
        if request.method == 'POST':
            date_input = request.form['date']
   
            # Convert date to a format your model can use
            date = pd.to_datetime(date_input)
       
            data_2021_features = feature_engineering(data_2021)
     
            feature_dates = pd.date_range(start='2022-01-01', end=date)

            # Initialize 'your_data' with the initial features
            your_data, prediction_features = feature_engineering_for_prediction(data_2021_features, '2022-01-01')
            predicted_receipts = None
             
         
            
            for start_date in feature_dates:
                # Scale the features
                scaled_features = x_scaler.transform(prediction_features.loc[:, prediction_features.columns != 'Year'].values)
                
                # Perform prediction
                with torch.no_grad():
                    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
                    prediction = model(scaled_features_tensor).item()
                    predicted_receipts = y_scaler.inverse_transform(np.array([prediction]).reshape(-1, 1)).flatten()
                
                # Update prediction_features with the predicted Receipt_Count
            
                prediction_features["Receipt_Count"] = predicted_receipts
                
               
                # Append the predicted features to your_data
                your_data = pd.concat([your_data, prediction_features], ignore_index=True)
                
                
                # Update prediction_features for the next iteration
                your_data, prediction_features = feature_engineering_for_prediction(your_data, start_date)
            
            # get the monthly aggregated data till that day
            predicted_receipts_for_month = your_data[(your_data['Month'] == date.month) & (your_data['Year'] == date.year)].Receipt_Count.sum()
            # Prepare data for visualization

            predicted_data= str(int(predicted_receipts_for_month))
            time_series_return = your_data['Year'].astype(str) + "-" + your_data['Month'].astype(str)+"-" +your_data['Day'].astype(str)
            all_data = your_data['Receipt_Count']
            print(time_series_return)
            # data = {
            #     'prediction': str(int(predicted_receipts_for_month)),
            #     'historical_data': your_data.to_dict(orient='records')
            # }
            print(predicted_receipts_for_month)
            return render_template('index.html', result=predicted_data,
                                    time_series_return =time_series_return.to_list(),
                                    all_data= all_data.to_list(),
                                    date_input = date_input)
        else:
            return render_template('index.html')

        #return jsonify(data)
    # except Exception as e:

    #     return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
