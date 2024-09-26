from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

app = Flask(__name__)

# Load the training data
train_data = pd.read_csv("Data_Train.csv")

# Convert 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' to datetime objects
train_data['Date_of_Journey'] = pd.to_datetime(train_data['Date_of_Journey'], format='%d/%m/%Y', errors='coerce')

# Extract day and month from 'Date_of_Journey'
train_data['Journey_Day'] = train_data['Date_of_Journey'].dt.day
train_data['Journey_Month'] = train_data['Date_of_Journey'].dt.month

# Extract hour and minute from 'Dep_Time' and 'Arrival_Time'
train_data['Dep_Hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
train_data['Dep_Minute'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
train_data['Arrival_Hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data['Arrival_Minute'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute

# Drop the original date and time columns
train_data.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

# Convert 'Duration' to total minutes
def convert_duration(duration):
    total_minutes = 0
    for part in duration.split():
        if 'h' in part:
            total_minutes += int(part.replace('h', '')) * 60
        elif 'm' in part:
            total_minutes += int(part.replace('m', ''))
    return total_minutes

train_data['Duration'] = train_data['Duration'].apply(convert_duration)

# Encode categorical features
label_encoder_airline = LabelEncoder()
label_encoder_source = LabelEncoder()
label_encoder_destination = LabelEncoder()

train_data['Airline'] = label_encoder_airline.fit_transform(train_data['Airline'])
train_data['Source'] = label_encoder_source.fit_transform(train_data['Source'])
train_data['Destination'] = label_encoder_destination.fit_transform(train_data['Destination'])

# Define the input features and target variable
input_features = ['Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute', 'Arrival_Hour', 'Arrival_Minute', 'Airline', 'Source', 'Destination']
X = train_data[input_features]
y = train_data['Price']

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Define the route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    journeyDate = request.form['journeyDate']
    depTime = request.form['depTime']
    arrivalTime = request.form['arrivalTime']
    airline = request.form['airline']
    source = request.form['source']
    destination = request.form['destination']

    # Process form data
    journey_day = pd.to_datetime(journeyDate, format='%d/%m/%Y').day
    journey_month = pd.to_datetime(journeyDate, format='%d/%m/%Y').month
    dep_hour = pd.to_datetime(depTime, format='%H:%M').hour
    dep_minute = pd.to_datetime(depTime, format='%H:%M').minute
    arrival_hour = pd.to_datetime(arrivalTime, format='%H:%M').hour
    arrival_minute = pd.to_datetime(arrivalTime, format='%H:%M').minute
    airline_encoded = label_encoder_airline.transform([airline])[0]
    source_encoded = label_encoder_source.transform([source])[0]
    destination_encoded = label_encoder_destination.transform([destination])[0]

    # Create a DataFrame for user input
    user_input = pd.DataFrame([[journey_day, journey_month, dep_hour, dep_minute, arrival_hour, arrival_minute, airline_encoded, source_encoded, destination_encoded]], columns=input_features)

    # Initialize an empty list to store predictions
    predictions = []

    # Predict for the provided date and the following seven days
    for day in range(0, 7):
        prediction_date = pd.to_datetime(journeyDate, format='%d/%m/%Y') + timedelta(days=day)

        # Update the user input DataFrame with the current prediction date
        user_input['Journey_Day'] = prediction_date.day
        user_input['Journey_Month'] = prediction_date.month

        # Make prediction using the trained model
        predicted_price = rf_model.predict(user_input)

        # Append prediction to list
        predictions.append({'Date': prediction_date.strftime('%Y-%m-%d'), 'Predicted_Price': predicted_price[0]})

    # Pass predictions to result.html for rendering
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
