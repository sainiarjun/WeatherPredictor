import streamlit as st
from meteostat import Point, Daily
import torch
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta



# Set page title and layout
st.set_page_config(page_title="Weather Prediction App", layout="wide")

# Title of the app
st.title("Weather Prediction App")

# Input date (Allow user input)
date_input = st.date_input("Please enter a date", datetime.today())

# Function to fetch weather data and make predictions
def fetch_weather_predictions(date_input):
    try:
        date_object = datetime.strptime(str(date_input), "%Y-%m-%d")
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
        return
    
    previous_day = date_object - timedelta(days=1)
    date_one_month_ago = date_object - relativedelta(months=1)
    end = datetime(previous_day.year, previous_day.month, previous_day.day)
    start = datetime(date_one_month_ago.year, date_one_month_ago.month, date_one_month_ago.day)

    # Location (New York City)
    location = Point(40.7128, -74.0060)
    
    # Fetch data from Meteostat API
    input_data = Daily(location, start, end)
    input_data = input_data.fetch()
    input_data = pd.DataFrame(input_data)
    
    # Preprocessing
    input_data.drop(['wpgt', 'tsun', 'wdir'], axis=1, inplace=True)
    input_data.fillna(0, inplace=True)
    input_data.replace([float('inf'), -float('inf')], 0, inplace=True)

    # Load the model
    model = torch.jit.load('model_scripted.pt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Scaling the input data
    scaler = QuantileTransformer(output_distribution='uniform')
    input_data_scaled = scaler.fit_transform(input_data)
    input_data_scaled = input_data_scaled[-30:]
    input_data_scaled = torch.tensor(input_data_scaled).view(1, 30, 7).float()
    input_data_scaled = input_data_scaled.to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(input_data_scaled)
    predictions_rescaled = predictions.cpu().numpy()
    predictions_rescaled = scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, input_data.shape[1]))

    result = dict(zip(input_data.columns, predictions_rescaled[0]))
    legend = {
        'tavg': "Avg Temp in °C",
        'tmin': 'Min Temp in °C',
        'tmax': 'Max Temp in °C',
        'prcp': 'Precipitation in mm',
        'snow': 'Snow in mm',
        'wspd': 'Windspeed in km/h',
        'pres': 'Atmospheric Pressure in hPa'
    }

    return result, legend

# Fetch and display results when the user selects a date
if st.button("Get Predictions"):
    result, legend = fetch_weather_predictions(date_input)
    
    # Display the results
    st.subheader(f"Weather Predictions for {date_input}")
    
    for key, value in result.items():
        st.write(f"{legend[key]}: {value:.2f}")
