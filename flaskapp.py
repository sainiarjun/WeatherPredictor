from flask import Flask, render_template_string, request
import torch
import pandas as pd
from meteostat import Point, Daily
from sklearn.preprocessing import QuantileTransformer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Initialize the Flask app
app = Flask(__name__)

# Function to fetch weather data and make predictions
def fetch_weather_predictions(date_input):
    try:
        date_object = datetime.strptime(date_input, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}
    
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
    predictions_rescaled = scaler.inverse_transform(predictions_rescaled.reshape(-1, input_data.shape[1]))

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

# Define routes
@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Weather Prediction App</title></head>
        <body>
            <h1>Weather Prediction App</h1>
            <form method="post" action="/predict">
                <label for="date">Enter a Date (YYYY-MM-DD):</label>
                <input type="date" id="date" name="date" required>
                <button type="submit">Get Predictions</button>
            </form>
        </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    date_input = request.form.get('date')
    if not date_input:
        return "Please provide a valid date."
    
    result, legend = fetch_weather_predictions(date_input)
    if "error" in result:
        return result["error"]

    predictions_html = '<h2>Weather Predictions</h2>'
    for key, value in result.items():
        predictions_html += f'<p>{legend[key]}: {value:.2f}</p>'

    return predictions_html

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8501)
