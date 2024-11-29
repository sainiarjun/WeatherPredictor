# Use an official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app and model
COPY app.py /app/
COPY requirements.txt /app/
COPY model_scripted.pt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
