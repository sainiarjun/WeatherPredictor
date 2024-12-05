# Use an official Python image
FROM public.ecr.aws/docker/library/python:slim-bullseye 

# Set working directory
WORKDIR /app

# Copy app and model
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address", "0.0.0.0", "--logger.level=debug"]
