import streamlit as st
import os
import sys
import traceback

st.set_page_config(page_title="Debug App", layout="wide")
st.title("Debugging Information")

# System Information
st.write("## System Information")
st.write(f"Python Version: {sys.version}")
st.write(f"Current Working Directory: {os.getcwd()}")
st.write(f"Directory Contents: {os.listdir('.')}")

# Environment Variables
st.write("## Environment Variables")
for key, value in os.environ.items():
    st.write(f"{key}: {value}")

# Dependency Check
st.write("## Dependency Check")
try:
    import torch
    st.write(f"Torch Version: {torch.__version__}")
except Exception as e:
    st.error(f"Torch Import Error: {str(e)}")
    st.error(traceback.format_exc())

try:
    import pandas
    st.write(f"Pandas Version: {pandas.__version__}")
except Exception as e:
    st.error(f"Pandas Import Error: {str(e)}")
    st.error(traceback.format_exc())

# Model Loading Attempt
st.write("## Model Loading")
try:
    import torch
    model_path = 'model_scripted.pt'
    st.write(f"Model Path: {os.path.abspath(model_path)}")
    st.write(f"Model Exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        model = torch.jit.load(model_path)
        st.success("Model loaded successfully")
    else:
        st.error("Model file not found")
except Exception as e:
    st.error(f"Model Loading Error: {str(e)}")
    st.error(traceback.format_exc())