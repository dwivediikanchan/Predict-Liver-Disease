import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🧬",
    layout="wide"
)


# -----------------------------------------------------
# BACKGROUND + SIDEBAR + HEADER STYLING
# -----------------------------------------------------
def set_background(img_path="liver_bg.jpg", fallback="https://images.unsplash.com/photo-1582719478250-0f6a4f6b4c7a?q=80"):

    # Try reading local image
    try:
        with open(img_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            background_url = f"data:image_
