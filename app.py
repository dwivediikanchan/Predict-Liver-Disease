import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("project-data.csv", sep=";")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -----------------------------------------
# TRAIN MODEL
# -----------------------------------------
@st.cache_resource
def train_model(df):

    target_col = "category"

    # Encode target column
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))

    # Split X and y
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode any categorical features (none in your dataset except target)
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Convert all features to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill missing
    X = X.fillna(X.mean())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns), le


model, scaler, feature_names, label_encoder = train_model(df)

# -----------------------------------------
# STREAMLIT UI DESIGN
# -----------------------------------------

st.set_page_config(
    page_title="Liver Disease Prediction",
    layout="wide",
    page_icon="🧬"
)

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🩺 Prediction", "📊 About Model"])

# Add styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        padding: 10px;
        color: #3b82f6;
    }
    .sub-title {
        font-size: 22px;
        font-weight:600;
        margin-top: 15px;
    }
    .result-success {
        background-color: #d1fae5;
        padding: 15px;
        border-radius: 10px;
        color: #065f46;
        font-size: 20px;
    }
    .result-danger {
        background-color: #fee2e2;
        padding: 15px;
        border-radius: 10px;
        color: #991b1b;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HOME PAGE
# -----------------------------------------
if page == "🏠 Home":
    st.markdown("<div class='main-title'>🧬 Liver Disease Prediction System</div>", unsafe_allow_html=True)

    st.write("""
    This interactive app predicts liver disease categories using machine learning.

    ### Features:
    - 📌 User-friendly UI  
    - 📊 Automatic feature scaling  
    - 🤖 Random Forest model  
    - 🎯 Instant prediction  
    """)

# -----------------------------------------
# PREDICTION PAGE
# -----------------------------------------
elif page == "🩺 Prediction":
    st.markdown("<div class='main-title'>Patient Data Input</div>", unsafe_allow_html=True)

    st.markdown("<div class='sub-title'>Enter patient medical values</div>", unsafe_allow_html=True)

    # Two-column layout
    col1, col2 = st.columns(2)
    user_data = {}

    for i, col in enumerate(feature_names):
        if i % 2 == 0:
            user_data[col] = col1.number_input(
                f"{col.replace('_', ' ').title()}",
                min_value=0.0,
                step=0.1
            )
        else:
            user_data[col] = col2.number_input(
                f"{col.replace('_', ' ').title()}",
                min_value=0.0,
                step=0.1
            )

    st.markdown("---")

    if st.button("🔍 Predict Result", use_container_width=True):

        input_df = pd.DataFrame([user_data])

        # Encode & scale input
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(input_df.mean())
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]

        # Display result
        if pred_label.lower() == "healthy":
            st.markdown(f"<div class='result-success'>🟢 Prediction: {pred_label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-danger'>🔴 Prediction: {pred_label}</div>", unsafe_allow_html=True)

        st.subheader("📋 Patient Summary")
        st.json(user_data)

# -----------------------------------------
# ABOUT MODEL
# -----------------------------------------
elif page == "📊 About Model":
    st.markdown("<div class='main-title'>📊 Model Information</div>", unsafe_allow_html=True
