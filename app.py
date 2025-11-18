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

    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found in dataset")
        return None, None, None

    # Encode target column (category)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))

    # Split X and y
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Convert all object columns in X to numeric if they exist
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Ensure all features numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill missing values
    X = X.fillna(X.mean())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
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

    return model, scaler, list(X.columns)

model, scaler, feature_names = train_model(df)

# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------
st.set_page_config(page_title="Liver Disease Prediction", layout="centered")

st.title("🧬 Liver Disease Prediction App")
st.write("Enter patient details below to predict liver disease category.")

user_data = {}

st.subheader("Patient Input Features")

# Create number inputs for all feature columns
for col in feature_names:
    user_data[col] = st.number_input(
        f"{col.replace('_', ' ').title()}",
        min_value=0.0,
        step=0.1
    )

# Prediction button
if st.button("Predict"):

    # Convert dictionary to DataFrame row
    input_df = pd.DataFrame([user_data])

    # Encode any categorical columns
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = LabelEncoder().fit_transform(input_df[col].astype(str))

    # Fill missing values
    input_df = input_df.fillna(input_df.mean())

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    pred = model.predict(input_scaled)[0]

    # Map back to original label
    categories = df["category"].unique().tolist()
    decoded_label = categories[pred]

    st.success(f"### 🩺 Prediction: **{decoded_label}**")
