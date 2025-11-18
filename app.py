import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load & Train Model
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("project-data.csv", sep=';')
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def train_model(df):
    # Encoding Target
    le = LabelEncoder()
    df["Dataset"] = le.fit_transform(df["Dataset"])

    # Splitting X & Y
    X = df.drop('category', axis=1)
    y = df['category']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns)

# Load Data & Train Model
df = load_data()
model, scaler, feature_names = train_model(df)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Liver Disease Prediction App", layout="centered")

st.title("🧬 Liver Disease Prediction")
st.markdown("Enter patient details below to check whether they have liver disease.")

# Create Input Form
user_inputs = {}

for col in feature_names:
    user_inputs[col] = st.number_input(
        f"Enter {col}",
        min_value=0.0,
        step=0.1
    )

if st.button("Predict"):
    user_data = np.array(list(user_inputs.values())).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]

    if prediction == 1:
        st.error("⚠️ **The patient is likely to have Liver Disease**")
    else:
        st.success("✅ **The patient is NOT likely to have Liver Disease**")
