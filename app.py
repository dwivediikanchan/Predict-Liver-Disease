import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ---------------------- CONFIG ----------------------
DATA_PATH = "project-data.csv"
LOCAL_BG_FILENAME = "liver_bg.jpg"
FALLBACK_BG_URL = "https://images.unsplash.com/photo-1582719478250-0f6a4f6b4c7a?q=80&w=1400&auto=format&fit=crop"


st.set_page_config(page_title="Liver Disease Predictor", page_icon="🧬", layout="wide")


# ---------------------- BACKGROUND IMAGE ----------------------
def apply_background():
    try:
        with open(LOCAL_BG_FILENAME, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            bg_url = f"data:image/jpg;base64,{encoded}"
    except:
        bg_url = FALLBACK_BG_URL

    css = f"""
    <style>
    .stApp {{
        background-image: url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main-card {{
        background: rgba(255,255,255,0.85);
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.10);
    }}

    /* Attractive Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #670000, #300000);
        color: white;
    }}

    [data-testid="stSidebar"] * {{
        color: white !important;
        font-size: 16px;
    }}

    .sidebar-title {{
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding-bottom: 15px;
    }}

    .sidebar-item {{
        font-size: 18px !important;
        padding: 6px 0px;
    }}

    .result-good {{
        background:#e5ffe9;
        padding:14px;
        border-radius:10px;
        color:#064e3b;
        font-weight:bold;
        text-align:center;
        font-size:20px;
    }}

    .result-bad {{
        background:#ffe5e5;
        padding:14px;
        border-radius:10px;
        color:#7f1d1d;
        font-weight:bold;
        text-align:center;
        font-size:20px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


apply_background()


# ---------------------- LOAD DATA & TRAIN MODEL ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def train_model(df):
    df = df.copy()

    target = "category"
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target].astype(str))

    X = df.drop(columns=[target])
    y = df[target]

    # Encode any object columns
    for col in X.select_dtypes(include="object"):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, list(X.columns), le


df = load_data()
model, scaler, features, label_encoder = train_model(df)


# ---------------------- SIDEBAR ----------------------
st.sidebar.markdown("<div class='sidebar-title'>🧬 Liver AI</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Home", "🔍 Predict", "📊 Model Info"],
    label_visibility="collapsed"
)


# ---------------------- HOME PAGE ----------------------
if page == "🏠 Home":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("## 🧬 Liver Disease Prediction System")
    
    st.write("""
    A simple and clean interface to predict liver disease category using AI.
    
    Navigate to **Predict** from the left sidebar to enter patient values.
    """)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- PREDICT PAGE ----------------------
elif page == "🔍 Predict":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("## 🔍 Predict Liver Condition")

    left, right = st.columns([2, 1])

    with left:
        inputs = {}
        for col in features:
            inputs[col] = st.number_input(col.replace("_", " ").title(), value=None, step=0.1)

    predict_btn = st.button("Predict Now", use_container_width=True)

    if predict_btn:
        if None in inputs.values():
            st.error("❗ Please fill **all fields** before predicting.")
        else:
            ip = pd.DataFrame([inputs])

            ip = ip.apply(pd.to_numeric, errors="coerce").fillna(ip.mean())
            scaled = scaler.transform(ip.values)

            pred = model.predict(scaled)[0]
            label = label_encoder.inverse_transform([pred])[0]

            st.markdown("### Result:")
            if str(label).lower() in ["healthy", "normal", "no"]:
                st.markdown(f"<div class='result-good'>🟢 {label}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-bad'>🔴 {label}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- MODEL INFO PAGE ----------------------
elif page == "📊 Model Info":
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("## 📊 Model Information")

    st.write("**Model:** RandomForestClassifier (300 trees, depth 12)")
    st.write(f"**Features:** {len(features)}")
    st.write(", ".join(features))

    st.markdown("</div>", unsafe_allow_html=True)
