import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# Config
# ----------------------
DATA_PATH = "project-data.csv"
LOCAL_BG_FILENAME = "liver_bg.jpg"  # optional: place your chosen image here
FALLBACK_BG_URL = "https://images.unsplash.com/photo-1582719478250-0f6a4f6b4c7a?q=80&w=1400&auto=format&fit=crop&ixlib=rb-4.0.3&s=1b6f9b0b6f9c4e8b1f1f0f1e0f1e1f1f"  # tasteful medical-style image (Unsplash)

st.set_page_config(page_title="Liver Disease Predictor", page_icon="🧬", layout="wide")

# ----------------------
# Helpers
# ----------------------
def get_background_css(image_url=None, local_image_path=None, blur_px=0):
    """
    Returns CSS to place an image as the app background.
    If local_image_path exists it will be encoded and used; otherwise image_url is used.
    """
    img_data = None
    if local_image_path:
        try:
            with open(local_image_path, "rb") as f:
                img_data = f.read()
        except Exception:
            img_data = None

    if img_data:
        b64 = base64.b64encode(img_data).decode()
        url = f"data:image/jpg;base64,{b64}"
    else:
        url = image_url

    css = f"""
    <style>
    .stApp {{
        background-image: url("{url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        backdrop-filter: blur({blur_px}px);
    }}
    /* Add a translucent overlay so text is readable */
    .app-overlay {{
        background: rgba(255,255,255,0.82);
        padding: 18px;
        border-radius: 12px;
    }}
    .card {{
        background: rgba(255,255,255,0.92);
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }}
    .result-good {{
        background-color: #e6ffed;
        color: #064e3b;
        padding: 14px;
        border-radius: 8px;
        font-weight: 700;
    }}
    .result-bad {{
        background-color: #fff1f2;
        color: #7f1d1d;
        padding: 14px;
        border-radius: 8px;
        font-weight: 700;
    }}
    </style>
    """
    return css

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def train_model(df):
    """
    Cleans dataframe, encodes target, trains RandomForest and returns:
    - model, scaler, feature_names, label_encoder, class_names, metrics_dict
    """
    df = df.copy()
    target_col = "category"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {df.columns.tolist()}")

    # Encode target labels (store mapping)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    class_names = list(le.classes_)

    # Features
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If any object columns in features -> label encode them
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    # Force numeric, coerce errors -> NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill NaNs with column mean
    X = X.fillna(X.mean())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model - tuned reasonably for small/medium dataset
    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    # Optional quick metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    metrics = {"train_score": float(train_score), "test_score": float(test_score)}

    return model, scaler, list(X.columns), le, class_names, metrics

# ----------------------
# Load & Train (with graceful error handling)
# ----------------------
try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data from `{DATA_PATH}`. Error: {e}")
    st.stop()

try:
    model, scaler, feature_names, label_encoder, class_names, metrics = train_model(df)
except Exception as e:
    st.error(f"Error during model training: {e}")
    st.stop()

# ----------------------
# Apply background (local fallback to remote)
# ----------------------
st.markdown(get_background_css(image_url=FALLBACK_BG_URL, local_image_path=LOCAL_BG_FILENAME, blur_px=2), unsafe_allow_html=True)

# ----------------------
# Layout & Navigation
# ----------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose page", ["Home", "Predict", "Model Info", "Download"])

# Top bar info
st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h1 style="margin:0;padding:0;">🧬 Liver Disease Predictor</h1>
      <div style="font-size:14px;color:#444">Interactive demo • Random Forest • Clean UI</div>
    </div>
""", unsafe_allow_html=True)

# ----------------------
# HOME PAGE
# ----------------------
if page == "Home":
    st.markdown("<div class='app-overlay card'>", unsafe_allow_html=True)
    st.markdown("### Welcome 👋")
    st.write("""
    This app predicts liver disease **category** using a Random Forest model trained on your dataset.
    - Use the **Predict** page to enter patient values and get an instant prediction.
    - The app will use a local image `liver_bg.jpg` as the background if present; otherwise a high-quality fallback image is shown.
    """)
    st.markdown("---")

    # Quick dataset preview & stats
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(8), use_container_width=True)

    st.markdown("#### Feature summary")
    st.write(df.describe().T)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# PREDICTION PAGE
# ----------------------
elif page == "Predict":
    st.markdown("<div class='app-overlay card'>", unsafe_allow_html=True)
    st.markdown("### 🩺 Prediction")

    st.write("Enter patient values below (numbers only). You can use the up/down arrows to set values quickly.")

    # split into left (inputs) and right (live summary + result)
    left, right = st.columns([2, 1])

    # Build inputs with sensible default values from dataset mean
    defaults = df.drop(columns=["category"]).apply(pd.to_numeric, errors="coerce").fillna(0).mean().to_dict()

    with left:
        st.markdown("#### Patient Inputs")
        user_inputs = {}
        for col in feature_names:
            # sensible min, max based on data (if possible)
            col_series = pd.to_numeric(df[col], errors="coerce")
            cmin = float(col_series.min()) if pd.notna(col_series.min()) else 0.0
            cmax = float(col_series.max()) if pd.notna(col_series.max()) else cmin + 100.0
            cmean = float(defaults.get(col, 0.0))

            # number input
            user_inputs[col] = st.number_input(
                label=col.replace("_", " ").title(),
                value=round(cmean, 2),
                min_value=0.0,
                step=0.1,
                format="%.3f"
            )

        st.markdown("---")
        submitted = st.button("🔍 Predict")

    with right:
        st.markdown("#### Patient Summary")
        st.json(user_inputs)

        st.markdown("#### Prediction Result")
        # placeholder for result
        result_box = st.empty()
        prob_box = st.empty()

    # After submit -> predict
    if submitted:
        try:
            input_df = pd.DataFrame([user_inputs], columns=feature_names)
            input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(input_df.mean())

            X_scaled = scaler.transform(input_df.values)
            pred_proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
            pred = model.predict(X_scaled)[0]
            label = label_encoder.inverse_transform([pred])[0]

            # display
            if pred_proba is not None:
                confidence = float(np.max(pred_proba))
                prob_box.info(f"Confidence: {confidence:.2%}")
            else:
                prob_box.info("Confidence: (model does not support predict_proba)")

            if isinstance(label, str) and label.lower() in ["healthy", "normal", "no"]:
                result_box.markdown(f"<div class='result-good'>🟢 Prediction: **{label}**</div>", unsafe_allow_html=True)
            else:
                result_box.markdown(f"<div class='result-bad'>🔴 Prediction: **{label}**</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# MODEL INFO PAGE
# ----------------------
elif page == "Model Info":
    st.markdown("<div class='app-overlay card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Model & Dataset Info")

    st.write("**Trained model:** RandomForestClassifier")
    st.write(f"**Features used ({len(feature_names)}):** {', '.join(feature_names)}")
    st.write(f"**Target classes ({len(class_names)}):** {', '.join(class_names)}")
    st.write("**Quick metrics:**")
    st.metric("Train accuracy", f"{metrics['train_score']:.3f}")
    st.metric("Test accuracy", f"{metrics['test_score']:.3f}")

    st.markdown("#### Raw dataset head")
    st.dataframe(df.head(12), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# DOWNLOAD PAGE (Export)
# ----------------------
elif page == "Download":
    st.markdown("<div class='app-overlay card'>", unsafe_allow_html=True)
    st.markdown("### 📥 Export & Download")

    st.write("You can download a small sample of the dataset or the trained model (pickles).")

    if st.button("Download sample CSV (first 100 rows)"):
        csv_bytes = df.head(100).to_csv(index=False).encode("utf-8")
        st.download_button("Click to download CSV", data=csv_bytes, file_name="project_sample.csv", mime="text/csv")

    # Model export (pickl

