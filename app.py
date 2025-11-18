# app.py — Ultra Premium Liver Disease App (Version C)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import time
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Liver Disease Prediction", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# STYLES & ANIMATIONS (ULTRA PREMIUM)
# -------------------------
# All CSS/animations live here. No external images required.
ultra_css = """
<style>

/* Moving gradient background */
:root{
  --g1: #071a52;
  --g2: #0f172a;
  --accent: #00f0ff;
}
[data-testid="stAppViewContainer"]{
  background: linear-gradient(120deg, rgba(7,26,82,0.95) 0%, rgba(15,23,42,0.95) 50%, rgba(3,7,18,0.95) 100%);
  background-size: 400% 400%;
  animation: gradientShift 18s ease infinite;
  min-height: 100vh;
}

/* Gradient animation */
@keyframes gradientShift {
  0%{background-position:0% 50%}
  50%{background-position:100% 50%}
  100%{background-position:0% 50%}
}

/* Top navbar */
.topbar {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  padding:10px 20px;
  border-radius:10px;
  margin-bottom:18px;
}
.brand {
  display:flex; align-items:center; gap:12px;
}
.brand .logo {
  background: linear-gradient(135deg,#00e0ff,#8b5cf6);
  width:48px; height:48px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  box-shadow: 0 8px 30px rgba(11,12,40,0.6);
  font-weight:900; color:black;
}
.brand .title {
  font-size:20px; font-weight:800; color:white;
  letter-spacing:0.4px;
}

/* typing effect for subtitle */
@keyframes typing { from { width: 0 } to { width: 100% } }
@keyframes blink { 50% { border-color: transparent } }

.typing {
  color: #cfeffd;
  font-weight:600;
  white-space:nowrap;
  overflow:hidden;
  border-right: .12em solid #cfeffd;
  width:20ch;
  animation: typing 2.4s steps(20, end), blink .8s step-end infinite;
}

/* Sidebar styling (Streamlit modern DOM) */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(7,26,82,0.98), rgba(10,10,20,0.98));
  color: #cfeffd;
  padding: 18px;
  border-right: 1px solid rgba(255,255,255,0.03);
}
section[data-testid="stSidebar"] .css-1d391kg { color: #cfeffd; } /* label fix */

/* Sidebar header */
.sidebar-title {
  font-size:22px; font-weight:800; color:#00f0ff; margin-bottom:8px;
}

/* floating card (glass) */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border-radius:14px;
  padding:18px;
  box-shadow: 0 12px 40px rgba(2,6,23,0.6);
  backdrop-filter: blur(8px) saturate(1.2);
  border: 1px solid rgba(255,255,255,0.05);
}

/* input fields styling */
input, textarea, .stNumberInput input {
  background: rgba(255,255,255,0.02) !important;
  color: #e6f9ff !important;
  border: 1px solid rgba(0,240,255,0.12) !important;
  border-radius:8px !important;
}

/* fancy button */
.stButton>button {
  background: linear-gradient(90deg,#00e0ff,#7c3aed) !important;
  color: #021024 !important;
  font-weight:800;
  padding:10px 22px;
  border-radius:12px;
  box-shadow: 0 8px 20px rgba(125,58,237,0.24);
}

/* result card (3D glowing) */
.result-3d {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
  border-radius:14px;
  padding:22px;
  text-align:center;
  color:#e6fbff;
  box-shadow:
    0 6px 18px rgba(1,5,20,0.6),
    inset 0 1px 0 rgba(255,255,255,0.02);
  transform-style: preserve-3d;
  perspective: 1000px;
  border: 1px solid rgba(0,240,255,0.08);
}

/* glowing badge */
.glow-badge {
  display:inline-block;
  padding:10px 20px;
  border-radius:999px;
  font-weight:900;
  color:#021024;
  background: linear-gradient(90deg,#00f0ff,#a855f7);
  box-shadow: 0 10px 30px rgba(167,90,255,0.12), 0 4px 20px rgba(0,240,255,0.06);
  transform: translateZ(60px) rotateX(6deg);
}

/* confidence bar */
.conf-bar {
  width:100%;
  height:14px;
  background: rgba(255,255,255,0.06);
  border-radius:10px;
  overflow:hidden;
  margin-top:14px;
}
.conf-fill {
  height:100%;
  background: linear-gradient(90deg,#00e0ff,#7c3aed);
  width:0%;
  transition: width 1.3s ease;
}

/* small helpers */
.kv {font-weight:700; color:#bfefff}
.sep { height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent); margin:12px 0; border-radius:6px; }

</style>
"""

st.markdown(ultra_css, unsafe_allow_html=True)

# -------------------------
# Helper: load or train model
# -------------------------
@st.cache_data
def load_dataset(path="project-data.csv"):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def train_or_load_model(df, model_path="liver_model.joblib"):
    """
    Try to load a joblib model (model, scaler, label_encoder).
    If not found, train a RandomForest on df and return model, scaler, le, class_names, metrics.
    """
    try:
        loaded = joblib.load(model_path)
        # Expecting tuple (model, scaler, label_encoder)
        if isinstance(loaded, (list,tuple)) and len(loaded) >= 3:
            model, scaler, le = loaded[0], loaded[1], loaded[2]
            # no metrics available from file
            metrics = {}
            class_names = list(le.classes_) if hasattr(le, "classes_") else []
            return model, scaler, le, class_names, metrics
    except Exception:
        pass

    # fallback: train from dataframe
    target = "category"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing in dataset.")

    dfc = df.copy()
    le = LabelEncoder()
    dfc[target] = le.fit_transform(dfc[target].astype(str))
    class_names = list(le.classes_)

    X = dfc.drop(columns=[target])
    y = dfc[target]

    # encode object features
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtrain, Xtest, ytrain, ytest = train_test_split(Xs, y, test_size=0.18, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    model.fit(Xtrain, ytrain)

    metrics = {"train_acc": float(model.score(Xtrain, ytrain)), "test_acc": float(model.score(Xtest, ytest))}

    # save a copy for future
    try:
        joblib.dump((model, scaler, le), model_path)
    except Exception:
        # ignore saving errors
        pass

    return model, scaler, le, class_names, metrics

# -------------------------
# Load data + model (safe)
# -------------------------
try:
    df = load_dataset()
except Exception as e:
    st.error(f"Could not load dataset 'project-data.csv'. Error: {e}")
    st.stop()

try:
    model, scaler, label_encoder, class_names, metrics = train_or_load_model(df)
except Exception as e:
    st.error(f"Model load/train failed: {e}")
    st.stop()

# -------------------------
# Topbar (brand + typing header)
# -------------------------
top_left, top_right = st.columns([3,1])
with top_left:
    st.markdown("""
    <div class="topbar">
      <div class="brand">
        <div class="logo">🩺</div>
        <div>
          <div class="title">Liver Disease Prediction</div>
          <div class="typing">Predicting liver health, responsibly</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with top_right:
    # small metrics card
    st.markdown("<div class='card' style='text-align:right'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kv'>Classes: <span style='font-weight:900;color:#cfefff'>{len(class_names)}</span></div>", unsafe_allow_html=True)
    if metrics:
        st.markdown(f"<div class='kv'>Test Acc: <span style='font-weight:900;color:#cfefff'>{metrics.get('test_acc',0):.2f}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Sidebar navigation (icons)
# -------------------------
st.sidebar.markdown("<div class='sidebar-title'>🧭 Ultra Menu</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", options=["🧪Home","🔍Predict","🔍Model"], index=0)

# -------------------------
# HOME PAGE (disease info, NO dataset preview)
# -------------------------
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:4px;color:#e8fbff'>About Liver Disease</h2>", unsafe_allow_html=True)
    st.write("""
**Liver disease** encompasses many conditions that cause the liver to be damaged or inflamed.
The liver performs vital functions: detoxifying blood, storing energy, producing proteins, and helping digestion.
Common causes include viral infections, alcohol-related liver disease, fatty liver (NAFLD), autoimmune disorders, and genetic conditions.

**Common lab indicators used in diagnosis:**
- ALT (alanine aminotransferase) — rises with liver cell injury  
- AST (aspartate aminotransferase) — together with ALT helps detect severity  
- ALP (alkaline phosphatase) — can indicate bile duct problems  
- Bilirubin — high levels cause jaundice  
- Albumin & Protein — low levels can show reduced synthetic function  
""")
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    st.write("Use the **Predict** page to input patient measurements and get an AI-based risk prediction. This tool is for informational purposes only and not a substitute for professional medical advice.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PREDICT PAGE (glass-input cards + animated prediction)
# -------------------------
elif page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:6px;color:#e8fbff'>Predict Liver Condition</h2>", unsafe_allow_html=True)
    st.write("Enter patient test values below. All fields are required for accurate prediction.")

    # Build input cards (grouped)
    cols = st.columns((1,1))
    left = cols[0]
    right = cols[1]

    # We'll use the columns that exist in your dataset (exclude target)
    feature_list = [c for c in df.columns if c != "category"]

    # Split features roughly to left/right
    half = (len(feature_list) + 1) // 2
    left_feats = feature_list[:half]
    right_feats = feature_list[half:]

    left_inputs = {}
    right_inputs = {}

    with left:
        st.markdown("<div style='margin-bottom:8px'><strong>Inputs (Left)</strong></div>", unsafe_allow_html=True)
        for f in left_feats:
            # sensible default: empty (user types)
            left_inputs[f] = st.number_input(label=f.replace("_"," ").title(), value=0.0, format="%.3f")

    with right:
        st.markdown("<div style='margin-bottom:8px'><strong>Inputs (Right)</strong></div>", unsafe_allow_html=True)
        for f in right_feats:
            right_inputs[f] = st.number_input(label=f.replace("_"," ").title(), value=0.0, format="%.3f")

    # Combine
    user_inputs = {**left_inputs, **right_inputs}
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    submit_col1, submit_col2 = st.columns([3,1])
    with submit_col1:
        st.write("")

    with submit_col2:
        predict_clicked = st.button("Predict Now")

    # Placeholder boxes for result & confidence animation
    result_placeholder = st.empty()
    conf_placeholder = st.empty()

    if predict_clicked:
        # validate inputs (all entered non-null — here we allow 0.0 but could require >0)
        try:
            # create dataframe for scaling
            X_in = pd.DataFrame([user_inputs])
            X_in = X_in.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            Xs = scaler.transform(X_in.values)

            # predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(Xs)[0]
                pred_idx = int(np.argmax(probs))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])
            else:
                pred_idx = int(model.predict(Xs)[0])
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence = 0.0

            # animated display
            with result_placeholder.container():
                st.markdown("<div class='result-3d'>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:20px;color:#cfefff;margin-bottom:8px'>Prediction</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='glow-badge'>{pred_label}</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'>Confidence</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='conf-bar'><div class='conf-fill' id='conf_fill'></div></div>", unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv'>Advice</div>", unsafe_allow_html=True)
                # simple advice text
                if str(pred_label).lower() in ["healthy","normal","no","low risk"]:
                    st.markdown("<div style='padding:8px'>Low predicted risk. Consider routine follow-up and healthy lifestyle measures.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='padding:8px;color:#ffd4d4'>Predicted elevated risk — consult a hepatologist for further evaluation and diagnostics.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # animate the confidence bar width using JS injection trick via st.markdown
            percent = int(confidence * 100)
            # small delay for UX
            time.sleep(0.22)
            js = f"""
            <script>
            const el = document.querySelector('.conf-fill');
            if (el) {{
              el.style.width = '{percent}%';
            }}
            </script>
            """
            st.markdown(js, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# MODEL INFO PAGE
# -------------------------
elif page == "Model":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#e8fbff'>Model Details</h2>", unsafe_allow_html=True)
    st.write("Model: RandomForestClassifier (either loaded from `liver_model.joblib` or trained automatically from `project-data.csv`).")
    if metrics:
        st.write(f"Train accuracy: **{metrics.get('train_acc',0):.3f}**, Test accuracy: **{metrics.get('test_acc',0):.3f}**")
    st.write(f"Classes: {', '.join(class_names)}")
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    st.write("Feature list (used for prediction):")
    st.write([c for c in df.columns if c != "category"])
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# END
# -------------------------
