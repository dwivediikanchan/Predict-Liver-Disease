# app.py — Ultra Animated Liver Disease App (All animations combined)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Liver Disease Prediction", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helper: load dataset & model (cached)
# -------------------------
@st.cache_data
def load_dataset(path="project-data.csv"):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def train_or_load_model(df, model_path="liver_model.joblib"):
    # Try load joblib first
    try:
        m = joblib.load(model_path)
        if isinstance(m, (list, tuple)) and len(m) >= 3:
            model, scaler, le = m[0], m[1], m[2]
            class_names = list(le.classes_) if hasattr(le, "classes_") else []
            return model, scaler, le, class_names, {}
    except Exception:
        pass

    # train from dataset
    if "category" not in df.columns:
        raise ValueError("Dataset must have a 'category' column as target.")

    d = df.copy()
    le = LabelEncoder()
    d["category"] = le.fit_transform(d["category"].astype(str))
    class_names = list(le.classes_)

    X = d.drop(columns=["category"])
    y = d["category"]

    # encode any object features
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtrain, Xtest, ytrain, ytest = train_test_split(Xs, y, test_size=0.18, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    model.fit(Xtrain, ytrain)

    metrics = {"train_acc": float(model.score(Xtrain, ytrain)), "test_acc": float(model.score(Xtest, ytest))}

    # Save for speed if possible
    try:
        joblib.dump((model, scaler, le), model_path)
    except Exception:
        pass

    return model, scaler, le, class_names, metrics

# -------------------------
# LOTTIE / external assets
# We'll embed lottie-player via CDN and remote JSONs (from LottieFiles)
# If your deployment blocks external resources, lottie may not load.
# -------------------------
LOTTIE_MEDICAL = "https://assets8.lottiefiles.com/packages/lf20_0yfsb3a1.json"  # generic medical animation
LOTTIE_LIVER = "https://assets10.lottiefiles.com/packages/lf20_ck6ciotq.json"    # organ/scan style (fallback)
LOTTIE_LOADING = "https://assets4.lottiefiles.com/packages/lf20_jtbfg2nb.json"

# -------------------------
# CSS + animations (combined)
# - Futuristic: neon pulses, glow
# - Soft: fade-ins, gentle shadows
# - Medical: neat icons, badges
# - Dashboard: animated progress/conf bars
# -------------------------
st.markdown(
    """
    <style>
    /* App background gradient & subtle animation */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, rgba(7,26,82,1) 0%, rgba(8,15,36,1) 50%, rgba(2,6,23,1) 100%);
        background-size: 300% 300%;
        animation: bgShift 14s ease infinite;
        color: #e6fbff;
        min-height: 100vh;
    }
    @keyframes bgShift {
      0% {background-position: 0% 50%;}
      50% {background-position: 100% 50%;}
      100% {background-position: 0% 50%;}
    }

    /* Sidebar styling (neon gradient) */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(12,10,34,0.95), rgba(18,20,46,0.95));
        border-right: 1px solid rgba(255,255,255,0.03);
        color: #cfefff;
        padding: 18px;
    }
    .sidebar-title { font-size:22px; color:#00e0ff; font-weight:800; margin-bottom:8px; }

    /* Topbar brand */
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:12px;}
    .brand { display:flex; align-items:center; gap:12px;}
    .logo { width:44px; height:44px; border-radius:10px; background: linear-gradient(90deg,#00e0ff,#a855f7); display:flex; align-items:center; justify-content:center; font-weight:900; color:#021024; box-shadow:0 8px 30px rgba(128,90,213,0.12); }

    .title { font-size:20px; font-weight:800; color:#cfefff; }

    /* typing subtitle */
    .typing { color:#cfefff; opacity:0.95; font-weight:600; font-size:14px; }

    /* cards (glass) */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        border-radius:14px;
        padding:18px;
        box-shadow: 0 12px 40px rgba(2,6,23,0.6);
        backdrop-filter: blur(6px) saturate(1.1);
        border: 1px solid rgba(255,255,255,0.03);
        color:#e6fbff;
    }

    /* floating/slide-in */
    .fade-in { animation: fadeIn 0.9s ease both; }
    @keyframes fadeIn {
      from {opacity:0; transform: translateY(8px);}
      to {opacity:1; transform: translateY(0);}
    }

    /* neon button */
    .stButton>button {
        background: linear-gradient(90deg,#00e0ff,#7c3aed) !important;
        color: #021024 !important;
        font-weight:800 !important;
        padding:10px 22px !important;
        border-radius:10px !important;
        box-shadow: 0 8px 30px rgba(124,58,237,0.12) !important;
    }

    /* glowing badge */
    .glow-badge {
      display:inline-block;
      padding:8px 18px;
      border-radius:999px;
      background: linear-gradient(90deg,#00e0ff,#a855f7);
      color:#021024;
      font-weight:900;
      box-shadow:0 10px 30px rgba(165,90,255,0.12);
    }

    /* result "3D" card */
    .result-card {
      border-radius:12px;
      padding:18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      box-shadow: 0 18px 60px rgba(2,6,23,0.6);
      border: 1px solid rgba(0,240,255,0.06);
      text-align:center;
    }

    /* animated conf bar container */
    .conf-bar { width:100%; height:14px; background: rgba(255,255,255,0.04); border-radius:8px; overflow:hidden; margin-top:12px; }
    .conf-fill { height:100%; width:0%; background: linear-gradient(90deg,#00e0ff,#7c3aed); transition: width 1.4s ease; }

    /* soft hover card */
    .hover-card:hover { transform: translateY(-6px); transition: all 0.35s ease; }

    /* small text helpers */
    .kv { color:#bfefff; font-weight:700; }
    .sep { height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent); margin:10px 0; border-radius:6px; }

    /* responsive adjustments */
    @media (max-width: 640px) {
        .brand .title { font-size:16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Insert Lottie webcomponent script (CDN)
# -------------------------
# This injects the <lottie-player> web component so we can embed Lottie easily.
st.markdown(
    """
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load dataset & model
# -------------------------
try:
    df = load_dataset()
except Exception as e:
    st.error(f"Could not load 'project-data.csv': {e}")
    st.stop()

try:
    model, scaler, label_enc, class_names, metrics = train_or_load_model(df)
except Exception as e:
    st.error(f"Model training/loading failed: {e}")
    st.stop()

# -------------------------
# Topbar (brand + typing subtitle)
# -------------------------
c1, c2 = st.columns([3,1])
with c1:
    st.markdown(
        """
        <div class="topbar fade-in">
          <div class="brand">
            <div class="logo">🩺</div>
            <div>
              <div class="title">Liver disease Prediction</div>
              <div class="typing">Detecting liver risk with modern ML</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    # show small metrics
    st.markdown("<div class='card fade-in' style='text-align:right'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kv'>Classes: <strong>{len(class_names)}</strong></div>", unsafe_allow_html=True)
    if metrics:
        st.markdown(f"<div class='kv'>Test Acc: <strong>{metrics.get('test_acc',0):.2f}</strong></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Sidebar navigation with emojis
# -------------------------
st.sidebar.markdown("<div class='sidebar-title'>🧭 Navigation</div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    options=[
        "🏠 Home",
        "🧪 Predict",
        "📊 Model Info"
    ],
    index=0
)

# -------------------------
# HOME PAGE
# -------------------------
if page == "🏠 Home":
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#e8fbff;margin:6px;'>About Liver Disease</h2>", unsafe_allow_html=True)
    st.write(
        """
        **Liver disease** covers many conditions that damage the liver. The liver filters blood, stores energy,
        and produces essential proteins. Damage may be caused by viruses, alcohol, fatty liver disease, autoimmune disorders,
        or genetic conditions.

        **Common lab indicators:**
        - ALT (alanine aminotransferase)
        - AST (aspartate aminotransferase)
        - ALP (alkaline phosphatase)
        - Bilirubin
        - Albumin and total protein

        This app is for **informational** purposes, not a substitute for professional medical advice.
        """
    )
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # Show a Lottie animation on Home to add premium feel
    st.markdown(
        f"""
        <div style="display:flex;gap:18px;align-items:center">
          <div style="flex:1">
            <lottie-player src="{LOTTIE_MEDICAL}" background="transparent" speed="1"  style="width:100%;max-width:450px;" loop autoplay></lottie-player>
          </div>
          <div style="flex:1;padding:6px;">
            <h3 style="color:#cfefff">Why early testing matters</h3>
            <p style="color:#dff8ff">
              Early detection lets clinicians act earlier, improving outcomes and preventing progression to cirrhosis.
              Use the Predict panel to input test results.
            </p>
            <div style="margin-top:8px;">
              <span class="glow-badge">Machine Learning powered</span>
              <span style="display:inline-block;width:10px"></span>
              <span class="glow-badge">Fast</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PREDICT PAGE
# -------------------------
elif page == "🧪 Predict":
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#e8fbff;margin:6px'>🧪 Predict Liver Condition</h2>", unsafe_allow_html=True)
    st.write("Fill in patient test values below. Use realistic lab numbers for best results.")

    # Build input layout; use dataset columns excluding 'category'
    feature_list = [c for c in df.columns if c != "category"]
    half = (len(feature_list) + 1) // 2
    left_feats = feature_list[:half]
    right_feats = feature_list[half:]

    left_col, right_col = st.columns(2)
    left_inputs, right_inputs = {}, {}

    with left_col:
        for f in left_feats:
            left_inputs[f] = st.number_input(label=f.replace("_", " ").title(), value=0.0, format="%.3f")

    with right_col:
        for f in right_feats:
            right_inputs[f] = st.number_input(label=f.replace("_", " ").title(), value=0.0, format="%.3f")

    # Animated lottie (scan) and Predict button
    st.markdown("<div style='display:flex;gap:18px;align-items:center;margin-top:12px'>", unsafe_allow_html=True)
    st.markdown(f'<div style="flex:1"><lottie-player src="{LOTTIE_LIVER}" background="transparent" speed="1" style="width:100%;max-width:320px;" loop autoplay></lottie-player></div>', unsafe_allow_html=True)
    # Action column
    action_html = """
        <div style="flex:1;">
            <div style="padding:12px;">
                <h3 style="color:#cfefff;margin-bottom:6px">Ready to predict</h3>
                <p style="color:#dff8ff;margin-top:0;">Press Predict to run the model. A confidence bar and a 3D result card will animate.</p>
            </div>
        </div>
    """
    st.markdown(action_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button (neon)
    if st.button("🔍 Predict Now"):
        # Combine features
        user_inputs = {**left_inputs, **right_inputs}
        X_in = pd.DataFrame([user_inputs])
        X_in = X_in.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Scale
        try:
            Xs = scaler.transform(X_in.values)
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            st.stop()

        # Predict with subtle animation (progress)
        progress = st.progress(0)
        for p in range(0, 101, 10):
            time.sleep(0.04)
            progress.progress(p)

        # Prediction + confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xs)[0]
            idx = int(np.argmax(probs))
            label = label_enc.inverse_transform([idx])[0]
            confidence = float(probs[idx])
        else:
            idx = int(model.predict(Xs)[0])
            label = label_enc.inverse_transform([idx])[0]
            confidence = 0.0

        # Show animated 3D card and animate confidence bar via JS injection
        st.markdown("<div class='result-card fade-in hover-card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:16px;color:#cfefff'>Result</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:22px;margin-bottom:8px'><span class='glow-badge'>{label}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:6px' class='kv'>Confidence: <strong>{confidence:.2%}</strong></div>", unsafe_allow_html=True)
        st.markdown("<div class='conf-bar'><div class='conf-fill'></div></div>", unsafe_allow_html=True)
        # Advice
        if str(label).lower() in ["healthy", "normal", "no", "low risk"]:
            st.markdown("<div style='margin-top:12px;color:#cfefff'>Low predicted risk. Continue routine care.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-top:12px;color:#ffd6d6'>Elevated risk — please consult a hepatologist for further evaluation.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Animate confidence fill using JS snippet (manipulate conf-fill width)
        pct = int(confidence * 100)
        js = f"""<script>
            const el = document.querySelector('.conf-fill');
            if(el){{el.style.width = '{pct}%';}}
            </script>"""
        st.markdown(js, unsafe_allow_html=True)

# -------------------------
# MODEL INFO PAGE
# -------------------------
elif page == "📊 Model Info":
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#e8fbff;margin:6px'>🤖 Model Information</h2>", unsafe_allow_html=True)

    st.write("**Model:** RandomForestClassifier (auto-loaded or trained at startup)")
    if metrics:
        st.write(f"**Train accuracy:** {metrics.get('train_acc',0):.3f} — **Test accuracy:** {metrics.get('test_acc',0):.3f}")
    st.write(f"**Classes:** {', '.join(class_names)}")
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    st.write("**Features used:**")
    st.write([c for c in df.columns if c != "category"])
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# End
# -------------------------
