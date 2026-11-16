import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="Mobile Phone Tier Classifier",
    page_icon="📱",
    layout="wide"
)

# ==============================================================================
# 1. Model & Preprocessor Loader (with Automatic Fallback Training)
# ==============================================================================
@st.cache_resource
def load_or_train_model():
    """
    Loads saved model.keras, scaler.pkl, and label_encoder.pkl.
    If not found, trains the simple Sequential ANN once and caches it.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    model_path = "model.keras"
    scaler_path = "scaler.pkl"
    encoder_path = "label_encoder.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
        model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        return model, scaler, label_encoder

    # Train model if not yet saved
    df = pd.read_csv("test_tiered.csv")
    X = df[["battery_power", "clock_speed", "int_memory", "n_cores", "ram"]]
    y = df["tier"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(32, activation="relu", input_shape=(5,)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    backtracking = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=32,
        callbacks=[backtracking],
        verbose=0
    )

    # Save artifacts for future sessions
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    return model, scaler, label_encoder

# Load artifacts
model, scaler, label_encoder = load_or_train_model()

# ==============================================================================
# 2. UI Layout & User Inputs
# ==============================================================================

st.title("📱 Mobile Phone Tier Classifier")
st.markdown(
    "Predict whether a phone configuration belongs to **Budget**, **Mid-Range**, or **Flagship** "
    "using an **Artificial Neural Network (Sequential Model)** with live Softmax confidence percentages."
)

st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("⚙️ Enter Phone Specifications")

    # Battery
    battery_power = st.slider(
        "🔋 Battery Capacity (mAh)",
        min_value=3000,
        max_value=6000,
        value=4800,
        step=50,
        help="Typical range: 3500 mAh (Budget) to 5500 mAh (Flagship)"
    )

    # Clock Speed
    clock_speed = st.slider(
        "⚡ CPU Clock Speed (GHz)",
        min_value=1.0,
        max_value=3.5,
        value=2.8,
        step=0.1,
        help="Higher clock speed corresponds to faster processing power"
    )

    # Storage
    int_memory = st.select_slider(
        "💾 Internal Storage Capacity (GB)",
        options=[32, 64, 128, 256, 512],
        value=256,
        help="Storage capacity of the device"
    )

    # Cores
    n_cores = st.selectbox(
        "🧠 Number of CPU Cores",
        options=[4, 6, 8],
        index=2,
        help="Number of processing cores in the chipset"
    )

    # RAM
    ram_options = {
        "2 GB (2048 MB)": 2048,
        "3 GB (3072 MB)": 3072,
        "4 GB (4096 MB)": 4096,
        "6 GB (6144 MB)": 6144,
        "8 GB (8192 MB)": 8192,
        "12 GB (12288 MB)": 12288,
        "16 GB (16384 MB)": 16384
    }
    ram_label = st.selectbox(
        "🚀 RAM Capacity",
        options=list(ram_options.keys()),
        index=4,  # default 8 GB
        help="System RAM capacity"
    )
    ram = ram_options[ram_label]

    predict_btn = st.button("🔍 Classify Phone Tier", use_container_width=True, type="primary")

with col_right:
    st.subheader("📊 Softmax Classification Output")

    # Run inference
    input_data = np.array([[battery_power, clock_speed, int_memory, n_cores, ram]])
    input_scaled = scaler.transform(input_data)

    # Softmax probabilities
    probs = model.predict(input_scaled, verbose=0)[0]
    classes = label_encoder.classes_

    pred_idx = np.argmax(probs)
    predicted_tier = classes[pred_idx]
    predicted_conf = probs[pred_idx] * 100.0

    # Color badge for predicted tier
    tier_colors = {
        "Flagship": ("#4CAF50", "🌟 Flagship Phone"),
        "Mid-Range": ("#2196F3", "⚖️ Mid-Range Phone"),
        "Budget": ("#FF9800", "💰 Budget Phone")
    }
    color, badge_text = tier_colors.get(predicted_tier, ("#9E9E9E", predicted_tier))

    st.markdown(
        f"""
        <div style="background-color: {color}1A; border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center; margin-bottom: 20px;">
            <h4 style="margin: 0; color: {color};">Predicted Tier</h4>
            <h2 style="margin: 5px 0 0 0; color: {color};">{badge_text}</h2>
            <p style="margin: 5px 0 0 0; font-size: 16px; color: #555;">Confidence: <b>{predicted_conf:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Softmax Probability Breakdown")
    prob_df = pd.DataFrame({
        "Tier": classes,
        "Probability (%)": [p * 100.0 for p in probs]
    }).set_index("Tier")

    # Progress bars for each tier
    for cls, prob in zip(classes, probs):
        pct = prob * 100.0
        st.write(f"**{cls}**: `{pct:.2f}%`")
        st.progress(float(prob))

    st.divider()

    # Visual Bar Chart
    st.markdown("### Softmax Distribution Chart")
    st.bar_chart(prob_df, y="Probability (%)", color="#1f77b4")

st.divider()
st.caption("Built with Streamlit & TensorFlow Keras (Sequential ANN with Backtracking).")
