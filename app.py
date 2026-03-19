import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ⚠️ CRITICAL FIX (Keras compatibility)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from tensorflow.keras.applications.resnet50 import preprocess_input

# Page config
st.set_page_config(page_title="Surveillance Detection System", layout="centered")

# Title
st.title("🔍 Surveillance Detection System")
st.markdown("Upload an image to analyze the situation (Safe / Danger / Isolated)")

MODEL_PATH = "resnet50_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1-GjozOJ-D3-8lqZPg8xEVzBeCEP0SI8S"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

model = load_model()

# Class names
class_names = ['dangerimages', 'isolatedimages', 'safeimages']

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100

    st.subheader("🔎 Prediction Result")

    if predicted_class == "dangerimages":
        st.error(f"⚠️ Danger Detected ({confidence:.2f}%)")
    elif predicted_class == "isolatedimages":
        st.warning(f"🚶 Isolated Situation ({confidence:.2f}%)")
    else:
        st.success(f"✅ Safe Situation ({confidence:.2f}%)")

    st.subheader("📊 Confidence Scores")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
