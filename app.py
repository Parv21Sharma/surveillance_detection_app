import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import gdown

st.set_page_config(page_title="Surveillance Detection System", layout="centered")

st.title("🔍 Surveillance Detection System")
st.markdown("Upload an image to analyze the situation (Safe / Danger / Isolated)")

MODEL_PATH = "final_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1tNEqflcekNm3VCmyaxJcAX1bLG5msp0v"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

class_names = ['dangerimages', 'isolatedimages', 'safeimages']

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

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
