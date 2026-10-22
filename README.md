# Surveillance Detection System

A deep learning-based surveillance image classification application built using TensorFlow, Keras, and Streamlit. The system classifies uploaded surveillance images into three categories: **Safe, Isolated, and Dangerous**.

## 🚀 Live Demo

Try the application here: [Surveillance Detection App](https://surveillancedetectionapp-7dlhzcyjbqqy2f3dmdxx3j.streamlit.app/)

## ✨ Features

- Upload surveillance images in JPG, JPEG, or PNG format.
- Classify images into Safe, Isolated, or Dangerous categories.
- Display the predicted class and confidence scores.
- Use a trained deep learning model for image classification.
- Simple, interactive web interface built with Streamlit.
- Automatically download the trained model when it is not available locally.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model Architecture:** ResNet50-based image classification
- **Web Framework:** Streamlit
- **Image Processing:** Pillow, NumPy
- **Model Download:** gdown

## 🧠 How It Works

1. The user uploads a surveillance image.
2. The image is converted to RGB format and resized to 224 × 224 pixels.
3. ResNet50-compatible preprocessing is applied.
4. The trained model generates predictions for the three classes.
5. The application displays the predicted category and corresponding confidence scores.

## 📂 Project Structure

```text
surveillance_detection_app/
├── README.md
├── app.py
├── requirements.txt
└── survillence2(ResNet50).ipynb
```

- `app.py` — Streamlit application and image prediction pipeline.
- `requirements.txt` — Required Python packages.
- `survillence2(ResNet50).ipynb` — Jupyter notebook for model development and experimentation.
- `README.md` — Project documentation.

## 📊 Model and Dataset

### Dataset Creation

A manually curated image dataset was created for training and evaluating the surveillance detection system. The dataset contains three classes:

- **Safe:** Surveillance scenes classified as safe.
- **Isolated:** Scenes depicting isolated situations.
- **Dangerous:** Scenes depicting potentially dangerous situations.

The dataset was organized into class-specific directories to support image loading and classification using TensorFlow/Keras.

### Data Preparation

- Images were collected and manually organized according to their respective categories.
- Images were resized to 224 × 224 pixels.
- ResNet50-compatible preprocessing was applied before model inference.
- The dataset was divided into training, validation, and testing subsets for model development and evaluation.

### Model Architecture

The project uses transfer learning with a ResNet50-based image classification model to classify surveillance images into the three categories.

The application processes an uploaded image and returns the predicted class along with confidence scores.

### Future Dataset Improvements

- Expand the dataset with more diverse surveillance scenes.
- Improve class balance and image diversity.
- Evaluate generalization using unseen images and appropriate classification metrics.

## 🔒 Limitations and Responsible Use

- Predictions are estimates and should not be treated as definitive assessments of safety.
- The system should not be used as the sole basis for emergency response or other high-stakes decisions.
- Surveillance images may contain identifiable individuals and should be handled responsibly.
- Performance should be evaluated on diverse, unseen data before real-world deployment.

## 🔮 Future Improvements

- Expand and diversify the dataset.
- Compare ResNet50 with MobileNetV2 and EfficientNet.
- Evaluate precision, recall, F1-score, and confusion matrices.
- Explore privacy-preserving preprocessing, such as face blurring.
- Optimize inference latency for resource-constrained edge devices.

## 👨‍💻 Author

**Parv Sharma**

GitHub: [Parv21Sharma](https://github.com/Parv21Sharma)

---

If you find this project interesting, feel free to explore the repository and its implementation.
