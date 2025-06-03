import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Facial Emotion Recognition using RNN")

# Load the pre-trained model
model = load_model("emotion_rnn_model.h5")
st.success("Model loaded successfully!")

# Image upload for prediction
st.subheader("Upload an Image for Emotion Prediction")
uploaded_file = st.file_uploader("Choose a grayscale image (48x48)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = load_img(uploaded_file, color_mode="grayscale", target_size=(48, 48))
    img_array = img_to_array(img).reshape(1, 48, 48) / 255.0
    prediction = model.predict(img_array)
    predicted_label = emotion_labels[np.argmax(prediction)]

    st.image(img, caption=f"Predicted Emotion: {predicted_label}", width=150)
    st.write(f"**Prediction:** {predicted_label}")
    
    # Display prediction probabilities
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(emotion_labels, prediction[0])
    ax.set_xticklabels(emotion_labels, rotation=45)
    ax.set_ylabel('Probability')
    st.pyplot(fig)
