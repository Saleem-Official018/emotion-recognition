import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Facial Emotion Recognition using RNN")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48))
    X = np.stack(pixels.values)
    X = X / 255.0
    X = X.reshape(-1, 48, 48)
    y = to_categorical(data['emotion'], num_classes=7)
    return train_test_split(X, y, test_size=0.2)

X_train, X_test, y_train, y_test = load_data()

# Build and train model
def build_and_train_model():
    model = Sequential([
        SimpleRNN(128, input_shape=(48, 48), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("emotion_rnn_model.h5")
    return model

# Load or train model
if not os.path.exists("emotion_rnn_model.h5"):
    with st.spinner("Training model... this may take a while"):
        model = build_and_train_model()
    st.success("Model trained and saved.")
else:
    model = load_model("emotion_rnn_model.h5")
    st.success("Model loaded.")

# Show random samples
if st.checkbox("Show Random Samples with Labels"):
    st.subheader("Random Emotion Samples")
    fig, axes = plt.subplots(2, 7, figsize=(14, 5))
    for i, ax in enumerate(axes.flatten()):
        index = np.random.randint(0, len(X_train))
        image = X_train[index].reshape(48, 48)
        label = np.argmax(y_train[index])
        ax.imshow(image, cmap='gray')
        ax.set_title(emotion_labels[label])
        ax.axis('off')
    st.pyplot(fig)

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
