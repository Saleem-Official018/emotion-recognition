# Emotion Recognition using Deep Learning

A deep learning project using RNN to recognize emotions from user-uploaded images. Detects facial expressions and outputs the identified emotion.

## Features
- Real-time emotion detection from images
- Support for 7 basic emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- User-friendly web interface using Streamlit
- Pre-trained model for accurate predictions
- Visual probability distribution of emotions

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Saleem-Official018/emotion-recognition.git
cd emotion-recognition
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Getting Started
👉 To get started:

1. Make sure you have the pre-trained model file (`emotion_rnn_model.h5`) in the project directory.
2. Run the following command to launch the app:
```bash
streamlit run app.py
```
3. The app will automatically open in your default web browser

## Usage
1. Upload a grayscale image (48x48 pixels) using the file uploader
2. The app will automatically process the image and show:
   - The uploaded image with the predicted emotion
   - A bar chart showing the probability distribution across all emotions

## Project Structure
```
emotion-recognition/
├── app.py              # Main application file
├── emotion_rnn_model.h5 # Pre-trained model
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Model Details
- Architecture: SimpleRNN with dropout layers
- Input: 48x48 grayscale images
- Output: 7 emotion classes
- Pre-trained model is included in the repository

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
