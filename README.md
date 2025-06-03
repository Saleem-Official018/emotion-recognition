# Emotion Recognition using Deep Learning

A deep learning project using RNN to recognize emotions from user-uploaded images. Detects facial expressions and outputs the identified emotion.

## Features
- Real-time emotion detection from images
- Support for 7 basic emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- User-friendly web interface
- Pre-trained model for accurate predictions

## Dataset
📌 This project uses the FER-2013 Facial Expression Recognition dataset, available here:  
🔗 https://www.kaggle.com/datasets/deadskull7/fer2013

## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Flask
- NumPy
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

1. Download the pre-trained model file (`emotion_rnn_model.h5`) and place it in the project directory.
2. Run the following command to launch the app:
```bash
python app.py
```
3. Open your web browser and navigate to `http://localhost:5000`

## Usage
1. Click on the "Choose File" button to select an image
2. Click "Upload" to process the image
3. The detected emotion will be displayed along with the processed image

## Project Structure
```
emotion-recognition/
├── app.py              # Main application file
├── emotion_rnn_model.h5 # Pre-trained model
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
