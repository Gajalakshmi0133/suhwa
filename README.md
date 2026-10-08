# Suhwa — AI-Powered Sign Language Recognition

Suhwa is an AI-powered web application that uses computer vision and deep learning to recognize sign language gestures in real time and convert them into text.

The project is designed to support accessible communication by providing a simple interface for detecting and interpreting hand gestures through a webcam.

## Features

- Real-time sign language gesture recognition
- Webcam-based hand gesture detection
- Hand landmark detection and processing
- Deep learning-based gesture classification
- Conversion of recognized gestures into text
- Interactive web interface
- Real-time prediction feedback

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python, JavaScript |
| Backend | Flask |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | TensorFlow, Keras |
| Frontend | HTML, CSS, JavaScript |
| Version Control | Git, GitHub |

## System Workflow

```text
Webcam Input
     ↓
Video Frame Capture
     ↓
Hand Detection
     ↓
Landmark / Gesture Processing
     ↓
Deep Learning Model
     ↓
Gesture Classification
     ↓
Text Output


How It Works
1. The application captures video through the user's webcam.
2. MediaPipe and OpenCV process the video frames and detect hand movements.
3. Relevant gesture information is extracted from the detected hand.
4. The trained TensorFlow/Keras model classifies the gesture.
5. The recognized sign is displayed as text through the web interface.

Installation

Clone the repository:
git clone https://github.com/Gajalakshmi0133/Suhwa.git
cd Suhwa

Create a virtual environment:
python -m venv venv

Activate the environment on Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Open the application in your browser:
http://127.0.0.1:5000

Project Structure
Suhwa/
├── app.py
├── templates/
├── static/
├── model/
├── dataset/
├── requirements.txt
└── README.md

The project structure may vary depending on the implementation.

Use Cases
Suhwa can be used as a foundation for:
- Assistive communication systems
- Sign language learning applications
- Accessibility-focused applications
- Gesture-based human-computer interaction
- Educational technology

Future Enhancements
- Expand the supported sign vocabulary
- Improve recognition accuracy
- Add sentence-level sign recognition
- Add text-to-speech output
- Support multiple sign languages
- Optimize the model for mobile and real-time deployment

Project Status
Completed — Academic / Portfolio Project
The current implementation focuses on real-time gesture recognition and text conversion. The system can be further extended with additional gestures, improved models, and voice-based output.

Author
Gajalakshmi K
B.Sc. Computer Science
GitHub


Suhwa is an AI-powered web application that uses computer vision and deep learning to recognize sign language gestures in real time and convert them into text.
