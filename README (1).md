
# Multimodal Sentiment Analysis App

This project is a web application that performs sentiment analysis on text, images, audio, and video using various machine learning models. It utilizes Streamlit for the web interface and integrates several libraries for audio processing, image recognition, and sentiment analysis.

## Table of Content

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Installation

To set up the project, follow these steps:


1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yashwanth17230/Multimodal_Sentiment_Analysis.git
   
   ```


2. **Create a Virtual Environment** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```



3. **Install Dependencies**:

   Make sure you have `pip` installed, then run:
   ```bash
      pip install -r requirements.txt
    ```

4. **Download Required Models**:

   Ensure you have the following files in the project directory:
   
   - `face_emotion.json`: The model architecture for emotion detection.
   
   - `face_emotion.h5`: The pre-trained weights for the emotion detection model.
   
   - `haarcascade_frontalface_default.xml`: The Haar Cascade classifier for face detection.
   
   - `emotions.txt`: A text file containing emotion keywords and their corresponding labels.
## Usage


1. **Run the Application**:


   After installing the dependencies and ensuring all required files are in place, you can start the Streamlit application by running:
   ```bash
      streamlit run app.py
   ```


2. **Access the Application**:


   Open your web browser and go to `http://localhost:8501` to access the application.


3. **Choose Analysis Type**:

   Use the sidebar to select the type of analysis you want to perform:

   - **Text Analysis**: Enter text to analyze sentiment and emotions.
   
   - **Image Analysis**: Upload an image to detect emotions.
   
   - **Audio Analysis**: Upload an audio file or record audio to analyze sentiment and emotions.
   
   - **Video Analysis**: Use the webcam to analyze emotions in real-time.
## Dependencies

The project requires the following Python packages:

- streamlit
- speech_recognition
- pydub
- opencv-python
- numpy
- tensorflow
- Pillow
- nltk
- streamlit-webrtc


You can find the complete list of dependencies in the `requirements.txt` file.