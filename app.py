import streamlit as st
import io
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image
import os
import time

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Text Analysis'

# Load emotions from a file
def load_emotions(file_path):
    emotions = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                if ':' in clear_line:
                    word, emotion = clear_line.split(':')
                    emotions[word.strip()] = emotion.strip().capitalize()
    except FileNotFoundError:
        st.error(f"Emotions file not found at {file_path}. Please check the path.")
    return emotions

# Ensure the file path is correct
emotions_dict = load_emotions('emotions.txt')

# Text Analysis
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        sentiment = "Negative sentiment"
        text_color = "red"
        bg_color = "#FFCCCB"
    elif pos > neg:
        sentiment = "Positive sentiment"
        text_color = "green"
        bg_color = "#88E788"
    else:
        sentiment = "Neutral sentiment"
        text_color = "black"
        bg_color = "yellow"
    
    detected_emotions = set()
    for word in sentiment_text.split():
        word = word.strip().lower()
        if word in emotions_dict:
            detected_emotions.add(emotions_dict[word])
    
    return sentiment, text_color, bg_color, list(detected_emotions)

# Audio Analysis
def recognize_audio(uploaded_audio):
    recognizer = sr.Recognizer()
    text = ""

    audio_file = io.BytesIO(uploaded_audio.read())
    audio = AudioSegment.from_file(audio_file, format="wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio.export(temp_audio_file.name, format="wav")

    with sr.AudioFile(temp_audio_file.name) as source:
        st.write('Clearing background noise...')
        recognizer.adjust_for_ambient_noise(source, duration=3)
        st.write('Analysing uploaded wav file...')
        recordedAudio = recognizer.record(source)

    duration_seconds = len(audio) / 1000  # Duration in seconds
    st.write(f'Done! Duration: {duration_seconds} seconds')

    try:
        st.write('Printing the message...')
        text = recognizer.recognize_google(recordedAudio, language='en-US')
        st.write('Your message: {}'.format(text)) 
    except Exception as ex:
        st.write(ex)

    return text

# Record audio from microphone
def record_audio():
    recognizer = sr.Recognizer()
    text = ""

    with sr.Microphone() as source:
        st.write('Clearing background noise...')
        recognizer.adjust_for_ambient_noise(source, duration=3)
        st.write('Start Speaking...')

        start_time = time.time()
        recordedAudio = recognizer.listen(source)
        end_time = time.time()

        st.write(f'Done recording! Time taken: {round(end_time - start_time, 2)} seconds')

    try:
        st.write('Printing the message...')
        text = recognizer.recognize_google(recordedAudio, language='en-US')
        st.write('Your message: {}'.format(text)) 
    except Exception as ex:
        st.write(ex)

    return text

# Image Analysis
def analyze_image(uploaded_image):
    image = Image.open(uploaded_image)
    img_array = np.array(image)
    
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        st.write("No faces detected.")
        return []
    
    emotions_detected = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            finalout = emotion_dict[maxindex]
            confidence = prediction[maxindex] * 100
            emotions_detected.append(f"{finalout.capitalize()} ({confidence:.2f}%)")
            cv2.putText(img_array, f"{finalout.capitalize()} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    st.image(img_array, caption='Processed Image', use_column_width=True)
    return emotions_detected

# Video Analysis
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                confidence = prediction[maxindex] * 100
                output = f"{finalout.capitalize()} ({confidence:.2f}%)"
                
                # Debugging: Print the output to the console
                print(output)
                
                # Ensure the text is within the frame
                label_position = (x, y - 10 if y - 10 > 10 else y + 10)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return img

# Load models and classifiers
emotion_dict = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

json_file = open('face_emotion.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

classifier.load_weights("face_emotion.h5")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Streamlit UI
st.markdown("""
    <style>
    /* Enhanced Navigation Elements */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        color: white;
        background-color: #28a745;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .stTextInput>div>div>input {
        background-color: #f8f9fa;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 10px;
    }
    .stAlert {
        background-color: #e6f4ea;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #00008B; /* Dark Blue */
    }
    </style>
    """, unsafe_allow_html=True)

# Main function
def main():
    st.sidebar.title("Choose Analysis Type")
    page = st.sidebar.selectbox("", ["Text Analysis", "Image Analysis", "Audio Analysis", "Video Analysis"], index=["Text Analysis", "Image Analysis", "Audio Analysis", "Video Analysis"].index(st.session_state.page))

    st.title("Multimodal Sentiment Analysis App")

    # Text Analysis
    if page == "Text Analysis":
        st.header("Text Sentiment Analysis")
        st.write("Enter text to analyze sentiment and emotions:")
        user_input = st.text_area("Text")
        if st.button("Analyze Text"):
            sentiment_result, text_color, bg_color, detected_emotions = sentiment_analyse(user_input)
            st.markdown(
                f"""
                <div style='border: 2px solid {text_color}; background-color: {bg_color}; padding: 10px; border-radius: 5px;'>
                    <span style='color:{text_color}; font-weight: bold;'>Detected Sentiment: {sentiment_result}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.success(f"Detected Emotions: {', '.join(detected_emotions)}")

    # Image Analysis
    elif page == "Image Analysis":
        st.header("Image Emotion Analysis")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            if st.button("Analyze Image"):
                emotions_detected = analyze_image(uploaded_image)
                # Display detected emotions
                st.markdown(
                    f"""
                    <div style='border: 2px solid green; background-color: #e6f4ea; padding: 10px; border-radius: 5px;'>
                        <span style='color:green; font-weight: bold;'>Detected Emotions: {', '.join(emotions_detected)}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    # Audio Analysis
    elif page == "Audio Analysis":
        st.header("Audio Sentiment Analysis")
        audio_option = st.selectbox("Choose Audio Option", ["Upload Audio", "Record Audio"])

        if audio_option == "Upload Audio":
            uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
            if uploaded_audio is not None:
                if st.button("Analyze Audio"):
                    text = recognize_audio(uploaded_audio)
                    sentiment_result, text_color, bg_color, detected_emotions = sentiment_analyse(text)
                    st.markdown(
                        f"""
                        <div style='border: 2px solid {text_color}; background-color: {bg_color}; padding: 10px; border-radius: 5px;'>
                            <span style='color:{text_color}; font-weight: bold;'>Detected Sentiment: {sentiment_result}</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.success(f"Detected Emotions: {', '.join(detected_emotions)}")

        elif audio_option == "Record Audio":
            st.write("Record an Audio and get the predicted Sentiment & Emotions")
            if st.button("Start Recording"):
                text = record_audio()
                sentiment_result, text_color, bg_color, detected_emotions = sentiment_analyse(text)
                st.markdown(
                    f"""
                    <div style='border: 2px solid {text_color}; background-color: {bg_color}; padding: 10px; border-radius: 5px;'>
                        <span style='color:{text_color}; font-weight: bold;'>Detected Sentiment: {sentiment_result}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.success(f"Detected Emotions: {', '.join(detected_emotions)}")

    # Video Analysis
    elif page == "Video Analysis":
        st.header("Video Emotion Analysis")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == '__main__':
    main()