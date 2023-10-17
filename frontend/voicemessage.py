import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import speech_recognition as sr
import numpy as np
import tempfile
import os

# Preprocessing: Clean and preprocess the text
# Feature Extraction: TF-IDF vectorization
tfidf_vectorizer = joblib.load('models/audioTfidfVectorizer3.sav')  # Load the pre-trained vectorizer

# Load the pre-trained Support Vector Classifier (SVC) model
svc_classifier = joblib.load('models/audiomodel2.sav')

# Function to convert audio to text and make predictions using the loaded model
def process_audio(audio_file, classifier, vectorizer):
    try:
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Recognize the audio from the audio_file
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)

        # Preprocess the transcribed text (clean and preprocess)
        preprocessed_text = preprocess_text(transcribed_text)

        # Vectorize the preprocessed text using the loaded vectorizer
        text_vector = tfidf_vectorizer.transform([preprocessed_text])

        # Make predictions using the loaded classifier
        prediction = classifier.predict(text_vector)

        # Return the prediction
        return prediction[0]
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None

# Function to preprocess text (you can customize this based on your needs)
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove leading and trailing whitespaces
    text = text.strip()
    # Rejoin tokens into a single string
    text = ' '.join(text.split())

    return text

# Streamlit app
def main():
    st.title("Audio Text Classification")

    # File upload widget
    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        # Create a temporary directory to save the uploaded audio file
        temp_dir = tempfile.mkdtemp()
        audio_file_path = os.path.join(temp_dir, "uploaded_audio.wav")

        # Save the uploaded audio file to the temporary directory
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(uploaded_file.read())

        # Process the audio file on button click
        if st.button("Process Audio"):
            predicted_label_svc = process_audio(audio_file_path, svc_classifier, tfidf_vectorizer)
            st.subheader("Prediction Result:")
            if predicted_label_svc == 1:
                st.error("Spam")
            else:
                st.success("Not Spam")

if __name__ == '__main__':
    main()
