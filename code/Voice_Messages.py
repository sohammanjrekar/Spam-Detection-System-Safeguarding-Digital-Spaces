import streamlit as st
import speech_recognition as sr

st.title("Spam Detection in Voice Messages")
st.write("Build a system to detect spam content in voice messages.")

# Record voice message
recorded_audio = st.file_uploader("Record voice message", type=["wav"])

# Button to trigger spam detection
if st.button("Detect Spam"):
    if recorded_audio is not None:
        # Call your voice message spam detection function here
        is_spam = detect_voice_spam(recorded_audio)
        if is_spam:
            st.error("This voice message contains spam content.")
        else:
            st.success("This voice message is clean.")
    else:
        st.warning("Please record a voice message.")

# Example function (replace with your actual detection logic)
def detect_voice_spam(audio_file):
    # Placeholder logic using speech recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        if "spam" in text.lower():
            return True
    return False
