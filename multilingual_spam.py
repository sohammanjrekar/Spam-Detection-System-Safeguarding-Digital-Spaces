import streamlit as st

# Set the title and description
st.title("Multilingual Spam Detection")
st.write("Develop a spam detection system that can handle multiple languages.")

# Text input for user to enter a message
user_input = st.text_area("Enter your message", "")

# Language selection dropdown
selected_language = st.selectbox("Select language", ["English", "Spanish", "French"])

# Button to trigger spam detection
if st.button("Detect Spam"):
    if user_input:
        # Call your multilingual spam detection function here
        is_spam = detect_multilingual_spam(user_input, selected_language)
        if is_spam:
            st.error("This message is identified as spam.")
        else:
            st.success("This message is not spam.")
    else:
        st.warning("Please enter a message.")

# Example function (replace with your actual detection logic)
def detect_multilingual_spam(message, language):
    # Placeholder logic
    if len(message) > 50:
        return True
    return False
