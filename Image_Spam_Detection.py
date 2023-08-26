import streamlit as st
from PIL import Image

st.title("Image Spam Detection")
st.write("Develop an algorithm to detect spam in image files.")

# Upload image
image = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

# Button to trigger spam detection
if st.button("Detect Spam"):
    if image is not None:
        image = Image.open(image)
        # Call your image spam detection function here
        is_spam = detect_image_spam(image)
        if is_spam:
            st.error("This image contains spam content.")
        else:
            st.success("This image is clean.")
    else:
        st.warning("Please upload an image.")

# Example function (replace with your actual detection logic)
def detect_image_spam(image):
    # Placeholder logic
    if image.width > 800 or image.height > 800:
        return True
    return False
