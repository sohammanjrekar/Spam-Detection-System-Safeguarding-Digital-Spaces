import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image

# Set the title and description
st.title("Spam Detection Project Showcase")
st.write("Explore different spam detection project ideas using Streamlit.")

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Spam detection', [
    "Multilingual Spam Detection",
    "Spam Detection in Code Repositories",
    "Image Spam Detection",
    "Temporal Analysis of Spam Patterns",
    "Spam Detection in Voice Messages",
    "Spam Detection in Social Media Comments",
    "Enhanced Email Spam Detection",
    "Spam Detection in Video Comments",
    "Chatbot-based Spam Detection",
    "Deep Learning for Spam Detection",
    "Spam Detection for Medical Texts",
    "Spam Detection in Fake News",
    "Spam Detection in IoT Data Streams",
    "User-Specific Spam Filtering",
    "Adversarial Spam Detection"
],
        icons=['activity', 'heart', 'person','list-task'],
        default_index=0)



if option_menu == "Multilingual Spam Detection":
    # Insert the code template for Multilingual Spam Detection here
    pass
elif option_menu == "Spam Detection in Code Repositories":
    # Insert the code template for Spam Detection in Code Repositories here
    pass
elif option_menu == "Image Spam Detection":
    # Insert the code template for Image Spam Detection here
    pass
elif option_menu == "Temporal Analysis of Spam Patterns":
    # Insert the code template for Temporal Analysis of Spam Patterns here
    pass
elif option_menu == "Spam Detection in Voice Messages":
    # Insert the code template for Spam Detection in Voice Messages here
    pass
elif option_menu == "Spam Detection in Social Media Comments":
    # Insert the code template for Spam Detection in Social Media Comments here
    pass
elif option_menu == "Enhanced Email Spam Detection":
    # Insert the code template for Enhanced Email Spam Detection here
    pass
elif option_menu == "Spam Detection in Video Comments":
    # Insert the code template for Spam Detection in Video Comments here
    pass
elif option_menu == "Chatbot-based Spam Detection":
    # Insert the code template for Chatbot-based Spam Detection here
    pass
elif option_menu == "Deep Learning for Spam Detection":
    # Insert the code template for Deep Learning for Spam Detection here
    pass
elif option_menu == "Spam Detection for Medical Texts":
    # Insert the code template for Spam Detection for Medical Texts here
    pass
elif option_menu == "Spam Detection in Fake News":
    # Insert the code template for Spam Detection in Fake News here
    pass
elif option_menu == "Spam Detection in IoT Data Streams":
    # Insert the code template for Spam Detection in IoT Data Streams here
    pass
elif option_menu == "User-Specific Spam Filtering":
    # Insert the code template for User-Specific Spam Filtering here
    pass
elif option_menu == "Adversarial Spam Detection":
    # Insert the code template for Adversarial Spam Detection here
    pass
else:
    st.write("Select a project from the sidebar to get started.")