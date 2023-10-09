import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

# Load the pre-trained TF-IDF vectorizer
tfidf_vectorizer = joblib.load('models/emailTfidfVectorizer.sav')

# Load the pre-trained ensemble model
ensemble_model = joblib.load('models/emailmodel.sav')

# Preprocess the text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])

    # Tokenization (split text into words)
    tokens = word_tokenize(text)

    # Remove stopwords (common words like "the", "and", "is")
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Rejoin tokens into a single string
    text = ' '.join(tokens)

    return text

# Create the Streamlit app
def email_main():
    st.title("Spam Detection in GMail")
    image = Image.open('images/sms.png')

    # columns
    # no inputs from the user
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Spam Detection in GMail', width=200)
    with col2:
        # Input text box
        user_input = st.text_area("Enter a GMail:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a GMail.")
        else:
            # Preprocess the user input
            preprocessed_input = preprocess_text(user_input)

            # Vectorize the preprocessed input using the pre-trained TF-IDF vectorizer
            input_vector = tfidf_vectorizer.transform([preprocessed_input])

            # Make predictions using the pre-trained ensemble model
            ensemble_predictions = predict_ensemble(input_vector)

            # Display the prediction result
            st.subheader("Prediction Result:")
            if ensemble_predictions == 1:
                st.error("Spam")
            else:
                st.success("Not Spam")

# Make predictions using the pre-trained ensemble model
def predict_ensemble(input_vector):
    ensemble_predictions = ensemble_model.predict(input_vector)
    return ensemble_predictions.astype(int)
