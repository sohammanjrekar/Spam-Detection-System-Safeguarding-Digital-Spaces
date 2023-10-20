import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image

# Load the pre-trained TF-IDF vectorizer
tfidf_vectorizer = joblib.load('models/jobTfidfVectorizer.sav')

# Load the pre-trained ensemble model
ensemble_model = joblib.load('models/jobmodel.sav')

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
def main():
    st.title("Spam Detection in Job Post")
    image = Image.open('images/sms.png')

    # columns
    # no inputs from the user
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption='Spam Detection in Job Post', width=200)
        # Input text boxes
        description = st.text_area("Enter Job Description:", height=100)
        company_profile = st.text_area("Enter Company Profile:", height=100)
        requirements = st.text_area("Enter Job Requirements:", height=100)

    if st.button("Predict"):
        if description.strip() == "" or company_profile.strip() == "" or requirements.strip() == "":
            st.warning("Please enter values for all input fields.")
        else:
            # Concatenate user inputs into a single text
            user_input = f"{description} {company_profile} {requirements}"

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

if __name__ == '__main__':
    main()
