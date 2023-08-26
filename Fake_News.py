import streamlit as st

st.title("Spam Detection in Fake News")
st.write("Develop a system to identify spammy or sensationalist news articles.")

# Text input for user to enter news article
user_article = st.text_area("Enter news article", "")

# Button to trigger spam detection
if st.button("Detect Spam"):
    if user_article:
        # Call your fake news detection function here
        is_spam = detect_fake_news(user_article)
        if is_spam:
            st.error("This news article is identified as spam.")
        else:
            st.success("This news article is not spam.")
    else:
        st.warning("Please enter a news article.")

# Example function (replace with your actual detection logic)
def detect_fake_news(article):
    # Placeholder logic
    if "shocking revelation" in article.lower():
        return True
    return False
