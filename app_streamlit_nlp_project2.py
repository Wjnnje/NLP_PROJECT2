import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load your data or use a sample dataset
# Example:
data = pd.read_csv('C:/Users/jega_/NLP_PROJECT2/df_new_format.csv')

# Assuming your data has 'comment' and 'sentiment' columns
# Example data:
new_df_clean=data[data['score'] != '']
new_df_clean['comment'] = new_df_clean['comment'].fillna('')


# Tokenize and vectorize the comments
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
text_counts = cv.fit_transform(new_df_clean['comment'])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(text_counts, new_df_clean['sentiment'], test_size=0.25, random_state=5)

# Train the model
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)

# Streamlit App
def main():
    st.title("Comment Sentiment Prediction App")

    # Sidebar with options
    page_options = ["Home", "Sentiment Prediction"]
    page = st.sidebar.selectbox("Choose a page", page_options)

    if page == "Home":
        st.header("Welcome to the Comment Sentiment Prediction App")
        st.write("Use the sidebar to navigate to different pages.")

    elif page == "Sentiment Prediction":
        st.header("Sentiment Prediction Page")
        st.write("Enter a comment below to predict its sentiment:")

        # User input for prediction
        user_input = st.text_area("Enter your comment here:")

        if st.button("Predict Sentiment"):
            # Tokenize and vectorize the user input
            user_input_counts = cv.transform([user_input])

            # Predict the sentiment
            predicted_sentiment = MNB.predict(user_input_counts)

            # Display the prediction message
            if predicted_sentiment[0] == 1:
                st.write("You wrote a positive review about this service. Thanks!")
            elif predicted_sentiment[0] == -1:
                st.write("You wrote a negative review about this service.")
            else:
                st.write("Thank you for your review.")

if __name__ == "__main__":
    main()