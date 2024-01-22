import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np

from transformers import AutoModelWithLMHead, AutoTokenizer



from sklearn.svm import SVC


import pickle
# Create sentiment analysis pipeline
classification = pipeline(
    task="sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

# Load Course dataset

df=pd.read_csv("df_mergedV4.csv",sep=',')
filtered_documents=df.copy()

import string

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Add any other preprocessing steps as needed

    return text

model_filename = 'svm_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

with open(model_filename, 'rb') as model_file:
     loaded_model = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
     loaded_vectorizer = pickle.load(vectorizer_file)

def summarize(text, max_length=150):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds[0]

# Streamlit App
def main():
    st.title("Home services Application")

     # Sidebar with options
    page_options = ["Home", "Prediction","Summary","Explanation"]
    page = st.sidebar.selectbox("Choose a page", page_options)

    

    if page == "Home":
        
        st.image("https://global.hitachi-solutions.com/wp-content/uploads/2022/01/Webinar-NLP-In-RCG.png", width=750, caption="Welcome to the NLP App")
        
        st.markdown(
        """
        # Welcome to our NLP application on home services.
        In this application, you will find different use case of NLP techniques on a scrapped data base.
        Use The sidebar to navigate to different pages.

        Have Fun !

        Janany & Mathilde
        """
    )
    
    elif page == "Prediction":
        st.header("Sentiment Prediction")
        st.write("Enter a comment below to predict its sentiment:")

        # User input for prediction
        user_input = st.text_area("Enter your text here:")

        if st.button("Analyze Sentiment"):
            # Perform sentiment analysis
            result = classification(user_input)
            # Determine the overall sentiment
            overall_sentiment = max(result[0], key=lambda x: x['score'])
            if overall_sentiment['label']=='negative':
                st.write("Sorry to hear that your experience with this service was not up to your expectations. We appreciate your feedback.")
            elif overall_sentiment['label']=='positif':
                st.write("We are glad of you experience with this service. We appreciate your feedback.")
            else :
                st.write("We appreciate your feedback.")
            st.write(f"Overall Sentiment: {overall_sentiment['label']}")

            with st.expander("Show Detailed Scores"):
                st.write("Sentiment Scores:")
                for score in result[0]:
                    st.write(f"{score['label']}: {score['score']}")
        # Add rating prediction below sentiment analysis
        st.header("Prediction")
        st.write("Enter a comment below to predict its rating:")
        # User input for rating prediction
        user_input_rating = st.text_area("Enter your review here:")

        if st.button("Predict Rating"):
             # Preprocess the user input text
            user_input_rating_processed = preprocess_text(user_input_rating)

            # Vectorize the input text using the loaded TF-IDF vectorizer
            user_input_vectorized = loaded_vectorizer.transform([user_input_rating_processed])

            # Predict the rating using the loaded SVM model
            rating_prediction = loaded_model.predict([user_input_rating_processed])[0]

            st.write(f"Predicted Rating: {rating_prediction}")



    elif page == "Explanation":
        st.header("Explanation Page")
        st.write("Welcome to the Explanation Page. Here, we'll provide some details about the app and its functionality.")

        # Add explanation text
        st.subheader("How the Sentiment Prediction Works:")
        st.write("Sentiment prediction is based on the bert multilingual cased sentiment available on Hugging Face. The model compute as result a dictionnary with the sentiment label as the key and its probability as value. In our application, we select the overall sentiment.")
        
        # Add code snippets
        st.subheader("Code Snippet: Sentiment Prediction")
        st.code("""
        classification = pipeline(
                                    task="sentiment-analysis",
                                    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
                                    return_all_scores=True)

        user_input = st.text_area("Enter your text here:")
        
        if st.button("Analyze Sentiment"):
            result = classification(user_input)
            # Process the sentiment analysis results
            # Display the overall sentiment and detailed scores
            """)

        st.subheader("How the Service Retrieval Works:")
        st.write("This functionnality is based on Tf-idf vectorization of each description of services. We then compute the cosine similiarity between the user's query and the available services. Finally, this score is weighted by the average score of each services, in order to provide the best services.")
        
        # Add code snippets
        st.subheader("Code Snippet: Service Retrieval")
        st.code("""
        query_summary = st.text_area("✏️ Enter your request :")
        
        if st.button("Retrieve Services"):
            similarity_scores_with_impact, top_documents = retrieve_top_documents(query_summary, k=10)
            # Process and display the top service retrieval results
            """)

        st.subheader("How the Question Answering Chatbot Works:")
        st.write("our Chatbot relieves on the combination of TF-idf and question answering model from Hugging Face.")
        
        # Add code snippets
        st.subheader("Code Snippet: Question Answering Chatbot")
        st.code("""
        qa = pipeline('question-answering')
        prompt = st.chat_input("What is up?")
        
        if prompt and st.session_state.context_string == []:
            similarity_scores_with_impact, top_documents = retrieve_top_documents(prompt, k=10)
                # we compute top 10 results that matches the user's query using tf-idf
            context_string = create_context_string(top_documents)
                #we store those info into a context string
            st.session_state.context_string = context_string
            answer = qa(context=context_string, question=prompt)
                # using the qa model, we provide the context and user's query
            # Display the chatbot response
            """)
        
        st.subheader("How the Summary works:")
        st.write("The summary is based on")
        
        # Add code snippets
        st.subheader("Code Snippet: Review summary")
        st.code("""
        
         user_review = st.text_area("Enter your review here:")

        if st.button("Generate Summary"):
            # Perform summarization (you may need to replace this with your summarization logic)
            # For this example, let's assume a simple summary by taking the first 50 characters of the review
            summary = summarize(user_review)
            
            st.subheader("Review Summary:")
            st.write(summary)
            """)

    




        




if __name__ == "__main__":
    main()
