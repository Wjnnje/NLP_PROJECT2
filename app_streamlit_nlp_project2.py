import streamlit as st
from transformers import pipeline

# Create sentiment analysis pipeline
classification = pipeline(
    task="sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

# Streamlit App
def main():
    st.title("NLP App")

     # Sidebar with options
    page_options = ["Home", "Sentiment Prediction"]
    page = st.sidebar.selectbox("Choose a page", page_options)

    

    if page == "Home":
        st.header("Welcome to the NLP App")
        st.write("Use the sidebar to navigate to different pages.")
    
    elif page == "Sentiment Prediction":
        st.header("Sentiment Prediction Page")
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

if __name__ == "__main__":
    main()
