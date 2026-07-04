import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

tfidf = pickle.load(open('notebook/vectorizer.pkl', 'rb'))
model = pickle.load(open('notebook/model.pkl', 'rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    st.header("Spam" if result == 1 else "Not Spam")