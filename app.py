import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = { value: key for key, value in word_index.items()}

## load pre-trained model with ReLu activation

model = load_model('simple_rnn_imdb.h5')

## 1. Decode reviews
def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

## 2. Function to preprocess the user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## prediction function
def predict_sentiment(review):
    # preprocessed = preprocess_text(review)
    # print(preprocessed)
    prediction = model.predict(review)
    print(prediction)
    sentiment = 'Positive' if prediction > 0.8 else 'Negative'
    
    return sentiment

import streamlit as st

st.title('IMDb Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')
if st.button('Classify'):
    with st.spinner('Classifying...'):
        processed_input = preprocess_text(user_input)
        sentiment = predict_sentiment(processed_input)
    if(sentiment=='Positive'):
        st.success(f'Sentiment: {sentiment}')
    else:
        st.error(f'Sentiment: {sentiment}')
else:
    st.write('Please enter a movie review.')