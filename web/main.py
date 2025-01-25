import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('model.h5')  
tokenizer = joblib.load('tokenizer.pkl')  

# Function to make predictions
def predict_sentiment(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200) # Adjust maxlen if needed
    prediction = model.predict(padded_sequence)
    if prediction > 0.5:
        return "Positive Review 😁"
    else:
        return "Negative Review 😒"

# Streamlit app layout
st.title("TMDB Sentiment Analysis using LSTM")
st.write("Enter a movie review to get its sentiment:")

user_input = st.text_area("Enter review here")
if st.button("Predict"):
    sentiment = predict_sentiment(user_input)
    st.write(f"Sentiment: **{sentiment}**")
