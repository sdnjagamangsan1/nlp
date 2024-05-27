import streamlit as st
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Define the text data and their corresponding sentiment labels
text_data = [
    "I love the new iPhone 13. The camera is amazing!",
    "The customer service at this store is terrible.",
    "I don't have any opinion on this laptop brand."
]
sentiment_labels = ["positive", "negative", "neutral"]

# Process the text data with the NER model and extract named entities
named_entities_data = []
for text in text_data:
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    named_entities_data.append(' '.join(named_entities))

# Create a count vectorizer for named entities
vectorizer = CountVectorizer()
vectorizer.fit(named_entities_data)
X = vectorizer.transform(named_entities_data)

# Create a logistic regression model for sentiment analysis
lr = LogisticRegression()
lr.fit(X, sentiment_labels)

# Streamlit UI
st.title('Aplikasi Praktikum NLP Ida Hafizah')
st.write('Selamat datang di aplikasi praktikum berbasis Streamlit!')

# Input untuk nama
name = st.text_input('Masukkan nama Anda:')
if name:
    st.write(f'Halo, {name}!')

# Input untuk teks analisis sentimen
user_input = st.text_area("Masukkan kalimat untuk analisis sentimen:")

if st.button("Analisis"):
    if user_input:
        # Process the user input with the NER model
        doc = nlp(user_input)
        named_entities = [ent.text for ent in doc.ents]
        user_named_entities_data = ' '.join(named_entities)
        
        # Transform the user input to the vectorizer format
        user_X = vectorizer.transform([user_named_entities_data])
        
        # Predict the sentiment
        prediction = lr.predict(user_X)
        
        # Display the prediction
        st.write(f"Prediksi Sentimen: {prediction[0]}")
    else:
        st.write("Silakan masukkan kalimat untuk analisis sentimen.")

# Contoh Prediksi Sentimen untuk Teks Baru
st.write("Berikut Contoh Prediksi Sentimen yang dapat dicoba:")
new_text_data = [
    "I love the new iPhone 13. The camera is amazing!",
    "The customer service at this store is terrible.",
    "I don't have any opinion on this laptop brand."
]

new_named_entities_data = []
for text in new_text_data:
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    new_named_entities_data.append(' '.join(named_entities))

new_X = vectorizer.transform(new_named_entities_data)
predictions = lr.predict(new_X)

for text, pred in zip(new_text_data, predictions):
    st.write(f"Teks: {text}")
