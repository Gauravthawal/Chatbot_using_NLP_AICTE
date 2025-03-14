import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Custom Streamlit Styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #1e1e2f;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .chat-container {
            background: #292b3a;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.3);
            margin-bottom: 15px;
        }
        .user-message {
            text-align: right;
            color: #ffffff;
            background: #0078D7;
            padding: 12px;
            border-radius: 12px;
            margin: 5px 0;
            font-weight: bold;
        }
        .bot-message {
            text-align: left;
            color: #ffffff;
            background: #44476a;
            padding: 12px;
            border-radius: 12px;
            margin: 5px 0;
            font-weight: bold;
        }
        .sidebar .sidebar-content {
            background-color: #282a36;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load intents from the JSON file
file_path = "intents.json"
if not os.path.exists(file_path):
    st.error("Error: 'intents.json' file not found.")
    st.stop()

with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent.get('patterns', []):
        tags.append(intent['tag'])
        patterns.append(pattern)

# Ensure patterns are not empty before fitting
if patterns:
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)
else:
    st.error("Error: No patterns found in 'intents.json'.")
    st.stop()

def chatbot(input_text):
    if not input_text.strip():
        return "Please enter a valid message."
    input_text_vectorized = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vectorized)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

st.sidebar.title("🔹 Chatbot Options")
menu = ["Home", "Chat History", "About"]
choice = st.sidebar.radio("Navigate", menu)

if choice == "Home":
    st.title("🤖 AI Chatbot")
    st.markdown("### Welcome! Type your message below:")
    
    chat_history = st.container()
    with chat_history:
        user_input = st.text_input("You:", "", key="input")
        if user_input:
            response = chatbot(user_input)
            st.markdown(f'<div class="chat-container"><div class="user-message">{user_input}</div><div class="bot-message">{response}</div></div>', unsafe_allow_html=True)

elif choice == "Chat History":
    st.title("📜 Conversation History")
    if os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "r", encoding="utf-8") as file:
            history = csv.reader(file)
            for row in history:
                if len(row) == 3:
                    st.markdown(f'<div class="chat-container"><div class="user-message">User: {row[0]}</div><div class="bot-message">Chatbot: {row[1]}</div></div>', unsafe_allow_html=True)
    else:
        st.warning("No conversation history found.")

elif choice == "About":
    st.title("ℹ️ About This Chatbot")
    st.write("This chatbot uses NLP and Logistic Regression to respond intelligently to user queries.")
    st.write("It features a sleek and modern UI with an interactive chat experience.")
    st.write("--GRT")
