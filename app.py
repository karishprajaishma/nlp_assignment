import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import soundfile as sf
# import asyncio

# from gtts import gTTS
import os
import torch
import pandas as pd
# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


torch.classes.__path__ = []
torch.classes.__path__ = []

def encode_special_characters(text):
    encoded_text = ''
    special_characters = {'&': '%26', '=': '%3D', '+': '%2B', ' ': '%20'}
    for char in text.lower():
        encoded_text += special_characters.get(char, char)
    return encoded_text

def fetch_news_articles(company_name):
    query_encoded = encode_special_characters(company_name)
    url = f"https://news.google.com/search?q={query_encoded}&hl=en-US&gl=US&ceid=US%3Aen"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = soup.find_all('article')
    links = [article.find('a')['href'] for article in articles]
    links = [link.replace("./articles/", "https://news.google.com/articles/") for link in links]

    news_text = [article.get_text(separator='\n') for article in articles]
    news_text_split = [text.split('\n') for text in news_text]

    news_df = pd.DataFrame({
        'Title': [text[2] for text in news_text_split],
        'Source': [text[0] for text in news_text_split],
        'Time': [text[3] if len(text) > 3 else 'Missing' for text in news_text_split],
        'Author': [text[4].split('By ')[-1] if len(text) > 4 else 'Missing' for text in news_text_split],
        'Link': links
    })
    
    return news_df

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(text)

# Streamlit app
st.title("News Summarization and Text-to-Speech Application")

company_name = st.text_input("Enter the company name:")
if company_name:
    news_articles = fetch_news_articles(company_name)
    
    # Limit the number of articles to 10 or fewer
    news_articles = news_articles.head(10)
    
    for index, article in news_articles.iterrows():
        st.write(f"**Title:** {article['Title']}")
        st.write(f"**Source:** {article['Source']}")
        st.write(f"**Time:** {article['Time']}")
        st.write(f"**Author:** {article['Author']}")
        st.write(f"**Link:** {article['Link']}")
        
        # Analyze sentiment of the title
        sentiment = analyze_sentiment(article['Title'])
        st.write(f"**Sentiment:** {sentiment[0]['label']} (Confidence: {sentiment[0]['score']:.2f})")
        
        # Add a button for text-to-speech
        if st.button(f"Read Aloud: {article['Title']}", key=f"read_aloud_{index}"):
            st.audio(f"https://translate.google.com/translate_tts?ie=UTF-8&q={article['Title']}&tl=en&client=tw-ob", format='audio/mp3')
        
        st.write("---")  # Separator between articles