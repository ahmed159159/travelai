# import library
import streamlit as st
import pandas as pd
import json
import requests
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import folium
from streamlit_folium import folium_static

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()
DOBBY_API_KEY = os.getenv('DOBBY_API_KEY')
FOURSQUARE_API_KEY = os.getenv('FOURSQUARE_API_KEY')

# Foursquare base URL
FOURSQUARE_URL = "https://api.foursquare.com/v3/places/search"

# Dobby query function
def dobby_query(prompt, api_key=DOBBY_API_KEY):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "accounts/fireworks/models/sentientfoundation/dobby-unhinged-llama-3-3-70b-new",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()['choices'][0]['message']['content']

# Foursquare search function
def foursquare_search(query, lat, lon, radius=3000):
    headers = {
        "accept": "application/json",
        "Authorization": FOURSQUARE_API_KEY,
        "X-Places-Api-Version": "1970-01-01"
    }
    params = {
        "query": query,
        "ll": f"{lat},{lon}",
        "radius": radius,
        "limit": 20
    }
    response = requests.get(FOURSQUARE_URL, headers=headers, params=params)
    result = response.json()
    df = pd.json_normalize(result['results'])
    return df

# Main App
def main():
    st.sidebar.title("Travel Recommendation App Demo")
    destination = st.sidebar.text_input('Destination:')
    min_rating = st.sidebar.number_input('Minimum Rating:', value=4.0, min_value=0.5, max_value=5.0, step=0.5)
    radius = st.sidebar.number_input('Search Radius in meter:', value=3000, min_value=500, max_value=50000, step=100)

    if destination:
        # Get coordinates from Foursquare for the city
        df_location = foursquare_search(destination, lat=0, lon=0, radius=10000)  # lat/lon dummy for initial lookup
        if df_location.empty:
            st.error("Destination not found")
            return
        initial_latitude = df_location['geocodes.main.latitude'].iloc[0]
        initial_longitude = df_location['geocodes.main.longitude'].iloc[0]

        # Search different categories
        df_hotel = foursquare_search(f'Hotel near {destination}', initial_latitude, initial_longitude, radius)
        df_hotel['type'] = 'Hotel'
        df_restaurant = foursquare_search(f'Restaurant near {destination}', initial_latitude, initial_longitude, radius)
        df_restaurant['type'] = 'Restaurant'
        df_tourist = foursquare_search(f'Tourist attraction near {destination}', initial_latitude, initial_longitude, radius)
        df_tourist['type'] = 'Tourist'

        df_place = pd.concat([df_hotel, df_restaurant, df_tourist], ignore_index=True)

        # Preprocess dataframe for display
        df_place['combined_info'] = df_place.apply(lambda row: f"Type: {row['type']}, Name: {row['name']}. Rating: {row.get('rating', 'N/A')}. Address: {row.get('location.formatted_address', '')}.", axis=1)

        # Vectorstore for retrieval
        loader = DataFrameLoader(df_place, page_content_column='combined_info')
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2", model_kwargs={'device':'cpu'})
        vectorstore = FAISS.from_documents(texts, embeddings)

        def maps():
            st.header("🌏 Travel Recommendation App 🌏")
            places_type = st.radio('Looking for: ', ["Hotels 🏨", "Restaurants 🍴", "Tourist Attractions ⭐"])
            type_colour = {'Hotel':'blue', 'Restaurant':'green', 'Tourist':'orange'}
            type_icon = {'Hotel':'home', 'Restaurant':'cutlery', 'Tourist':'star'}

            if places_type == 'Hotels 🏨': df_show = df_hotel
            elif places_type == 'Restaurants 🍴': df_show = df_restaurant
            else: df_show = df_tourist

            for index, row in df_show.iterrows():
                location = [row['geocodes.main.latitude'], row['geocodes.main.longitude']]
                mymap = folium.Map(location=[initial_latitude, initial_longitude], zoom_start=12)
                content = f"{row['name']}<br>Rating: {row.get('rating', 'N/A')}<br>Address: {row.get('location.formatted_address', '')}"
                iframe = folium.IFrame(content, width=300, height=125)
                popup = folium.Popup(iframe, max_width=300)
                icon = folium.Icon(color=type_colour[row['type']], icon=type_icon[row['type']])
                folium.Marker(location=location, popup=popup, icon=icon).add_to(mymap)
                st.write(f"## {index + 1}. {row['name']}")
                folium_static(mymap)

        def chatbot():
            class Message(BaseModel):
                actor: str
                payload: str

            USER = "user"
            ASSISTANT = "ai"
            MESSAGES = "messages"
            if MESSAGES not in st.session_state:
                st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

            for msg in st.session_state[MESSAGES]:
                st.chat_message(msg.actor).write(msg.payload)

            query = st.chat_input("Enter a prompt here")
            if query:
                st.session_state[MESSAGES].append(Message(actor=USER, payload=query))
                st.chat_message(USER).write(query)

                # Use Dobby to answer
                context_text = '\n'.join(df_place['combined_info'].tolist())
                prompt = f"Context: {context_text}\nUser question: {query}\nAnswer with 3 recommendations, addresses, and websites sorted by rating."
                response = dobby_query(prompt)

                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
                st.chat_message(ASSISTANT).write(response)

        method = st.sidebar.radio(" ", ["Search 🔎", "ChatBot 🤖"], key="method_app")
        if method == "Search 🔎":
            maps()
        else:
            chatbot()

if __name__ == '__main__':
    main()
