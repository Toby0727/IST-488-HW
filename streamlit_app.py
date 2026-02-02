import streamlit as st
lab1 = st.Page('labs/lab1.py' , title = 'lab 1')
lab2 = st.Page('labs/lab2.py' , title = 'lab 2', default=True)
pg = st.navigation([lab1, lab2])
st.set_page_config(page_title= 'IST 488 Lab', initial_sidebar_state='expanded')

import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None