import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests

secret_key = st.secrets.OPENAI_API_KEY
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None
    
st.title("🌐 URL Summarizer")

openai_api_key = secret_key
client = OpenAI(api_key=openai_api_key)

url_input = st.text_input("Enter a URL to summarize")

# Sidebar controls for summary type and model selection
st.sidebar.header("Summary Options")
summary_type = st.sidebar.selectbox(
    "Type of summary",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ],
)

use_advanced = st.sidebar.checkbox("Use advanced model")

generate = st.sidebar.button("Generate Summary")

if generate and url_input:
    document = read_url_content(url_input)
    if document:
        # Choose model based on user selection
        if use_advanced:
            model = "gpt-4o"
        else:
            model = 'gpt-3.5-turbo'

        # Include the summary type explicitly in the LLM instructions
        instruction = (
            f"{summary_type}. Provide the summary only and do not include the original document text."
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
            {"role": "user", "content": f"{instruction}\n\nDocument:\n\n{document}"},
        ]

        with st.spinner("Generating summary..."):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

        summary = response.choices[0].message.content

        st.subheader("Summary")
        st.write(summary)
    else:
        st.error("Could not read the URL. Please check the URL and try again.")
elif generate and not url_input:
    st.error("Please enter a URL to summarize.")



