import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai

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

# LLM selection dropdown
llm_choice = st.sidebar.selectbox(
    "Select LLM",
    ["OpenAI", "Gemini"],
    key="llm_selection"
)

# Language selection dropdown
language = st.sidebar.selectbox(
    "Select Output Language",
    ["English", "Chinese", "French", "Turkish"],
    key="output_language"
)

generate = st.sidebar.button("Generate Summary")

if generate and url_input:
    with st.spinner("📖 Reading URL content..."):
        document = read_url_content(url_input)
    
    if document:
        # Include the summary type explicitly in the LLM instructions
        instruction = (
            f"{summary_type}. Provide the summary only and do not include the original document text. Output the summary in {language}."
        )

        try:
            with st.spinner(f"✨ Generating summary with {llm_choice}..."):
                if llm_choice == "OpenAI":
                    # Load and validate OpenAI key
                    openai_api_key = st.secrets.OPENAI_API_KEY
                    if not openai_api_key or not openai_api_key.strip():
                        raise ValueError("OpenAI API key is empty or invalid")
                    client = OpenAI(api_key=openai_api_key)
                    if use_advanced:
                        model = "gpt-4o"
                    else:
                        model = 'gpt-3.5-turbo'

                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                        {"role": "user", "content": f"{instruction}\n\nDocument:\n\n{document}"},
                    ]

                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                    summary = response.choices[0].message.content

                elif llm_choice == "Gemini":
                    # Load and validate Gemini key
                    gemini_api_key = st.secrets.GEMINI_API_KEY
                    if not gemini_api_key or not gemini_api_key.strip():
                        raise ValueError("Gemini API key is empty or invalid")
                    genai.configure(api_key=gemini_api_key)
                    
                    # Choose model based on user selection
                    if use_advanced:
                        model_name = "gemini-2.0-flash"
                    else:
                        model_name = "gemini-1.5-flash"
                    
                    model = genai.GenerativeModel(model_name)
                    
                    prompt = f"{instruction}\n\nDocument:\n\n{document}"
                    response = model.generate_content(prompt)
                    summary = response.text

                st.subheader("Summary")
                st.write(summary)
        except KeyError as e:
            st.error(f"Missing API key: {e}. Please configure the {llm_choice} API key in Streamlit secrets.")
        except ValueError as e:
            st.error(f"Invalid API key: {e}")
    else:
        st.error("Could not read the URL. Please check the URL and try again.")
elif generate and not url_input:
    st.error("Please enter a URL to summarize.")



