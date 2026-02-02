import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai
from anthropic import Anthropic

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

# LLM selection dropdown
llm_choice = st.sidebar.selectbox(
    "Select LLM",
    ["OpenAI", "Gemini", "Claude"],
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
    document = read_url_content(url_input)
    if document:
        # Validate API key before running
        try:
            if llm_choice == "OpenAI":
                st.secrets.OPENAI_API_KEY
            elif llm_choice == "Gemini":
                st.secrets.GEMINI_API_KEY
            elif llm_choice == "Claude":
                st.secrets.CLAUDE_API_KEY
        except KeyError:
            st.error(f"Missing API key for {llm_choice}. Please configure the {llm_choice} API key in Streamlit secrets.")
        else:
            # Include the summary type explicitly in the LLM instructions
            instruction = (
                f"{summary_type}. Provide the summary only and do not include the original document text. Output the summary in {language}."
            )

            with st.spinner("Generating summary..."):
                if llm_choice == "OpenAI":
                    # Choose model based on user selection
                    if use_advanced:
                        model = "gpt-4o"
                    else:
                        model = 'gpt-3.5-turbo'

                    openai_api_key = st.secrets.OPENAI_API_KEY
                    client = OpenAI(api_key=openai_api_key)

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
                    genai.configure(api_key=st.secrets.GEMINI_API_KEY)
                    
                    # Choose model based on user selection
                    if use_advanced:
                        model_name = "gemini-2.0-flash"
                    else:
                        model_name = "gemini-1.5-flash"
                    
                    model = genai.GenerativeModel(model_name)
                    
                    prompt = f"{instruction}\n\nDocument:\n\n{document}"
                    response = model.generate_content(prompt)
                    summary = response.text

                elif llm_choice == "Claude":
                    client_claude = Anthropic(api_key=st.secrets.CLAUDE_API_KEY)
                    
                    # Choose model based on user selection
                    if use_advanced:
                        model_name = "claude-3-opus-20250219"
                    else:
                        model_name = "claude-3-5-sonnet-20241022"
                    
                    message = client_claude.messages.create(
                        model=model_name,
                        max_tokens=1024,
                        messages=[
                            {"role": "user", "content": f"{instruction}\n\nDocument:\n\n{document}"}
                        ]
                    )
                    summary = message.content[0].text

            st.subheader("Summary")
            st.write(summary)
    else:
        st.error("Could not read the URL. Please check the URL and try again.")
elif generate and not url_input:
    st.error("Please enter a URL to summarize.")



