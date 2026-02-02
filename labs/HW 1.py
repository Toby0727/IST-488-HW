import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def validate_api_key(api_key):
    """Validate the API key by making a test call"""
    if not api_key:
        st.session_state.api_key_valid = False
        return
    
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        st.session_state.api_key_valid = True
        st.success("✅ API key is valid!", icon="✓")
    except Exception as e:
        st.session_state.api_key_valid = False
        st.error(f"❌ Invalid API key: {str(e)}")


st.title("📄 Document Question Answering")

openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    on_change=lambda: validate_api_key(st.session_state.openai_api_key),
    key="openai_api_key",
)

# Initialize session state if needed
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

if not st.session_state.api_key_valid:
    st.info("Please enter a valid OpenAI API key to continue.", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

uploaded_file = st.file_uploader(
    "Upload a document (.txt or .pdf)", type=("txt", "pdf")
)

question = st.text_area(
    "Ask a question about the document",
    disabled=not uploaded_file,
)

if uploaded_file and question:

    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "txt":
        document = uploaded_file.read().decode("utf-8")

    elif file_extension == "pdf":
        document = read_pdf(uploaded_file)

    else:
        st.error("Unsupported file type.")
        st.stop()

    messages = [
        {
            "role": "user",
            "content": f"Here is the document:\n\n{document}\n\n---\n\n{question}",
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    st.write(response.choices[0].message.content)


