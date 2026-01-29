import streamlit as st

secret_key = st.secrets.OPENAI_API_KEY

from openai import OpenAI
from PyPDF2 import PdfReader

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text




st.title("📄 Document Question Answering")

openai_api_key = secret_key


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


