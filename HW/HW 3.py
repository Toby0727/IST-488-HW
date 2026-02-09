import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Page config
st.set_page_config(page_title="Lab 3: Streaming Chatbot", initial_sidebar_state="expanded")

# ===== BIG TITLE =====
st.title("CHURCH BOT 🤖⛪️")
st.markdown("---")

# ===== DESCRIPTION =====
st.write("""
### 📚 How This Chatbot Works

**Church Bot** is an educational assistant designed to explain topics in a way that 10-year-olds can understand!

**Conversation Memory:**
- I use a **message-based buffer** that keeps the last 6 messages (3 exchanges)
- This means I remember your recent questions but forget older ones to save on processing
- The system prompt (my personality and instructions) is always kept, so I stay consistent!

**Context from URL:**
- You can provide a URL in the sidebar
- I'll read the content and use it to answer your questions
- This context is always included, even when buffering older messages

Ask me anything! 🎓
""")

st.markdown("---")

# ===== HELPER FUNCTION: READ URL =====
def read_url_content(url):
    """Read and extract text content from a URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit to 5000 characters
        
    except requests.RequestException as e:
        return f"Error: {str(e)}"

# ===== SIDEBAR: URL INPUT =====
st.sidebar.subheader("📎 URL Context (Optional)")

url_1 = st.sidebar.text_input(
    "Enter URL 1:",
    placeholder="https://example.com/article-1",
    help="Optional first URL for context"
)

url_2 = st.sidebar.text_input(
    "Enter URL 2:",
    placeholder="https://example.com/article-2",
    help="Optional second URL for context"
)

st.sidebar.divider()

# Combine URLs (use one or both if provided)
urls = [u for u in (url_1, url_2) if u]


# ===== LLM VENDOR SELECTION =====
st.sidebar.title("Settings")
st.sidebar.subheader("🤖 AI Model Selection")

llm_choice = st.sidebar.selectbox(
    "Select AI Vendor:",
    options=["OpenAI", "Gemini"],
    index=0
)

# Advanced model toggle
use_advanced = st.sidebar.checkbox(
    "Use Advanced Model",
    value=False,
    help="Use premium models (GPT-4o or Gemini-3-Flash-Preview)"
)

# Display which model will be used
if llm_choice == "OpenAI":
    if use_advanced:
        model_display = "gpt-4o (Premium)"
        model_option = "gpt-4o"
    else:
        model_display = "gpt-3.5-turbo (Standard)"
        model_option = "gpt-3.5-turbo"
elif llm_choice == "Gemini":
    if use_advanced:
        model_display = "gemini-3-flash-preview (Premium)"
        model_option = "gemini-3-flash-preview"
    else:
        model_display = "gemini-2.5-flash (Standard)"
        model_option = "gemini-2.5-flash"

st.sidebar.info(f"**Current Model:** {model_display}")

st.sidebar.divider()

# ===== BUFFER CONFIG =====
st.sidebar.subheader("💾 Conversation Memory")
buffer_size = 3  # 3 exchanges = 6 messages
st.sidebar.write("**Keeping last 3 exchanges (6 messages)**")

# ===== HELPER FUNCTIONS =====
def count_tokens_approximate(messages):
    """Approximate token count: 1 token ≈ 4 characters"""
    total_chars = 0
    for message in messages:
        total_chars += len(message.get("role", ""))
        total_chars += len(message.get("content", ""))
        total_chars += 20
    return total_chars // 4

def get_buffered_messages(all_messages, buffer_size=3):
    """
    Keep system prompt + last N user/assistant message pairs
    System prompt is ALWAYS kept!
    """
    if len(all_messages) == 0:
        return []
    
    # Extract system prompt (should be first message)
    system_prompt = all_messages[0] if all_messages[0]["role"] == "system" else None
    
    # Get conversation messages (everything after system prompt)
    conversation = all_messages[1:] if system_prompt else all_messages
    
    # If conversation is short, return system + all conversation
    if len(conversation) <= buffer_size * 2:
        return [system_prompt] + conversation if system_prompt else conversation
    
    # Keep only last (buffer_size * 2) conversation messages
    buffered_conversation = conversation[-(buffer_size * 2):]
    
    # Return system prompt + buffered conversation
    return [system_prompt] + buffered_conversation if system_prompt else buffered_conversation

# ===== BUILD SYSTEM PROMPT WITH URL CONTEXT =====
base_instructions = """You are a helpful educational assistant that explains things in a way that 10-year-olds can understand.

After answering each question:
1. Give a clear, simple answer that a 10-year-old can understand
2. Vary how you offer more info - don't always use the same phrase

Ways to offer more info (mix these up):
- "Want to know more about this?"
- "Should I explain that part in more detail?"
- "Curious about how that works?"
- "There's more cool stuff about this - interested?"
- "I can tell you more if you'd like!"
- Sometimes just end naturally without always asking

If the user wants more information:
- Provide additional details in a friendly, conversational way
- Keep it simple and fun for a 10-year-old

If the user doesn't want more or changes topic:
- Be friendly and ready for their next question
- Use emojis to keep it fun and engaging!

Remember: Always use simple words and fun examples that kids can relate to!"""

# Read URL content if provided
url_context = ""

if urls:
    with st.spinner("📖 Reading content from URL(s)..."):
        url_sections = []
        for index, url in enumerate(urls, start=1):
            content = read_url_content(url)
            if not content.startswith("Error"):
                url_sections.append(f"URL {index}: {url}\n{content}")
            else:
                st.sidebar.error(f"❌ {content}")
        if url_sections:
            url_context = "\n\nCONTEXT FROM URL(S):\n" + "\n\n".join(url_sections) + "\n"
            st.sidebar.success("✅ URL loaded successfully!")

# Combine instructions with URL context
if url_context:
    system_content = base_instructions + url_context + "\n\nUse the context from the URL to answer questions when relevant, but keep explanations simple for 10-year-olds."
else:
    system_content = base_instructions

# Create system prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": system_content
}

# Initialize session state WITH system prompt
if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT]

# Display all previous messages (skip system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't show system prompt to user
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask me anything!"):
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Create buffered messages (ALWAYS includes system prompt)
    buffered_messages = get_buffered_messages(st.session_state.messages, buffer_size)
    
    # Display statistics
    tokens_in_buffer = count_tokens_approximate(buffered_messages)
    total_tokens = count_tokens_approximate(st.session_state.messages)
    
    st.sidebar.divider()
    st.sidebar.write("**📊 Buffer Statistics:**")
    st.sidebar.write(f"Messages in buffer: {len(buffered_messages)}")
    st.sidebar.write(f"Total messages: {len(st.session_state.messages)}")
    st.sidebar.write(f"Approx tokens in buffer: ~{tokens_in_buffer}")
    st.sidebar.write(f"Approx total tokens: ~{total_tokens}")
    st.sidebar.write(f"System prompt included: ✅")
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        try:
            if llm_choice == "OpenAI":
                # Initialize OpenAI client
                openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
                if not openai_api_key or not openai_api_key.strip():
                    st.error("❌ OpenAI API key is missing or invalid")
                    response = ""
                else:
                    client = OpenAI(api_key=openai_api_key)
                    
                    stream = client.chat.completions.create(
                        model=model_option,
                        messages=buffered_messages,
                        stream=True
                    )
                    response = st.write_stream(stream)
            
            elif llm_choice == "Gemini":
                # Initialize Gemini
                gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
                if not gemini_api_key or not gemini_api_key.strip():
                    st.error("❌ Gemini API key is missing or invalid")
                    response = ""
                else:
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel(model_option)
                    
                    # Convert messages to Gemini format (skip system message)
                    gemini_messages = []
                    system_content = ""
                    
                    for msg in buffered_messages:
                        if msg["role"] == "system":
                            system_content = msg["content"]
                        elif msg["role"] == "user":
                            gemini_messages.append({
                                "role": "user",
                                "parts": [msg["content"]]
                            })
                        elif msg["role"] == "assistant":
                            gemini_messages.append({
                                "role": "model",
                                "parts": [msg["content"]]
                            })
                    
                    # Prepend system instructions to first user message
                    if gemini_messages and system_content:
                        gemini_messages[0]["parts"][0] = f"{system_content}\n\n{gemini_messages[0]['parts'][0]}"
                    
                    # Create chat and send message
                    chat = model.start_chat(history=gemini_messages[:-1])
                    gemini_response = chat.send_message(gemini_messages[-1]["parts"][0])
                    response = gemini_response.text
                    st.markdown(response)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            response = ""
    
    # Save assistant response (only if not empty)
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})