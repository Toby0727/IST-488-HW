from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

import chromadb

st.set_page_config(page_title="HW 5: Short-Term Memory RAG Chatbot", initial_sidebar_state="expanded")
st.title("HW 5: Short-Term Memory RAG Chatbot")
st.markdown("---")

st.write("""
### 📚 How This Chatbot Works
**Memory & Context:**
- I retrieve context **once per topic** and answer a series of follow-up questions using it
- I use a **vector database** to find relevant Syracuse University organization information
- Ask me anything and I'll keep digging deeper with you!
""")
st.markdown("---")

# ===== LOAD VECTOR DATABASE =====
db_path = Path(__file__).parent / "chroma_db"
if not db_path.exists():
    st.error("❌ Vector database not found!")
    st.stop()

try:
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_collection(name="HW4Collection")
    st.sidebar.success(f"✅ Vector DB: {collection.count()} chunks loaded")
    st.sidebar.info("🤖 Using: GPT-4o")
except Exception as e:
    st.error(f"❌ Error loading vector database: {str(e)}")
    st.stop()

st.sidebar.divider()

# ===== MEMORY BUFFER SETTINGS =====
st.sidebar.subheader("💾 Conversation Memory")
buffer_size = st.sidebar.slider("Remember last N interactions:", 1, 10, 5)
st.sidebar.write(f"**Remembering:** Last {buffer_size} exchanges ({buffer_size * 2} messages)")
st.sidebar.divider()

# ===== OPENAI CLIENT =====
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

client = st.session_state.openai_client  # reuse single instance
st.session_state.HW4_VectorDB = collection

# ===== RETRIEVE FUNCTION =====
def relevant_club_info(query, n_results=5):
    embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(docs) if docs else ""

    filenames = set(m.get("filename", "Unknown") for m in results.get("metadatas", [[]])[0])
    files_used = ", ".join(sorted(filenames)) if filenames else "None"

    return {"context": context, "files_used": files_used, "results": results}

# ===== BUFFER FUNCTION =====
def get_buffered_messages(all_messages, buffer_size=5):
    if len(all_messages) <= 1:
        return all_messages
    system_prompt = all_messages[0] if all_messages[0]["role"] == "system" else None
    conversation = all_messages[1:] if system_prompt else all_messages
    buffered = conversation[-(buffer_size * 2):]
    return ([system_prompt] + buffered) if system_prompt else buffered

# ===== SESSION STATE INIT =====
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": """You are a Syracuse University organization information assistant.

INSTRUCTIONS:
1. You will receive retrieved context from the organization database once per topic.
2. Use that context to answer the user's initial question AND any follow-up questions.
3. After answering, ALWAYS end your response by asking ONE relevant follow-up question 
   to help the user learn more about the topic — based on what's available in the context.
   Format it clearly like: "**Want to know more?** [your follow-up question here]"
4. If the user answers or asks something new, continue the conversation using the same context.
5. Only re-retrieve from the database if the user clearly changes topic.
6. Be conversational, helpful, and curious."""
    }]

if "current_context" not in st.session_state:
    st.session_state.current_context = None
if "current_files" not in st.session_state:
    st.session_state.current_files = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ===== DISPLAY CHAT HISTORY =====
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ===== CHAT INPUT =====
if prompt := st.chat_input("Ask me anything about Syracuse University organizations"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ===== DECIDE WHETHER TO RE-RETRIEVE =====
    # Re-retrieve only if no context yet, or if user seems to be changing topic
    should_retrieve = st.session_state.current_context is None

    if not should_retrieve:
        # Ask the LLM if this is a new topic or a follow-up
        topic_check = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You determine if a message is a follow-up to the previous conversation or a new topic. Reply with only 'followup' or 'newtopic'."},
                {"role": "user", "content": f"Previous context was about: {st.session_state.current_files}\n\nNew message: {prompt}"}
            ],
            max_tokens=10,
            temperature=0
        )
        decision = topic_check.choices[0].message.content.strip().lower()
        if "newtopic" in decision:
            should_retrieve = True

    # ===== RETRIEVE IF NEEDED =====
    if should_retrieve:
        with st.spinner("🔍 Searching organization database..."):
            retrieval = relevant_club_info(prompt)
            st.session_state.current_context = retrieval["context"]
            st.session_state.current_files = retrieval["files_used"]
            st.session_state.last_results = retrieval["results"]

    # ===== BUILD API MESSAGES =====
    buffered = get_buffered_messages(st.session_state.messages, buffer_size)

    context_msg = {
        "role": "system",
        "content": f"""Retrieved context from Syracuse University organization pages: {st.session_state.current_files}

{st.session_state.current_context}

Use this context to answer AND to generate a relevant follow-up question at the end of your response."""
    }

    # Fix: don't append prompt again — it's already in buffered messages
    api_messages = buffered + [context_msg]

    # ===== STREAM RESPONSE =====
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=api_messages,
            stream=True,
            temperature=0.7
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ===== SIDEBAR STATS =====
st.sidebar.divider()
st.sidebar.write("**📊 Memory Statistics:**")
total_exchanges = (len(st.session_state.messages) - 1) // 2
st.sidebar.write(f"Total exchanges: {total_exchanges}")
st.sidebar.write(f"Messages in buffer: {len(get_buffered_messages(st.session_state.messages, buffer_size)) - 1}")
st.sidebar.write(f"Total messages: {len(st.session_state.messages) - 1}")
if total_exchanges > buffer_size:
    st.sidebar.warning(f"⚠️ {total_exchanges - buffer_size} older exchanges not in active memory")

if st.session_state.current_files:
    st.sidebar.divider()
    st.sidebar.write("**📁 Current Context Source:**")
    st.sidebar.info(st.session_state.current_files)

# ===== SHOW RETRIEVED CHUNKS =====
st.sidebar.divider()
if st.sidebar.checkbox("Show retrieved documents"):
    if st.session_state.last_results:
        st.sidebar.write("### 📄 Retrieved Pages:")
        for i, meta in enumerate(st.session_state.last_results["metadatas"][0], 1):
            fname = meta.get("filename", "Unknown")
            cidx = meta.get("chunk_index", 0)
            total = meta.get("total_chunks", 1)
            st.sidebar.write(f"{i}. **{fname}** (chunk {cidx+1}/{total})")
    else:
        st.sidebar.write("No retrieval performed yet.")

# ===== CLEAR CHAT =====
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.session_state.current_context = None
    st.session_state.current_files = None
    st.session_state.last_results = None
    st.rerun()