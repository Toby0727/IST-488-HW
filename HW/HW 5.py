from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path

# Ensure ChromaDB uses a compatible SQLite
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

import chromadb

# Page config
st.set_page_config(page_title="HW 5: Short-Term Memory RAG Chatbot", initial_sidebar_state="expanded")

# ===== BIG TITLE =====
st.title("HW 5: Short-Term Memory RAG Chatbot")
st.markdown("---")

# ===== DESCRIPTION =====
st.write("""
### 📚 How This Chatbot Works

**Memory & Context:**
- I remember the **last N conversations** (2N messages: N from you, N from me)
- I use a **vector database** to find relevant Syracuse University organization information
- I combine your conversation history with retrieved documents to give better answers

**Ask me about:**
- Syracuse University student organizations
- Club activities, meetings, and events
- Organization contact information

Let's chat! 🎓
""")

st.markdown("---")

# ===== LOAD PRE-BUILT VECTOR DATABASE =====
db_path = Path(__file__).parent / "chroma_db"

if not db_path.exists():
    st.error("❌ Vector database not found! Make sure 'chroma_db' folder is deployed.")
    st.stop()

try:
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_collection(name="HW4Collection")
    
    chunk_count = collection.count()
    st.sidebar.success(f"✅ Vector DB: {chunk_count} chunks loaded")
    st.sidebar.info(f"🤖 Using: GPT-4o")
    
except Exception as e:
    st.error(f"❌ Error loading vector database: {str(e)}")
    st.info("Try rebuilding the database with build_database.py")
    st.stop()

st.sidebar.divider()

# ===== MEMORY BUFFER SETTINGS =====
st.sidebar.subheader("💾 Conversation Memory")

buffer_size = st.sidebar.slider(
    "Remember last N interactions:",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Number of conversation exchanges to remember"
)

st.sidebar.write(f"**Remembering:** Last {buffer_size} exchanges ({buffer_size * 2} messages)")

st.sidebar.divider()

# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# Store collection in session state
st.session_state.HW4_VectorDB = collection

# ===== HELPER FUNCTION: RETRIEVE RELEVANT CLUB INFO =====
def relevant_club_info(query, n_results=5):
    """
    Run a vector search against the organization collection and return context.
    """
    query_embedding = st.session_state.openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    results = st.session_state.HW4_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    relevant_docs = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(relevant_docs) if relevant_docs else ""

    filenames = set()
    for metadata in results.get("metadatas", [[]])[0]:
        filenames.add(metadata.get("filename", "Unknown"))
    files_used = ", ".join(sorted(filenames)) if filenames else "None"

    return {
        "context": context,
        "files_used": files_used,
        "results": results
    }

# ===== HELPER FUNCTION: BUFFER MESSAGES =====
def get_buffered_messages(all_messages, buffer_size=5):
    """
    Keep only the last N user/assistant message pairs
    Always keeps the system prompt (first message)
    
    Args:
        all_messages: Full conversation history
        buffer_size: Number of exchanges to keep
    
    Returns:
        Buffered message list with system prompt + recent exchanges
    """
    if len(all_messages) <= 1:  # Only system prompt or empty
        return all_messages
    
    # Separate system prompt from conversation
    system_prompt = all_messages[0] if all_messages[0]["role"] == "system" else None
    conversation = all_messages[1:] if system_prompt else all_messages
    
    # Keep only last N exchanges (N*2 messages)
    max_messages = buffer_size * 2
    if len(conversation) <= max_messages:
        buffered_conversation = conversation
    else:
        buffered_conversation = conversation[-max_messages:]
    
    # Return system prompt + buffered conversation
    if system_prompt:
        return [system_prompt] + buffered_conversation
    else:
        return buffered_conversation

# Initialize chat history (with system prompt)
if "messages" not in st.session_state:
    system_prompt = {
        "role": "system",
        "content": """You are a Syracuse University organization information assistant.

CRITICAL INSTRUCTIONS:
1. You have access to a vector database of Syracuse University organization profiles
2. You will receive relevant context from these profiles for each user question
3. When answering:
   - If info IS in the context: Cite the organization name and answer based on the context
   - If info is NOT in the context: Say "I didn't find this in the organization database, but..." and provide general knowledge
4. Be conversational and helpful
5. Remember the conversation history - reference previous topics when relevant

Always be clear about whether you're using the organization database or general knowledge."""
    }
    st.session_state.messages = [system_prompt]

# Display chat history (skip system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Syracuse University organizations"):
    
    # Add user message to FULL history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ===== STEP 1: QUERY VECTOR DATABASE =====
    with st.spinner("🔍 Searching organization database..."):
        retrieval = relevant_club_info(prompt, n_results=5)
        context = retrieval["context"]
        files_used = retrieval["files_used"]
        results = retrieval["results"]
    
    # ===== STEP 2: CREATE BUFFERED CONVERSATION =====
    # Get buffered messages (system + last N exchanges)
    buffered_messages = get_buffered_messages(st.session_state.messages, buffer_size)
    
    # ===== STEP 3: ADD RETRIEVED CONTEXT (NO FUNCTION CALLING) =====
    retrieval_message = {
        "role": "system",
        "content": f"""Retrieved context from Syracuse University organization pages: {files_used}

{context}

Instructions: Use this context when answering. If the answer is not in the context, say so and provide general knowledge."""
    }

    # Create messages for API call
    api_messages = buffered_messages + [retrieval_message, {"role": "user", "content": prompt}]
    
    # ===== STEP 4: GET LLM RESPONSE (GPT-4o) =====
    with st.chat_message("assistant"):
        client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
        
        stream = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o
            messages=api_messages,
            stream=True,
            temperature=0.7
        )
        
        response = st.write_stream(stream)
    
    # Save assistant response to FULL history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Store results for sidebar display
    st.session_state.last_results = results

# ===== SIDEBAR: BUFFER STATISTICS =====
st.sidebar.divider()
st.sidebar.write("**📊 Memory Statistics:**")

total_exchanges = (len(st.session_state.messages) - 1) // 2  # Exclude system prompt
st.sidebar.write(f"Total exchanges: {total_exchanges}")
st.sidebar.write(f"Messages in buffer: {len(get_buffered_messages(st.session_state.messages, buffer_size)) - 1}")
st.sidebar.write(f"Total messages: {len(st.session_state.messages) - 1}")

if total_exchanges > buffer_size:
    forgotten = total_exchanges - buffer_size
    st.sidebar.warning(f"⚠️ {forgotten} older exchanges not in active memory")

# ===== SIDEBAR: SHOW RETRIEVED CHUNKS =====
st.sidebar.divider()

if st.sidebar.checkbox("Show retrieved documents"):
    if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
        results = st.session_state.last_results
        st.sidebar.write("### 📄 Retrieved Pages:")
        
        for i, metadata in enumerate(results["metadatas"][0], 1):
            filename = metadata.get("filename", "Unknown")
            chunk_idx = metadata.get("chunk_index", 0)
            total = metadata.get("total_chunks", 1)
            st.sidebar.write(f"{i}. **{filename}** (chunk {chunk_idx + 1}/{total})")
    else:
        st.sidebar.write("No retrieval performed yet.")

# ===== CLEAR CHAT BUTTON =====
st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Chat History"):
    # Keep only the system prompt
    st.session_state.messages = [st.session_state.messages[0]]
    if hasattr(st.session_state, 'last_results'):
        del st.session_state.last_results
    st.rerun()