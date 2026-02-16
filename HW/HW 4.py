from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
import sys
import shutil
from pathlib import Path

# Ensure ChromaDB uses a compatible SQLite
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

import chromadb


# Page config
st.set_page_config(page_title="HW 4: RAG Chatbot", initial_sidebar_state="expanded")
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

# ===== BIG TITLE =====
st.title("HW 4: Chatbot using RAG with HTML Documents")
st.markdown("---")

# ===== CHUNKING FUNCTION =====
# CHUNKING METHOD: Simple Split with Sentence Boundary Detection
# 
# This method splits each document into EXACTLY 2 chunks (mini-documents).
# 
# WHY THIS METHOD:
# 1. Simple Split: Divides document at the midpoint to create two equal-sized chunks
# 2. Sentence Boundary Detection: Adjusts the split point to avoid breaking in the
#    middle of a sentence, which preserves semantic meaning
# 3. Benefits:
#    - Maintains context within each chunk (no broken sentences)
#    - Creates manageable chunk sizes for embeddings
#    - Doubles our searchable documents (2 per HTML file)
#    - Better retrieval precision - queries can match more specific sections
# 4. Trade-offs:
#    - Information at the split point might be separated
#    - But sentence boundary detection minimizes this issue
def chunk_text(text):
    """
    Split text into exactly 2 chunks, attempting to break at sentence boundaries.
    
    Args:
        text: The full text to split
        
    Returns:
        List of exactly 2 text chunks
    """
    text_length = len(text)
    
    # Calculate midpoint
    midpoint = text_length // 2
    
    # Find the nearest sentence boundary around the midpoint
    # Search in a window of +/- 500 characters
    search_start = max(0, midpoint - 500)
    search_end = min(text_length, midpoint + 500)
    search_text = text[search_start:search_end]
    
    # Look for sentence-ending punctuation near the midpoint
    best_split = midpoint
    for punct in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
        pos = search_text.find(punct, len(search_text) // 2 - 250)
        if pos != -1:
            # Found a good sentence boundary
            best_split = search_start + pos + len(punct)
            break
    
    # Create the two chunks
    chunk1 = text[:best_split].strip()
    chunk2 = text[best_split:].strip()
    
    return [chunk1, chunk2]


# ===== VECTOR DATABASE SETUP =====
# PERSISTENCE STRATEGY: Only create the vector DB if it doesn't already exist
# This allows the app to run multiple times without re-processing documents
# and wasting API calls for embeddings
#
# The vector DB is stored in: HW/chroma_db/
# - This is a persistent database that survives app restarts
# - ChromaDB uses SQLite to store the embeddings and metadata
#
# CREATION LOGIC:
# 1. Check if database already exists and has data
# 2. If yes: Skip processing and use existing embeddings
# 3. If no: Process all HTML files, chunk them, and create embeddings

# Create ChromaDB client with persistent storage in the HW folder
db_path = Path(__file__).parent / "chroma_db"
db_path.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(db_path))
collection = chroma_client.get_or_create_collection(name="HW4Collection")

# Create OpenAI client for embeddings
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# CHECK 1: Does the vector database already have data?
# If yes, we'll skip re-processing to save time and API costs
try:
    existing_count = collection.count()
except Exception:
    st.sidebar.warning("ChromaDB failed to load existing data. Rebuilding index.")
    try:
        chroma_client.delete_collection(name="HW4Collection")
    except Exception:
        try:
            shutil.rmtree(db_path, ignore_errors=True)
        except Exception:
            pass

    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_or_create_collection(name="HW4Collection")
    existing_count = 0

# Define the path to HTML files
html_folder = Path(__file__).parent / "hw4_data"
html_files = list(html_folder.glob("*.html")) if html_folder.exists() else []

# CHECK 2: Does the database have the expected number of chunks?
# Each HTML file creates 2 chunks, so expected count = num_files * 2
expected_count = len(html_files) * 2

# Rebuild if the database doesn't have the expected number of chunks
if existing_count < expected_count:
    st.sidebar.info(f"Vector DB needs updating: {existing_count}/{expected_count} chunks found")
    try:
        chroma_client.delete_collection(name="HW4Collection")
    except Exception:
        shutil.rmtree(db_path, ignore_errors=True)

    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_or_create_collection(name="HW4Collection")
    existing_count = 0
else:
    st.sidebar.success(f"✅ Vector DB already exists with {existing_count} chunks")

# ONLY CREATE VECTOR DB IF IT DOESN'T EXIST
if existing_count == 0:
    if html_folder.exists() and html_folder.is_dir():
        st.sidebar.info("Processing HTML files with chunking...")
        
        # Process each HTML file WITH CHUNKING
        for html_file in html_files:
            try:
                # Read HTML and extract text
                st.sidebar.info(f"Processing {html_file.name}...")
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                # Parse HTML and extract text using BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text_content = soup.get_text(separator=" ", strip=True)
                
                # CHUNK THE TEXT INTO 2 MINI-DOCUMENTS
                # Each HTML file will create exactly 2 chunks
                if text_content.strip():
                    chunks = chunk_text(text_content)
                    st.sidebar.info(f"  → Split into {len(chunks)} chunks")
                    
                    # Add each chunk to the collection
                    for i, chunk in enumerate(chunks):
                        try:
                            # Create embedding for this chunk
                            embedding = st.session_state.openai_client.embeddings.create(
                                input=chunk,
                                model="text-embedding-3-small"
                            ).data[0].embedding
                            
                            # Add to ChromaDB collection with unique ID per chunk
                            collection.add(
                                documents=[chunk],
                                embeddings=[embedding],
                                ids=[f"{html_file.name}_chunk_{i}"],
                                metadatas=[{
                                    "filename": html_file.name,
                                    "chunk_index": i,
                                    "total_chunks": len(chunks)
                                }]
                            )
                        except Exception as e:
                            st.sidebar.warning(f"Error embedding chunk {i}: {str(e)}")
                            continue
                    
                    st.sidebar.success(f"✅ Loaded: {html_file.name} ({len(chunks)} chunks)")
                    
            except Exception as e:
                st.sidebar.error(f"Error loading {html_file.name}: {str(e)}")


# Store collection in session state
st.session_state.HW4_VectorDB = collection

st.sidebar.write(
    f"📚 Chunks in database: {st.session_state.HW4_VectorDB.count()}"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Syracuse University organizations"):
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ALWAYS QUERY THE VECTOR DATABASE FIRST
    # Step 1: Create embedding for user's question
    query_embedding = st.session_state.openai_client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    ).data[0].embedding
    
    # Step 2: Search the vector database for relevant chunks
    results = st.session_state.HW4_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=5  # Get top 5 most relevant chunks
    )
    
    # Step 3: Extract the relevant context
    relevant_docs = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(relevant_docs) if relevant_docs else ""
    
    # Get unique filenames from chunks
    filenames = set()
    for metadata in results.get("metadatas", [[]])[0]:
        filenames.add(metadata.get("filename", "Unknown"))
    files_used = ", ".join(sorted(filenames))
    
    # Step 4: Create system prompt that handles BOTH cases
    system_prompt = """You are a Syracuse University organization information assistant with access to organization profiles.

CRITICAL INSTRUCTIONS:
1. You will be provided with context from Syracuse University organization pages
2. FIRST, check if the answer is in the provided context
3. If the answer IS in the context:
   - Start with: "Based on [organization name]..." or "According to [organization page]..."
   - Cite which specific organization(s) you're using
   - Answer using ONLY the information from the context
   
4. If the answer is NOT in the context:
   - Start with: "I didn't find this in the organization pages, but..."
   - Then provide an answer using your general knowledge
   - Be helpful and informative
   
5. Be clear about which case you're in (found in docs vs. using general knowledge)

Examples:
- Found in docs: "Based on the Alpha Phi Alpha profile, their meeting times are..."
- Not found: "I didn't find this in the organization pages, but I can help explain..."
"""

    enhanced_prompt = f"""Context from Syracuse University organization pages: {files_used}

{context}

User Question: {prompt}

Instructions: 
- If the answer is in the context above, cite your sources and answer based on the organization pages
- If the answer is NOT in the context, say "I didn't find this in the organization pages, but..." and provide a helpful answer using general knowledge"""
    
    # Step 5: Get response from ChatGPT
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ],
            stream=True
        )
        response = st.write_stream(stream)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Store results for sidebar display
    st.session_state.last_results = results

# Sidebar to show retrieved chunks
if st.sidebar.checkbox("Show retrieved chunks"):
    if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
        results = st.session_state.last_results
        st.sidebar.write("### Retrieved Pages:")
        
        for i, metadata in enumerate(results["metadatas"][0], 1):
            filename = metadata.get("filename", "Unknown")
            chunk_idx = metadata.get("chunk_index", 0)
            total = metadata.get("total_chunks", 1)
            st.sidebar.write(f"{i}. **{filename}** (chunk {chunk_idx + 1}/{total})")
        
    else:
        st.sidebar.write("No retrieval performed yet.")

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
