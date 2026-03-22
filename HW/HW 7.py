__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import re
import time
import pandas as pd
import streamlit as st

# ── Lazy imports (cached so they only load once) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_chroma():
    """Load the pre-built ChromaDB. Run database.py first."""
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("news_articles")


@st.cache_data(show_spinner=False)
def load_dataframe():
    df = pd.read_csv("HW/news.csv")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["title"] = df["Document"].apply(_extract_title)
    df["description"] = df["Document"].apply(_extract_description)
    return df


@st.cache_data(show_spinner=False)
def embed_query(query: str) -> list:
    model = load_embedding_model()
    return model.encode([query], show_progress_bar=False).tolist()


# ── Text parsing helpers ─────────────────────────────────────────────────────
def _extract_title(doc: str) -> str:
    for marker in ["Description:", "content:"]:
        if marker in doc:
            return doc.split(marker)[0].strip()
    return doc[:120].strip()


def _extract_description(doc: str) -> str:
    if "Description:" in doc:
        after = doc.split("Description:", 1)[1]
        if "content:" in after:
            after = after.split("content:", 1)[0]
        return after.strip()
    return doc[120:400].strip()


# ── RAG retrieval ────────────────────────────────────────────────────────────
def retrieve(query: str, collection, n_results: int = 10,
             company_filter: str | None = None) -> list[dict]:
    query_vec = embed_query(query)
    where = {"company": company_filter} if company_filter else None
    results = collection.query(
        query_embeddings=query_vec,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )
    articles = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        articles.append({**meta, "similarity": round(1 - dist, 3)})
    return articles


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _articles_to_context(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"[{i}] {a['company']} | {a['date'][:10] if a['date'] else 'N/A'}\n"
            f"    Title: {a['title']}\n"
            f"    Summary: {a['description']}\n"
            f"    URL: {a['url']}\n"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are LexPulse, an elite news intelligence assistant for a global law firm.
Your job is to monitor news about the firm's clients and surface what matters most to attorneys.

When ranking news, prioritize:
1. Regulatory/legal risk (lawsuits, investigations, fines, compliance issues)
2. M&A activity, major deals, or ownership changes
3. Leadership changes (CEO, board)
4. Financial distress or major earnings surprises
5. Reputational events (scandals, public controversies)
6. Product launches or market-moving announcements

Always cite article numbers [1], [2], etc. when referencing specific stories.
Be concise, precise, and professional — you are writing for busy lawyers."""


def call_gemini(messages: list[dict]) -> str:
    from google import genai
    from google.genai import types
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return "❌ GEMINI_API_KEY not set in Streamlit secrets."
    client = genai.Client(api_key=api_key)
    prompt = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text


def call_openai(messages: list[dict]) -> str:
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        return "❌ OPENAI_API_KEY not set in Streamlit secrets."
    client = OpenAI(api_key=api_key)
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(model="gpt-4o-mini", messages=full_messages)
    return response.choices[0].message.content


def call_llm(messages: list[dict], llm_choice: str) -> tuple[str, float]:
    start = time.time()
    if llm_choice == "gemini-2.0-flash":
        response = call_gemini(messages)
    else:
        response = call_openai(messages)
    return response, round(time.time() - start, 2)


# ── Intent detection ──────────────────────────────────────────────────────────
def detect_intent(query: str) -> tuple[str, str | None]:
    q = query.lower()
    if any(p in q for p in ["most interesting", "top news", "best stories",
                              "most important", "biggest news", "rank"]):
        return "rank", None
    m = re.search(
        r"(?:news about|find news on|tell me about|what.*about|stories about|"
        r"articles about|coverage of|headlines on)\s+(.+?)(?:\?|$)", q
    )
    if m:
        return "search", m.group(1).strip()
    return "general", None


def build_rag_prompt(user_query: str, collection,
                     company_filter: str | None = None) -> tuple[str, list[dict]]:
    intent, entity = detect_intent(user_query)
    search_q = entity if entity else user_query
    if company_filter:
        search_q = f"{company_filter} {search_q}"

    articles = retrieve(search_q, collection, n_results=10, company_filter=company_filter)
    context = _articles_to_context(articles)

    if intent == "rank":
        instruction = (
            "The user wants a RANKED LIST of the most interesting/important news stories "
            "for a law firm monitoring its clients. Rank from most to least significant. "
            "For each story: provide rank, company, title, one-sentence legal/business relevance, "
            "and the URL. Explain your ranking criteria briefly at the top."
        )
    elif intent == "search":
        instruction = (
            f"The user is looking for news about '{entity}'. "
            "Return all relevant articles, grouped by relevance. "
            "For each: company, title, key point, URL."
        )
    else:
        instruction = "Answer the user's question using only the articles provided below."

    augmented = (
        f"{instruction}\n\n"
        f"RETRIEVED ARTICLES:\n{context}\n\n"
        f"USER QUESTION: {user_query}"
    )
    return augmented, articles


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
  --ink: #0f0f0f;
  --cream: #f5f0e8;
  --gold: #b8922a;
  --red: #9b1d1d;
  --rule: #d4c9b0;
  --muted: #6b6355;
}

.masthead {
  border-bottom: 3px double var(--ink);
  padding-bottom: 0.5rem;
  margin-bottom: 1.5rem;
  text-align: center;
}
.masthead h1 {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0;
}
.masthead .tagline {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  color: var(--muted);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 0.2rem;
}
.edition-line {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  padding: 0.2rem 0;
  text-align: center;
  margin-bottom: 1.5rem;
  letter-spacing: 0.08em;
}
.chat-user {
  background: #0f0f0f;
  color: #f5f0e8;
  border-radius: 2px 12px 12px 12px;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0 0.5rem 3rem;
  font-size: 0.95rem;
}
.chat-bot {
  background: white;
  border-left: 3px solid var(--gold);
  padding: 0.75rem 1rem;
  margin: 0.5rem 3rem 0.5rem 0;
  font-size: 0.9rem;
  line-height: 1.6;
  box-shadow: 2px 2px 0 var(--rule);
}
.model-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.4rem;
}
.timing {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  color: #aaa;
  text-align: right;
  margin-top: 0.3rem;
}
.source-card {
  background: white;
  border: 1px solid var(--rule);
  border-top: 2px solid var(--gold);
  padding: 0.6rem 0.8rem;
  margin: 0.3rem 0;
  font-size: 0.78rem;
}
.sc-company {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--gold);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.sc-title { font-weight: 500; }
.sc-url {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  color: var(--muted);
  word-break: break-all;
}
.sc-sim {
  float: right;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  color: var(--red);
}
</style>
""", unsafe_allow_html=True)


# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div class="tagline">Global Law Firm Intelligence Platform</div>
  <h1>LexPulse</h1>
  <div class="tagline">Client News Monitor · Powered by RAG</div>
</div>
""", unsafe_allow_html=True)

from datetime import datetime
st.markdown(
    f'<div class="edition-line">EDITION: {datetime.now().strftime("%B %d, %Y")}'
    f' &nbsp;|&nbsp; DATABASE: 1,290 ARTICLES · 177 CLIENTS</div>',
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading news database…"):
    collection = load_chroma()
    df = load_dataframe()

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl_col1, ctrl_col2 = st.columns([1, 2])

with ctrl_col1:
    llm_choice = st.radio(
        "Model",
        ["gemini-2.0-flash", "gpt-4o-mini"],
        horizontal=True,
    )

with ctrl_col2:
    companies = ["All clients"] + sorted(df["company_name"].dropna().unique().tolist())
    company_sel = st.selectbox("Filter by client", companies)
    company_filter = None if company_sel == "All clients" else company_sel

# Quick queries
st.markdown("**Quick queries:**")
qcols = st.columns(5)
quick_queries = [
    "Find the most interesting news",
    "Find regulatory or legal risk news",
    "Find M&A and deal news",
    "Find leadership changes",
    "Find earnings surprises",
]
for col, qq in zip(qcols, quick_queries):
    with col:
        if st.button(qq, use_container_width=True):
            st.session_state["hw7_quick"] = qq

if st.button("🗑 Clear chat"):
    st.session_state["hw7_messages"] = []
    st.session_state["hw7_sources"] = []
    st.rerun()

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "hw7_messages" not in st.session_state:
    st.session_state["hw7_messages"] = []
if "hw7_sources" not in st.session_state:
    st.session_state["hw7_sources"] = []

# ── Layout ────────────────────────────────────────────────────────────────────
chat_col, source_col = st.columns([3, 1])

with chat_col:
    for msg in st.session_state["hw7_messages"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🔍 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot">'
                f'<div class="model-badge">▸ {msg.get("model", "")} · {msg.get("elapsed", "")}s</div>'
                f'{msg["content"].replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )

    user_input = st.chat_input("Ask about your clients' news…")

    if "hw7_quick" in st.session_state:
        user_input = st.session_state.pop("hw7_quick")

    if user_input:
        st.session_state["hw7_messages"].append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-user">🔍 {user_input}</div>', unsafe_allow_html=True)

        with st.spinner("Searching & generating…"):
            augmented_prompt, retrieved_articles = build_rag_prompt(
                user_input, collection, company_filter=company_filter
            )
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["hw7_messages"][:-1]
            ]
            history.append({"role": "user", "content": augmented_prompt})
            response, elapsed = call_llm(history, llm_choice)

        st.session_state["hw7_messages"].append({
            "role": "assistant",
            "content": response,
            "model": llm_choice,
            "elapsed": elapsed,
        })
        st.session_state["hw7_sources"] = retrieved_articles
        st.rerun()

with source_col:
    st.markdown(
        '<p style="font-family:IBM Plex Mono;font-size:0.65rem;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#6b6355;border-bottom:1px solid #d4c9b0;'
        'padding-bottom:0.3rem;">Retrieved Sources</p>',
        unsafe_allow_html=True,
    )
    if st.session_state["hw7_sources"]:
        for a in st.session_state["hw7_sources"][:8]:
            st.markdown(
                f'<div class="source-card">'
                f'<span class="sc-sim">{int(a["similarity"]*100)}%</span>'
                f'<div class="sc-company">{a["company"]}</div>'
                f'<div class="sc-title">{a["title"][:65]}{"…" if len(a["title"])>65 else ""}</div>'
                f'<div class="sc-url"><a href="{a["url"]}" target="_blank">{a["url"][:40]}…</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="font-family:IBM Plex Mono;font-size:0.65rem;color:#aaa;">'
            'Sources appear here after your first query.</p>',
            unsafe_allow_html=True,
        )