__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CSV_PATH = "news.csv"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "news_articles"


def extract_title(doc: str) -> str:
    for marker in ["Description:", "content:"]:
        if marker in doc:
            return doc.split(marker)[0].strip()
    return doc[:120].strip()


def extract_description(doc: str) -> str:
    if "Description:" in doc:
        after = doc.split("Description:", 1)[1]
        if "content:" in after:
            after = after.split("content:", 1)[0]
        return after.strip()
    return doc[120:400].strip()


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["title"] = df["Document"].apply(extract_title)
    df["description"] = df["Document"].apply(extract_description)
    df["text_for_embed"] = (
        df["company_name"] + " | " + df["title"] + " | " + df["description"]
    )
    print(f"  {len(df)} articles loaded across {df['company_name'].nunique()} companies")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding articles (this takes ~30s)...")
    embeddings = model.encode(
        df["text_for_embed"].tolist(),
        show_progress_bar=True,
        batch_size=64,
    ).tolist()

    print("Building ChromaDB...")
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    # Drop existing collection if rebuilding
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("  Dropped existing collection")

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"doc_{i}" for i in range(len(df))],
        documents=df["text_for_embed"].tolist(),
        embeddings=embeddings,
        metadatas=[
            {
                "company": row["company_name"],
                "title": row["title"],
                "description": row["description"][:500],
                "url": str(row["URL"]),
                "date": row["Date"].isoformat() if pd.notna(row["Date"]) else "",
                "days_since_2000": int(row["days_since_2000"]),
            }
            for _, row in df.iterrows()
        ],
    )

    print(f"\n✅ Done. ChromaDB saved to {DB_PATH}/")
    print("   Commit ./chroma_db to your repo, then run: streamlit run HW7.py")


if __name__ == "__main__":
    main()