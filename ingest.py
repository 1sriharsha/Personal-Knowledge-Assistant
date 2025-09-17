import os
import fitz  # PyMuPDF
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from rag_pipeline import track_embedding, track_pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("text-embedding-3-large")

# ---- Extract text from PDF using PyMuPDF ----
def extract_text_from_pdf(file_path: str):
    """Extract text from each page in a PDF quickly using PyMuPDF."""
    doc = fitz.open(file_path)
    text = []
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

# ---- Chunk text manually ----
def chunk_text(text: str, chunk_size=800, overlap=50):
    """Split text into overlapping chunks for embeddings."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---- Ingest PDF into Pinecone ----
def ingest_pdf(file_path: str, batch_size=50):
    """Extract, chunk, embed in batches, and upsert into Pinecone."""
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        return 0

    chunks = chunk_text(text)
    total_tokens = 0
    vectors = []

    # Batch embeddings
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large"
        )

        total_tokens += embeddings.usage.total_tokens

        # Build vectors for Pinecone
        for j, emb in enumerate(embeddings.data):
            vectors.append({
                "id": f"{os.path.basename(file_path)}_{i+j}",
                "values": emb.embedding,
                "metadata": {"text": batch[j]}
            })

        # Upsert incrementally (prevents memory issues)
        index.upsert(vectors=vectors)
        vectors = []  # clear buffer

    # --- Track Metrics ---
    track_embedding(tokens=total_tokens, vectors=len(chunks))
    track_pinecone(writes=len(chunks))

    return len(chunks)
