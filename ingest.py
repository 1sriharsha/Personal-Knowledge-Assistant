import os
from PyPDF2 import PdfReader
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from rag_pipeline import track_embedding, track_pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("text-embedding-3-large")

def extract_text_from_pdf(file_path: str):
    """Extract all text from a PDF file using PyPDF2."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size=800, overlap=50):
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_path: str):
    """Load a PDF, split into chunks, embed, and store in Pinecone."""
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        return 0

    chunks = chunk_text(text)

    # Create embeddings
    embeddings = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-large"
    )

    # Build Pinecone vectors
    vectors = []
    for i, emb in enumerate(embeddings.data):
        vectors.append({
            "id": f"{os.path.basename(file_path)}_{i}",
            "values": emb.embedding,
            "metadata": {"text": chunks[i]}
        })

    # Upsert into Pinecone
    index.upsert(vectors=vectors)

    # --- Track Metrics ---
    embedding_tokens = embeddings.usage.total_tokens
    track_embedding(tokens=embedding_tokens, vectors=len(chunks))
    track_pinecone(writes=len(chunks))

    return len(chunks)
