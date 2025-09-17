import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from PyPDF2 import PdfReader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def embed_text(text: str):
    """Generate embeddings from OpenAI model"""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks for embedding"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_path):
    """Extract text from PDF and store embeddings in Pinecone"""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    chunks = chunk_text(full_text)

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        vectors.append((f"chunk-{i}", embedding, {"text": chunk}))

    index.upsert(vectors, namespace="")
    return len(chunks)
