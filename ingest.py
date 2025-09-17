import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from rag_pipeline import track_embedding, track_pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("text-embedding-3-large")

def ingest_pdf(file_path: str):
    """Load a PDF, split into chunks, embed, and store in Pinecone."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        return 0

    # Prepare embeddings
    texts = [chunk.page_content for chunk in chunks]
    embeddings = client.embeddings.create(
        input=texts,
        model="text-embedding-3-large"
    )

    # Build Pinecone vectors
    vectors = []
    for i, emb in enumerate(embeddings.data):
        vectors.append({
            "id": f"{os.path.basename(file_path)}_{i}",
            "values": emb.embedding,
            "metadata": {"text": texts[i]}
        })

    # Upsert into Pinecone
    index.upsert(vectors=vectors)

    # --- Track Metrics ---
    embedding_tokens = embeddings.usage.total_tokens
    track_embedding(tokens=embedding_tokens, vectors=len(chunks))
    track_pinecone(writes=len(chunks))

    return len(chunks)
