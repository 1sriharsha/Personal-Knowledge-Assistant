import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ---- Initialize clients ----
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("text-embedding-3-large")

# ---- Metrics tracker ----
_metrics = {
    "query_count": 0,
    "embedding_requests": 0,
    "embedding_tokens": 0,
    "vectors_stored": 0,
    "pinecone_reads": 0,
    "pinecone_writes": 0,
    "total_tokens": 0,
    "chat_completion_requests": 0,
    "openai_cost": 0.0,
    "pinecone_cost": 0.0,
}

# Pricing estimates (update with actual rates)
OPENAI_EMBED_COST_PER_1K = 0.0001
OPENAI_CHAT_COST_PER_1K = 0.0005
PINECONE_PRICE_PER_READ = 0.0001
PINECONE_PRICE_PER_WRITE = 0.0004

def track_embedding(tokens: int, vectors: int):
    _metrics["embedding_requests"] += 1
    _metrics["embedding_tokens"] += tokens
    _metrics["vectors_stored"] += vectors
    _metrics["openai_cost"] += (tokens / 1000) * OPENAI_EMBED_COST_PER_1K

def track_chat(tokens: int):
    _metrics["chat_completion_requests"] += 1
    _metrics["total_tokens"] += tokens
    _metrics["openai_cost"] += (tokens / 1000) * OPENAI_CHAT_COST_PER_1K

def track_pinecone(reads=0, writes=0):
    _metrics["pinecone_reads"] += reads
    _metrics["pinecone_writes"] += writes
    _metrics["pinecone_cost"] += (
        reads * PINECONE_PRICE_PER_READ + writes * PINECONE_PRICE_PER_WRITE
    )

def get_metrics():
    return _metrics

# ---- RAG Answer ----
def rag_answer(query: str) -> str:
    _metrics["query_count"] += 1

    # Embed query
    emb = client.embeddings.create(input=query, model="text-embedding-3-large")
    query_vector = emb.data[0].embedding
    track_embedding(tokens=emb.usage.total_tokens, vectors=0)

    # Pinecone search
    search_result = index.query(vector=query_vector, top_k=3, include_metadata=True)
    track_pinecone(reads=1)

    if not search_result.matches:
        return "No documents found. Please upload a PDF first."

    context = " ".join([m.metadata["text"] for m in search_result.matches])

    # Chat completion
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    tokens_used = completion.usage.total_tokens
    track_chat(tokens=tokens_used)

    return completion.choices[0].message.content
