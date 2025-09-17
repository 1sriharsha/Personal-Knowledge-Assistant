import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def rag_answer(query, top_k=3):
    # 1. Embed the query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    # 2. Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=""
    )

    if not results.get("matches"):
        return "No relevant documents found."

    # 3. Collect context
    context = " ".join([match["metadata"]["text"] for match in results["matches"]])

    # 4. Ask GPT with context
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )

    return completion.choices[0].message.content

