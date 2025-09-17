from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

CHROMA_PATH = "chroma_store"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Chroma vector store
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

# Use a small model for free hosting (fast & lightweight)
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

def rag_answer(question: str, k: int = 4) -> str:
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a helpful assistant. Answer the question based only on the context below.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question: {question}
    Answer:
    """

    result = qa_model(prompt, max_new_tokens=200)
    return result[0]["generated_text"]
