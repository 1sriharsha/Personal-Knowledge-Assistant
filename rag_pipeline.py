from ingest import faiss_db, embedding_model, load_or_create_faiss
from transformers import pipeline

# Ensure FAISS is loaded on startup
db = load_or_create_faiss()

qa_model = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

def rag_answer(question: str, k: int = 4) -> str:
    global faiss_db
    if faiss_db is None or len(faiss_db.index_to_docstore_id) == 0:
        return "No documents found. Please upload a PDF first."

    docs = faiss_db.similarity_search(question, k=k)

    if docs:
        context = "\n\n".join([d.page_content[:400] for d in docs])
    else:
        all_docs = faiss_db.similarity_search("", k=8)
        context = " ".join([d.page_content[:400] for d in all_docs]) if all_docs else ""

    if not context:
        return "I don't know."

    prompt = f"""
    You are a helpful assistant. Answer the question based only on the context below.
    If the question is very broad (like 'what is this document about?'), summarize the content.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question: {question}
    Answer:
    """

    result = qa_model(prompt, max_new_tokens=200)
    return result[0]["generated_text"]
