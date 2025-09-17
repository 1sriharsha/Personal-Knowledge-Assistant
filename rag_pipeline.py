from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

CHROMA_PATH = "chroma_store"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

def rag_answer(question: str, k: int = 4) -> str:
    docs = db.similarity_search(question, k=k)

    if docs:
        context = "\n\n".join([d.page_content[:400] for d in docs])
    else:
        # Fallback: summarize the whole document if no relevant context found
        all_docs = db.similarity_search("", k=8)
        if not all_docs:
            return "No documents found. Please upload a PDF first."
        context = " ".join([d.page_content[:400] for d in all_docs])

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
