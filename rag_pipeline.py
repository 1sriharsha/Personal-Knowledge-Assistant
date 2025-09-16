from langchain_community.vectorstores import Chroma
from transformers import pipeline
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_store"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

# Small free Hugging Face model for Q&A
qa_model = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1)

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

    result = qa_model(prompt, max_new_tokens=200, do_sample=True, temperature=0.3)
    return result[0]["generated_text"]
