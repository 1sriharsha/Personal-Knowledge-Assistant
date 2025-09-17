import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# ---- Setup ----
FAISS_PATH = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_db = None

# QA model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# ---- FAISS Helpers ----
def load_or_create_faiss():
    global faiss_db
    if os.path.exists(FAISS_PATH):
        faiss_db = FAISS.load_local(
            FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
        )
    return faiss_db

def save_faiss():
    global faiss_db
    if faiss_db:
        faiss_db.save_local(FAISS_PATH)

def ingest_pdf(file_path: str):
    global faiss_db
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    if faiss_db is None:
        faiss_db = FAISS.from_documents(splits, embedding_model)
    else:
        faiss_db.add_documents(splits)

    save_faiss()
    return len(splits)

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

# ---- Streamlit UI ----
st.set_page_config(page_title="Personal Knowledge Assistant", layout="wide")
st.title("Personal PDF Assistant")

# Load FAISS index at startup
load_or_create_faiss()

uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        chunks = ingest_pdf(file.name)
        st.success(f"✅ Ingested {chunks} chunks from {file.name}")

    if faiss_db:
        st.info(f"FAISS index has {len(faiss_db.index_to_docstore_id)} chunks")

st.divider()

query = st.text_input("Ask a question about your knowledge base:")
if query:
    with st.spinner("Thinking..."):
        answer = rag_answer(query)
        st.write("### 🤖 Answer:")
        st.write(answer)
