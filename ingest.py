import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_PATH = "faiss_index"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_db = None

def load_or_create_faiss():
    """Load FAISS index if it exists, else return None."""
    global faiss_db
    if os.path.exists(FAISS_PATH):
        faiss_db = FAISS.load_local(
            FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    return faiss_db

def save_faiss():
    """Persist FAISS index to disk."""
    global faiss_db
    if faiss_db:
        faiss_db.save_local(FAISS_PATH)

def ingest_pdf(file_path: str):
    """Load PDF, split into chunks, and add to FAISS index."""
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
