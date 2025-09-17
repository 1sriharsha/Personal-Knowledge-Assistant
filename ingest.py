from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_store"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    db.add_texts(texts=texts, metadatas=metadatas)

    return len(splits)


