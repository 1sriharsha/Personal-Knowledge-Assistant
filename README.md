# Personal Knowledge Assistant

A free, open-source app that lets you upload your own documents (PDFs) and ask questions about them using **RAG (Retrieval-Augmented Generation)**.

### 🚀 Features
- Upload PDFs → automatically split & store in vector DB (Chroma).
- Ask natural language questions.
- Answers grounded in your documents.
- Runs entirely free on Hugging Face Spaces.

### Stack
- Streamlit (UI)
- LangChain (retrieval orchestration)
- ChromaDB (vector store)
- SentenceTransformers (embeddings)
- Falcon-7B-Instruct (open LLM)

### How to Run (Locally)
```bash
pip install -r requirements.txt
streamlit run app.py
