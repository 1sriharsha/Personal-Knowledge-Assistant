import streamlit as st
import os
from ingest import ingest_pdf
from rag_pipeline import rag_answer

st.set_page_config(page_title="Personal Knowledge Assistant", layout="wide")

st.title("Personal Knowledge Assistant")

# Upload documents
uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        chunks = ingest_pdf(file.name)
        st.success(f"✅ Ingested {chunks} chunks from {file.name}")

st.divider()

# Ask questions
st.subheader("Ask a question about your knowledge base:")
query = st.text_input("Type your question here...")

if query:
    with st.spinner("Thinking..."):
        answer = rag_answer(query)
    st.write("### 🤖 Answer:")
    st.write(answer)
