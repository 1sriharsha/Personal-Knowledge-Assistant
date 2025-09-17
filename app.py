import streamlit as st
from ingest import ingest_pdf, faiss_db, load_or_create_faiss
from rag_pipeline import rag_answer

st.set_page_config(page_title="Personal Knowledge Assistant", layout="wide")
st.title("Personal PDF Assistant")



# Load FAISS index on startup
load_or_create_faiss()


# Upload PDFs
uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        chunks = ingest_pdf(file.name)
        st.success(f"Ingested {chunks} chunks from {file.name}")

    if faiss_db:
        st.info("FAISS index is ready and persisted.")

st.divider()

# Ask questions
st.subheader("Ask a question about your knowledge base:")
query = st.text_input("Type your question here...")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = rag_answer(query)
            st.write("###Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error while answering: {e}")
