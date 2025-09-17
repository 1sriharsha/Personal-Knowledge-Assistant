import streamlit as st
from ingest import ingest_pdf
from rag_pipeline import rag_answer
import os

st.set_page_config(page_title="Personal Knowledge Assistant", layout="wide")
st.title("Personal Knowledge Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("temp.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    chunks = ingest_pdf(file_path)
    st.success(f"Ingested {chunks} chunks from {uploaded_file.name}")

query = st.text_input("Ask a question about the uploaded documents:")

if st.button("Get Answer"):
    if query:
        answer = rag_answer(query)
        st.markdown(f"### Answer:\n{answer}")
    else:
        st.warning("Please enter a question.")
