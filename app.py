import streamlit as st
import os
from ingest import ingest_pdf
from rag_pipeline import rag_answer

st.set_page_config(
    page_title="📘 Personal Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar (Navigation + Document Management) ---
with st.sidebar:
    st.markdown("<h2 style='color:#0A66C2;'>📂 Document Manager</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_" + uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            chunks = ingest_pdf(file_path)
            st.success(f"Ingested {chunks} chunks from {uploaded_file.name}")

    st.markdown("---")
    st.info("💡 Upload documents here and ask questions in the main panel.")


# --- Main Panel (Chat Interface) ---
st.markdown("<h1 style='color:#0A66C2;'>Personal Knowledge Assistant</h1>", unsafe_allow_html=True)
st.caption("Ask questions across your uploaded documents. Powered by OpenAI + Pinecone.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
query = st.text_input("Type your question:")

if st.button("Ask"):
    if query:
        with st.spinner("🔎 Searching your documents..."):
            try:
                answer = rag_answer(query)
                st.session_state.chat_history.append(("🧑 You", query))
                st.session_state.chat_history.append(("🤖 Assistant", answer))
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Display Chat History ---
st.markdown("### Conversation")
for role, msg in st.session_state.chat_history:
    if "You" in role:
        st.markdown(
            f"<div style='text-align:right; background-color:#E6F0FA; padding:10px; border-radius:10px; margin:5px;'>"
            f"<b>{role}</b>: {msg}</div>", unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='text-align:left; background-color:#F9F9F9; padding:10px; border-radius:10px; margin:5px;'>"
            f"<b>{role}</b>: {msg}</div>", unsafe_allow_html=True
        )
