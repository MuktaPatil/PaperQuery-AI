import streamlit as st
import fitz  # PyMuPDF
import textwrap
import time
from transformers import pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperQuery",
    page_icon="📄",
    layout="centered"
)

# ── Title & description ───────────────────────────────────────────────────────
st.title("📄 PaperQuery")
st.markdown("**Ask questions about any research paper — powered by DistilBERT QA**")
st.markdown("---")

# ── Load QA model (cached so it only loads once) ─────────────────────────────
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_qa_pipeline()

# ── Helper functions ──────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def answer_question(question, chunks):
    best_answer = ""
    best_score = 0.0
    for chunk in chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)
            if result["score"] > best_score:
                best_score = result["score"]
                best_answer = result["answer"]
        except:
            continue
    return best_answer or "Sorry, I couldn't find an answer in the PDF."

# ── PDF source selection ──────────────────────────────────────────────────────
st.subheader("📂 Choose a PDF")

source = st.radio(
    "Select PDF source:",
    ["📘 Use sample paper", "📤 Upload your own PDF"],
    horizontal=True
)

pdf_bytes = None
pdf_name = None

if source == "📘 Use sample paper":
    st.info("Using the sample paper: **Seizure Detection and Probability Prediction using Random Forests**")
    try:
        with open("Seizure Detection and Probability Prediction using Random Forests.pdf", "rb") as f:
            pdf_bytes = f.read()
        pdf_name = "Sample Paper"
    except FileNotFoundError:
        st.error("⚠️ Sample PDF not found. Please make sure it's in the same folder as app.py.")

else:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        pdf_name = uploaded_file.name

# ── Process PDF & Q&A ─────────────────────────────────────────────────────────
if pdf_bytes:
    with st.spinner("🔍 Processing PDF..."):
        text = extract_text_from_pdf(pdf_bytes)
        chunks = chunk_text(text)

    st.success(f"✅ **{pdf_name}** loaded — {len(chunks)} chunks ready")
    st.markdown("---")

    # Chat history stored in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    question = st.chat_input("Ask a question about the paper...")

    if question:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(question, chunks)
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown("👆 Select a PDF source above to get started.")
