# bot8.py — Free / Lightweight PDF & DOCX Insight Generator (No API keys)
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import io
import re
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import docx

# Lightweight NLP & summarization
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Ensure NLTK punkt is available (quiet)
nltk.download("punkt", quiet=True)

# =============================
# Streamlit config & styles
# =============================
st.set_page_config(page_title="Doc Insight (Free)", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 15px; }
    h1 { font-size: 1.5rem; }
    .muted { color: #6b7280; }
    .card { padding: 0.9rem; border: 1px solid #eee; border-radius: 10px; background: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Utilities: extraction & chunking
# =============================
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Fast PDF extraction using PyMuPDF."""
    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return ""
    parts = []
    for page in pdf:
        txt = page.get_text("text") or ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
    except Exception:
        return ""
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks of roughly max_chars."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# =============================
# Caching utilities (Streamlit)
# =============================
@st.cache_resource(show_spinner=False)
def get_vectorizer(max_features: int = 2000) -> TfidfVectorizer:
    # small max_features keeps memory low on Streamlit Cloud
    return TfidfVectorizer(stop_words="english", max_features=max_features)

@st.cache_data(show_spinner=False)
def compute_tfidf(corpus: List[str], max_features: int = 2000):
    vec = get_vectorizer(max_features=max_features)
    X = vec.fit_transform(corpus)
    return vec, X

@st.cache_data(show_spinner=False)
def lexrank_summary(text: str, sentences_count: int = 3) -> str:
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sentences = summarizer(parser.document, sentences_count)
        return " ".join([str(s) for s in sentences])
    except Exception:
        # fallback: return first N sentences
        sents = re.split(r'(?<=[.!?])\s+', text)
        return " ".join(sents[:sentences_count])

# =============================
# Session-state storage
# =============================
if "docs" not in st.session_state:
    # docs: {doc_name: {"text": str, "chunks": [str], "summary_short": str, "summary_long": str}}
    st.session_state.docs: Dict[str, Dict] = {}

if "chunks" not in st.session_state:
    # chunks_list: list of chunk strings
    st.session_state.chunks: List[str] = []
    st.session_state.chunk_doc_map: List[str] = []  # map chunk idx -> doc name

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

# =============================
# UI: Upload + Process
# =============================
st.title("📄 Document Insight Generator — Free (No API keys)")

st.info("This free version uses TF-IDF + extractive summarization (LexRank). No API keys required.")

uploads = st.file_uploader(
    "Upload PDF / DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key="uploader_free_v1"
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per file guard

if uploads:
    with st.spinner("Processing uploaded documents..."):
        new_chunks = []
        new_map = []
        for uf in uploads:
            name = uf.name
            try:
                content = uf.read()
            except Exception as e:
                st.error(f"Could not read {name}: {e}")
                continue

            if len(content) > MAX_FILE_SIZE:
                st.warning(f"{name} is larger than 10 MB — skipping for performance.")
                continue

            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf_bytes(content)
            else:
                text = extract_text_from_docx(content)

            if not text:
                st.warning(f"No text extracted from {name}.")
                continue

            # chunk the text
            chunks = chunk_text(text, max_chars=1200, overlap=200)
            # compute summaries (short + long) using extractive summarizer
            # sentences_count derived from text length
            approx_sent_count = max(3, min(12, len(re.split(r'(?<=[.!?])\s+', text)) // 20))
            short_sum = lexrank_summary(text, sentences_count=max(2, approx_sent_count // 3))
            long_sum = lexrank_summary(text, sentences_count=max(5, approx_sent_count))

            # store per-doc
            st.session_state.docs[name] = {
                "text": text,
                "chunks": chunks,
                "summary10": short_sum,
                "summary100": long_sum
            }

            # append chunk-level index entries
            for ch in chunks:
                new_chunks.append(ch)
                new_map.append(name)

        # if new chunks exist, rebuild TF-IDF matrix across all chunks
        if new_chunks:
            st.session_state.chunks.extend(new_chunks)
            st.session_state.chunk_doc_map.extend(new_map)
            # compute TF-IDF for all chunks (keeps vectorizer cached)
            vectorizer = get_vectorizer(max_features=3000)
            tfidf_matrix = vectorizer.fit_transform(st.session_state.chunks)
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.vectorizer = vectorizer

    st.success("Documents processed and indexed.")

st.markdown("---")

# =============================
# Query interface
# =============================
st.subheader("🔎 Search & Inspect Documents")

query = st.text_input("Enter a keyword, phrase, or question (e.g., 'revenue growth', 'risk factors')", key="query_input")

top_k = st.slider("Number of top chunks to show", min_value=1, max_value=10, value=4)

if query:
    if not st.session_state.chunks or st.session_state.tfidf_matrix is None:
        st.info("No documents indexed yet. Upload files first.")
    else:
        with st.spinner("Searching..."):
            # vectorize query and compute cosine similarity to chunk matrix
            q_vec = st.session_state.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, st.session_state.tfidf_matrix).ravel()
            top_idxs = np.argsort(sims)[::-1][:top_k]
            results = [(int(i), float(sims[i]), st.session_state.chunk_doc_map[i], st.session_state.chunks[i]) for i in top_idxs if sims[i] > 0]
        if not results:
            st.info("No relevant matches found for that query.")
        else:
            st.write("**Top matches (chunk-level)**")
            for idx, score, doc_name, chunk_text in results:
                with st.expander(f"{doc_name} — relevance {score:.3f}"):
                    st.write(chunk_text)
                    st.download_button(f"Download chunk {idx}", data=chunk_text, file_name=f"{doc_name}_chunk_{idx}.txt", mime="text/plain")

            # show best-matching document summary (aggregate)
            best_doc = results[0][2]
            st.subheader(f"📘 Best-matching document: {best_doc}")
            doc_data = st.session_state.docs.get(best_doc, {})
            st.markdown("**Short (extractive) summary:**")
            st.write(doc_data.get("summary10", ""))
            st.markdown("**Detailed (extractive) summary:**")
            st.write(doc_data.get("summary100", ""))

st.markdown("---")

# =============================
# Uploaded Documents viewer
# =============================
if st.session_state.docs:
    st.subheader("📚 Uploaded Documents")
    for name, meta in st.session_state.docs.items():
        with st.expander(name):
            st.markdown("**Short summary:**")
            st.write(meta.get("summary10", ""))
            st.markdown("**Detailed summary:**")
            st.write(meta.get("summary100", ""))
            st.markdown("**Extracted text (first 5000 chars):**")
            st.text_area("", value=meta["text"][:5000], height=220)
            st.download_button("⬇️ Download full extracted text", data=meta["text"], file_name=f"{name}_extracted.txt", mime="text/plain")

st.markdown("<div class='muted'>Developed by Aditya — Free TF-IDF + LexRank version (no API keys)</div>", unsafe_allow_html=True)
