# bot8.py — Ultra-Fast PDF & DOCX Insight Generator
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import io
import re
from typing import List, Tuple

import numpy as np
import streamlit as st
import fitz  # PyMuPDF (⚡Super-fast PDF extraction)
import docx

# FAISS (optional)
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =============================
# Streamlit UI Config
# =============================
st.set_page_config(page_title="AI Document Insight Generator", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 15px; }
    h1 { font-size: 1.6rem; }
    h2 { font-size: 1.3rem; }
    .muted { color: #6b7280; }
    .card { padding: 1rem; border: 1px solid #eee; border-radius: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.06); background: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Load ML Models (cached)
# =============================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_summarizer():
    # 3× faster than bart-large-cnn
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# =============================
# FAST TEXT EXTRACTION
# =============================
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Fast PDF extraction using PyMuPDF."""
    text_parts = []
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    for page in pdf:
        page_text = page.get_text("text")
        page_text = re.sub(r"\s+", " ", page_text).strip()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

# =============================
# Embeddings
# =============================
def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return vectors.astype(np.float32)

# =============================
# Summarization (Optimized)
# =============================
@st.cache_data(show_spinner=False)
def cached_summary(text: str, ratio: float):
    summarizer = get_summarizer()
    chunk_size = 3000  # large chunk size → fewer summarizer calls

    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summary_parts = []

    for ch in chunks:
        out = summarizer(
            ch,
            max_length=400,
            min_length=80,
            do_sample=False
        )
        summary_parts.append(out[0]["summary_text"])

    return " ".join(summary_parts)

# =============================
# Vector Store
# =============================
class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids = []
        self.vectors = []
        self.index = faiss.IndexFlatL2(dim) if HAVE_FAISS else None

    def add(self, vectors: np.ndarray, ids: List[str]):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if HAVE_FAISS and self.index is not None:
            self.index.add(vectors)
        self.vectors.extend([v for v in vectors])
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int = 5):
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # FAISS search (faster)
        if HAVE_FAISS and self.index is not None and len(self.ids) > 0:
            D, I = self.index.search(query_vec, min(k, len(self.ids)))
            return [
                (self.ids[idx], float(dist))
                for idx, dist in zip(I.ravel().tolist(), D.ravel().tolist())
                if 0 <= idx < len(self.ids)
            ]

        # fallback cosine similarity
        if not self.vectors:
            return []

        M = np.vstack(self.vectors)
        q = query_vec[0]
        sims = M @ q
        dists = 1 - sims
        order = np.argsort(dists)[:k]

        return [(self.ids[i], float(dists[i])) for i in order]

# =============================
# Session State
# =============================
if "docs" not in st.session_state:
    st.session_state.docs = {}

if "store" not in st.session_state:
    dim = get_embedder().get_sentence_embedding_dimension()
    st.session_state.store = VectorStore(dim)

# =============================
# UI – Upload Documents
# =============================
st.title("📄 AI-Powered Document Insight Generator")

uploads = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploads:
    with st.spinner("Processing documents…"):
        names, texts = [], []

        for uf in uploads:
            try:
                content = uf.read()

                if uf.name.endswith(".pdf"):
                    text = extract_text_from_pdf_bytes(content)
                else:
                    text = extract_text_from_docx(content)

                if not text:
                    st.warning(f"No text found in {uf.name}")
                    continue

                short_summary = cached_summary(text, ratio=0.1)
                long_summary = cached_summary(text, ratio=1.0)

                st.session_state.docs[uf.name] = {
                    "text": text,
                    "summary10": short_summary,
                    "summary100": long_summary
                }

                names.append(uf.name)
                texts.append(text)

            except Exception as e:
                st.error(f"Error processing {uf.name}: {e}")

        if texts:
            vectors = embed_texts(texts)
            st.session_state.store.add(vectors, names)

    st.success("All documents processed successfully!")

st.divider()

# =============================
# Query Interface
# =============================
st.subheader("🔍 Query Document Summaries")
user_q = st.text_input("Enter keywords or a question")

if user_q:
    qvec = embed_texts([user_q])
    results = st.session_state.store.search(qvec)

    if results:
        st.write("**Top Matches:**")
        for doc_id, dist in results:
            st.write(f"• {doc_id} — distance {dist:.4f}")

        best_doc = results[0][0]
        data = st.session_state.docs[best_doc]

        st.subheader(f"📘 Best Match: {best_doc}")

        keyword = user_q.lower()

        def filter_summary(summary):
            return " ".join([s for s in summary.split(". ") if keyword in s.lower()]) or summary

        st.markdown("**Short Summary (10%)**")
        st.text_area("", filter_summary(data["summary10"]), height=200)

        st.markdown("**Detailed Summary (100%)**")
        st.text_area("", filter_summary(data["summary100"]), height=300)

st.divider()

# =============================
# Uploaded Documents Viewer
# =============================
if st.session_state.docs:
    st.subheader("📚 Uploaded Documents")
    tabs = st.tabs(list(st.session_state.docs.keys()))

    for tab, name in zip(tabs, st.session_state.docs.keys()):
        with tab:
            st.text_area("Full Extracted Text", st.session_state.docs[name]["text"], height=300)
            st.download_button(
                "⬇️ Download Extracted Text",
                st.session_state.docs[name]["text"],
                file_name=f"{name}_extracted.txt",
                mime="text/plain"
            )

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Developed by Aditya • Powered by AI</div>", unsafe_allow_html=True)
