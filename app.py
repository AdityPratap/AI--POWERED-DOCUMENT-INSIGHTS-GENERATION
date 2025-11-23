# app.py — Unified Analyzer for PDF & DOCX with Summarization and Search
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import io
import re
from typing import List, Tuple
import numpy as np
import streamlit as st
import PyPDF2
import docx  # for .docx extraction

# Optional FAISS
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =============================
# Page Config & Styles
# =============================
st.set_page_config(page_title="PDF & DOCX Analyzer", page_icon="📄", layout="wide")

st.sidebar.markdown("🧩 **App Version:** 2.0 — PDF + DOCX Summarization Enabled")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 15px; }
    h1 { font-size: 1.6rem; }
    .card { padding: 1rem; border: 1px solid #eee; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,.06); background: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Cached Models
# =============================
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# =============================
# Utility Functions
# =============================
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    text_parts: List[str] = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = re.sub(r"\s+", " ", page_text).strip()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return vectors.astype(np.float32)

def summarize_text(text: str, ratio: float = 0.1, max_tokens: int = 512) -> str:
    """Summarize text to a target ratio."""
    summarizer = get_summarizer()
    # Split into manageable chunks (~1000 chars)
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summary_parts = []
    for ch in chunks:
        try:
            out = summarizer(ch, max_length=int(max_tokens * ratio), min_length=30, do_sample=False)
            summary_parts.append(out[0]['summary_text'])
        except Exception:
            continue
    return " ".join(summary_parts) if summary_parts else text[:1000]

# =============================
# Vector Store (FAISS or NumPy)
# =============================
class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.index = faiss.IndexFlatL2(dim) if HAVE_FAISS else None

    def add(self, vectors: np.ndarray, ids: List[str]):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.index is not None:
            self.index.add(vectors)
        self.vectors.extend([v for v in vectors])
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if self.index is not None and len(self.ids) > 0:
            D, I = self.index.search(query_vec, min(k, len(self.ids)))
            return [(self.ids[i], float(D[0][j])) for j, i in enumerate(I[0]) if 0 <= i < len(self.ids)]
        if not self.vectors:
            return []
        M = np.vstack(self.vectors)
        q = query_vec[0]
        sims = M @ q
        dists = 1.0 - sims
        order = np.argsort(dists)[:k]
        return [(self.ids[i], float(dists[i])) for i in order]

# =============================
# Session State Setup
# =============================
if "docs" not in st.session_state:
    st.session_state.docs = {}  # {name: {"text": str, "summary10": str, "summary100": str}}
if "store" not in st.session_state:
    dim = get_embedder().get_sentence_embedding_dimension()
    st.session_state.store = VectorStore(dim)

# =============================
# Main UI
# =============================
st.title("📄 PDF & DOCX Analyzer with Smart Summarization")

uploads = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploads:
    with st.spinner("🔄 Processing uploads..."):
        names, texts = [], []
        for uf in uploads:
            try:
                content = uf.read()
                if uf.name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf_bytes(content)
                else:
                    text = extract_text_from_docx(content)

                if not text:
                    st.warning(f"No readable text in {uf.name}.")
                    continue

                short_sum = summarize_text(text, ratio=0.1)
                long_sum = summarize_text(text, ratio=1.0)

                st.session_state.docs[uf.name] = {
                    "text": text,
                    "summary10": short_sum,
                    "summary100": long_sum,
                }

                names.append(uf.name)
                texts.append(text)
            except Exception as e:
                st.error(f"❌ Error processing {uf.name}: {e}")

        if texts:
            vecs = embed_texts(texts)
            st.session_state.store.add(vecs, names)
    st.success("✅ Upload & summarization complete!")

st.divider()

# =============================
# Query Section
# =============================
st.subheader("🔍 Search or Filter Summaries by Keyword")

user_q = st.text_input("Enter keyword(s) or a question to analyze your documents:")

if user_q:
    qvec = embed_texts([user_q])
    results = st.session_state.store.search(qvec, k=min(3, len(st.session_state.store.ids)))

    if not results:
        st.info("No indexed documents found. Please upload a file first.")
    else:
        st.write("**Top matches:**")
        for doc_id, dist in results:
            st.write(f"• {doc_id} — distance: {dist:.4f}")

        top_doc = results[0][0]
        doc_data = st.session_state.docs.get(top_doc, {})

        keyword = user_q.lower()
        short_sum = doc_data.get("summary10", "")
        long_sum = doc_data.get("summary100", "")

        # Keyword-based filtering
        filtered_short = " ".join([s for s in short_sum.split(". ") if keyword in s.lower()])
        filtered_long = " ".join([s for s in long_sum.split(". ") if keyword in s.lower()])

        st.markdown("### 🩵 Short Summary (10%)")
        st.text_area("Summary 10%", value=filtered_short or short_sum, height=200)

        st.markdown("### 💠 Detailed Summary (100%)")
        st.text_area("Summary 100%", value=filtered_long or long_sum, height=300)

st.divider()

# =============================
# View / Download Documents
# =============================
if st.session_state.docs:
    st.subheader("📚 Uploaded Documents & Summaries")
    tabs = st.tabs(list(st.session_state.docs.keys()))
    for tab, name in zip(tabs, st.session_state.docs.keys()):
        with tab:
            data = st.session_state.docs[name]
            st.markdown(f"### 📝 Full Text of {name}")
            st.text_area("Extracted Text", value=data["text"], height=250)

            st.markdown("**Short Summary (10%)**")
            st.text_area("10% Summary", value=data["summary10"], height=150)

            st.markdown("**Detailed Summary (100%)**")
            st.text_area("100% Summary", value=data["summary100"], height=200)

            st.download_button("⬇️ Download Extracted Text", data=data["text"],
                               file_name=f"{name}_extracted.txt", mime="text/plain")
            st.download_button("⬇️ Download Summaries",
                               data=f"SHORT SUMMARY (10%):\n\n{data['summary10']}\n\n---\n\nDETAILED SUMMARY (100%):\n\n{data['summary100']}",
                               file_name=f"{name}_summaries.txt", mime="text/plain")
else:
    st.info("Upload PDFs or DOCX files to begin analysis.")
