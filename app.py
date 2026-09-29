import os
import sys

import streamlit as st
import numpy as np
import faiss
import pymupdf

from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai


# =========================================================
# PROJECT PATH
# =========================================================

sys.path.append("src")

from chunker import create_chunks


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PaperWise",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# =========================================================
# LOAD RERANKER
# =========================================================

@st.cache_resource
def load_reranker():

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return model


# =========================================================
# GEMINI CLIENT
# =========================================================

api_key = os.getenv("GEMINI_API_KEY")

if api_key:

    gemini_client = genai.Client(
        api_key=api_key
    )

else:

    gemini_client = None


# =========================================================
# CUSTOM CSS ONLY
#
# IMPORTANT:
# We are NOT creating the interface with HTML.
# Streamlit native components create the interface.
# CSS only changes their appearance.
# =========================================================

st.markdown(
    """
    <style>

    /* ================================================
       MAIN BACKGROUND
       ================================================ */

    .stApp {
        background-color: #f5f7fa;
        color: #172033;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ================================================
       GLOBAL TEXT
       ================================================ */

    .stApp p,
    .stApp label,
    .stApp span {
        color: #374151;
    }

    h1, h2, h3, h4 {
        color: #111827 !important;
    }


    /* ================================================
       SIDEBAR
       ================================================ */

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #111827 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #374151 !important;
    }


    /* ================================================
       TEXT INPUT
       ================================================ */

    .stTextInput input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        min-height: 48px;
    }

    .stTextInput input::placeholder {
        color: #9ca3af !important;
    }


    /* ================================================
       BUTTONS
       ================================================ */

    .stButton button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
        border: 1px solid #d1d5db;
        background-color: #ffffff;
        color: #1f2937;
    }

    .stButton button:hover {
        border-color: #5962d8;
        color: #5962d8;
    }


    /* ================================================
       PRIMARY BUTTON
       ================================================ */

    button[kind="primary"] {
        background-color: #5962d8 !important;
        color: white !important;
        border: none !important;
    }

    button[kind="primary"]:hover {
        background-color: #4d56c9 !important;
        color: white !important;
    }


    /* ================================================
       FILE UPLOADER
       ================================================ */

    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px solid #e1e5eb;
        border-radius: 10px;
        padding: 8px;
    }


    /* ================================================
       EXPANDERS
       ================================================ */

    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e1e5eb;
        border-radius: 10px;
    }


    /* ================================================
       TABS
       ================================================ */

    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #6b7280;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #5962d8 !important;
    }


    /* ================================================
       METRICS
       ================================================ */

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e1e5eb;
        border-radius: 10px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }


    /* ================================================
       DIVIDERS
       ================================================ */

    hr {
        border-color: #e5e7eb;
    }


    /* ================================================
       INFO / SUCCESS / WARNING
       ================================================ */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_uploaded_pdf(pdf_bytes):

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        pages.append(
            {
                "page": page_number + 1,
                "text": text
            }
        )

    document.close()

    return pages


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_document(
    pdf_bytes,
    filename
):

    pages = extract_uploaded_pdf(
        pdf_bytes
    )

    chunks = create_chunks(
        pages,
        filename,
        chunk_size=5,
        overlap=2
    )

    return chunks


# =========================================================
# BUILD VECTOR INDEX
# =========================================================

def build_index():

    all_chunks = []

    for filename, pdf_bytes in (
        st.session_state.documents.items()
    ):

        chunks = process_document(
            pdf_bytes,
            filename
        )

        all_chunks.extend(
            chunks
        )

    if not all_chunks:

        return None, []

    # Assign global chunk IDs
    for index_number, chunk in enumerate(
        all_chunks
    ):

        chunk["chunk_id"] = index_number

    # Load embedding model
    model = load_embedding_model()

    # Extract text
    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    # Create embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(
        embeddings
    )

    # Embedding dimension
    dimension = embeddings.shape[1]

    # FAISS index
    index = faiss.IndexFlatIP(
        dimension
    )

    # Add vectors
    index.add(
        embeddings
    )

    return index, all_chunks


# =========================================================
# SEARCH DOCUMENTS
# =========================================================

def search_documents(
    question,
    top_k
):

    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):

        return []

    model = load_embedding_model()

    # Embed question
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Normalize
    faiss.normalize_L2(
        query_embedding
    )

    # Prevent asking FAISS for more vectors
    # than actually exist
    top_k = min(
        top_k,
        st.session_state.index.ntotal
    )

    # Search
    scores, indices = (
        st.session_state.index.search(
            query_embedding,
            top_k
        )
    )

    results = []

    for index_id, score in zip(
        indices[0],
        scores[0]
    ):

        chunk = st.session_state.chunks[
            int(index_id)
        ]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score)
            }
        )

    return results


# =========================================================
# RERANK RESULTS
# =========================================================

def rerank_results(
    question,
    results
):

    if not results:

        return []

    reranker = load_reranker()

    pairs = [
        [
            question,
            result["text"]
        ]
        for result in results
    ]

    scores = reranker.predict(
        pairs
    )

    for result, score in zip(
        results,
        scores
    ):

        result["rerank_score"] = float(
            score
        )

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results


# =========================================================
# CREATE LLM CONTEXT
# =========================================================

def create_context(
    results
):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE [{number}]
Document: {result['source']}
Page: {result['page']}

{result['text']}
"""
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# GEMINI GENERATION
# =========================================================

def generate_answer(
    question,
    context
):

    if gemini_client is None:

        return None, "NO_API_KEY"

    prompt = f"""
You are PaperWise, a document question-answering assistant.

Answer the user's question using ONLY the provided context.

RULES:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer is not available in the context, say:
   "The answer is not available in the provided documents."
5. Keep the answer concise and useful.
6. Cite information using [1], [2], [3], etc.
7. Only use citation numbers that exist in the context.
8. Put citations immediately after the statement they support.

DOCUMENT CONTEXT
==============================

{context}

==============================

QUESTION

{question}

==============================

ANSWER
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text, "SUCCESS"

    except Exception as error:

        error_text = str(error)

        if "503" in error_text:

            return None, "HIGH_DEMAND"

        return None, error_text


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # Brand
    st.title("📚 PaperWise")

    st.caption(
        "Research Intelligence Platform"
    )

    st.divider()

    # -----------------------------------------------------
    # DOCUMENT LIBRARY
    # -----------------------------------------------------

    st.subheader(
        "📂 Document Library"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload research papers, textbooks, lecture notes, etc."
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            file_bytes = uploaded_file.getvalue()

            # Add only if new
            if filename not in st.session_state.documents:

                st.session_state.documents[
                    filename
                ] = file_bytes

                # Index is now outdated
                st.session_state.indexed = False

                st.session_state.index = None

                st.session_state.chunks = []


    # -----------------------------------------------------
    # SHOW DOCUMENTS
    # -----------------------------------------------------

    if st.session_state.documents:

        st.write(
            f"**{len(st.session_state.documents)} "
            f"document(s) selected**"
        )

        for filename in list(
            st.session_state.documents.keys()
        ):

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                st.write(
                    f"📄 {filename}"
                )

            with col2:

                if st.button(
                    "×",
                    key=f"remove_{filename}"
                ):

                    del st.session_state.documents[
                        filename
                    ]

                    st.session_state.indexed = False

                    st.session_state.index = None

                    st.session_state.chunks = []

                    st.rerun()

    else:

        st.info(
            "Upload a PDF to get started."
        )


    # -----------------------------------------------------
    # CLEAR ALL
    # -----------------------------------------------------

    if st.session_state.documents:

        if st.button(
            "🗑️ Clear All Documents",
            use_container_width=True
        ):

            st.session_state.documents = {}

            st.session_state.chunks = []

            st.session_state.index = None

            st.session_state.indexed = False

            st.session_state.last_results = []

            st.rerun()


    # -----------------------------------------------------
    # INDEX
    # -----------------------------------------------------

    if st.session_state.documents:

        st.write("")

        if st.button(
            "⚡ Index Documents",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting → Chunking → Embedding → Indexing..."
            ):

                index, chunks = build_index()

                st.session_state.index = index

                st.session_state.chunks = chunks

                st.session_state.indexed = True

            st.success(
                f"Successfully indexed {len(chunks)} chunks."
            )


    st.divider()

    # -----------------------------------------------------
    # SEARCH SETTINGS
    # -----------------------------------------------------

    st.subheader(
        "⚙️ Search Settings"
    )

    retrieval_k = st.slider(
        "Chunks retrieved from FAISS",
        min_value=3,
        max_value=20,
        value=10,
        step=1
    )

    final_k = st.slider(
        "Chunks sent to LLM",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

    show_scores = st.checkbox(
        "Show similarity scores",
        value=True
    )

    st.divider()

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    st.subheader(
        "System Status"
    )

    if st.session_state.indexed:

        st.success(
            "✓ Vector index ready"
        )

    else:

        st.warning(
            "○ Documents not indexed"
        )

    if gemini_client:

        st.success(
            "✓ Gemini connected"
        )

    else:

        st.warning(
            "○ Gemini not configured"
        )


# =========================================================
# MAIN HEADER
# =========================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.title(
        "PaperWise"
    )

    st.caption(
        "Ask questions. Retrieve evidence. Understand papers."
    )

with header_right:

    if gemini_client:

        st.success(
            "Generator Connected"
        )

    else:

        st.warning(
            "Generator Offline"
        )


st.divider()


# =========================================================
# HERO
# =========================================================

st.header(
    "Your research, searchable."
)

st.write(
    "Upload research papers, lecture notes or technical "
    "documents and ask questions using a Retrieval-Augmented "
    "Generation pipeline grounded in your own documents."
)


# =========================================================
# PIPELINE
# =========================================================

st.subheader(
    "🧠 RAG Pipeline"
)

pipeline_columns = st.columns(11)

pipeline_steps = [
    "📄 Document",
    "→",
    "✂️ Chunk",
    "→",
    "🧠 Embed",
    "→",
    "🔎 FAISS",
    "→",
    "🎯 Rerank",
    "→",
    "🤖 Generate"
]

for column, step in zip(
    pipeline_columns,
    pipeline_steps
):

    with column:

        if step == "→":

            st.write(
                step
            )

        else:

            st.info(
                step
            )


st.divider()


# =========================================================
# QUESTION AREA
# =========================================================

st.subheader(
    "💬 Ask Your Documents"
)

question = st.text_input(
    "Question",
    placeholder=(
        "Example: What is the main objective of this paper?"
    ),
    label_visibility="collapsed"
)

ask_button = st.button(
    "🔍 Search & Ask",
    type="primary",
    use_container_width=True
)


# =========================================================
# ASK QUESTION
# =========================================================

if ask_button:

    if not st.session_state.indexed:

        st.warning(
            "Please upload a PDF and click "
            "'Index Documents' first."
        )

        st.stop()


    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # -----------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------

    with st.spinner(
        "🔎 Searching your documents..."
    ):

        retrieved = search_documents(
            question,
            retrieval_k
        )


    # -----------------------------------------------------
    # RERANKING
    # -----------------------------------------------------

    with st.spinner(
        "🎯 Reranking relevant passages..."
    ):

        reranked = rerank_results(
            question,
            retrieved
        )


    # -----------------------------------------------------
    # FINAL CONTEXT
    # -----------------------------------------------------

    final_results = reranked[
        :final_k
    ]

    st.session_state.last_results = (
        final_results
    )

    st.session_state.last_question = (
        question
    )


    # -----------------------------------------------------
    # CREATE CONTEXT
    # -----------------------------------------------------

    context = create_context(
        final_results
    )


    # =====================================================
    # METRICS
    # =====================================================

    st.write("")

    metric1, metric2, metric3, metric4 = st.columns(
        4
    )

    with metric1:

        st.metric(
            "Retrieved",
            len(retrieved)
        )

    with metric2:

        st.metric(
            "Final Context",
            len(final_results)
        )

    with metric3:

        st.metric(
            "Documents",
            len(st.session_state.documents)
        )

    with metric4:

        st.metric(
            "Embedding Size",
            "384D"
        )


    st.write("")


    # =====================================================
    # TABS
    # =====================================================

    answer_tab, sources_tab, analysis_tab = st.tabs(
        [
            "✨ Answer",
            "📖 Sources",
            "🔬 Retrieval Analysis"
        ]
    )


    # =====================================================
    # ANSWER TAB
    # =====================================================

    with answer_tab:

        st.subheader(
            "Answer"
        )

        if gemini_client is None:

            st.info(
                "Gemini is not configured. "
                "Your retrieval and reranking pipeline "
                "has still completed successfully."
            )

        else:

            with st.spinner(
                "🤖 Generating grounded answer..."
            ):

                answer, status = generate_answer(
                    question,
                    context
                )


            if status == "SUCCESS":

                st.success(
                    "Answer generated."
                )

                st.write(
                    answer
                )

            elif status == "HIGH_DEMAND":

                st.warning(
                    "Gemini is temporarily unavailable "
                    "because the model is experiencing "
                    "high demand."
                )

                st.info(
                    "The retrieval pipeline worked successfully. "
                    "Open the Sources tab to inspect the evidence "
                    "that would have been sent to Gemini."
                )

            else:

                st.error(
                    f"Generation error: {answer}"
                )


    # =====================================================
    # SOURCES TAB
    # =====================================================

    with sources_tab:

        st.subheader(
            "📖 Retrieved Evidence"
        )

        st.caption(
            "These passages were retrieved by FAISS "
            "and reordered by the cross-encoder."
        )


        for number, result in enumerate(
            final_results,
            start=1
        ):

            with st.expander(
                f"[{number}] {result['source']} — Page {result['page']}"
            ):

                st.write(
                    f"**Document:** "
                    f"{result['source']}"
                )

                st.write(
                    f"**Page:** "
                    f"{result['page']}"
                )

                st.write(
                    f"**Chunk ID:** "
                    f"{result['chunk_id']}"
                )

                if show_scores:

                    score1, score2 = st.columns(
                        2
                    )

                    with score1:

                        st.metric(
                            "FAISS Similarity",
                            f"{result['score']:.4f}"
                        )

                    with score2:

                        st.metric(
                            "Reranker Score",
                            f"{result['rerank_score']:.4f}"
                        )

                st.divider()

                st.write(
                    result["text"]
                )


    # =====================================================
    # RETRIEVAL ANALYSIS TAB
    # =====================================================

    with analysis_tab:

        st.subheader(
            "🔬 Retrieval Analysis"
        )

        st.write(
            "This shows how the system selected the "
            "evidence used for the answer."
        )


        for number, result in enumerate(
            final_results,
            start=1
        ):

            st.markdown(
                f"### Rank {number}"
            )

            st.write(
                f"**Document:** {result['source']}"
            )

            st.write(
                f"**Page:** {result['page']}"
            )

            st.write(
                f"**Chunk:** {result['chunk_id']}"
            )

            score1, score2 = st.columns(
                2
            )

            with score1:

                st.metric(
                    "FAISS Similarity",
                    f"{result['score']:.4f}"
                )

            with score2:

                st.metric(
                    "Reranker Score",
                    f"{result['rerank_score']:.4f}"
                )

            st.divider()


# =========================================================
# EMPTY STATE
# =========================================================

if not st.session_state.indexed:

    st.divider()

    st.subheader(
        "📄 Get Started"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.info(
            "1️⃣ Upload one or more PDF documents "
            "from the sidebar."
        )

    with col2:

        st.info(
            "2️⃣ Click 'Index Documents' to build "
            "the FAISS vector store."
        )

    with col3:

        st.info(
            "3️⃣ Ask questions and inspect the "
            "retrieved evidence."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PaperWise · Retrieval Augmented Generation · "
    "Sentence Transformers · FAISS · Cross-Encoder · Gemini"
)