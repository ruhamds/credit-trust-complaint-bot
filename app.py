"""
app.py
------
CrediTrust Financial — Complaint Intelligence Platform
Streamlit Application

Run:
    streamlit run app.py

Requirements:
    pip install streamlit groq chromadb sentence-transformers python-dotenv
    GROQ_API_KEY in .env file at project root.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import rag_pipeline, VALID_CATEGORIES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrediTrust Intelligence",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #080c14;
    color: #c8d0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0c1220;
    border-right: 1px solid #151e30;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; }

/* ── Wordmark ── */
.wordmark {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #e8c46a;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.wordmark-sub {
    font-size: 0.65rem;
    color: #2e3d58;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: -2px;
}

/* ── Page title ── */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.85rem;
    font-weight: 600;
    color: #e8edf5;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #2e3d58;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}

/* ── Input label ── */
.input-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #2e3d58;
    font-weight: 500;
    margin-bottom: 0.4rem;
}

/* ── Textarea ── */
.stTextArea textarea {
    background-color: #0c1220 !important;
    color: #c8d0e0 !important;
    border: 1px solid #151e30 !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    caret-color: #e8c46a;
    resize: none;
}
.stTextArea textarea:focus {
    border-color: #e8c46a44 !important;
    box-shadow: 0 0 0 1px #e8c46a22 !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #1e2c42 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    border-radius: 3px;
    letter-spacing: 0.05em;
    transition: all 0.12s ease;
    border: none !important;
}
[data-testid="baseButton-primary"] {
    background: #e8c46a !important;
    color: #080c14 !important;
    font-weight: 600 !important;
    padding: 0 1.4rem !important;
}
[data-testid="baseButton-primary"]:hover {
    background: #f0d080 !important;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: #2e3d58 !important;
    border: 1px solid #151e30 !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #2e3d58 !important;
    color: #4a5c78 !important;
}

/* ── Selectbox ── */
.stSelectbox label {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #2e3d58 !important;
}
[data-baseweb="select"] {
    background: #0c1220 !important;
}
[data-baseweb="select"] > div {
    background: #0c1220 !important;
    border-color: #151e30 !important;
    border-radius: 4px !important;
    color: #c8d0e0 !important;
    font-size: 0.85rem !important;
}

/* ── Slider ── */
.stSlider label {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #2e3d58 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #e8c46a !important;
}

/* ── Result container ── */
.result-block {
    border-top: 1px solid #151e30;
    padding-top: 1.8rem;
    margin-top: 1.8rem;
}
.result-question {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 500;
    color: #e8edf5;
    margin-bottom: 0.8rem;
    line-height: 1.4;
}

/* ── Stat bar ── */
.stat-bar {
    display: flex;
    gap: 2rem;
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #0f1825;
}
.stat-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #e8c46a;
}
.stat-key {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #2e3d58;
}

/* ── Answer text ── */
.answer-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #2e3d58;
    margin-bottom: 0.7rem;
}
.answer-body {
    font-size: 0.9rem;
    line-height: 1.8;
    color: #b0bcd0;
}
.answer-body strong, .answer-body b {
    color: #c8d8f0;
    font-weight: 500;
}
.answer-body ul, .answer-body ol {
    padding-left: 1.2rem;
    margin: 0.4rem 0;
}
.answer-body li {
    margin-bottom: 0.25rem;
}

/* ── Sources ── */
.sources-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #2e3d58;
    margin: 1.6rem 0 0.8rem 0;
}
.source-row {
    display: grid;
    grid-template-columns: 28px 1fr;
    gap: 0 1rem;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #0d1520;
}
.source-index {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    color: #e8c46a;
    font-weight: 600;
    padding-top: 2px;
}
.source-meta {
    font-size: 0.72rem;
    color: #2e3d58;
    margin-bottom: 0.3rem;
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    align-items: center;
}
.source-sim {
    color: #4a6080;
    font-variant-numeric: tabular-nums;
}
.source-excerpt {
    font-size: 0.82rem;
    color: #4a5c78;
    line-height: 1.6;
    font-style: italic;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: #151e30;
}
.empty-state-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: #1a2638;
    margin-bottom: 0.5rem;
}
.empty-state-sub {
    font-size: 0.8rem;
    color: #111820;
    line-height: 1.7;
}

/* ── Sidebar divider ── */
.sidebar-divider {
    border: none;
    border-top: 1px solid #151e30;
    margin: 1.2rem 0;
}

/* ── Alert override ── */
.stAlert {
    background: #0c1220 !important;
    border: 1px solid #151e30 !important;
    color: #4a5c78 !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="wordmark">CrediTrust</div>'
        '<div class="wordmark-sub">Complaint Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    selected_category = st.selectbox(
        label   = 'Product Category',
        options = sorted(VALID_CATEGORIES),
        index   = 1,
    )

    top_k = st.slider(
        'Retrieved sources', min_value=3, max_value=10, value=5,
        help='Number of complaint excerpts retrieved per query.'
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # History summary
    if st.session_state.history:
        st.markdown(
            f'<div style="font-size:0.7rem; text-transform:uppercase; '
            f'letter-spacing:0.12em; color:#2e3d58; margin-bottom:0.6rem">'
            f'Session</div>'
            f'<div style="font-size:0.8rem; color:#2e3d58">'
            f'{len(st.session_state.history)} queries</div>',
            unsafe_allow_html=True
        )
        if st.button('Clear session', use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown(
            '<div style="font-size:0.7rem; color:#151e30">No queries yet</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.65rem; color:#151e30; line-height:1.9">'
        'Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#1e2c42">llama-3.1-8b</span><br>'
        'Embeddings&nbsp;<span style="color:#1e2c42">all-MiniLM-L6-v2</span><br>'
        'Vector DB&nbsp;&nbsp;<span style="color:#1e2c42">ChromaDB</span><br>'
        'Corpus&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#1e2c42">CFPB 347k complaints</span>'
        '</div>',
        unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Complaint Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">CFPB Consumer Complaint Analysis · CrediTrust Financial</div>',
    unsafe_allow_html=True
)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-label">Question</div>', unsafe_allow_html=True)
question = st.text_area(
    label            = 'Question',
    placeholder      = f'Ask about {selected_category.lower()} complaints...',
    height           = 80,
    label_visibility = 'collapsed',
)

col_btn, col_ctx, _ = st.columns([1.2, 4, 4])
with col_btn:
    submit = st.button('Analyse →', type='primary', use_container_width=True)
with col_ctx:
    st.markdown(
        f'<div style="padding-top:0.55rem; font-size:0.75rem; color:#1e2c42">'
        f'{selected_category} · {top_k} sources</div>',
        unsafe_allow_html=True
    )

# ── Run pipeline ──────────────────────────────────────────────────────────────
if submit:
    if not question.strip():
        st.warning('Enter a question to continue.')
    else:
        with st.spinner(''):
            try:
                result = rag_pipeline(
                    question         = question.strip(),
                    product_category = selected_category,
                    top_k            = top_k,
                )
                st.session_state.history.insert(0, {
                    'question': question.strip(),
                    'category': selected_category,
                    'answer'  : result['answer'],
                    'chunks'  : result['chunks'],
                    'metrics' : {
                        'avg_sim'  : result['avg_similarity'],
                        'min_sim'  : result['min_similarity'],
                        'latency'  : result['latency_sec'],
                        'tokens'   : result['input_tokens'] + result['output_tokens'],
                    }
                })
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f'Pipeline error: {e}')

# ── Results ───────────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-state-title">No queries yet</div>'
        '<div class="empty-state-sub">'
        'Select a product category in the sidebar,<br>'
        'type a question above, and click Analyse.'
        '</div></div>',
        unsafe_allow_html=True
    )
else:
    for i, entry in enumerate(st.session_state.history):
        m = entry['metrics']

        st.markdown(f'<div class="result-block">', unsafe_allow_html=True)

        # Question
        st.markdown(
            f'<div class="result-question">{entry["question"]}</div>',
            unsafe_allow_html=True
        )

        # Stat bar
        st.markdown(
            f'<div class="stat-bar">'
            f'<div class="stat-item">'
            f'<div class="stat-value">{entry["category"]}</div>'
            f'<div class="stat-key">Category</div></div>'
            f'<div class="stat-item">'
            f'<div class="stat-value">{m["avg_sim"]}</div>'
            f'<div class="stat-key">Avg similarity</div></div>'
            f'<div class="stat-item">'
            f'<div class="stat-value">{m["min_sim"]}</div>'
            f'<div class="stat-key">Min similarity</div></div>'
            f'<div class="stat-item">'
            f'<div class="stat-value">{m["latency"]}s</div>'
            f'<div class="stat-key">Latency</div></div>'
            f'<div class="stat-item">'
            f'<div class="stat-value">{m["tokens"]:,}</div>'
            f'<div class="stat-key">Tokens</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Answer
        st.markdown('<div class="answer-label">Analysis</div>', unsafe_allow_html=True)
        st.markdown(entry['answer'])

        # Sources
        st.markdown('<div class="sources-label">Retrieved Sources</div>', unsafe_allow_html=True)
        for j, chunk in enumerate(entry['chunks'], 1):
            excerpt = chunk['text'][:280] + '...' if len(chunk['text']) > 280 else chunk['text']
            issue   = chunk['issue'][:90] if chunk['issue'] else '—'
            st.markdown(
                f'<div class="source-row">'
                f'<div class="source-index">{j:02d}</div>'
                f'<div>'
                f'<div class="source-meta">'
                f'<span>{chunk["company"]}</span>'
                f'<span>{chunk["state"]}</span>'
                f'<span class="source-sim">sim {chunk["similarity"]}</span>'
                f'</div>'
                f'<div style="font-size:0.7rem; color:#1e2c42; margin-bottom:0.3rem">{issue}</div>'
                f'<div class="source-excerpt">{excerpt}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)