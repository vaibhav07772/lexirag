"""LexiRAG — Streamlit UI"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
from datetime import datetime

from src.orchestrator.graph import LexiRAG


# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexiRAG — Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E40AF, #0891B2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background: #F0F9FF;
        border-left: 4px solid #0891B2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .citation {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .confidence-high {
        color: #059669;
        font-weight: 700;
    }
    .confidence-mid {
        color: #D97706;
        font-weight: 700;
    }
    .confidence-low {
        color: #DC2626;
        font-weight: 700;
    }
    .blocked-box {
        background: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .source-card {
        background: #F9FAFB;
        border-left: 3px solid #9CA3AF;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">⚖️ LexiRAG</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Production-grade Legal AI Assistant for Indian Law — Hybrid RAG + Guardrails</p>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    top_k = st.slider("Top K sources", 3, 10, 5)
    show_sources = st.checkbox("Show retrieved sources", value=True)
    show_debug = st.checkbox("Show debug info", value=False)

    st.markdown("---")
    st.markdown("### 🛡️ Safety Layers")
    st.markdown("""
    - **🔒 Input Guard** — PII + injection
    - **📚 Hybrid Retriever** — BM25 + Semantic + RRF
    - **🎯 Cross-Encoder Rerank**
    - **✅ Output Guard** — Citation validation
    """)

    st.markdown("---")
    st.markdown("### 📊 Coverage")
    st.markdown("""
    - Constitution of India
    - Indian Penal Code (IPC)
    - CrPC
    - Contract Act
    - 15,972 legal Q&A pairs
    """)

    st.markdown("---")
    st.caption("v0.1.0 • Hybrid RAG + LangGraph")


# ─────────────────────────────────────────────────────────
# MAIN INPUT
# ─────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])

with col1:
    question = st.text_input(
        "❓ Ask a legal question",
        value="",
        placeholder="e.g., What is Section 302 IPC?",
        label_visibility="collapsed",
    )

with col2:
    ask_button = st.button("🔍 Ask", use_container_width=True, type="primary")


# Example questions
st.markdown("**Try:**")
example_cols = st.columns(4)
examples = [
    "What is Section 302 IPC?",
    "How to file an FIR?",
    "What is Article 21?",
    "Grounds for divorce?",
]
for i, ex in enumerate(examples):
    with example_cols[i]:
        if st.button(ex, use_container_width=True, key=f"ex_{i}"):
            question = ex
            ask_button = True


# ─────────────────────────────────────────────────────────
# INIT SESSION STATE
# ─────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    with st.spinner("🔧 Initializing LexiRAG (loading models)..."):
        st.session_state.rag = LexiRAG()
        st.session_state.history = []


# ─────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────
if ask_button and question.strip():
    st.markdown("---")

    with st.spinner("🔍 Processing your question..."):
        result = st.session_state.rag.ask(question)

    # Save to history
    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    # ─── BLOCKED CASE ───
    if result["blocked"]:
        st.markdown(
            f'<div class="blocked-box">'
            f'<strong>🚫 Blocked by Input Guardrail</strong><br>'
            f'{result["block_reason"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # ─── ANSWER ───
        st.markdown("### ✅ Answer")
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # ─── CONFIDENCE + CITATIONS ───
        col_a, col_b, col_c = st.columns(3)

        conf = result["confidence"]
        if conf >= 0.7:
            conf_class = "confidence-high"
            conf_label = "🟢 High"
        elif conf >= 0.4:
            conf_class = "confidence-mid"
            conf_label = "🟡 Medium"
        else:
            conf_class = "confidence-low"
            conf_label = "🔴 Low"

        with col_a:
            st.markdown("**Confidence**")
            st.markdown(
                f'<span class="{conf_class}">{conf_label} — {conf:.2f}</span>',
                unsafe_allow_html=True,
            )
            st.progress(float(conf))

        with col_b:
            st.markdown("**Citations**")
            if result["citations"]:
                for c in result["citations"]:
                    st.markdown(
                        f'<span class="citation">{c}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No citations")

        with col_c:
            st.markdown("**Sources**")
            st.markdown(f"**{len(result['sources'])}** retrieved")

        # ─── SOURCES ───
        if show_sources and result["sources"]:
            st.markdown("### 📚 Retrieved Sources")
            for i, src in enumerate(result["sources"], 1):
                score = src["score"]
                score_color = "#059669" if score > 0 else "#DC2626"
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>[{i}]</strong> {src["source"]} — '
                    f'<span style="color:{score_color}">score: {score:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ─── DEBUG ───
        if show_debug:
            with st.expander("🔍 Debug Info"):
                st.json({
                    "question": question,
                    "confidence": result["confidence"],
                    "citations": result["citations"],
                    "sources_count": len(result["sources"]),
                    "blocked": result["blocked"],
                })


# ─────────────────────────────────────────────────────────
# LANDING STATE
# ─────────────────────────────────────────────────────────
elif not st.session_state.get("history"):
    st.markdown("---")
    st.markdown("### 🎯 How It Works")

    cols = st.columns(4)
    with cols[0]:
        st.markdown("#### 🛡️ Input Guard")
        st.caption("Detects PII + prompt injection before LLM call")
    with cols[1]:
        st.markdown("#### 📚 Hybrid Retrieval")
        st.caption("BM25 + Semantic search + RRF fusion")
    with cols[2]:
        st.markdown("#### 🎯 Reranking")
        st.caption("Cross-Encoder reranks for precision")
    with cols[3]:
        st.markdown("#### ✅ Output Guard")
        st.caption("Validates citations, detects PII leaks")

    st.info("👆 Enter a legal question above and click **Ask** to start.")