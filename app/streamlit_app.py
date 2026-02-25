"""
Streamlit demo UI for the RAG system.

Run: streamlit run app/streamlit_app.py
"""
import os
import sys
import time

import streamlit as st

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from docint.config import load_config
from docint.pipeline import RAGPipeline
from docint.verification.groundedness import GroundednessChecker


@st.cache_resource
def load_pipeline():
    """Load and cache the RAG pipeline."""
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        st.error("OPENAI_API_KEY not set. Please set it in your environment.")
        st.stop()
    
    config = load_config('configs/default.yaml')
    config.openai_api_key = api_key
    
    pipeline = RAGPipeline(config)
    pipeline.ingest_directory('eval_data/documents')
    
    checker = GroundednessChecker(api_key=api_key)
    
    return pipeline, checker


def main():
    st.set_page_config(
        page_title="Doc Intelligence",
        page_icon="📚",
        layout="wide",
    )
    
    st.title("📚 Doc Intelligence")
    st.markdown("*RAG-powered document Q&A*")
    
    # Load pipeline
    with st.spinner("Loading pipeline..."):
        pipeline, checker = load_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        verify_groundedness = st.checkbox(
            "Verify groundedness",
            value=False,
            help="Check if answer is supported by context (adds ~1-2s)",
        )
        
        show_sources = st.checkbox(
            "Show source chunks",
            value=True,
        )
        
        st.divider()
        
        st.header("📊 Stats")
        stats = pipeline.get_stats()
        st.metric("Chunks indexed", stats.get("chunk_count", 0))
        st.metric("Retrieval type", stats.get("retriever", {}).get("type", "simple"))
        
        st.divider()
        
        st.header("📝 Example Questions")
        examples = [
            "What was TechCorp's Q3 2024 revenue?",
            "How many employees does TechCorp have?",
            "What security certifications does CloudScale have?",
            "How much does CloudScale storage cost?",
            "What was the SecureNet acquisition price?",
        ]
        
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.question = ex
    
    # Main area
    question = st.text_input(
        "Ask a question about the documents:",
        value=st.session_state.get("question", ""),
        placeholder="e.g., What was TechCorp's Q3 2024 revenue?",
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and question:
        with st.spinner("Searching..."):
            start = time.time()
            result = pipeline.query(question)
            query_time = time.time() - start
        
        # Answer
        st.header("💬 Answer")
        st.markdown(result.answer)
        
        # Groundedness check
        if verify_groundedness:
            with st.spinner("Verifying groundedness..."):
                check = checker.check(
                    question=question,
                    answer=result.answer,
                    context_chunks=result.retrieval.contexts,
                )
            
            if check.is_grounded:
                st.success(f"✅ Grounded (confidence: {check.confidence:.0%})")
            else:
                st.warning(f"⚠️ Possibly hallucinated (confidence: {check.confidence:.0%})")
                if check.unsupported_claims:
                    st.markdown("**Unsupported claims:**")
                    for claim in check.unsupported_claims:
                        st.markdown(f"- {claim}")
        
        # Metrics
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total time", f"{query_time:.2f}s")
        with col2:
            st.metric("Retrieval", f"{result.latency_ms.get('retrieval_ms', 0):.0f}ms")
        with col3:
            st.metric("Generation", f"{result.latency_ms.get('generation_ms', 0):.0f}ms")
        
        # Sources
        if show_sources:
            st.divider()
            st.header("📄 Sources")
            
            for i, r in enumerate(result.retrieval.results[:3]):
                with st.expander(
                    f"Source {i+1} (score: {r.score:.3f}) - {r.metadata.get('filename', 'unknown')}"
                ):
                    st.markdown(r.content)
    
    elif search_button:
        st.warning("Please enter a question.")


if __name__ == "__main__":
    main()