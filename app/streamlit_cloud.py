"""Simplified Streamlit app for cloud deployment."""
import os
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Doc Intelligence", page_icon="📚")

st.title("📚 Doc Intelligence")
st.caption("RAG-powered document Q&A")

# Check API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set")
    st.stop()

client = OpenAI(api_key=api_key)

# Sample documents (embedded for demo)
DOCS = """
# TechCorp Annual Report 2024

## Financial Highlights
- Total revenue: $4.2 billion (23% YoY growth)
- Q1 2024: $920 million
- Q2 2024: $1.05 billion
- Q3 2024: $1.15 billion (strongest quarter)
- Q4 2024: $1.08 billion
- Cloud services: $2.1 billion (50% of revenue)
- R&D spending: $240 million

## Regional Revenue
- North America: 62% ($2.6 billion)
- Europe/EMEA: 28%
- Asia-Pacific: 10%

## Workforce
- Total employees: 12,500 (up from 10,200)
- New engineering centers: Austin, Texas and Dublin, Ireland

## Acquisitions
- SecureNet: $320 million

## 2025 Outlook
- Projected revenue: $4.8 to $5.0 billion

# CloudScale Documentation

## Storage Pricing
- Object storage: $0.023 per GB/month
- Infrequent access: $0.0125 per GB/month

## Compute Instances
- Up to 64 vCPUs and 256GB RAM

## Security Certifications
- SOC 2 Type II
- ISO 27001
- HIPAA
- PCI DSS
"""

question = st.text_input("Ask a question:", placeholder="What was TechCorp's Q3 2024 revenue?")

if st.button("�� Search") and question:
    with st.spinner("Searching..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Answer questions based only on this context. Be concise.\n\nContext:\n{DOCS}"},
                {"role": "user", "content": question}
            ]
        )
        st.subheader("💬 Answer")
        st.write(response.choices[0].message.content)
        st.success("✅ Grounded in context")

st.sidebar.header("📝 Example Questions")
st.sidebar.markdown("""
- What was TechCorp's Q3 2024 revenue?
- How many employees does TechCorp have?
- What security certifications does CloudScale have?
- What was the SecureNet acquisition price?
""")

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This is a demo of Doc Intelligence, a RAG system for document Q&A.

[View full project on GitHub](https://github.com/SAMithila/doc-intelligence)
""")
