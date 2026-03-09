"""Cloud RAG app with Pinecone vector database."""
import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

st.set_page_config(page_title="Doc Intelligence", page_icon="📚", layout="wide")

st.title("📚 Doc Intelligence")
st.caption("RAG-powered document Q&A")

# Check API keys
openai_key = os.environ.get("OPENAI_API_KEY")
pinecone_key = os.environ.get("PINECONE_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY not set")
    st.stop()

if not pinecone_key:
    st.error("PINECONE_API_KEY not set")
    st.stop()

# Initialize clients
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("doc-intelligence")

# Sample documents
DOCUMENTS = {
    "techcorp": """# TechCorp Annual Report 2024
Total revenue: $4.2 billion (23% YoY growth)
Q1 2024: $920 million, Q2 2024: $1.05 billion, Q3 2024: $1.15 billion, Q4 2024: $1.08 billion
Cloud services: $2.1 billion (50% of revenue), R&D spending: $240 million
Employees: 12,500. SecureNet acquisition: $320 million.
Regional: North America 62%, Europe 28%, Asia-Pacific 10%.""",
    
    "cloudscale": """# CloudScale Documentation
Storage: Object storage $0.023/GB/month, Infrequent access $0.0125/GB/month
Compute: Up to 64 vCPUs and 256GB RAM
Security: SOC 2 Type II, ISO 27001, HIPAA, PCI DSS certified.""",
    
    "acme": """# Acme Corporation Q4 2024
Revenue: $890 million (28% YoY growth)
Breakdown: Product Sales $520M (58%), Services $280M (31%), Licensing $90M (11%)
Regional: Americas $490M (55%), EMEA $267M (30%), Asia-Pacific $133M (15%)
Profitability: Gross margin 42%, Operating margin 18%, Net income $142M
Employees: 8,200 total.""",
    
    "dataflow": """# DataFlow AI Series A
ARR: $2.4 million, Customers: 45 enterprises
LTV/CAC ratio: 7.1x, Payback period: 8 months
Team: 38 employees, Founded 2023.""",
    
    "nexus": """# Nexus Cloud Platform
GPU Instances: GPU1 $2.10/hr, GPU4 $8.40/hr, GPU8 $24.00/hr (8x NVIDIA A100)
Storage: SSD $0.12/GB/month, HDD Archive $0.02/GB/month
Certifications: SOC 2 Type II, ISO 27001, HIPAA, PCI DSS, FedRAMP."""
}


def get_embedding(text: str) -> list:
    """Get OpenAI embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def index_documents():
    """Index all documents to Pinecone."""
    vectors = []
    for doc_id, content in DOCUMENTS.items():
        embedding = get_embedding(content)
        vectors.append({
            "id": doc_id,
            "values": embedding,
            "metadata": {"content": content}
        })
    index.upsert(vectors=vectors)
    return len(vectors)


def search(query: str, top_k: int = 3) -> list:
    """Search Pinecone for relevant documents."""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match.metadata["content"] for match in results.matches]


def generate_answer(query: str, context: str) -> str:
    """Generate answer using GPT."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer based only on this context. Be concise.\n\nContext:\n{context}"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


# Sidebar
st.sidebar.header("📝 Example Questions")
st.sidebar.markdown("""
- What was TechCorp's Q3 2024 revenue?
- What is Acme's revenue breakdown?
- How much does Nexus GPU8 cost?
- What is DataFlow AI's LTV/CAC ratio?
- What certifications does CloudScale have?
""")

st.sidebar.header("⚙️ Admin")
if st.sidebar.button("🔄 Re-index Documents"):
    with st.spinner("Indexing..."):
        count = index_documents()
        st.sidebar.success(f"Indexed {count} documents")

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
RAG system with Pinecone vector search.

[GitHub](https://github.com/SAMithila/doc-intelligence)
""")

# Main content
st.subheader("Ask a question:")
question = st.text_input("", placeholder="e.g., What is Acme's revenue breakdown?", label_visibility="collapsed")

if st.button("🔍 Search", type="primary"):
    if question:
        with st.spinner("Searching..."):
            # Retrieve relevant documents
            relevant_docs = search(question, top_k=3)
            context = "\n\n".join(relevant_docs)
            
            # Generate answer
            answer = generate_answer(question, context)
            
            st.subheader("💬 Answer")
            st.write(answer)
            st.success("✅ Grounded in context")
            
            # Show sources
            with st.expander("📄 Sources"):
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(doc[:300] + "..." if len(doc) > 300 else doc)
    else:
        st.warning("Please enter a question")
