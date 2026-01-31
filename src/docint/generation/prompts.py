"""
Prompt templates for RAG generation.

Baseline: Simple context + question prompt.
"""

RAG_PROMPT_TEMPLATE = """Answer the question based on the provided context. 
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""


RAG_PROMPT_WITH_CITATIONS = """Answer the question based on the provided context.
Include citations in [Source N] format when referencing specific information.
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer (with citations):"""


def format_context(chunks: list[str], include_numbers: bool = True) -> str:
    """Format context chunks for prompt."""
    if include_numbers:
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[Source {i}]\n{chunk}")
        return "\n\n".join(formatted)
    else:
        return "\n\n---\n\n".join(chunks)


def build_rag_prompt(
    question: str,
    context_chunks: list[str],
    include_citations: bool = False,
) -> str:
    """Build complete RAG prompt."""
    context = format_context(context_chunks, include_numbers=include_citations)
    
    template = RAG_PROMPT_WITH_CITATIONS if include_citations else RAG_PROMPT_TEMPLATE
    
    return template.format(
        context=context,
        question=question,
    )
