"""
HyDE (Hypothetical Document Embeddings) for query expansion.

Instead of embedding the query directly, we generate a hypothetical
answer and embed that. This often matches better with real documents.
"""
from dataclasses import dataclass
import openai

from docint.embeddings.base import BaseEmbedder


@dataclass
class HyDEResult:
    """Result of HyDE expansion."""
    original_query: str
    hypothetical_document: str
    embedding: list[float]


class HyDEExpander:
    """
    HyDE query expander.

    Generates a hypothetical document that would answer the query,
    then uses that document's embedding for retrieval.

    Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496
    """

    HYDE_PROMPT = """Given the question, write a short paragraph that would answer it.
Write as if you're quoting from a document that contains the answer.
Do not say "I don't know" - make up a plausible answer.

Question: {query}

Answer paragraph:"""

    def __init__(
        self,
        api_key: str,
        embedder: BaseEmbedder,
        model: str = "gpt-4o-mini",
    ):
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._embedder = embedder

    def expand(self, query: str) -> HyDEResult:
        """
        Generate hypothetical document and its embedding.

        Args:
            query: The user's query

        Returns:
            HyDEResult with hypothetical document and embedding
        """
        # Generate hypothetical document
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": self.HYDE_PROMPT.format(query=query)
            }],
            temperature=0.7,  # Some creativity for varied hypotheticals
            max_tokens=150,
        )

        hypothetical_doc = response.choices[0].message.content.strip()

        # Embed the hypothetical document
        embedding = self._embedder.embed_single(hypothetical_doc)

        return HyDEResult(
            original_query=query,
            hypothetical_document=hypothetical_doc,
            embedding=embedding,
        )


class MultiQueryExpander:
    """
    Generates multiple query variations for better retrieval.

    Instead of searching with one query, generate several
    variations and combine results.
    """

    EXPANSION_PROMPT = """Generate 3 different versions of this question.
Each version should ask the same thing but with different wording.
Return only the questions, one per line.

Original question: {query}

Variations:"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ):
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def expand(self, query: str) -> list[str]:
        """
        Generate query variations.

        Args:
            query: Original query

        Returns:
            List of query variations (including original)
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": self.EXPANSION_PROMPT.format(query=query)
            }],
            temperature=0.7,
            max_tokens=150,
        )

        variations_text = response.choices[0].message.content.strip()
        variations = [q.strip()
                      for q in variations_text.split('\n') if q.strip()]

        # Include original query
        return [query] + variations[:3]
