"""
Hallucination detection via groundedness checking.

Verifies that generated answers are supported by retrieved context.
"""
from dataclasses import dataclass
import openai


@dataclass
class GroundednessResult:
    """Result of groundedness check."""
    is_grounded: bool
    confidence: float  # 0-1
    supported_claims: list[str]
    unsupported_claims: list[str]
    explanation: str


class GroundednessChecker:
    """
    Checks if an answer is grounded in the provided context.

    Uses LLM to verify each claim in the answer has support
    in the retrieved chunks.
    """

    CHECK_PROMPT = """You are a fact-checker. Your job is to verify if an answer is fully supported by the given context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER TO VERIFY: {answer}

Analyze the answer and determine:
1. Is every claim in the answer supported by the context?
2. Are there any claims that are NOT in the context (hallucinations)?

Respond in this exact format:
GROUNDED: [YES/NO]
CONFIDENCE: [0.0-1.0]
SUPPORTED_CLAIMS:
- [claim 1 that is supported]
- [claim 2 that is supported]
UNSUPPORTED_CLAIMS:
- [claim that is NOT in context]
- [another unsupported claim]
EXPLANATION: [brief explanation of your assessment]

If there are no unsupported claims, write "None" under UNSUPPORTED_CLAIMS.
If there are no supported claims, write "None" under SUPPORTED_CLAIMS."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def check(
        self,
        question: str,
        answer: str,
        context_chunks: list[str],
    ) -> GroundednessResult:
        """
        Check if answer is grounded in context.

        Args:
            question: The original question
            answer: The generated answer to verify
            context_chunks: Retrieved context chunks

        Returns:
            GroundednessResult with verification details
        """
        context = "\n\n---\n\n".join(context_chunks)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": self.CHECK_PROMPT.format(
                    context=context,
                    question=question,
                    answer=answer,
                )
            }],
            temperature=0,
            max_tokens=500,
        )

        result_text = response.choices[0].message.content
        return self._parse_result(result_text)

    def _parse_result(self, text: str) -> GroundednessResult:
        """Parse the LLM response into structured result."""
        lines = text.strip().split('\n')

        is_grounded = False
        confidence = 0.5
        supported = []
        unsupported = []
        explanation = ""

        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("GROUNDED:"):
                is_grounded = "YES" in line.upper()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except:
                    confidence = 0.5
            elif line.startswith("SUPPORTED_CLAIMS:"):
                current_section = "supported"
            elif line.startswith("UNSUPPORTED_CLAIMS:"):
                current_section = "unsupported"
            elif line.startswith("EXPLANATION:"):
                current_section = "explanation"
                explanation = line.split(
                    ":", 1)[1].strip() if ":" in line else ""
            elif line.startswith("- "):
                claim = line[2:].strip()
                if claim.lower() != "none":
                    if current_section == "supported":
                        supported.append(claim)
                    elif current_section == "unsupported":
                        unsupported.append(claim)
            elif current_section == "explanation" and line:
                explanation += " " + line

        return GroundednessResult(
            is_grounded=is_grounded,
            confidence=confidence,
            supported_claims=supported,
            unsupported_claims=unsupported,
            explanation=explanation.strip(),
        )


class CitationExtractor:
    """
    Extracts which context chunks support which parts of the answer.
    """

    CITATION_PROMPT = """Given an answer and context chunks, identify which chunks support each claim.

CONTEXT CHUNKS:
{chunks}

ANSWER: {answer}

For each factual claim in the answer, identify which chunk number (1, 2, 3, etc.) supports it.
If a claim is not supported by any chunk, mark it as [UNSUPPORTED].

Format:
CLAIM: [the claim]
SOURCE: [chunk number or UNSUPPORTED]

List all claims:"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def extract(
        self,
        answer: str,
        context_chunks: list[str],
    ) -> list[dict]:
        """
        Extract citations for claims in the answer.

        Returns list of {claim, source_chunk_index, supported}
        """
        # Format chunks with numbers
        numbered_chunks = "\n\n".join(
            f"[Chunk {i+1}]: {chunk[:500]}"
            for i, chunk in enumerate(context_chunks)
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": self.CITATION_PROMPT.format(
                    chunks=numbered_chunks,
                    answer=answer,
                )
            }],
            temperature=0,
            max_tokens=500,
        )

        return self._parse_citations(response.choices[0].message.content)

    def _parse_citations(self, text: str) -> list[dict]:
        """Parse citation response."""
        citations = []
        lines = text.strip().split('\n')

        current_claim = None

        for line in lines:
            line = line.strip()

            if line.startswith("CLAIM:"):
                current_claim = line.split(":", 1)[1].strip()
            elif line.startswith("SOURCE:") and current_claim:
                source = line.split(":", 1)[1].strip()

                if "UNSUPPORTED" in source.upper():
                    citations.append({
                        "claim": current_claim,
                        "source_chunk": None,
                        "supported": False,
                    })
                else:
                    # Extract chunk number
                    try:
                        chunk_num = int(''.join(filter(str.isdigit, source)))
                        citations.append({
                            "claim": current_claim,
                            "source_chunk": chunk_num - 1,  # 0-indexed
                            "supported": True,
                        })
                    except:
                        citations.append({
                            "claim": current_claim,
                            "source_chunk": None,
                            "supported": False,
                        })

                current_claim = None

        return citations
