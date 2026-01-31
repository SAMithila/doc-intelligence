"""
LLM generation for RAG responses.

Baseline: Simple OpenAI completion.
"""
from dataclasses import dataclass
import openai

from docint.generation.prompts import build_rag_prompt


@dataclass
class GenerationResult:
    """Result of generation."""
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    @property
    def cost_estimate(self) -> float:
        """Estimate cost in USD (GPT-4o-mini pricing)."""
        # Approximate pricing: $0.15/1M input, $0.60/1M output
        input_cost = self.prompt_tokens * 0.15 / 1_000_000
        output_cost = self.completion_tokens * 0.60 / 1_000_000
        return input_cost + output_cost


class Generator:
    """
    LLM generator for RAG responses.
    
    Baseline implementation using OpenAI.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        
        self._client = openai.OpenAI(api_key=api_key)
    
    def generate(
        self,
        question: str,
        context_chunks: list[str],
        include_citations: bool = False,
    ) -> GenerationResult:
        """
        Generate answer using RAG.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            include_citations: Whether to ask for citations
            
        Returns:
            GenerationResult with answer and usage info
        """
        # Build prompt
        prompt = build_rag_prompt(
            question=question,
            context_chunks=context_chunks,
            include_citations=include_citations,
        )
        
        # Call LLM
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        
        return GenerationResult(
            answer=response.choices[0].message.content or "",
            model=self._model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
