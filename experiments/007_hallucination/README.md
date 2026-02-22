# Hallucination Detection

Added verification to check if answers are grounded in context.

## The Problem

RAG can still hallucinate. The LLM might make up facts that aren't in the retrieved chunks.

Example:
- Context: "Q3 revenue was $1.15B"
- Question: "What was Q3 profit margin?"
- Bad answer: "Profit margin was 23%" ← Made up!

## Solution

LLM-based groundedness checker that:
1. Extracts claims from the answer
2. Checks each claim against context
3. Flags unsupported claims

## Results

| Test Case | Detection |
|-----------|-----------|
| Fully grounded answer | ✅ Correctly passed |
| Fully hallucinated answer | ✅ Caught (confidence: 0.1) |
| Partial hallucination | ✅ Identified which claim was unsupported |

## Latency

Adds ~1-2 seconds per answer for verification.

## Decision

Available as optional verification step. Not on by default since it adds latency, but useful for:
- High-stakes queries
- Debugging retrieval issues
- Building trust with users (show citations)