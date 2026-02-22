# Hallucination Detection Notes

## What I tried

Used LLM to verify its own outputs. Sounds circular but works because:
1. Generation is creative (might hallucinate)
2. Verification is analytical (checking facts against text)

## What surprised me

The "CEO's name" query worked well - RAG correctly said "context doesn't provide this info" instead of making up a name. The groundedness checker confirmed this was grounded (saying you don't know IS grounded).

## Edge cases

- Partial hallucinations are tricky. If 3 claims are correct and 1 is wrong, is it grounded? Current approach: NO, any unsupported claim = not grounded.
- Paraphrasing vs hallucination. "12,500 employees" vs "about 12,500 staff" - both grounded. Checker handles this.

## Limitations

- Adds latency (1-2s per verification)
- Uses extra API calls
- Can't catch subtle hallucinations (wrong date, slightly off numbers)

## When to use

- Production: Spot-check a sample, not every query
- Debugging: Find why answers are wrong
- High-stakes: Medical, legal, financial queries
