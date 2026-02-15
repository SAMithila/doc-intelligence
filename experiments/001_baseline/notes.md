# Baseline Experiment Notes

**Goal:** Get simplest RAG working, measure what breaks

## Hypothesis

Fixed-size chunking + semantic search should handle basic factual queries.

## What I Actually Tested

5 queries against TechCorp annual report and CloudScale docs:
1. Q3 revenue
2. Storage cost
3. Employee count
4. Security certs
5. Acquisition price

## Results

4/5 passed. Q3 revenue failed.

## What Surprised Me

The Q3 failure wasn't an embedding problem or prompt problem. It was chunking.

I spent 2 hours checking:
- Embedding similarity scores (looked fine)
- Prompt template (standard RAG prompt)
- Top-k setting (tried 3, 5, 10)

Then I finally printed the actual retrieved chunks:

```
Chunk 3: "ervices revenue hit $580 million..."
```

The chunk started with "ervices". Took me way too long to realize fixed chunking cut "services" in half.

## What I Learned

Always inspect retrieved chunks, not just final answers. The retrieval step is where most RAG failures happen.

## Next Steps

Try recursive chunking that respects sentence/paragraph boundaries.