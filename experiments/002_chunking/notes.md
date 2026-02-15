# Chunking Experiment Notes

**Goal:** Fix Q3 revenue failure by using smarter chunking

## Hypothesis

Recursive chunking that splits on paragraph boundaries will keep section headers with their content.

## What I Tried

Implemented RecursiveChunker with split hierarchy:
1. "\n\n" (double newline - section break)
2. "\n" (single newline - paragraph)
3. ". " (sentence)
4. " " (word - last resort)

## Results

Q3 query: FAIL → PASS

Before (fixed chunking):
```
Chunk: "ervices revenue hit $580 million..."
```

After (recursive chunking):
```
Chunk: "Q3 2024 was our strongest quarter, with revenue reaching $1.15 billion. Cloud services revenue hit $580 million..."
```

The section stays together.

## Chunk Count Change

- Fixed: 15 chunks
- Recursive: 17 chunks

Slightly more chunks but way better quality.

## What Surprised Me

The recursive chunker sometimes creates small chunks (like 150 chars) when a section is naturally short. I expected more uniform sizes.

Not a problem in practice - small chunks just get lower retrieval scores if they're not relevant.

## Edge Case I Found

Headers like "## Q3 2024" are short enough that they could become their own chunk. I considered adding logic to merge headers with following content, but the current approach naturally keeps them together because it splits on double newlines first.

## Decision

Ship recursive chunking as default. The 2 extra chunks are negligible cost.