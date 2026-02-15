# Evaluation Framework Notes

**Goal:** Build proper evaluation because I realized my tests were too easy

## The Problem

Days 1-5, I kept getting 100% accuracy. Felt good but something was off.

All my test queries were like "What was TechCorp Q3 2024 revenue?" - specific, clear, obvious.

Of course they worked.

## What I Built

30 queries split into categories:

**Easy (10):** Specific questions, single fact answers
- "What was Q3 revenue?" 
- "How many employees?"

**Vague (10):** Short queries, ambiguous
- "Q3 revenue"
- "storage pricing"
- "employee count"

**Hard (10):** Multi-hop, comparison, calculation
- "Which quarter had highest revenue and by how much?"
- "What % of revenue came from cloud?"

## Results That Changed Everything

| Category | Accuracy |
|----------|----------|
| Easy | 90% |
| Vague | 50% |
| Hard | 90% |

**Vague queries were the weak point all along.**

## Specific Failures

1. **"compute instances"** - Retrieved CloudScale chunk about VM types but it didn't mention the specific "64 vCPU, 256GB" specs. Turns out that info isn't in my test docs clearly. Not a retrieval failure - data gap.

2. **"R&D budget"** - Got "$210 million" (Q1 R&D) instead of "$240 million" (annual R&D). Retrieved wrong chunk because both mention R&D.

3. **"quarterly growth"** - Query needs to combine Q1, Q2, Q3, Q4 data. Retrieved Q3 chunk only. Multi-hop retrieval problem.

## What I Learned

1. **Easy tests are worthless** - They pass with any reasonable implementation
2. **Vague queries expose retrieval weaknesses** - Short queries embed poorly
3. **Multi-hop is hard** - Single retrieval can't gather info from multiple chunks

## Impact on Previous Experiments

Went back and tested hybrid search + HyDE on the full 30 queries:

| Approach | Overall | Vague |
|----------|---------|-------|
| Semantic | 77% | 50% |
| Hybrid | 83% | 60% |
| HyDE | 83% | 70% |

Now I can actually see that hybrid and HyDE help. Before, everything was 100% because tests were too easy.

## Regret

Should have built this evaluation framework on day 1, not day 6.