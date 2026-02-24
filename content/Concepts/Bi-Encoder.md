---
type: concept
aliases: [bi-encoder, dual encoder, two-tower model]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# Bi-Encoder

> [!definition] Bi-Encoder
> A **bi-encoder** (dual encoder / two-tower model) encodes queries and documents **independently** into dense vectors, then computes relevance via a simple similarity function (dot product or cosine). Documents can be pre-encoded offline.

```
Query → [Encoder_Q] → q ∈ ℝ^d
                                   → sim(q, d) = q · d
Document → [Encoder_D] → d ∈ ℝ^d
```

Key advantage: document vectors are computed once and indexed → retrieval is just nearest neighbor search in vector space.

See [[Dense Retrieval]] for full details, training, and indexing.

## Appears In

- [[IR-L05 - Neural IR Intro & Reranking]], [[IR-L06 - Dense Retrieval]]
