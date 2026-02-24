---
type: concept
aliases: [cross-encoder, cross encoder]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# Cross-Encoder

> [!definition] Cross-Encoder
> A **cross-encoder** processes the query and document **jointly** as a single input to a transformer. It can model fine-grained interactions between query and document tokens via self-attention, making it highly effective but computationally expensive.

```
Input:  [CLS] query tokens [SEP] document tokens [SEP]
                    ↓
              [Transformer]
                    ↓
              [CLS] → relevance score
```

> [!formula] Scoring
> $$s(q, d) = \sigma(\mathbf{w}^\top \cdot \text{BERT}_{[\text{CLS}]}([q; d]))$$

## Cross-Encoder vs [[Bi-Encoder]]

| Property | Cross-Encoder | Bi-Encoder |
|----------|:---:|:---:|
| Query-doc interaction | Full (self-attention) | None (independent encoding) |
| Effectiveness | Higher | Lower |
| Latency | High ($O(n)$ per query) | Low (pre-compute docs) |
| Use case | Reranking top-k | First-stage retrieval |
| Can pre-compute docs? | ❌ | ✅ |

> [!tip] In Practice
> Cross-encoders are too expensive for full-collection retrieval. They're used as **rerankers** in [[Multi-Stage Ranking]]: first retrieve top-k with [[BM25]] or [[Dense Retrieval]], then rerank with cross-encoder.

## Appears In

- [[IR-L05 - Neural IR Intro & Reranking]]
