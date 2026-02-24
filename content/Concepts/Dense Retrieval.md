---
type: concept
aliases: [dense retrieval, dense passage retrieval, Dense Retrieval and Bi-Encoders]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# Dense Retrieval

> [!definition] Dense Retrieval
> **Dense retrieval** encodes queries and documents into dense low-dimensional vectors using neural encoders (typically transformers), then retrieves by finding the nearest document vectors to a query vector using similarity measures like dot product or cosine similarity.

## Architecture: [[Bi-Encoder]]

```
Query: "best pizza NYC"     Document: "Joe's Pizza is a famous..."
        ↓                              ↓
   [BERT Encoder]                [BERT Encoder]
        ↓                              ↓
    q ∈ ℝ^768                     d ∈ ℝ^768
        ↘                            ↙
         similarity(q, d) = q · d
```

> [!formula] Scoring
> $$s(q, d) = \mathbf{q}^\top \mathbf{d} = E_Q(\text{query})^\top \cdot E_D(\text{doc})$$
> 
> Document encodings are pre-computed and indexed → only the query needs encoding at search time.

## Key Models

| Model | Key Innovation |
|-------|---------------|
| [[DPR]] | In-batch negatives training, dual BERT encoders |
| [[ColBERT]] | Late interaction: token-level MaxSim matching |
| ANCE | Asynchronous hard negative mining from own index |
| TAS-Balanced | Topic-aware sampling for balanced training |

## Training

- **Contrastive loss**: Push relevant pairs together, push irrelevant apart
- **In-batch negatives**: Other documents in the batch serve as negatives (efficient)
- **Hard negative mining**: Use BM25 or the model itself to find challenging negatives
- **Knowledge distillation**: Train bi-encoder to match cross-encoder scores

## Indexing & Search

Pre-compute all document embeddings → use [[Approximate Nearest Neighbor]] (ANN) for fast search:
- **FAISS** (Facebook): IVF, PQ, HNSW
- **Product Quantization**: Compress vectors for memory efficiency
- **HNSW**: Graph-based ANN with high recall

## Advantages vs Limitations

✅ Semantic matching (handles synonyms, paraphrases)
✅ Pre-computed document embeddings → fast retrieval
❌ Requires GPU for encoding
❌ Large index size (dense vectors)
❌ Weaker zero-shot generalization than BM25 (BEIR benchmark)

## Connections

- Contrasted with: [[BM25]] (sparse), [[Learned Sparse Retrieval]] (learned sparse)
- First-stage for: [[Neural Reranking]] pipelines
- Extended by: [[ColBERT]] (late interaction), hybrid retrieval

## Appears In

- [[IR-L06 - Dense Retrieval]]
