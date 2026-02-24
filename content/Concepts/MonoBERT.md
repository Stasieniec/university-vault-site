---
type: concept
aliases: [MonoBERT]
course: [IR]
tags: [neural-ir, transformer, bert]
status: complete
---

# MonoBERT

> [!definition] MonoBERT
> **MonoBERT** is a cross-encoder ranking model that uses the BERT architecture to score the relevance of a document to a query. It treats ranking as a sequence-pair classification task.

> [!intuition] The Input Format
> MonoBERT takes a query $q$ and a document $d$ and concatenates them:
> $$[CLS] \, q \, [SEP] \, d \, [SEP]$$
> The model utilizes self-attention across the combined sequence, allowing every token in the query to interact with every token in the document. The final relevance score is derived from the representation of the $[CLS]$ token.

## Key Technical Details

- **Architecture**: Standard BERT (base or large).
- **Scoring**: A linear layer is applied to the final hidden state of the $[CLS]$ token to output a single scalar (relevance score).
- **Training**: Typically fine-tuned using a cross-entropy loss on datasets like **MS MARCO**.
- **Negative Sampling**: Trained using positive pairs (relevant q-d) and negative pairs (irrelevant q-d, often sampled from top BM25 results).

### Pros and Cons

| Feature | Description |
|---------|-------------|
| **Strength** | Superior semantic matching; handles lexical mismatch (synonyms, paraphrase). |
| **Weakness** | Very high latency. Cannot pre-compute document embeddings (must run for every q-d pair). |
| **Usage** | Restricted to reranking top-k results (e.g., $k \leq 100$). |

## Connections

- Instance of: [[Neural Reranking]], [[Multi-Stage Ranking]].
- Successor: [[monoT5]] (uses T5 instead of BERT).
- Alternative: [[DPR]] (Bi-encoder approach).

## Appears In

- [[IR-L05 - Neural IR]]
