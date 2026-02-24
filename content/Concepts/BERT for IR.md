---
type: concept
aliases: [BERT for IR, Neural IR with BERT]
course: [IR]
tags: [neural-ir]
status: complete
---

# BERT for IR

> [!definition] BERT for IR
> **BERT for IR** refers to the application of the Bidirectional Encoder Representations from Transformers (BERT) architecture to search tasks. BERT allows the system to understand the context of query and document terms, moving beyond exact keyword matching to semantic understanding.

> [!formula] Primary Architectures
> 1. **Cross-Encoder (MonoBERT)**: Query and Document are concatenated as input: `[CLS] Query [SEP] Document [SEP]`.
>    - Score = $\text{MLP}(\text{BERT}(q, d))$
>    - High accuracy, very slow (must run for every pair).
> 2. **Bi-Encoder (DPR/Dense Retrieval)**: Query and Document are encoded separately.
>    - Score = $\text{Enc}_q(q) \cdot \text{Enc}_d(d)$
>    - Fast (uses ANN / Faiss), lower accuracy than cross-encoders.
> 3. **Late Interaction (ColBERT)**: Encodes both separately but keeps multiple vectors per token.
>    - Score = $\sum \max(\text{similarity between vector sets})$
>    - Good balance of speed and accuracy.

> [!intuition] Context Matters
> In keyword IR, "bank" in "river bank" and "bank account" are the same. BERT "reads" the whole sentence and creates different embeddings for these two "banks." This allows the search engine to understand the user's *intent* rather than just their *words*.

## The Paradigm

- **Pre-training**: Learn general language patterns from massive corpora (Wikipedia, books).
- **Fine-tuning**: Train the model on IR-specific data (like MS MARCO) to distinguish between relevant and irrelevant documents.

## Connections

- **Foundations**: Transformer architecture, Attention mechanisms.
- **Used in**: [[DeepCT]], [[DeepImpact]], [[uniCOIL]], [[Retrieval-Augmented Generation]].
- **Components of**: [[Multi-Stage Ranking]].

## Appears In

- [[IR-L05 - Learning to Rank]]
- [[IR-L06 - Neural IR]]
