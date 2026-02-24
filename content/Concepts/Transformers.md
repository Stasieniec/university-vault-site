---
type: concept
aliases: [Transformers, Attention architecture, Transformer Model]
course: [IR, RL]
tags: [foundations, deep-learning, nlp, key-formula, exam-topic]
status: complete
---

# Transformers

> [!definition] Transformers
> The **Transformer** is a deep learning architecture based entirely on **attention mechanisms**, dispensing with recurrence (RNNs) and convolutions (CNNs). It allows for massive parallelization and state-of-the-art performance in sequence modeling.

> [!formula] Scaled Dot-Product Attention
> The "heart" of the transformer is the attention function mapping a query ($Q$), keys ($K$), and values ($V$):
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
> 
> where $d_k$ is the dimensionality of the keys.

> [!intuition] The Search Analogy
> Think of a transformer like a search engine:
> 1. You have a **Query** (what you are looking for).
> 2. You compare it against **Keys** (the labels or "indices" of all available info).
> 3. You take a weighted sum of the **Values** (the actual content) based on how well their Keys matched your Query.

## Key Components
1. **Multi-Head Attention**: Runs multiple attention layers in parallel to capture different types of relationships (e.g., syntax vs. semantics).
2. **Positional Encoding**: Since the model has no recurrence, it lacks an inherent sense of word order. Sinusoidal or learned encodings are added to inputs to inject position information.
3. **Encoder-Decoder Structure**:
   - **Encoder**: Bi-directional context (e.g., BERT).
   - **Decoder**: Uni-directional/autoregressive context (e.g., GPT).
   - **Combined**: For translation or summarization (e.g., T5).

## Connections
- Foundation for: BERT (used in [[COIL]], [[ColBERT]]), T5 (used in [[monoT5]], [[DocT5Query]], [[DSI]]), and GPT.
- Replaced: LSTMs and GRUs in most NLP and [[Neural Reranking]] tasks.

## Appears In
- [[IR-L05 - Neural IR 2]], [[IR-L06 - Neural IR 3]]
- [[RL-L08 - Advanced RL]]
