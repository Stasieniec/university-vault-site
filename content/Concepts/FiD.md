---
type: concept
aliases: [Fusion-in-Decoder]
course: [IR]
tags: [neural-ir, generative-retrieval, exam-topic]
status: complete
---

# FiD

## Definition

> [!definition] FiD (Fusion-in-Decoder)
> **FiD** (Izacard & Grave, 2021) is a retrieval-augmented seq2seq architecture for open-domain QA that **encodes each retrieved passage independently** alongside the question, then **fuses all passage representations in the decoder** via cross-attention. By separating encoding (per-passage, parallel) from fusion (joint, in the decoder), FiD scales to **100+ passages** where naive concatenation hits context limits and where [[RAG]]'s marginalization is slow.

## Intuition

> [!intuition] Split the work: encode locally, fuse globally
> Naive [[Retrieval-Augmented Generation|RAG]] concatenates all passages into one prompt — the self-attention cost grows **quadratically** with the total number of tokens, so you can only afford a handful of passages. Lewis et al.'s [[RAG]] avoids this by marginalizing over documents, but inference is slow and each output is grounded in one document at a time.
>
> FiD's trick: run the encoder **separately** on each (question, passage) pair. Encoding $k$ passages of length $L$ costs $O(k \cdot L^2)$ instead of $O((kL)^2)$ — **linear** in the number of passages. The decoder then attends over the *concatenation of all passage encodings*, so cross-attention can **synthesize evidence across passages** to produce answers not stated verbatim in any single one.

## Mathematical Formulation

Given question $q$ and retrieved passages $\{z_1, \dots, z_k\}$, FiD builds one encoder representation per passage and concatenates them for the decoder:

$$
H_i = \text{Encoder}\big(\texttt{question}: q \;\Vert\; \texttt{title}: t_i \;\Vert\; \texttt{context}: z_i\big), \qquad i = 1,\dots,k
$$

$$
H = \big[\,H_1 \,\Vert\, H_2 \,\Vert\, \cdots \,\Vert\, H_k\,\big], \qquad
P(y \mid q, z_{1:k}) = \prod_{j=1}^{|y|} \text{Decoder}\big(y_j \mid y_{<j},\, H\big)
$$

where:
- $q$ — the input question; $z_i$ — the $i$-th retrieved passage (with title $t_i$), prepended with special markers (`question:`, `title:`, `context:`)
- $H_i \in \mathbb{R}^{L_i \times d}$ — encoder hidden states for passage $i$, computed **independently** (no attention across passages in the encoder)
- $\Vert$ — concatenation; $H \in \mathbb{R}^{(\sum_i L_i) \times d}$ — the full set of token representations the decoder attends to
- $\text{Decoder}$ — autoregressive T5 decoder; its **cross-attention** ranges over all of $H$, so a single answer can draw on multiple passages
- $y$ — the generated answer; $y_{<j}$ — previously generated tokens

> [!formula] Cost: linear vs. quadratic in passages
> $$\text{Encoding: } O\!\Big(\textstyle\sum_i L_i^{\,2}\Big) \approx O(k L^2) \quad\text{vs.}\quad \text{Concatenation self-attention: } O\!\Big(\big(\textstyle\sum_i L_i\big)^{2}\Big) \approx O(k^2 L^2)$$
> Independent encoding makes the dominant self-attention cost **linear** in the number of passages $k$; only the (much cheaper) decoder cross-attention sees all $k L$ tokens jointly.

## Key Properties / Variants

- **Scalability:** Performance on Natural Questions **improves monotonically** with passage count up to 100+ passages, unlike concatenation which saturates/degrades at the context limit.
- **Cross-document synthesis:** Decoder cross-attention fuses evidence; answers can be assembled from facts spread across several passages.
- **Architectural simplicity:** Standard seq2seq (T5) with only modified input formatting — no new parameters, no marginalization machinery.
- **Disconnected retriever:** Vanilla FiD uses a **fixed** retriever ([[BM25]] or [[DPR]]); there is **no end-to-end gradient flow** to the retriever (contrast with end-to-end [[RAG]]). [[Atlas]] later closes this gap with a learnable, periodically reindexed dense [[Bi-Encoder]].
- **Inference cost grows with $k$:** Decoder cross-attention scans all passage tokens each step; tractable but linear in $k$.
- **Non-adaptive:** Always retrieves a fixed number of passages regardless of query difficulty (contrast with adaptive [[Self-RAG]]).
- **Mitigates "lost in the middle":** Independent encoding sidesteps the positional attention decay that plagues single-context concatenation (each passage is encoded as if at the start).

```pseudo
Algorithm: FiD (Fusion-in-Decoder) — Inference
────────────────────────────────────────────────
Input: question q, retriever R, corpus D, generator (Encoder, Decoder), k
  1. Z ← R.retrieve(q, D, top_k = k)          # fixed retriever (BM25 / DPR)
  2. for each z_i in Z:                         # independent, parallelizable
       x_i  ← "question: " + q +
              " title: " + title(z_i) +
              " context: " + text(z_i)
       H_i  ← Encoder(x_i)                      # per-passage hidden states
  3. H ← concat(H_1, ..., H_k)                  # fuse in decoder input
  4. y ← Decoder.generate(cross_attend_over = H)  # cross-attention over all passages
  return y
```

> [!warning] FiD vs. RAG — what "fusion" means
> In [[RAG]] (Lewis et al.), documents are **latent variables**: the model marginalizes $P(y\mid q)=\sum_z P(z\mid q)P(y\mid q,z)$ and gradients flow to the query encoder. In **FiD** there is **no marginalization** — all passages are fed jointly to the decoder and the retriever is frozen. FiD trades end-to-end learnability for a large gain in how many passages it can exploit (44.5 EM for RAG vs. up to 68.2 EM for FiD with gold passages on Natural Questions).

## Connections

- Compared with: [[RAG]] (marginalizes over latent documents, end-to-end), [[Self-RAG]] (adaptive retrieval), naive concatenation
- Extended by: [[Atlas]] (adds a learnable, periodically reindexed dense retriever on top of the FiD generator)
- Retriever used: [[BM25]] / [[DPR]] / [[Dense Retrieval]] feeding the [[Bi-Encoder]] front end
- Built on: [[Transformers]] (T5 seq2seq), decoder [[Cross-Encoder|cross-attention]]
- Part of: [[Retrieval-Augmented Generation]] family

## Appears In

- [[IR-L09 - RAG]]
