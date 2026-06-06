---
type: concept
aliases: [Atlas (RAG)]
course: [IR]
tags: [neural-ir, dense-retrieval, exam-topic]
status: complete
---

# Atlas

## Definition

> [!definition] Atlas
> **Atlas** (Izacard et al., 2023) is an end-to-end, jointly-trained [[Retrieval-Augmented Generation|retrieval-augmented]] language model designed for **few-shot** knowledge-intensive tasks. It pairs a learned dense [[Bi-Encoder]] retriever (the **Contriever**) with a [[FiD|Fusion-in-Decoder]] seq2seq generator (T5), and — crucially — **trains the retriever jointly with the generator** by treating the retrieved documents as latent variables and using the generator's own loss as the training signal for retrieval.
>
> Its defining engineering contribution is **periodic re-indexing**: because the document encoder is updated during training, the entire corpus must be periodically re-embedded so that documents are not permanently "unreachable" — the central failure mode of the original [[RAG]] model's frozen document encoder.

## Intuition

The original [[RAG]] (Lewis et al., 2020) only fine-tunes the **query** encoder; the document encoder is frozen so the 21M-document index does not have to be rebuilt every gradient step. The side effect: a document's embedding never moves, so if it starts far from every query it is **unreachable forever**, no matter how the query encoder learns.

Atlas removes this limitation. The retriever is a [[Contrastive Learning|contrastively pretrained]] dense [[Bi-Encoder]] whose **document side is also trainable**, and the index is rebuilt from scratch every few hundred steps. The clever part is **how** the retriever learns without relevance labels: there are no gold passages in the few-shot setting, so Atlas lets the **language model judge usefulness**. A passage is "good" if conditioning the generator on it raises the likelihood of the correct output. This signal is distilled back into the retriever — the generator effectively teaches the retriever which documents help it answer.

Combined with [[FiD]] (each passage is encoded independently with the query, then fused only in the decoder's cross-attention), Atlas scales to many passages and reaches strong performance with as few as 64 training examples — the regime where a *learnable* retriever matters most, because there is too little data to memorize answers in the generator's parameters.

## Mathematical Formulation

Atlas marginalizes the output over the top-$k$ retrieved documents, exactly like [[RAG]], but with **both** retriever and generator trainable:

$$
P(y \mid x) = \sum_{z \in \mathcal{Z}_k(x)} \underbrace{p_\eta(z \mid x)}_{\text{retriever}} \; \underbrace{p_\theta(y \mid x, z)}_{\text{generator (FiD)}}
$$

where:
- $x$ — input query/prompt, $y$ — target output, $z$ — a retrieved document
- $\mathcal{Z}_k(x)$ — the top-$k$ documents for $x$ under the retriever's MIPS (maximum inner product search) over the dense index
- $p_\eta(z \mid x) = \dfrac{\exp\big(\mathbf{q}(x)^\top \mathbf{d}(z)\big)}{\sum_{z' \in \mathcal{D}} \exp\big(\mathbf{q}(x)^\top \mathbf{d}(z')\big)}$ — dual-encoder retrieval score; $\mathbf{q}(\cdot)$ and $\mathbf{d}(\cdot)$ are the (shared-architecture) query/document encoders with parameters $\eta$
- $p_\theta(y \mid x, z)$ — generator likelihood, a [[FiD|Fusion-in-Decoder]] T5 with parameters $\theta$

**Retriever loss (the key idea).** The retriever has no relevance labels, so it is supervised by the **language model's perplexity** over the documents. Atlas's main objective (PDist — perplexity distillation) minimizes the KL divergence between the retriever's distribution and the posterior the LM assigns to each document, i.e. how much each document improves the likelihood of the gold output:

$$
p_{\text{LM}}(z \mid x, y) = \frac{\exp\big(\log p_\theta(y \mid x, z)\big)}{\sum_{z' \in \mathcal{Z}_k(x)} \exp\big(\log p_\theta(y \mid x, z')\big)}, \qquad
\mathcal{L}_{\text{retr}} = \mathrm{KL}\big(p_{\text{LM}}(z \mid x, y)\;\|\;p_\eta(z \mid x)\big)
$$

where:
- $p_{\text{LM}}(z \mid x, y)$ — a softmax over documents using the generator's log-likelihood of the *target* as the score: documents that make $y$ more likely get higher weight
- minimizing $\mathcal{L}_{\text{retr}}$ pulls the retriever's scores $p_\eta(z\mid x)$ toward the LM's usefulness ranking, so the retriever learns to fetch documents the generator actually finds helpful — no human relevance judgments required

## Key Properties / Variants

- **Joint, end-to-end training:** unlike [[RAG]] (frozen doc encoder) and [[FiD]] (retriever and generator disconnected — typically BM25), Atlas backpropagates a usefulness signal into **both** sides of the retriever.
- **Retriever:** Contriever-style dense [[Bi-Encoder]], contrastively pretrained, then fine-tuned with the LM signal.
- **Generator:** [[FiD|Fusion-in-Decoder]] (T5-base up to T5-XXL), encoding each of $k$ passages independently with the query, fusing in the decoder.
- **Few-shot strength:** designed for the low-data regime (e.g. 64 examples); a learnable retriever substitutes for the knowledge a large generator would otherwise have to store in parameters.
- **Retriever loss variants:** beyond perplexity distillation (PDist), the paper studies ADist (attention distillation from decoder cross-attention), EMDR² (end-to-end marginal-likelihood training of the retriever), and LOOP (leave-one-out perplexity). PDist/LOOP are the most robust.
- **Cost:** the document index must be periodically rebuilt; this is done **offline / every N steps**, not per step. Async index updates and over-retrieval (refresh on a stale index) keep it tractable.
- **Limitation:** re-indexing millions of documents is expensive; staleness between refreshes introduces a small approximation; still inherits RAG-family failure modes (semantic hallucination, lost-in-the-middle).

```pseudo
Algorithm: Atlas Training (joint retriever + generator)
────────────────────────────────────────────────────────
Initialize:
  retriever (dense bi-encoder, Contriever-pretrained, params η)
  generator (Fusion-in-Decoder T5, params θ)
  Build index: embed all docs z ∈ D with current doc encoder → MIPS index

Loop over training steps t = 1, 2, ...:
  Sample (x, y) from few-shot training set
  Retrieve top-k: Z_k(x) ← MIPS(q(x), index)        # may use stale index
  For each z in Z_k(x):
    encode (x, z) independently            # FiD encoder
  Fuse encodings in decoder → p_θ(y | x, z)
  # Generator update
  L_gen ← − log Σ_z p_η(z|x) p_θ(y | x, z)          # marginal likelihood
  # Retriever update (perplexity distillation)
  p_LM(z|x,y) ← softmax_z [ log p_θ(y | x, z) ]
  L_retr ← KL( p_LM(z|x,y) ‖ p_η(z|x) )
  Update θ, η by gradient descent on (L_gen + L_retr)

  if t mod N == 0:                                   # periodic re-indexing
     re-embed all docs z ∈ D with updated doc encoder
     rebuild MIPS index
```

## Connections

- Direct improvement on: [[RAG]] (Atlas unfreezes the document encoder and re-indexes; RAG kept it frozen)
- Generator architecture: [[FiD]] / [[Fusion-in-Decoder]]
- Retriever: dense [[Bi-Encoder]] / [[Dense Retrieval]], built with [[Contrastive Learning]]; uses [[DPR]]-style MIPS over the index
- Family member alongside: [[Self-RAG]] (adaptive retrieval via reflection tokens) — Atlas instead makes the retriever itself learnable
- Broader paradigm: [[Retrieval-Augmented Generation]]
- Search step relates to: [[Approximate Nearest Neighbor]] / [[ANN Search]] for the MIPS index

## Appears In

- [[IR-L09 - RAG]]
