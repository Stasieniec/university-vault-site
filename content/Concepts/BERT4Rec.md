---
type: concept
aliases: []
course: [RecSys]
tags: [sequential-rec, collaborative-filtering, exam-topic]
status: complete
---

# BERT4Rec

## Definition

> [!definition] BERT4Rec
> **BERT4Rec** (Sun et al., CIKM 2019) is a **bidirectional Transformer** recommender for [[Sequential Recommendation]]. Unlike causal/left-to-right models such as [[SASRec]] and [[GRU4Rec]], it uses **bidirectional self-attention** so every position can attend to both **left and right context**. It is trained with a **Cloze (masked-item) task** borrowed from BERT: randomly mask a fraction of items in a user's interaction sequence and predict them from the surrounding items. The motivation is that **causal (unidirectional) attention may miss patterns in loosely ordered interaction data**.

## Intuition

> [!intuition] Why mask instead of predict-the-next?
> A strict left-to-right model (SASRec, GRU4Rec) only ever conditions on the past, so it learns transitions in one direction. But user histories are often only **loosely ordered** — what comes after an item is as informative as what came before. By masking interior items and forcing the model to reconstruct them from **both sides**, BERT4Rec learns richer item representations.
>
> The Cloze objective also multiplies training signal: a single sequence of length $n$ with several masks yields several prediction targets, instead of just one next-item target per sequence.

## Mathematical Formulation

**Training (Cloze / masked-item task).** Given a sequence $S = [v_1, \dots, v_n]$, randomly replace a fraction $\rho$ of items with a special `[MASK]` token:
$$S = [v_1, \dots, v_n] \;\longrightarrow\; S_{\text{masked}} = [v_1, \dots, [\text{MASK}], \dots, v_n].$$

Each position embedding is the sum of an **item embedding** and a **positional embedding**, $h_i^{0} = v_i + p_i$. The sequence passes through $L$ stacked Transformer encoder blocks (Multi-Head Attention → Add & Norm → position-wise Feed-Forward → Add & Norm, with dropout), each with **bidirectional** connections (no causal mask). A projection head over the final layer produces, for each masked position, a **full-vocabulary softmax** over items.

**Loss — masked LM / cross-entropy over masked positions:**
$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P(v_i \mid S_{\text{masked}}; \theta)$$

where:
- $\mathcal{M}$ — set of **masked positions** in the sequence
- $v_i$ — the **true item** at masked position $i$
- $P(v_i \mid S_{\text{masked}}; \theta)$ — predicted probability from the Transformer + softmax over the **entire item vocabulary**
- $\theta$ — model parameters

**Inference ("mask at the end").** A next-item recommendation requires predicting position $n+1$, but during training the model never sees a mask at the very end. So at inference a `[MASK]` is **appended** to the history and its predicted distribution is the next item:
$$S = [v_1, \dots, v_n] \;\longrightarrow\; S_{\text{masked}} = [v_1, \dots, v_n, [\text{MASK}]].$$

Adding this **"mask-at-the-end" as a second training stage** raises performance, because it closes the train/inference mismatch between random-position masking and last-position prediction.

## Key Properties / Variants

- **Bidirectional vs. causal.** BERT4Rec is the bidirectional member of the Transformer-recommender family: SASRec uses a **causal mask** (each output $v_{t+1}$ from $v_1 \dots v_t$); RNN methods ([[GRU4Rec]]) chain left-to-right; BERT4Rec lets every position see every other.
- **Item + positional embeddings.** Like SASRec, the input is an item embedding plus a positional embedding; the difference is the attention pattern and the training objective.
- **Full-vocabulary softmax.** The Cloze loss uses cross-entropy over **all items**, not pairwise/sampled negatives — this is a key contrast with SASRec's original BCE-with-negative-sampling setup.
- **Strength / limitation (course comparison table).** Strength: leverages **bidirectional context**, outperforming SASRec on multiple datasets in the original paper. Limitation: can be **slower to train**, and the gains may vary.
- **The loss-vs-architecture caveat (very important for the exam).** [Klenitskiy & Vasilev, 2023] ("Turning Dross Into Gold: Is BERT4Rec Really Better Than SASRec?") show that when SASRec is retrained with a **full cross-entropy loss** or **BCE with many (3000) negatives** ("SASRec+"), it **beats** BERT4Rec on all metrics on ML-1M. The takeaway: BERT4Rec's apparent edge comes largely from its **loss function and the number of negatives** (full softmax avoids the **overconfidence** caused by too few negatives), **not** from bidirectionality per se. Losses (BPR / BCE / CE) are **model-agnostic** — any of these architectures can be trained with any of them.
- **Role in the generative-recommendation recap.** BERT4Rec is one of the classical "score-and-rank" sequential models: it differs from SASRec/GRU4Rec only in how it encodes the history $\mathcal{H}_t$, but keeps the same skeleton of encoding history → scoring catalogue items. Generative recommenders (TIGER, OneRec) instead **decode an item identifier** rather than scoring a fixed candidate set. BERT4Rec also appears as a baseline for test-time-reasoning methods (Think Before Recommend reports ~+6% NDCG@20 on top of it).

```pseudo
Algorithm: BERT4Rec (Cloze training + mask-at-the-end inference)
────────────────────────────────────────────────────────────────
Embeddings: each item v_i -> e(v_i) + p_i  (item + positional)

# --- Training (masked-item / Cloze) ---
Loop over user sequences S = [v_1, ..., v_n]:
  M <- sample a fraction ρ of positions to mask
  S_masked <- replace S[i] with [MASK] for i in M
  H <- L stacked bidirectional Transformer encoder blocks(S_masked)
  for i in M:
    P(· | S_masked) <- softmax(Projection(H_i))   # over full item vocab
  L_MLM <- -(1/|M|) Σ_{i∈M} log P(v_i | S_masked)
  update θ by gradient descent on L_MLM

# (optional) second stage: mask only the last position to match inference

# --- Inference (next-item) ---
S_masked <- [v_1, ..., v_n, [MASK]]
H <- encoder(S_masked)
scores <- softmax(Projection(H_{n+1}))     # distribution over items
recommend top-k items by score
```

## Connections

- Type of: [[Sequential Recommendation]] model; uses [[Self-Attention]] / [[Transformer Model]] encoder blocks
- Bidirectional counterpart of: [[SASRec]] (causal self-attention), [[GRU4Rec]] (RNN, left-to-right)
- Successor in lineage of: [[FPMC]] → [[GRU4Rec]] → [[SASRec]] → BERT4Rec
- Trained with: Cross-Entropy over masked positions (masked LM); contrast with [[BPR]] and BCE used by other sequential models
- Built on the [[Embedding Layer]] + positional embeddings recipe; full-vocabulary softmax over items
- Evaluated with: [[NDCG]], [[Hit Rate|HR@K]], [[MRR]] (ranking metrics)
- Foreshadows: [[Generative Recommendation]] (replaces scoring with identifier decoding), [[Semantic IDs]]
- Solves the limitation of: [[Matrix Factorization]] / [[Collaborative Filtering]] ignoring interaction order

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L03a - Sequential Recommendation Models]]
- [[RS-L04 - Generative Recommendation]]
