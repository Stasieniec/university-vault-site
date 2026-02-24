---
type: concept
aliases: [Negative Mining, Hard Negatives]
course: [IR]
tags: [neural-ir, training, optimization]
status: complete
---

# Hard Negative Mining

> [!definition] Hard Negative Mining
> **Hard Negative Mining** is the process of selecting "challenging" negative examples for training retrieval models—specifically documents that are similar to the query in some way (e.g., lexical overlap) but are ultimately non-relevant.

> [!intuition] Making the Model Sweat
> If you only train a model by comparing "Capital of France" to "How to bake a cake" (random negative), the model learns nothing useful.
> If you compare "Capital of France" to "The largest city in France" (hard negative), the model has to learn fine-grained semantic differences to correctly rank the actual answer higher.

## Common Strategies

| Strategy | Description |
|----------|-------------|
| **BM25 Negatives** | Use documents that have high BM25 scores but are not the ground-truth positive. These often share keywords but miss the intent. |
| **In-batch Negatives**| Every positive document $d_j$ in a batch is treated as a negative for query $q_i$ (where $i \neq j$). Cheap but depends on batch size. |
| **ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation)** | As training progresses, the model itself is used to retrieve its own "top mistakes." These are then used as negatives in the next step. |

## Why it Matters
Without hard negatives, [[DPR]] models often fail to outperform [[BM25]] because they never learn to distinguish between documents that "look right" and those that "are right."

## Connections

- Essential for: [[Contrastive Learning]].
- Used in: Training [[DPR]].
- Challenges: "False negatives" (selecting a document that is actually relevant but not labeled as such in the dataset).

## Appears In

- [[IR-L06 - Dense Retrieval]]
