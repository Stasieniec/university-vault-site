---
type: concept
aliases: [monoT5]
course: [IR]
tags: [neural-ir, reranking]
status: complete
---

# monoT5

> [!definition] monoT5
> **monoT5** is a neural reranking model based on the T5 transformer architecture. It formulates reranking as a sequence-to-sequence task where the model is prompted to produce a relevance label ("true" or "false") for a query-document pair.

> [!formula] monoT5 Relevance Scoring
> The model takes the input sequence:
> `Query: q Document: d Relevant:`
> 
> It then computes the probability of generating the token "true" versus "false":
> $$s_{d} = P(\text{"true"} \mid q, d)$$
> 
> This probability (or the logit before softmax) is used as the relevance score for sorting document $d$ in the final ranked list.

> [!intuition] Language Model as a Scorer
> Instead of using a classification head (like BERT-based rerankers), monoT5 treats the problem as a "natural language" completion. Since T5 is pre-trained on massive text corpora to predict tokens, it has a strong internal representation of whether a particular document content "follows" a query's intent.

## Connections
- Part of: [[Multi-Stage Ranking]] pipelines (usually the second stage after [[BM25]]).
- Scalability: Often paired with [[duoT5]] for even more precise (but computationally expensive) pairwise reranking.
- Architecture: Uses the T5 (Text-To-Text Transfer Transformer) encoder-decoder structure.

## Appears In
- [[IR-L05 - Neural IR 2]]
