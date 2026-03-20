---
type: concept
aliases: [Search-R1, SEARCH-R1]
course: [IR]
tags: [rag, agentic-search, reinforcement-learning, llm]
status: complete
---

# SEARCH-R1

> [!definition] SEARCH-R1
> A retrieval-augmented reasoning system that uses [[Reinforcement Learning]] to train LLMs to **interleave reasoning and search**. The model learns when to search and what to search for through outcome-based rewards, without requiring demonstration data.

## Core Innovation

Extends [[DeepSeek-R1]]'s pure RL approach to retrieval-augmented generation:
- Model generates reasoning traces with **search queries** embedded
- Search engine retrieves documents at each query
- Retrieved content is inserted into the generation
- Only final answer correctness provides reward

## Trajectory Format

```
<think>
[Reasoning about the problem]
<search>query to search engine</search>
</think>
<information>
[Retrieved documents inserted by system]
</information>
<think>
[Further reasoning with retrieved info]
</think>
<answer>
[Final answer]
</answer>
```

## Training Objective

> [!formula] **Masked RL Loss**
> $$\mathcal{L}(\theta) = -\sum_{t \in \mathcal{T}_{\text{model}}} \hat{A}_t \log \pi_\theta(y_t | y_{<t}, x)$$
>
> where $\mathcal{T}_{\text{model}}$ **excludes** tokens inside `<information>` tags (retrieved content).

**Why masking matters**:
- Retrieved content is not model-generated
- Prevents memorization of corpus
- Ensures gradients reflect reasoning quality, not retrieval

## Key Design Choices

| Choice | Recommendation | Rationale |
|--------|----------------|-----------|
| **Algorithm** | [[PPO]] over [[GRPO]] | Value function absorbs retrieval noise; GRPO suffers reward collapse |
| **Starting model** | Base (not instruct) | More malleable to RL |
| **Documents per search** | 3 | Balance coverage vs noise |
| **Reward** | Outcome + format | Sparse but clean signal |
| **Loss masking** | Mask retrieved content | Critical for generalization |

## Comparison with Static RAG

| Aspect | Static [[Retrieval-Augmented Generation|RAG]] | SEARCH-R1 |
|--------|------------|-----------|
| **Retrieval timing** | Before generation | During reasoning |
| **Query source** | User query | Model-generated |
| **Number of retrievals** | Fixed | Adaptive |
| **Training** | SFT | RL |
| **Multi-hop** | Limited | Natural |

## Results

SEARCH-R1 significantly outperforms static RAG on multi-hop reasoning benchmarks (HotpotQA, 2WikiMultiHop, MuSiQue), showing the value of iterative search.

## Connections

- Builds on [[DeepSeek-R1]] (pure RL for reasoning)
- Improves upon [[Retrieval-Augmented Generation]]
- Uses [[GRPO]] for optimization
- Related to ReAct and Self-RAG approaches

## Appears In

- [[IR-L13 - RL for Reasoning and Search]]
- Jin et al., "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning" (2025)
