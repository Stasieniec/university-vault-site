---
type: concept
aliases: [Agentic Search, Agentic RAG, Agent-based Retrieval]
course: [IR]
tags: [rag, agents, retrieval, reasoning]
status: complete
---

# Agentic Search

> [!definition] Agentic Search
> A retrieval paradigm where the language model has **agency over the retrieval process**, deciding when to search, what queries to issue, and how to incorporate results—in contrast to static [[Retrieval-Augmented Generation|RAG]] pipelines with fixed retrieve-then-read structure.

## Comparison with Static RAG

| Aspect | Static RAG | Agentic Search |
|--------|------------|----------------|
| **Control flow** | Fixed pipeline | Model-determined |
| **Retrieval timing** | Before generation | During reasoning |
| **Query source** | User query only | Model-generated subqueries |
| **Number of retrievals** | Fixed $k$ | Adaptive (0 to many) |
| **Multi-hop reasoning** | Limited | Natural |
| **Training** | SFT on QA pairs | Often RL with outcome reward |

## Key Capabilities

1. **Adaptive retrieval**: Search only when needed
2. **Query reformulation**: Generate targeted subqueries
3. **Iterative refinement**: Multiple search rounds
4. **Result synthesis**: Combine information across retrievals
5. **Self-evaluation**: Assess if more search is needed

## System 2 Analogy

Agentic search corresponds to **System 2 thinking** in cognitive science:
- Deliberate, slow processing
- Explicit reasoning about information needs
- Iterative refinement

Static RAG is more like **System 1**:
- Fast, automatic
- Fixed pattern execution
- No adaptation

## Example Systems

| System | Approach |
|--------|----------|
| **[[SEARCH-R1]]** | RL-trained interleaved reasoning + search |
| **Self-RAG** | Learned retrieval decisions via special tokens |
| **ReAct** | Reasoning-action interleaving |
| **Adaptive-RAG** | Query routing by complexity |

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| Simple factual QA | Static RAG |
| Multi-hop reasoning | Agentic Search |
| Latency-critical | Static RAG |
| Research queries | Agentic Search |

## Connections

- Extension of [[Retrieval-Augmented Generation]]
- Enabled by [[Reinforcement Learning]] training
- Related to tool-use in LLM agents
- Implemented in [[SEARCH-R1]]

## Appears In

- [[IR-L13 - RL for Reasoning and Search]]
- [[IR-L09 - RAG]] (mentioned as future direction)
