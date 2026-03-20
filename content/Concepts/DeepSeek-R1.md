---
type: concept
aliases: [DeepSeek-R1, DeepSeek R1]
course: [IR, RL]
tags: [llm, reasoning, reinforcement-learning, emergent-behavior]
status: complete
---

# DeepSeek-R1

> [!definition] DeepSeek-R1
> A large language model trained with **pure reinforcement learning** (no supervised fine-tuning on reasoning traces) that demonstrates emergent reasoning capabilities. Starting from a base model, it learns chain-of-thought reasoning solely through outcome-based reward signals.

## Key Breakthrough

**Pure RL without demonstrations**: Unlike prior work that required SFT on human-written reasoning traces, DeepSeek-R1 shows that reasoning behaviors can **emerge** from RL training alone.

## Emergent Behaviors

When trained with only final-answer correctness as reward:

1. **Extended thinking**: Model produces longer reasoning chains
2. **Self-verification**: "Let me check this calculation..."
3. **Backtracking**: Recognizing and correcting errors mid-reasoning
4. **Multi-step decomposition**: Breaking complex problems into parts

> [!intuition] **Why This Works**
> The reward signal creates selection pressure: trajectories with correct answers are reinforced. The model "discovers" that certain patterns (checking work, step-by-step reasoning) correlate with higher accuracy, so these behaviors are learned.

## Training Details

| Component | Choice |
|-----------|--------|
| **Base model** | DeepSeek-V3 (pre-trained, no SFT) |
| **Algorithm** | [[GRPO]] |
| **Reward** | Binary (correct = 1, incorrect = 0) |
| **Demonstrations** | None |

## Significance

1. **Scalable**: No need to collect expensive reasoning demonstrations
2. **Generalizable**: Reasoning transfers across domains
3. **Foundation for agentic systems**: Extended to [[SEARCH-R1]] with retrieval

## Connection to System 2 Thinking

DeepSeek-R1 exhibits "System 2" cognitive behavior:
- Slow, deliberate processing
- Explicit reasoning steps
- Self-monitoring and correction

This contrasts with "System 1" (fast, pattern-matching) typical of standard LLM inference.

## Appears In

- [[IR-L13 - RL for Reasoning and Search]]
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
