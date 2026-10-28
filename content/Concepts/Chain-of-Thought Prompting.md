---
type: concept
aliases: [chain-of-thought, CoT]
course: [AGAI]
tags: [llm, prompting, reasoning]
status: complete
---

# Chain-of-Thought Prompting

## Definition

> [!definition] Chain-of-thought (CoT) prompting (Wei et al., NeurIPS 2022)
> **Chain-of-thought prompting** elicits reasoning from a language model by getting it to **write out intermediate reasoning steps before the final answer**. In the original form, the few-shot examples in the prompt each show a worked chain of reasoning, and the model imitates that format on the new question.

## Intuition

> [!intuition] Tokens are thinking time
> A language model spends a fixed amount of computation per generated token. Asked for the answer straight away, it has to produce it in one step. Asked to reason first, it can spend many tokens on sub-steps, and each step it writes becomes context for the next. More tokens means more computation, and for multi-step problems (arithmetic word problems, logic, planning) that extra computation is what makes the difference.

> [!example] Standard versus chain-of-thought prompt (Wei et al.'s example)
> **Standard few-shot example:** "Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each. How many does he have now? A: 11."
> **Chain-of-thought example:** "... A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11."
> Shown examples of the second kind, the model writes its own step-by-step working for the next question, and gets more multi-step problems right.

## Key Properties / Variants

- **Prompting, without training.** CoT changes only the prompt, so it can elicit reasoning in models that were never tuned to reason. It relies on [[In-Context Learning]].
- **Zero-shot CoT** (background beyond the course material): appending an instruction such as "Let's think step by step" with no worked examples also triggers step-by-step reasoning (Kojima et al., 2022).
- **Scale-dependent** (background): Wei et al. found the gains appear mainly in large models; small models write fluent but wrong chains.
- **Search and verify.** Instead of one chain, sample many reasoning traces and pick or reward the best. Background beyond the course material: for instance by majority vote over final answers (self-consistency) or with a separate verifier model.
- **Decomposition variants for planning** (descriptions are background beyond the course material). Least-to-Most prompting splits a problem into easier sub-problems solved in order; Plan-and-Solve first writes a plan, then executes it.
- **From prompting to training.** Reasoning LLMs are fine-tuned so that generating a long chain of thought before answering is their default, often with reinforcement learning on outcome rewards, as in [[DeepSeek-R1]]. This moves effort from train-time compute to **test-time compute**.
- **In agents.** CoT is the "Thought" step of the [[ReAct]] loop. On its own it reasons only over what the model already knows, so it can build a confident argument on a hallucinated fact; interleaving actions and observations is ReAct's fix.
- **Faithfulness caveat** (background): a written chain is not guaranteed to be the computation that actually produced the answer, so reading it is useful but not proof of how the model decided.

## Connections

- Used inside: [[ReAct]], the planning and reasoning component of an [[LLM Agent]]
- Relies on: [[Large Language Models (LLM)]], [[In-Context Learning]]
- Trained versions: [[DeepSeek-R1]], [[GRPO]]

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
