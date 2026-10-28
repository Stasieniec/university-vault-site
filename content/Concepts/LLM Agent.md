---
type: concept
aliases: [LLM agents, LLM-based agent, LLM-based agents]
course: [AGAI]
tags: [agents, llm]
status: complete
---

# LLM Agent

## Definition

> [!definition] LLM agent
> An **LLM agent** is an [[Intelligent Agent]] whose decision-making core is a [[Large Language Models (LLM)|large language model]]: it **observes, reasons, plans and acts through natural language**. Around the model sit four components that turn a text predictor into an agent: a **role** (objectives and persona), **memory**, **tools**, and **planning**.

## Intuition

> [!intuition] Next-token prediction is secretly multi-task
> A pretrained LLM completing text already does many jobs: world knowledge ("The capital of the Netherlands is ..."), history ("The French Revolution happened in ..."), planning ("If I want to bake a cake, the first thing to do is ..."), arithmetic ("2 + 2 = ..."). An agent harnesses this by putting the model in the classic observe-act loop. The observations are text (a tool's output, a web page, a user message), the actions are text (a tool call, an answer), and the reasoning in between is text too.

The loop most LLM agents run is **Thought, Action, Observation**:
1. **Thought**: the model reasons about the objective and plans the next step.
2. **Action**: it emits a structured action, usually a tool call.
3. **Observation**: the tool's result is fed back into the context, and the loop repeats until the model answers.

This is the [[ReAct]] pattern. Its reasoning step is [[Chain-of-Thought Prompting|chain-of-thought]] applied inside a loop.

## Key Properties / Variants

**Reasoning.** Reasoning LLMs generate thought tokens before the final answer, which shifts effort from train-time compute to **test-time compute**: more tokens means more computation spent on the problem. Reasoning can be elicited by prompting (chain-of-thought, zero-shot or few-shot) in models not tuned for it, or improved by sampling many reasoning traces and rewarding the best ("search and verify"), or by fine-tuning token probabilities for reasoning, for example with reinforcement learning as in [[DeepSeek-R1]].

**Memory.** Remembering facts, previous interactions and context is what takes you from an LLM to an agent.
- *Short-term memory* is the context window. It is limited, so it needs context engineering: trimming or summarising old content.
- *Long-term memory* is knowledge stored in the parameters, or external databases the agent queries (see [[Retrieval-Augmented Generation]] and [[Agentic Search]]).

**Tools.** "Reasoning allows to think and plan. Tools allow agents to act." A tool is an external function the model can use, such as search or running code. Using tools involves: creating and documenting them so an LLM can use them; making the agent aware of which exist; choosing the right one; parsing and executing the call; and feeding the output back into the model. The Model Context Protocol (MCP) standardises how LLMs connect to tools. Toolformer (Schick et al., 2023) showed a model can teach itself when to call tools.

**Planning and reflection.** Reasoning enables identifying the steps to a goal: decomposing into sub-tasks and generating an action sequence. Early methods were prompting techniques (chain-of-thought, Least-to-Most, Plan-and-Solve); later ones fine-tune the model (supervised or with RL) to plan and to run the ReAct cycle better. Reflexion (Shinn et al., 2023) lets the agent reflect verbally on failed attempts and retry.

**Evaluation.** Evidence that an agent works comes from public benchmarks and leaderboards (coding, tool use, reasoning, open-ended knowledge, exam-style questions, harmfulness tests such as AgentHarm) and from human or automated evaluation, including LLM-as-judge.

**Risks.** Agents trained on a reward can exploit a misspecified one ([[Reward Hacking]]), and several LLM agents together raise multi-agent risks: miscoordination, conflict and collusion (see [[Multi-Agent System]]).

> [!example] Agent-centric information access
> Retrieval helps agents (as long-term memory), and agents help retrieval. Kanoulas et al. propose an ecosystem of many expert LLMs, where answering a query means estimating which agents have the expertise, retrieving the most relevant ones, querying them efficiently and combining their answers.

## Connections

- Is an: [[Intelligent Agent]] whose program is an LLM (compare the designs in [[Agent Program]])
- Core loop: [[ReAct]]; reasoning step: [[Chain-of-Thought Prompting]]
- Built on: [[Large Language Models (LLM)]], [[In-Context Learning]]
- Memory: [[Retrieval-Augmented Generation]], [[Agentic Search]], [[Self-RAG]]
- Trained or aligned with: [[Reinforcement Learning]], [[Reinforcement Learning from Human Feedback]], [[DeepSeek-R1]]
- Many of them: [[Multi-Agent System]]
- Failure mode: [[Reward Hacking]]

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
