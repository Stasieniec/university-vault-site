---
type: concept
aliases: [ReAct prompting, Reason and Act]
course: [AGAI]
tags: [agents, llm, prompting]
status: complete
---

# ReAct

## Definition

> [!definition] ReAct (Yao et al., ICLR 2023)
> **ReAct** ("Reasoning and Acting") is a prompting technique that makes a language model generate **verbal reasoning traces and actions, both as text, in an interleaved loop**: Thought, Action, Observation, repeated until the task is done. Actions call external tools; their outputs come back as observations and become part of the model's context.

## Intuition

> [!intuition] Think, do, look, think again
> Reasoning alone and acting alone each fail in a typical way. A model that only reasons ([[Chain-of-Thought Prompting|chain-of-thought]]) works from what is in its parameters, so it can reason fluently from facts it has made up. A model that only acts calls tools without a plan and loses track of what it is doing. ReAct alternates the two: thoughts decide what to do next and why, actions fetch real information, and observations ground the next thought in that information.

ReAct's three parts each play a role:

| Step | What it is | What it gives the agent |
|---|---|---|
| Thought | free-text reasoning | multi-step planning (chain-of-thought) |
| Action | text that invokes a tool | a way to interact with the outside world |
| Observation | the tool's output, appended to the prompt | memory of what has been found |

> [!example] A ReAct trace (illustrative)
> **Question:** Which is older, the University of Amsterdam or Leiden University?
> **Thought 1:** I need the founding year of each. Start with the University of Amsterdam.
> **Action 1:** search[University of Amsterdam]
> **Observation 1:** ... founded in 1632 as the Athenaeum Illustre ...
> **Thought 2:** Now Leiden.
> **Action 2:** search[Leiden University]
> **Observation 2:** ... founded in 1575 ...
> **Thought 3:** 1575 is earlier than 1632, so Leiden is older.
> **Action 3:** finish[Leiden University]

## Key Properties / Variants

- **Prompting only.** The original method needs no training: a few in-context examples of Thought/Action/Observation traces teach the format (see [[In-Context Learning]]). Models can also be fine-tuned, with supervised learning or RL, to run the cycle better.
- **Tools are text.** The model writes an action string; the agent framework parses it, executes the tool, and pastes the result back. This is the tool-calling loop of modern [[LLM Agent|LLM agents]].
- **Context as working memory.** Every thought and observation stays in the prompt, which is why short-term memory management (trimming, summarising) matters for long tasks.
- **Interpretability.** The reasoning trace shows why the agent took each action, which helps debugging and human oversight.
- **Evaluation in the paper** (background beyond the course material): knowledge-intensive question answering and fact verification with a simple Wikipedia search tool (HotpotQA, FEVER), and interactive decision-making in text-based household and web-shopping environments (ALFWorld, WebShop).
- **Limits.** Errors compound over many steps, a bad observation can derail the plan, and long traces fill the context window. Reflection methods such as Reflexion add a step where the agent critiques a failed attempt before retrying.

## Connections

- Reasoning component: [[Chain-of-Thought Prompting]]
- The agent pattern it defines: [[LLM Agent]] (Thought, Action, Observation loop)
- An instance of the observe-act loop of an [[Intelligent Agent]]
- Retrieval as an action: [[Agentic Search]], [[Retrieval-Augmented Generation]], [[SEARCH-R1]]
- Built on: [[Large Language Models (LLM)]], [[In-Context Learning]]

## Appears In

- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]]
