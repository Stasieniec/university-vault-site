---
type: moc
course: 5204AGAI6Y
tags: [moc]
status: active
canvas_id: 59984
canvas_url: https://canvas.uva.nl/courses/59984
exam_date: 2026-11-20
---

# Agentic Artificial Intelligence - Overview

> **Course:** Agentic Artificial Intelligence (5204AGAI6Y), 2026/27 Semester 1, Period 2. First year the course runs.
> **Programme:** MSc Artificial Intelligence, UvA
> **Credits:** 6 EC
> **Coordinator:** Fernando P. Santos (f.p.santos@uva.nl)
> **Teaching team:** Fernando Santos, Annet Onnes, Ece Erdem, Roman Chiva Gil, Dimitris Michailidis, Madhura Pawar, Alexandre Pires
> **Lectures:** Tuesdays and Thursdays 15:00-16:45
> **Werkcollege (lab):** Wednesdays. Stas: Group 5, 13:00-14:45, SP G2.13 (lab TA Dimitris Michailidis)
> **Exams:** midterm Fri 2026-11-20 13:00-15:00 (30%), final Fri 2026-12-18 13:00-15:00 (40%), resit Wed 2027-02-10 13:00-15:00. Format not announced yet.

## What the course is about

Three parts, each building on the last:

1. **Simple agents.** Reflex agents with tiny action spaces, and the complex dynamics they produce together: game theory (normal-form games, equilibria), social dilemmas, populations of agents and evolutionary game theory.
2. **Learning agents.** Reinforcement learning, multi-agent RL (MARL), deep MARL. Learning agents already show problems that LLM agents have (reward hacking), RL is used to train LLM agents, and some RL algorithms (AlphaZero) inspired techniques for LLM reasoning.
3. **Agentic systems.** LLM-based agents: reasoning, memory, tools, reflection, multi-agent architectures and orchestration, protocols, evaluation and benchmarks, simulacra of human behaviour, coding agents, and the risks of advanced multi-agent systems.

## Assessment

| Component | Weight | Dates | Notes |
|---|---|---|---|
| Midterm exam | 30% | Fri 20 Nov, 13:00-15:00 | Format and scope not announced yet |
| Final exam | 40% | Fri 18 Dec, 13:00-15:00 | The week plan in Lecture 1 says 19 Dec, the assessment slide says 18 Dec. Check DataNose closer to the date |
| Assignment 1: multi-agent RL | 15% | out Mon 2 Nov, due Fri 13 Nov 23:59 | Team of 4 |
| Assignment 2: agentic societies | 15% | out Mon 23 Nov, due Fri 4 Dec 23:59 | Team of 4, uses LLM API keys |
| Resit | | Wed 10 Feb 2027, 13:00-15:00 | |

- **Teams of 4**, all in the same werkcollege group. Formed on Canvas by Tue 27 Oct.
- **Labs are not graded.** They are open-ended notebooks: predict and justify, run, explain, modify. No coding questions; the questions ask for interpretation and evidence. A wrong prediction you can explain later is worth more than a lucky guess. Treated by the lecturer as the best practice for the midterm.
- **AI use is allowed**, with two conditions: you stay responsible for the analysis, and you are transparent about how you used it.
- **API keys** for LLMs come through UvA AI Chat, with a fixed budget per student. Needed for the labs in weeks 4-7 and Assignment 2.

## Readings

- Russell and Norvig, *Artificial Intelligence: A Modern Approach* (4th ed., 2022), Chapter 2, for Lecture 1.
- Shoham and Leyton-Brown, *Multiagent Systems* (2008).
- Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning* (MIT Press, 2024).
- Recent papers and the slides matter most (ReAct, Generative Agents, AutoGen and others). No need to buy the books.

---

## Weekly Schedule

| Week | Dates | Lectures | Werkcollege | Due |
|---|---|---|---|---|
| 1 | 26-30 Oct | Course intro, overview of agents and MAS; game theory, population dynamics, evolutionary game theory | Lab 1: game theory, evolutionary game theory, RL | |
| 2 | 2-6 Nov | Reinforcement learning and intro to MARL (Roman); equilibrium selection and MARL; deep RL and deep MARL | Lab 2: multi-agent deep RL | A1 out |
| 3 | 9-13 Nov | Multi-agent deep RL in practice; examples of MARL; transition to agentic systems (Giovanni) | Support for Assignment 1 | A1 due Fri 13 Nov |
| 4 | 16-20 Nov | How do agents decide? (Annet): reasoning; memory, tools, reflection | Lab 3: agent architecture | Midterm Fri 20 Nov |
| 5 | 23-27 Nov | Agent-centric information retrieval (Evangelos); multi-agent architectures, orchestration, protocols | Lab 4: agent societies | A2 out |
| 6 | 30 Nov-4 Dec | Evaluation and benchmarks; evaluation and failure in multi-agent LLMs; personas and simulacra of human behaviour (Dimitris) | Support for Assignment 2 | A2 due Fri 4 Dec |
| 7 | 7-11 Dec | Unpacking real-world applications: coding, Paprika (Andrea); challenges and risks of multi-agent advanced systems; the Hugging Face incident; exam preparation | Lab 5: exam preparation | |
| 8 | 14-18 Dec | | | Final exam |

### Week 1: Agents, games and populations

| | Topic | Notes |
|---|---|---|
| L1 (Tue 27 Oct) | Course intro; agents, multi-agent systems and LLM agents | [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]] |
| Reading | Russell and Norvig Ch 2, Intelligent Agents | [[AGAI-Book Ch2 - Intelligent Agents]] |
| Lab 1 (Wed 28 Oct) | Simple agents, complex dynamics: Nash equilibria, social dilemmas, repeated games, replicator dynamics, signalling, reward hacking | [[AGAI-Lab01 - Simple Agents, Complex Dynamics]] |
| L2 (Thu 29 Oct) | Game theory and solution concepts; populations of agents (evolutionary game theory) | Slides not on Canvas yet |

---

## Status

| Note | Status |
|---|---|
| [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]] | complete |
| [[AGAI-Book Ch2 - Intelligent Agents]] | complete |
| [[AGAI-Lab01 - Simple Agents, Complex Dynamics]] | complete |

---

## Concept Index

**Agents:** [[Intelligent Agent]] · [[Rational Agent]] · [[Task Environment]] · [[Agent Program]] · [[Multi-Agent System]]

**LLM agents:** [[LLM Agent]] · [[ReAct]] · [[Chain-of-Thought Prompting]] · [[Large Language Models (LLM)]] · [[Retrieval-Augmented Generation]]

**Game theory and populations:** [[Nash Equilibrium]] · [[Pareto Optimality]] · [[Prisoner's Dilemma]] · [[Stag Hunt]] · [[Repeated Game]] · [[Replicator Dynamics]]

**Rewards and learning:** [[Reward Hacking]] · [[Reward Signal]] · [[Value Iteration]] · [[Markov Decision Process]] · [[Reinforcement Learning]]
