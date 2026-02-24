---
type: moc
course: RL
tags: [moc]
---

# Reinforcement Learning — Overview

> **Course:** Reinforcement Learning 2025/2026 (Period 4)
> **Programme:** MSc Artificial Intelligence, UvA
> **Credits:** 6 EC (168 hours total)
> **Instructors:** Herke van Hoof, Christian Andersson Naesseth
> **Exam:** March 27, 2026 (65% of final grade, must score ≥ 5.0)

## Textbooks

- **Primary:** *Reinforcement Learning: An Introduction* (2nd ed.) — Sutton & Barto (RL:AI)
  - Chapters covered: 1-6, 8-11, 13, 16, 17
  - [Free PDF](http://incompleteideas.net/book/RLbook2020.pdf)
- **Secondary:** *A Survey on Policy Search for Robotics* — Deisenroth, Neumann, Peters
- Additional papers distributed via Canvas

## Assessment

| Component | Weight | Notes |
|-----------|--------|-------|
| 5 Coding Assignments | 2% each (10% total) | Groups of 2, late = -1 point |
| 5 Homework Sets | 4% each (20% total) | Groups of 2, due Thursdays 17:00 |
| Empirical RL Assignment | 5% | |
| Exam | 65% | Must score ≥ 5.0 |

---

## Weekly Schedule

### Week 1 — Foundations
| | Topic | Literature | Notes |
|---|-------|-----------|-------|
| L1 | Intro, MDPs & Bandits | RL:AI 1.1-1.4, 1.6, 2.1-2.4, 2.6-2.7, 3.1-3.3 | [[RL-L01 - Intro, MDPs & Bandits]] |
| L2 | Dynamic Programming | RL:AI 2.5, 3.4-3.8, 4 | [[RL-L02 - Dynamic Programming]] |
| **Book** | | | [[RL-Book Ch1 - Introduction]], [[RL-Book Ch2 - Multi-Armed Bandits]], [[RL-Book Ch3 - Finite MDPs]], [[RL-Book Ch4 - Dynamic Programming]] |
| **Exercises** | Ex 0.1-0.2, 1.1-1.3, 2.1-2.2 | | [[RL-ES01 - Exercise Set Week 1]] |
| **Homework** | HW 2.3-2.4 | Due 12/2 | [[RL-HW01 - Homework 1]] |
| **Coding** | CA1: Dynamic Programming | | [[RL-CA01 - Dynamic Programming]] |

### Week 2 — Tabular Methods
| | Topic | Literature | Notes |
|---|-------|-----------|-------|
| L3 | Monte Carlo Methods | RL:AI 5.1-5.7 | [[RL-L03 - Monte Carlo Methods]] |
| L4 | Temporal Difference Methods | RL:AI 6.1-6.5 | [[RL-L04 - Temporal Difference Learning]] |
| **Book** | | | [[RL-Book Ch5 - Monte Carlo Methods]], [[RL-Book Ch6 - Temporal-Difference Learning]] |
| **Exercises** | Ex 3.1-3.3, 4.1-4.2 | | [[RL-ES02 - Exercise Set Week 2]] |
| **Homework** | HW 3.4, 4.3 | Due 19/2 | [[RL-HW02 - Homework 2]] |
| **Coding** | CA2: Monte Carlo | | [[RL-CA02 - Monte Carlo]] |

### Week 3 — From Tabular to Approximation
| | Topic | Literature | Notes |
|---|-------|-----------|-------|
| L5 | From Tabular Learning to Approximation | RL:AI 9.1-9.3 | [[RL-L05 - Tabular to Approximation]] |
| L6 | On-policy TD Learning with Approximation | RL:AI 9.3-9.4, 9.7-9.8 | [[RL-L06 - On-Policy TD with Approximation]] |
| **Book** | | | [[RL-Book Ch9 - On-Policy Prediction with Approximation]] |
| **Exercises** | Ex 5.1-5.2, 6.1-6.4 | | [[RL-ES03 - Exercise Set Week 3]] |
| **Homework** | HW 5.3-5.4, 6.5 | Due 26/2 | [[RL-HW03 - Homework 3]] |
| **Coding** | CA3: Temporal Difference | | [[RL-CA03 - Temporal Difference]] |

### Week 4 — Off-Policy Approximation & Deep RL *(current)*
| | Topic | Literature | Notes |
|---|-------|-----------|-------|
| L7 | Off-policy RL with Approximation | RL:AI 10.1, 11.1-11.7 | [[RL-L07 - Off-Policy RL with Approximation]] |
| L8 | Deep RL (Value-Based Methods) | RL:AI 16.5; DQN & CQL papers | [[RL-L08 - Deep RL Value-Based]] |
| **Book** | | | [[RL-Book Ch10 - On-Policy Control with Approximation]], [[RL-Book Ch11 - Off-Policy Methods with Approximation]] |
| **Exercises** | Ex 7.1-7.3, 8.1 | | |
| **Homework** | HW 7.4, 8.2-8.3 | Due 5/3 | |
| **Coding** | | | |

### Week 5 — Policy Gradient Methods
| | Topic | Literature | Notes |
|---|-------|-----------|-------|
| L9 | REINFORCE | RL:AI 13.1-13.4, 13.7; Survey §2.4.1.2 | |
| L10 | PGT, DPG & Evaluation | RL:AI 13.5 + papers | |

### Week 6 — Advanced Methods
| | Topic | Literature | Notes |
|---|-------|-----------|-------|
| L11 | SAC, Decision Transformer, Decision Diffuser | Papers | |
| L12 | Planning and Learning | RL:AI 8.1-8.2, 8.8, 8.10-8.11, 8.13, 16.6 | |

### Week 7 — Wrap-Up
| | Topic | Literature | Notes |
|---|-------|-----------|-------|
| L13 | Partial Observability | RL:AI 17.3 | |
| L14 | Recap & Exam FAQ | | |
| **Exam** | March 27, 2026 | | |

---

## Exam Prep

- [[RL - Exam 2024 Analysis]]
- [[RL - Exam Cheat Sheet]] *(to be created)*

---

## Concept Index

Key concepts covered in this course (see [[Concepts/]] folder):

**Foundations:** [[Markov Decision Process]] · [[Bellman Equation]] · [[Value Function]] · [[Policy]] · [[Return]] · [[Discount Factor]] · [[Multi-Armed Bandit]]

**Tabular Methods:** [[Dynamic Programming]] · [[Policy Iteration]] · [[Value Iteration]] · [[Monte Carlo Methods]] · [[Temporal Difference Learning]] · [[SARSA]] · [[Q-Learning]] · [[Expected SARSA]]

**Exploration:** [[Exploration vs Exploitation]] · [[Epsilon-Greedy Policy]] · [[Exploring Starts]] · [[Upper Confidence Bound]] · [[Importance Sampling]]

**Approximation:** [[Function Approximation]] · [[Stochastic Gradient Descent]] · [[Semi-Gradient Methods]] · [[Linear Function Approximation]] · [[Feature Construction]] · [[Tile Coding]] · [[LSTD]]

**Deep RL:** [[Deep Q-Network (DQN)]] · [[Experience Replay]] · [[Target Network]] · [[Conservative Q-Learning (CQL)]]

**Off-Policy:** [[On-Policy vs Off-Policy]] · [[Deadly Triad]] · [[Gradient-TD Methods]]

**Policy Gradient:** *(weeks 5-6)* [[REINFORCE]] · [[Policy Gradient Theorem]] · [[Actor-Critic]] · [[Soft Actor-Critic (SAC)]]
