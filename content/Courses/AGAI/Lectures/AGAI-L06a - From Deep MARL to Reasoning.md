---
type: lecture
course: 5204AGAI6Y
week: 3
lecture: 6
date: 2026-11-12
book_sections: ["Albrecht, Christianos and Schäfer, MARL, Ch 9"]
topics:
  - "[[Monte Carlo Tree Search (MCTS)]]"
  - "[[AlphaGo Zero]]"
  - "[[Self-Play]]"
  - "[[Population-Based Training]]"
  - "[[Policy Space Response Oracles (PSRO)]]"
  - "[[REINFORCE]]"
  - "[[GRPO]]"
  - "[[DeepSeek-R1]]"
  - "[[Social Dilemma]]"
status: complete
---

# AGAI-L06a: From Deep MARL to Reasoning

> [!abstract] In short
> How do you learn a huge two-player zero-sum game like Go, where the only reward is win or lose at the end? Plain Monte Carlo tree search judges positions by random rollouts, a weak signal that assumes a random opponent. AlphaGo guides the search with two networks: a policy network that says which moves to explore and a value network that scores new positions. AlphaGo Zero learns both from self-play alone, and AlphaZero runs the same method on several games. Self-play needs symmetric roles; otherwise policies are trained against populations of other policies. PSRO does this with a meta-game, a normal-form game whose actions are the policies in the current populations; AlphaStar's league works like PSRO. The lecture ends by linking RL to LLMs and agentic AI.
>
> Feeds mock question 28 (the PSRO meta-game). MCTS and self-play themselves are explained in [[AGAI-L05 - Deep MARL, Part II#Large zero-sum games: search and self-play|Lecture 5]].

## Why plain MCTS struggles in Go

The setting is a two-player zero-sum board game with a sparse reward, a huge branching factor and a long horizon. [[Monte Carlo Tree Search (MCTS)|MCTS]] (selection, expansion, backpropagation, with $Q(s,a)$ the average outcome through $(s,a)$; see [[AGAI-L05 - Deep MARL, Part II#Monte Carlo tree search|Lecture 5]]) needs a way to judge a position it has just reached. In Go the usual ways fail:

- **Heuristics** that score positions require **manual annotation** by experts.
- **Random rollouts give a weak signal** about which lines are promising: a random game from a position says little about who wins under good play.
- **Random rollouts assume random opponents.** A move that wins against random replies can lose against a strong player.

The fix: learn the two things MCTS needs, **which moves are worth exploring** (a policy) and **how good a position is** (a value).

## AlphaGo: search guided by two networks

AlphaGo is "guided MCTS" with two networks:

- **Policy network** $\pi(a \mid s;\theta)$: which moves are promising? It **guides node selection**, so the search explores few of the many legal moves.
- **Value network** $V_\phi(s)$: what are the chances of winning from this position? It **evaluates newly expanded nodes**, so the search gets a learned estimate of a new position instead of relying on random rollouts alone.

How they are trained:

> [!formula] AlphaGo's training
> 1. **Policy, copy the experts:** supervised, reproduce expert moves with a cross-entropy loss, $L(\theta) = -\sum_{(s,a)}\log\pi(a \mid s;\theta)$.
> 2. **Policy, improve by self-play with [[REINFORCE]]:** $\theta \leftarrow \theta + \eta\, z \sum_t \nabla_\theta \log\pi(a_t \mid s_t;\theta)$, with $z$ the outcome of the self-play game (win or loss).
> 3. **Value, predict the winner:** squared error on the self-play outcomes, $L(\phi) = \big(z - V_\phi(s)\big)^2$.

Stage 2 is REINFORCE with a sparse reward: the only reward comes at the end, so every move of a won game becomes more likely and every move of a lost game less likely. The opponent is the policy itself, so it improves against ever-stronger play, which answers the "random opponent" problem.

## AlphaGo Zero and AlphaZero: no human games

**AlphaGo Zero** learns **entirely through self-play, with no human game examples**: it **starts from a random policy** instead of one trained on expert data, so stage 1 disappears. Self-play is possible because **the same policy can be used from the perspective of both agents** (the board shown from the side of the player to move). Search and network improve each other (detail from outside the course material): MCTS guided by the network picks better moves than the network alone, the network is trained towards the search's choices and the game outcomes, and the next search starts from a stronger guide. See [[AlphaGo Zero]].

**AlphaZero** applies the **same learning and search method to different games** (chess, shogi, Go). The only game-specific part is the engine for each game's legal moves, transitions and end conditions, which MCTS needs to simulate.

## When self-play is not enough: populations of policies

AlphaZero is self-play in a zero-sum game with a **homogeneous population**: one policy plays both sides. That **assumes the two agents have symmetric roles**, which is not always the case. When players have different roles, actions or observations, one policy cannot play both sides.

**[[Population-Based Training|Population-based training]]** tests one policy against a **population** of other policies, and the policies in a population can represent agents with **different roles, actions and observations**:

1. **Initialise** one population for each agent (for example with uniform policies).
2. **Evaluate** the current policies against the policies in the other populations.
3. **Modify** existing policies, by random changes or by copying the parameters of the best performers. Repeat.

This echoes the population dynamics of [[AGAI-L02 - Game Theory and Population Dynamics|week 1]]: what does well against the current populations spreads, as in [[Replicator Dynamics|replicator dynamics]].

A second reason for populations (from outside the course material): plain self-play only ever faces its latest self. In games with non-transitive strategies, like rock-paper-scissors, it can cycle and forget how to beat old strategies; a population keeps them.

## PSRO: populations plus a meta-game

**[[Policy Space Response Oracles (PSRO)]]** also keeps populations of policies, and adds a meta-game over them.

> [!definition] Meta-game
> A simplified **[[Normal-Form Game|normal-form game]]** that represents a more complex interaction between complex policies.
> - **Actions** in the meta-game = **policies** in the current populations
> - **Payoffs** between actions = **average returns** between those policies

Each cell holds every agent's average return when those two policies play each other, estimated by simulating many episodes of the real game. The real game can be huge; the meta-game has one row or column per stored policy.

PSRO repeats:

1. **Start** with random policies.
2. **Construct the meta-game** from the current populations (1 policy = 1 action).
3. **Use a meta-solver** to rank the policies, for example a [[Nash Equilibrium|Nash equilibrium]] of the meta-game. This gives a probability distribution over each agent's population (the meta-solution).
4. **Sample policies** for the other agents from the meta-solution.
5. **Train a new policy** for agent $i$ against them with single-agent RL in the real game, and **add it** to the population.

The new policy is an approximate [[Best Response|best response]] to the meta-solution; the "oracle" is the RL algorithm that produces it. Each generation the meta-game gains a row and a column, whose payoffs must be simulated before the next solve.

> [!example] PSRO on rock-paper-scissors (example from outside the course material)
> Both players start with {Rock}. The meta-solution is "always Rock", so the best response is Paper: {Rock, Paper}. Now the solution is "always Paper", best response Scissors: {Rock, Paper, Scissors}. The meta-game is now the full game, its Nash equilibrium mixes all three at 1/3, and nothing beats that mixture. Self-play against its latest self would cycle Rock, Paper, Scissors forever; PSRO stops because each new policy must beat a mixture of the earlier ones.

## AlphaStar: a PSRO-like league

AlphaStar (StarCraft II) is the example of **self-play with heterogeneous populations**: population-based training in a **league** that works like PSRO. (Outside the course material: StarCraft players choose different races and see only part of the map, so roles are asymmetric, which is why one self-play policy is not enough.)

## How the methods compare

| Method | Starts from | Trains against | Roles |
|---|---|---|---|
| AlphaGo | expert moves | itself (REINFORCE) | symmetric |
| AlphaGo Zero | random policy | itself | symmetric |
| AlphaZero | random policy | itself, several games | symmetric |
| Population-based training | e.g. uniform policies | other populations | any |
| PSRO | random policies | meta-solution mixture | any |
| AlphaStar | not covered | a PSRO-like league | asymmetric |

## From RL to agentic AI

The course then moves from RL to agentic AI, and the lecture links the two in three ways.

- **RL versus LLMs.** AlphaGo has two systems: **System 1**, the policy network (intuition, immediate action), and **System 2**, the long-horizon search. Current [[Large Language Models (LLM)|LLMs]] are argued to be **System 1 only**: they generate the next token without search.
- **RL plus LLMs.** Text generation as RL: the **state** is the prompt plus the tokens so far, the **action** is the next token, the **reward** is high if the completed response is good, the **policy** is the learned $\pi(a \mid s)$. Improving the policy with a policy gradient method ([[GRPO]], close to [[PPO]]) means better reasoning.
- **Same challenges.** Agents acting for different users, even when each is well aligned to its own user, have objectives that are neither identical nor completely opposed: **mixed-motive** settings, with room for mutual gain and for conflict ([[Social Dilemma|social dilemmas]]), the same problems as in MARL and Assignment 1.

## What to be able to answer

- Why plain MCTS is weak in Go, and which AlphaGo network fixes what (selection against evaluation of new nodes).
- How each network is trained, in words; what AlphaGo Zero drops and what AlphaZero generalises.
- What self-play assumes, and why populations are used instead.
- The PSRO meta-game (actions, payoffs), its steps, and what the meta-solver and the oracle do (mock Q28).

## Exam questions

> [!exam]- (a) Why is plain Monte Carlo tree search weak in a game like Go? (b) Name AlphaGo's two networks and say where each is used in the search. (c) How is each trained? (d) What does AlphaGo Zero change, and what makes that possible?
> **Key points:** heuristics need manual annotation, random rollouts give a weak signal and assume random opponents; policy network guides node selection, value network evaluates newly expanded nodes; policy: copy expert moves (cross-entropy), then REINFORCE in self-play weighted by the outcome $z$; value: squared error to $z$; AlphaGo Zero drops the expert data and starts from a random policy, possible because one policy can play from both players' perspectives.
>
> (a) MCTS has to judge the positions it reaches. Hand-made heuristics need manual annotation, and random rollouts give a weak signal about which lines are promising; they also assume a random opponent, while the agent must beat strong ones.
>
> (b) The policy network says which moves are promising and guides node selection, so only a few of the many legal moves are explored. The value network estimates the chance of winning and evaluates newly expanded nodes, a learned estimate instead of random rollouts alone.
>
> (c) The policy network first copies expert moves (supervised, cross-entropy), then improves by self-play with REINFORCE: every move of a won game becomes more likely, every move of a lost game less likely. The value network regresses the self-play outcome $z$ with a squared error.
>
> (d) AlphaGo Zero uses no human games: it starts from a random policy and learns only from self-play. This works because the same policy can be used from the perspective of both agents (symmetric roles, board shown from the mover's side).

> [!exam]- (a) What does self-play assume about the agents? Give a game where the assumption fails. (b) How does population-based training work? (c) Define the PSRO meta-game. (d) Give the steps of PSRO.
> **Key points:** one policy plays both sides, so roles must be symmetric (fails e.g. attacker against defender with different actions); one population per agent, evaluate against the other populations, modify by random changes or copying the best; meta-game = normal-form game, actions = policies in the populations, payoffs = average returns between them; start random, build meta-game, meta-solver (e.g. Nash), sample from the meta-solution, train a best response with single-agent RL and add it to the population.
>
> (a) Self-play uses the same policy for both agents, so it assumes symmetric roles with egocentric observations. It fails when the players differ, for example an attacker who moves units against a defender who builds walls, or StarCraft races with different units.
>
> (b) Initialise one population of policies per agent, evaluate the current policies against the policies in the other populations, then modify them by random changes or by copying the best performers' parameters, and repeat. The populations can hold agents with different roles, actions and observations.
>
> (c) A simplified normal-form game: each action is a policy in the current populations, and the payoffs are the average returns when those policies play each other, estimated by simulation.
>
> (d) Start with random policies; construct the meta-game; solve it with a meta-solver such as a Nash equilibrium to get a distribution over each population; sample the other agents' policies from that meta-solution; train a new policy for each agent against them with single-agent RL and add it to its population. Repeat with the grown populations.

## Flashcards

> [!card]- Why are random rollouts a weak signal for MCTS in Go?
> A random game says little about who wins under good play.

> [!card]- Which opponent do random rollouts implicitly assume?
> A random one, while the agent has to beat strong players.

> [!card]- Which MCTS step does AlphaGo's policy network guide?
> Node selection: which moves to explore.

> [!card]- What does AlphaGo's value network do in the search?
> Estimates the chance of winning at newly expanded nodes.

> [!card]- How is AlphaGo's policy network first trained?
> Supervised, to reproduce expert moves (cross-entropy loss).

> [!card]- In AlphaGo's REINFORCE stage, what happens to the moves of a lost self-play game?
> All become less likely, weighted by the outcome $z$.

> [!card]- What loss trains AlphaGo's value network?
> Squared error to the self-play outcome, $(z - V_\phi(s))^2$.

> [!card]- What does AlphaGo Zero drop from AlphaGo?
> The expert data: it starts from a random policy and learns only by self-play.

> [!card]- What does AlphaZero generalise?
> The same learning and search method to several games; only the game engine changes.

> [!card]- What does AlphaZero-style self-play assume about the two agents?
> Symmetric roles, so one policy can play both sides.

> [!card]- Why can population-based training handle asymmetric games?
> Policies in the populations can have different roles, actions and observations.

> [!card]- How does population-based training modify its policies?
> Random changes, or copying the best performers' parameters.

> [!card]- What is the PSRO meta-game?
> A normal-form game whose actions are the policies in the current populations.

> [!card]- What are the payoffs of the PSRO meta-game?
> Average returns when the policies play each other, estimated by simulation.

> [!card]- What does PSRO's meta-solver produce?
> A distribution over each agent's population, e.g. a Nash equilibrium of the meta-game.

> [!card]- How does PSRO train a new policy?
> Single-agent RL against the others' policies sampled from the meta-solution.

> [!card]- What kind of training is AlphaStar's league?
> Population-based training with heterogeneous populations, PSRO-like.

> [!card]- Why are current LLMs called System 1 only?
> They generate the next token without search, where AlphaGo also searches (System 2).

> [!card]- When text generation is framed as RL, what are the state and the action?
> State: the prompt and tokens so far. Action: the next token.

> [!card]- True or false: AlphaGo's value network chooses which nodes MCTS explores.
> **False.** The policy network guides selection; the value network evaluates new nodes.

> [!card]- True or false: in a PSRO meta-game, the actions are moves of the underlying game.
> **False.** The actions are whole policies from the current populations.

> [!card]- True or false: AlphaGo Zero's policy is first trained on expert games.
> **False.** It starts from a random policy.

> [!card]- Multiple selection: which are steps of PSRO? (A) Construct a meta-game from the current populations (B) Solve it with a meta-solver (C) Decompose the team value into utilities (D) Train a new policy against policies sampled from the meta-solution
> **A, B and D.** C is value decomposition.

> [!card]- Multiple selection: which weaknesses of plain MCTS does AlphaGo address? (A) Heuristics need manual annotation (B) Random rollouts give a weak signal (C) Random rollouts assume random opponents (D) The game has no fixed rules
> **A, B and C.** Go has fixed rules, which MCTS relies on.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L05 - Deep MARL, Part II]] (training modes, value decomposition, agent modelling, parameter sharing, MCTS, self-play)
- **Earlier:** [[AGAI-L04 - Deep RL and Intro to Deep MARL]] (REINFORCE, PPO); [[AGAI-L02 - Game Theory and Population Dynamics]] (normal-form games, Nash equilibrium, replicator dynamics)
- **Lab:** [[AGAI-Lab02 - Learning Agents]]
- **Concepts:** [[Monte Carlo Tree Search (MCTS)]], [[AlphaGo Zero]], [[Self-Play]], [[Population-Based Training]], [[Policy Space Response Oracles (PSRO)]], [[Normal-Form Game]], [[Nash Equilibrium]], [[Best Response]], [[REINFORCE]], [[Policy Gradient Methods]], [[GRPO]], [[DeepSeek-R1]], [[Large Language Models (LLM)]], [[LLM Agent]], [[Social Dilemma]], [[Replicator Dynamics]]
- **Source:** Agentic Artificial Intelligence, Lecture 6 part I, "From Deep MARL to Reasoning", MSc AI, UvA, 12 November 2026. Part II is Giovanni Sileno's guest lecture on definitions of agents.
- **Reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapter 9
- **Papers named:** Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (NeurIPS 2017); the AlphaStar team, "AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning" (DeepMind, 2019)
- **Mock exam:** AGAI Mock Midterm 2026, Deep MARL question 28
