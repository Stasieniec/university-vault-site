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
> How do you learn a huge two-player zero-sum game like Go, where the only reward is win or lose at the end? Monte Carlo tree search explores the game tree, but its random rollouts are a weak signal and assume a random opponent. AlphaGo guides the search with two networks: a policy network (first copying expert moves, then improved by REINFORCE in self-play) that chooses which moves to explore, and a value network that scores new positions. AlphaGo Zero drops the expert data and learns from self-play alone, and AlphaZero runs the same method on chess, shogi and Go. Self-play needs symmetric roles; when roles differ, population-based training trains policies against populations of other policies, and PSRO does this through a meta-game, a normal-form game whose actions are the policies in the current populations (AlphaStar's league is PSRO-like). The lecture closes by tying RL to agentic AI: an LLM is a next-token policy that RL can improve, it arguably lacks AlphaGo's search, and LLM agents face the same cooperation and conflict problems as MARL agents.
>
> Feeds mock question 28 (the PSRO meta-game) and the Deep MARL part of the midterm in general.

## What this builds on

[[AGAI-L05 - Deep MARL, Part II|Lecture 5]] ended on the setting used here: **zero-sum games with two agents and fully observable states and actions**, such as chess, shogi, backgammon and Go. They are hard because the reward is sparse (win or lose only at the end), the branching factor is large (many legal moves per position) and the horizon is long, so learning needs an efficient way to explore the game tree. Lecture 5 also gave the requirements of [[Self-Play|self-play]] (symmetric roles, egocentric observations) and [[Monte Carlo Tree Search (MCTS)|MCTS]], a tree search that simulates possible outcomes of the game and updates value estimates of the actions at each state. Each simulation has three steps: **selection and simulation** (walk down the tree, choosing an action at each node), **expansion** (add a new leaf $\hat s^l$ to the tree) and **backpropagation** (send the outcome back up the visited path).

One detail matters for what follows. In plain MCTS, once the search adds a new leaf, it plays a **random rollout** from there to the end of the game (both sides choosing random moves), and that game's outcome $u$ is what backpropagation sends up the tree. Every $(s,a)$ on the path is updated with it:

$$N(s,a) \leftarrow N(s,a) + 1, \qquad Q(s,a) \leftarrow Q(s,a) + \frac{G - Q(s,a)}{N(s,a)}$$

where $G$ is this simulation's outcome (the $u$ above), $N(s,a)$ counts how often $a$ was tried in $s$, and $Q(s,a)$ stays the average outcome through $(s,a)$. Selection picks actions from these statistics with ε-greedy or [[Upper Confidence Bound|UCB]].

## Why plain MCTS struggles in Go

MCTS needs some way to judge a position it has just reached. In a game as large as Go, the usual ways have three problems:

1. **Hand-made heuristics need manual annotation.** A function that scores positions directly has to be designed or labelled by human experts.
2. **Random rollouts give a weak signal** about which trajectories are most promising. The outcome of a random game from a position says little about who is winning under good play.
3. **Random rollouts assume random opponents.** A move that wins against random replies may lose against a strong player, and a strong player is exactly who the agent will face.

The fix is to learn the two things MCTS needs: **which moves are worth exploring** (a policy) and **how good a position is** (a value).

## AlphaGo: search guided by two networks

**AlphaGo** (Silver et al., Nature 2016) is "guided MCTS" trained with expert data. It learns two networks:

- a **policy network** $\pi(a \mid s;\theta)$: which moves are promising? It guides node selection in the tree;
- a **value network** $V_\phi(s)$: what are the chances of winning from this position? It estimates the value of newly explored states.

They are trained in three stages.

> [!formula] AlphaGo's training
> **1. Copy the experts** (supervised, cross-entropy loss):
> $$L(\theta) = -\sum_{(s,a) \sim D}\log\pi(a \mid s;\theta)$$
> **2. Improve by self-play with [[REINFORCE]]:**
> $$\theta \leftarrow \theta + \eta\, z\sum_t \nabla_\theta\log\pi(a_t \mid s_t;\theta)$$
> **3. Learn to predict the winner** (squared error):
> $$L(\phi) = \big(z - V_\phi(s)\big)^2$$
> - $D$: a dataset of positions $s$ and the moves $a$ expert human players made in them
> - $\eta$: the learning rate; $t$ runs over the moves of one self-play game
> - $z$: the **outcome of the self-play game** for the player whose moves are updated (in the paper $+1$ for a win, $-1$ for a loss)
> - $V_\phi(s)$: the value network's estimate of the outcome from position $s$

Stage 1 makes the expert's move likely in each position. Stage 2 is REINFORCE with a sparse reward: the only reward arrives at the end, so the return from every move is the outcome $z$, and every move of a won game becomes more likely while every move of a lost game becomes less likely. The opponent is the policy itself (in the paper, randomly chosen earlier versions of it), so the policy keeps improving beyond the experts against ever-stronger play, which answers the "random opponent" problem. Stage 3 regresses the value network on the outcomes of those self-play games, so it answers "who wins from here?" without a rollout.

Inside the search, each network replaces one weak part of plain MCTS:

```pseudo
Algorithm: One simulation of network-guided MCTS (AlphaGo)
──────────────────────────────────────────────────────────
s ← root (the current position)
While s is already in the tree:                    // selection
  choose a using Q(s,a), N(s,a) and the policy network π(a | s; θ)
                                                   // promising moves are explored first
  s ← the position after a
Add s to the tree                                  // expansion (InitializeNode)
u ← V_φ(s)                                         // value network scores the new leaf
For each (s,a) on the path, from the leaf up:      // backpropagation
  N(s,a) ← N(s,a) + 1
  Q(s,a) ← Q(s,a) + (u − Q(s,a)) / N(s,a)          // u seen from the side that played a:
                                                   // in a zero-sum game, my +u is your −u
```

The policy network narrows the search (few of the many legal moves get explored), and the value network shortens it (no need to play to the end). (Outside the course material: the exact selection formula is in [[AlphaGo Zero]], and the original AlphaGo scored a leaf by mixing the value network's estimate with the outcome of a fast rollout; AlphaGo Zero uses the value network alone.)

## AlphaGo Zero and AlphaZero: no human games

**AlphaGo Zero** (Silver et al., Nature 2017, "Mastering the game of Go without human knowledge") learned **entirely through self-play, with no human game examples**. It starts from a **random policy** instead of one trained on expert data, so stage 1 and its need for expert data disappear. Self-play is possible because **the same policy can be used from the perspective of both agents**: the board is presented from the side of the player to move, so one network can play black and white.

The training loop, as the AlphaGo Zero paper describes it (beyond the lecture's level of detail):

```pseudo
Algorithm: AlphaGo Zero / AlphaZero training loop
─────────────────────────────────────────────────
Initialise one network with random weights: policy head p(a | s), value head v(s)
Repeat:
  Play a game against itself: at every move, run MCTS guided by the network
    and play according to the search's visit counts
  For each position s of the game, store (s, π_MCTS(s), z)
    // π_MCTS: how often MCTS tried each move; z: final outcome for the player to move in s
  Train the network on stored positions:
    p(· | s) towards π_MCTS(s)     (cross-entropy)
    v(s)     towards z             (squared error)
```

Search improves on the network's moves, the network learns to imitate the improved moves, and the next search starts from a stronger guide.

**AlphaZero** (Silver et al., Science 2018) drops "Go" from the name because it is the **same learning and search method applied to different games**: chess, shogi and Go. The only game-specific part is the engine that enforces each game's legal moves, transitions and end conditions. MCTS needs this engine, because every simulation steps through the game's rules.

## When self-play is not enough: populations of policies

AlphaZero is **self-play in a zero-sum game with a homogeneous population**: a single policy plays both sides. That **assumes the two agents have symmetric roles**, which is not always the case. In many games the players have different roles, different actions or different observations, and one policy cannot simply play both sides.

**[[Population-Based Training|Population-based training]]** tests one policy against a **population** of other policies. The policies in a population can represent agents with **different roles, actions and observations**. The general procedure:

1. **Initialise** one population for each agent (for example with uniform random policies).
2. **Evaluate** the current policies against the policies in the other populations.
3. **Modify** existing policies, for example through random changes, or by copying the parameters of the best-performing policies.
4. Repeat from step 2.

This brings back the population dynamics of [[AGAI-L02 - Game Theory and Population Dynamics|week 1]]: as in [[Replicator Dynamics|replicator dynamics]], what does well against the current populations spreads (by copying the best performers), and random changes add variation.

## PSRO: populations plus a meta-game

**[[Policy Space Response Oracles (PSRO)]]** (Lanctot et al., NeurIPS 2017) also keeps populations of policies. What it adds is a **meta-game** over those policies: PSRO solves it and trains each new policy against the resulting mixture of the existing ones.

> [!definition] Meta-game
> A meta-game is a simplified **[[Normal-Form Game|normal-form game]]** that represents a more complex interaction between complex policies.
> - **Actions** in the meta-game = **policies** in the complex interaction
> - **Payoffs** between actions = **average returns** between the policies

Each cell holds every agent's average return for that pair of policies (for two agents, a pair of numbers such as "1, 2"), estimated by letting the policies play many episodes of the real game. The real game can be huge, but the meta-game has only one row or column per stored policy.

PSRO repeats five steps:

1. **Start** with random policies.
2. **Construct the meta-game** from the current populations: one policy is one action.
3. **Use a meta-solver** to rank the policies, for example by computing a [[Nash Equilibrium|Nash equilibrium]] of the meta-game. The result is a probability distribution $\delta_i^k$ over each agent's population.
4. **Sample policies** for the other agents from the meta-solution.
5. **Train a new policy** for agent $i$ against those sampled policies, with single-agent RL in the real game, and add it to the population.

```pseudo
Algorithm: Policy space response oracles (PSRO)
───────────────────────────────────────────────
Initialize populations Π_i^1 for all agents i ∈ I (e.g. random policies)
For each generation k = 1, 2, 3, ...:
  Construct meta-game M^k from current populations {Π_i^k}_{i ∈ I}
                     // entry: average returns when the chosen policies play each other
  Use meta-solver on M^k to obtain distributions {δ_i^k}_{i ∈ I}
  For each agent i ∈ I:                            // train best-response policies (the oracle)
    For each episode e = 1, 2, 3, ...:
      Sample policies for the other agents π_{−i} ~ δ_{−i}^k
      Use single-agent RL to train π_i' with respect to π_{−i} in the underlying game G
  Grow populations Π_i^{k+1} ← Π_i^k ∪ {π_i'}
```

The new policy $\pi_i'$ is an approximate [[Best Response|best response]] to the others' meta-game distribution: the "oracle" in the name is the RL algorithm that produces it. After each generation the meta-game gains one row and one column, for $\pi_1'$ and $\pi_2'$. Their payoffs against every older policy are still unknown, so the next generation has to simulate those new pairings before it can solve the larger meta-game.

> [!example] PSRO on rock-paper-scissors (example from outside the course material)
> Win = +1, loss = −1, draw = 0, and both players start with the population {Rock}.
> - **Generation 1.** The meta-game is a single cell, Rock against Rock. Its Nash equilibrium is "always Rock", and the best response to Rock is Paper. Populations: {Rock, Paper}.
> - **Generation 2.** In the 2 × 2 meta-game, Paper beats Rock and draws with itself, so the Nash equilibrium is "always Paper". The best response is Scissors. Populations: {Rock, Paper, Scissors}.
> - **Generation 3.** The meta-game is now the full game. Its Nash equilibrium mixes all three with probability 1/3, and against that mixture no new policy does better than a draw, so PSRO stops at the game's Nash equilibrium.
>
> Self-play that always best-responds to its latest self would go Rock, Paper, Scissors, Rock, ... forever. PSRO stops cycling because each new policy has to beat a mixture of the earlier ones.

## AlphaStar: a PSRO-like league

**AlphaStar** (DeepMind, 2019) reached Grandmaster level in StarCraft II, a real-time strategy game, using multi-agent RL. It is the example of **self-play with heterogeneous populations**: population-based training in a **league** that works like PSRO.

(Outside the course material: StarCraft II players pick one of three races with different units and see only part of the map, so the roles are asymmetric. AlphaStar's agents started by imitating human games, then trained in a league of frozen copies of past agents plus "exploiter" agents that hunt for the main agents' and the league's weaknesses. Opponents were picked by how often the learner lost to them, the role PSRO gives to the meta-solver.)

## How the methods compare

| Method | Starts from | Trains against | Roles |
|---|---|---|---|
| AlphaGo | expert moves | itself (REINFORCE) | symmetric |
| AlphaGo Zero | random policy | itself | symmetric |
| AlphaZero | random policy | itself, in three games | symmetric |
| Population-based training | e.g. uniform policies | other populations | any |
| PSRO | random policies | meta-solution mixture | any |
| AlphaStar | human games | a league | asymmetric |

## From RL to agentic AI

After two weeks on RL the course moves to agentic AI, and the lecture asks three questions about RL and [[Large Language Models (LLM)|LLMs]] to say why RL came first.

**RL versus LLMs: alternative approaches?** Games can serve as a proxy for studying reasoning. AlphaGo has **two systems**:

- **System 1:** the policy network, intuition: immediate feedback and action;
- **System 2:** the long-horizon rollouts of the search, thinking ahead before acting.

What do current LLMs lack? The lecture's answer is that they are **still System 1 only**: they generate the next token, with no search behind it. Two sources back this up. Thore Graepel's opinion piece "Don't be fooled: LLMs don't reason" (2 October 2026) argues that ten years after AlphaGo's match against Go champion Lee Sedol, today's AI still does not use the machinery that made that win possible. Yann LeCun, commenting on it: "True reasoning must involve a search. LLMs don't possess this capability." (The System 1 and System 2 terms come from Daniel Kahneman's psychology of fast intuitive and slow deliberate thinking; background from outside the course material.)

**RL plus LLMs: complementary approaches?** Text generation can be written as an RL problem:

- **State:** the prompt and the tokens generated so far
- **Action:** the next token
- **Reward:** high if the completed response is good
- **Policy:** choose the next token from the learned probability $\pi(a \mid s)$

Improving the policy then means better reasoning. [[DeepSeek-R1]] (Nature, September 2025, "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning") does this with a policy gradient algorithm, [[GRPO]], a close relative of [[PPO]]. The structure resembles AlphaGo's stage 2: one reward at the end of a long sequence of actions, used to make the actions of good responses more likely.

**RL and LLMs: equivalent challenges?** The Cooperative AI Foundation's report "Multi-Agent Risks from Advanced AI" (Hammond et al., February 2025) points out that in the vast majority of real-world strategic interactions, agents' objectives are **neither identical nor completely opposed**. If AI agents are well aligned to their users or deployers, some degree of both cooperation and competition should be expected, as in human society. These **mixed-motive** settings allow mutual gains but also conflict from selfish incentives. Conflict arises in any **[[Social Dilemma|social dilemma]]**, where selfish incentives diverge from the collective good. Social dilemmas are an old problem, but AI could help actors pursue selfish incentives by overcoming technical, legal or social barriers. These are the same cooperation and conflict problems as in Assignment 1. Everything else in this lecture was zero-sum; agents deployed in the world mostly face mixed motives.

## What to be able to answer

- The three challenges of plain MCTS in Go, and which network fixes which.
- AlphaGo's three training stages and losses (cross-entropy on expert moves, REINFORCE with outcome $z$, squared error for the value), and where each network enters the search.
- What AlphaGo Zero drops (expert data, start from random) and what AlphaZero adds (same method, several games, only the engine changes).
- Why self-play needs symmetric roles; the steps of population-based training.
- The PSRO meta-game (actions, payoffs), its five steps and pseudocode, and what the meta-solver and oracle do.
- AlphaStar as a PSRO-like league with heterogeneous populations.
- The System 1 and System 2 reading of AlphaGo; LLM generation as an RL problem (state, action, reward, policy); mixed-motive settings and social dilemmas.

Model answers are in the exam questions.

## Exam questions

> [!exam]- Mock exam Q28 (DMARL 7, 2 points, select one): In Policy Space Response Oracles (PSRO), what is the meta-game? (a) A stochastic game whose states are the observation histories of the agents (b) A normal-form game whose actions are the policies in the current populations (c) A common-reward game whose value function is decomposed into utilities (d) A search tree whose nodes are expanded by Monte Carlo tree search
> **Key points:** (b).
>
> PSRO keeps a population of policies per agent. The meta-game is a simplified normal-form game over those policies: each action is one policy in the current populations, and the payoff of a pair of actions is the average return when those policies play each other, estimated by simulation. A meta-solver (for example a Nash equilibrium) turns it into a distribution over each population, against which a new best-response policy is trained. (a) describes no part of PSRO, (c) describes value decomposition and (d) MCTS.

> [!exam]- Why is plain MCTS weak in a game like Go, and how does AlphaGo fix it? Give AlphaGo's training stages with their losses, and say where each network is used in the search.
> **Key points:** heuristics need manual annotation, random rollouts give a weak signal and assume random opponents; policy network trained by cross-entropy on expert moves, then REINFORCE in self-play with outcome $z$; value network trained with $(z - V_\phi(s))^2$; policy network guides node selection, value network evaluates expanded nodes.
>
> MCTS has to judge positions it reaches. Scoring them with hand-made heuristics requires manual annotation by experts, and scoring them by random rollouts gives a weak signal about which lines are promising, because a random game says little about play between good players; random rollouts also assume a random opponent, while the agent will meet strong ones.
>
> AlphaGo learns a policy network $\pi(a \mid s;\theta)$ and a value network $V_\phi(s)$. The policy network is first trained to reproduce expert moves with the cross-entropy loss $L(\theta) = -\sum_{(s,a) \sim D}\log\pi(a \mid s;\theta)$. It is then improved by self-play with REINFORCE, $\theta \leftarrow \theta + \eta z\sum_t\nabla_\theta\log\pi(a_t \mid s_t;\theta)$, where $z$ is the outcome of the self-play game: moves of won games become more likely, moves of lost games less likely. The value network predicts the chance of winning from a position and is trained on self-play outcomes with $L(\phi) = (z - V_\phi(s))^2$.
>
> In the search, the policy network guides node selection, so the most promising moves are explored first, and the value network evaluates newly expanded nodes, replacing the weak rollout estimate.

> [!exam]- How do AlphaGo Zero and AlphaZero differ from AlphaGo? What does self-play assume, and what is done when the assumption fails?
> **Key points:** AlphaGo Zero learns entirely through self-play with no human games and starts from a random policy; self-play works if one policy can be used from both agents' perspectives; AlphaZero applies the same learning and search method to chess, shogi and Go, with only the game engine changing; it assumes symmetric roles; otherwise population-based training trains against populations of policies that can have different roles, actions and observations.
>
> AlphaGo first trains its policy on expert games. AlphaGo Zero removes that stage: it starts from a random policy and learns only from games against itself. This is possible because the same policy can play from the perspective of either player.
>
> AlphaZero uses the same learning and search method for chess, shogi and Go. Only the game engine changes, which enforces each game's legal moves, transitions and end conditions.
>
> All of these are self-play in a zero-sum game with a homogeneous population: one policy plays both sides, which assumes the two agents have symmetric roles. When roles, actions or observations differ, population-based training is used instead: one population per agent is initialised, the policies are evaluated against the other populations, and they are modified by random changes or by copying the parameters of the best performers.

> [!exam]- Describe PSRO: the meta-game, the meta-solver, the oracle step, and why the method improves on plain self-play.
> **Key points:** populations of policies per agent, started at random; meta-game = normal-form game with policies as actions and average returns as payoffs; meta-solver (e.g. Nash) gives distributions $\delta_i^k$ over populations; oracle: for each agent, sample the others' policies from $\delta_{-i}^k$ and train a new policy with single-agent RL in the real game, then add it; each new policy must beat a mixture of earlier ones, so it does not cycle like best responses to the latest self.
>
> PSRO keeps a population $\Pi_i$ of policies for each agent, initialised for example with random policies. In each generation $k$ it constructs the meta-game $M^k$: a normal-form game in which each agent's actions are the policies in its population, and the payoff of a profile is the average return when those policies play each other in the real game, estimated by simulation.
>
> A meta-solver ranks the policies by solving $M^k$, for example with a Nash equilibrium, and returns a distribution $\delta_i^k$ over each population. Then, for each agent $i$, the oracle step samples policies for the other agents from $\delta_{-i}^k$ in each episode and trains a new policy $\pi_i'$ against them with single-agent RL in the underlying game. The new policy, an approximate best response to the meta-solution, is added: $\Pi_i^{k+1} = \Pi_i^k \cup \{\pi_i'\}$. The meta-game grows by one policy per agent, and the new pairings are simulated in the next generation.
>
> Self-play only responds to the current version of itself, so in games with non-transitive strategies like rock-paper-scissors it can cycle. PSRO's new policies have to do well against the meta-solution's mixture of all earlier policies, so the populations keep the ability to beat old strategies; AlphaStar's league uses a similar idea.

> [!exam]- How can LLM text generation be written as an RL problem, and how does this connect the course's RL material to agentic AI?
> **Key points:** state = prompt and tokens so far, action = next token, reward = high if the completed response is good, policy = learned $\pi(a \mid s)$; improving the policy = better reasoning, e.g. DeepSeek-R1 with GRPO (close to PPO); AlphaGo has System 1 (policy network) and System 2 (long-horizon search), while LLMs are argued to be System 1 only; multi-agent LLM systems face mixed-motive cooperation and conflict problems, such as social dilemmas.
>
> As an RL problem, the state is the prompt plus the tokens generated so far, the action is the next token, the policy is the model's learned next-token distribution $\pi(a \mid s)$, and the reward is high when the completed response is good. Improving the policy means better reasoning. DeepSeek-R1 does this with GRPO, a policy gradient algorithm close to PPO. Like AlphaGo's REINFORCE stage, it learns from one reward at the end of a long sequence of actions.
>
> RL is also the comparison point. AlphaGo combines a System 1, the policy network's immediate intuition, with a System 2, long-horizon rollouts in the search. Current LLMs are still System 1 only, predicting the next token without search: Graepel argues that today's AI does not use the machinery behind AlphaGo's win, and LeCun adds that true reasoning must involve a search.
>
> Finally, the multi-agent problems carry over. Most real interactions between agents, including AI agents well aligned to their users, are mixed-motive: objectives are neither identical nor completely opposed, so there is room for mutual gain and for conflict. Social dilemmas, where selfish incentives diverge from the collective good, are the same cooperation problems studied with MARL.

## Flashcards

> [!card]- What does plain MCTS do after adding a new leaf?
> Plays a random rollout to the end of the game and backs up the outcome.

> [!card]- What is the drawback of hand-made heuristics for estimating state values in MCTS?
> They require manual annotation.

> [!card]- Why is a random rollout a weak signal for MCTS?
> A random game says little about which lines are promising under good play.

> [!card]- What opponent do random rollouts implicitly assume?
> A random opponent, while the agent must beat strong ones.

> [!card]- Which question does AlphaGo's policy network answer?
> Which moves are promising?

> [!card]- Which MCTS step does AlphaGo's policy network guide?
> Node selection.

> [!card]- What does AlphaGo's value network estimate?
> The chance of winning from a given position.

> [!card]- In AlphaGo's search, which nodes does the value network evaluate?
> Newly expanded nodes.

> [!card]- How is AlphaGo's policy network first trained?
> Supervised, to reproduce expert moves, with a cross-entropy loss.

> [!card]- How is AlphaGo's policy network improved after the expert stage?
> Self-play with REINFORCE, weighting each game's moves by the outcome $z$.

> [!card]- In AlphaGo's REINFORCE update, what is $z$?
> The outcome of the self-play game.

> [!card]- Why does every move of a self-play game get the same weight $z$ in AlphaGo's REINFORCE update?
> The only reward is at the end, so every move's return is the outcome.

> [!card]- What loss trains AlphaGo's value network?
> $(z - V_\phi(s))^2$, squared error to the self-play outcome.

> [!card]- How does AlphaGo's policy network make the search cheaper?
> Only the promising few of the many legal moves get explored.

> [!card]- How does AlphaGo's value network make the search cheaper?
> It scores new positions, so no rollout to the end of the game is needed.

> [!card]- What does AlphaGo Zero start from instead of an expert-trained policy?
> A random policy.

> [!card]- What training data does AlphaGo Zero use?
> Only games of self-play, no human game examples.

> [!card]- When is self-play possible, according to AlphaGo Zero's design?
> When the same policy can be used from the perspective of both agents.

> [!card]- What does AlphaZero change relative to AlphaGo Zero?
> The same method is applied to chess, shogi and Go.

> [!card]- In AlphaZero, what is specific to each game?
> The engine enforcing legal moves, transitions and end conditions.

> [!card]- What assumption about the two agents does AlphaZero's self-play make?
> They have symmetric roles.

> [!card]- What does population-based training test a policy against?
> A population of other policies.

> [!card]- Why can population-based training handle asymmetric games?
> Policies in the populations can have different roles, actions and observations.

> [!card]- Name the steps of population-based training.
> Initialise one population per agent; evaluate against other populations; modify policies; repeat.

> [!card]- How are policies modified in population-based training?
> Random changes, or copying the parameters of the best-performing policies.

> [!card]- What is a meta-game in PSRO?
> A simplified normal-form game representing the interaction between complex policies.

> [!card]- What is the name for a normal-form game whose actions are policies and whose payoffs are their average returns?
> A meta-game.

> [!card]- In a PSRO meta-game, what are the actions?
> The policies in the current populations.

> [!card]- In a PSRO meta-game, what are the payoffs?
> The average returns when the policies play each other.

> [!card]- What does PSRO's meta-solver produce?
> A distribution over each agent's population, e.g. a Nash equilibrium.

> [!card]- In PSRO, how is agent $i$'s new policy trained?
> Single-agent RL in the real game against others sampled from the meta-solution.

> [!card]- What is the "oracle" in Policy Space Response Oracles?
> The RL algorithm producing a best response to the meta-solution.

> [!card]- Why does PSRO avoid the cycling of naive self-play?
> Each new policy must beat a mixture of all earlier policies.

> [!card]- What happens to the meta-game after a PSRO generation?
> It gains one policy per agent, whose payoffs must be simulated.

> [!card]- What kind of training is AlphaStar's league an example of?
> Population-based training with heterogeneous populations, PSRO-like.

> [!card]- In AlphaGo, what plays the role of System 1?
> The policy network: intuition, immediate feedback and action.

> [!card]- In AlphaGo, what plays the role of System 2?
> The long-horizon rollouts of the search.

> [!card]- Why are current LLMs called "System 1 only"?
> They generate the next token without any search behind it.

> [!card]- According to Yann LeCun, what must true reasoning involve?
> A search, which LLMs lack.

> [!card]- In LLM generation as RL, what is the state?
> The prompt plus the tokens generated so far.

> [!card]- In LLM generation as RL, what is the action?
> The next token.

> [!card]- In LLM generation as RL, what is the reward?
> High if the completed response is good.

> [!card]- Which RL algorithm does DeepSeek-R1 use to improve reasoning?
> GRPO, a policy gradient method close to PPO.

> [!card]- What is a setting called where agents' objectives are neither identical nor completely opposed?
> A mixed-motive setting.

> [!card]- What is a social dilemma?
> A situation where selfish incentives diverge from the collective good.

> [!card]- True or false: AlphaGo Zero's policy is first trained on expert games.
> **False.** It starts from a random policy and learns only from self-play.

> [!card]- True or false: AlphaGo's value network chooses which nodes MCTS selects.
> **False.** The policy network guides selection; the value network evaluates expanded nodes.

> [!card]- True or false: AlphaZero uses a different learning method for chess and for Go.
> **False.** The learning and search method is the same; only the game engine differs.

> [!card]- True or false: AlphaGo's value network is trained with a cross-entropy loss.
> **False.** It is trained with squared error to the self-play outcome $z$.

> [!card]- True or false: Self-play assumes the two agents have symmetric roles.
> **True.** One policy has to play both sides.

> [!card]- True or false: In a PSRO meta-game, the actions are the moves of the underlying game.
> **False.** The actions are whole policies from the current populations.

> [!card]- True or false: PSRO trains each new policy against the most recent policy only.
> **False.** It trains against policies sampled from the meta-solution over the whole population.

> [!card]- True or false: In PSRO the payoffs of the meta-game are average returns between policies.
> **True.** They are estimated by simulating the policies against each other.

> [!card]- True or false: Policies in population-based training must share the same actions and observations.
> **False.** They can represent agents with different roles, actions and observations.

> [!card]- Multiple selection: which are steps of PSRO? (A) Construct a meta-game from the current populations (B) Use a meta-solver to get distributions over populations (C) Decompose the team value into utilities (D) Train a new policy against policies sampled from the meta-solution
> **A, B and D.** C is value decomposition.

> [!card]- Multiple selection: which challenges of plain MCTS does AlphaGo address? (A) Heuristics need manual annotation (B) Random rollouts give a weak signal (C) Random rollouts assume random opponents (D) The game has no fixed rules
> **A, B and C.** D is false: Go has fixed rules, and MCTS relies on them.

> [!card]- Multiple selection: which are true of AlphaGo Zero and AlphaZero? (A) They learn without human game examples (B) They start from a random policy (C) They need asymmetric roles (D) AlphaZero applies one method to several games
> **A, B and D.** C: self-play assumes symmetric roles.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Previous lecture:** [[AGAI-L05 - Deep MARL, Part II]] (training modes, value decomposition, agent modelling, parameter sharing, MCTS steps, self-play requirements)
- **Earlier:** [[AGAI-L04 - Deep RL and Intro to Deep MARL]] (REINFORCE, PPO); [[AGAI-L02 - Game Theory and Population Dynamics]] (normal-form games, Nash equilibrium, replicator dynamics, social dilemmas)
- **Lab:** [[AGAI-Lab02 - Learning Agents]]
- **Concepts introduced here:** [[Self-Play]], [[Population-Based Training]]
- **Related concepts:** [[Monte Carlo Tree Search (MCTS)]], [[AlphaGo Zero]], [[Policy Space Response Oracles (PSRO)]], [[Normal-Form Game]], [[Nash Equilibrium]], [[Best Response]], [[REINFORCE]], [[Policy Gradient Methods]], [[Value Function]], [[Deep Reinforcement Learning]], [[GRPO]], [[DeepSeek-R1]], [[Reinforcement Learning from Human Feedback]], [[Large Language Models (LLM)]], [[LLM Agent]], [[Social Dilemma]], [[Replicator Dynamics]]
- **Source:** Agentic Artificial Intelligence, Lecture 6 part I, "From Deep MARL to Reasoning", MSc AI, UvA, 12 November 2026. Part II is Giovanni Sileno's guest lecture on definitions of agents.
- **Reading:** Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches* (MIT Press, 2024), Chapter 9 (exam material for the midterm): self-play in zero-sum games, MCTS, AlphaZero, population-based training, PSRO (Algorithm 26), AlphaStar
- **Papers named:** Silver et al., "Mastering the game of Go with deep neural networks and tree search" (Nature 2016); Silver et al., "Mastering the game of Go without human knowledge" (Nature 2017); Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (Science 2018); Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (NeurIPS 2017); the AlphaStar team, "AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning" (DeepMind, 30 October 2019); Graepel, "Don't be fooled: LLMs don't reason" (opinion piece, 2 October 2026); Guo et al., "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning" (Nature 645, 2025); Hammond et al., "Multi-Agent Risks from Advanced AI" (Cooperative AI Foundation Technical Report 1, February 2025)
- **Mock exam:** AGAI Mock Midterm 2026, Deep MARL question 28
