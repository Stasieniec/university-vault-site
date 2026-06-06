---
type: concept
aliases: [DT]
course: [RL]
tags: [model-based-rl, planning, exam-topic]
status: complete
---

# Decision-Time Planning

## Definition

> [!definition] Decision-Time Planning
> **Decision-time planning** is the use of a model (learned or given) to select an action *for the current state*, at the moment a decision must be made. Rather than computing a global policy ahead of time, the agent runs a focused planning computation rooted at the current state $S_t$, commits to a single action $A_t$, executes it, and then **discards** most of the work and re-plans from the next state.
>
> It contrasts with [[Background Planning]], which uses simulated experience to improve a *global* value function or policy for *all* states, ahead of acting. Decision-time planning instead concentrates all compute on the small, immediately-relevant part of the state space.

## Intuition

Think of playing chess. You do not pre-compute the best move for every conceivable board position. Instead, from your *current* position you read ahead — "if I play here, the opponent plays there, then..." — using a model of the rules to imagine futures. You spend your thinking budget only on lines reachable from where you are now, pick the best move, play it, and on the next turn you start the look-ahead afresh from the new position.

This is exactly the setting where the values produced by planning are useful *only as an input to selecting the current action*. Once $A_t$ is chosen and executed, the action-value estimates $\hat{q}(S_t, \cdot)$ are typically thrown away (though smarter methods like MCTS can re-use the relevant subtree). Because the model is consulted on demand and the search is shallow/narrow relative to the whole state space, decision-time planning is feasible even in enormous domains (e.g. Go, with $\approx 10^{170}$ states) where computing a global policy is hopeless.

Decision-time planning is most natural when responses are not needed too quickly and the agent can afford to deliberate per step. Background planning is preferred when low-latency reactions are required, since the policy is already available.

## Mathematical Formulation

The agent builds a forward model and, at each step, estimates action values from the current state by simulation, then acts greedily on those estimates:

$$A_t = \arg\max_{a} \hat{q}(S_t, a), \qquad \hat{q}(S_t, a) = \frac{1}{K}\sum_{i=1}^{K} G^{(i)}\big(S_t, a\big)$$

where:
- $S_t$ — the current (root) state at decision time
- $a$ — a candidate action evaluated only at $S_t$
- $\hat{q}(S_t, a)$ — the action-value estimate obtained *by planning from $S_t$*, e.g. via simulated rollouts
- $G^{(i)}(S_t, a)$ — the return (or terminal outcome $v_i$) of the $i$-th simulated trajectory that starts by taking $a$ in $S_t$ and then follows a rollout/tree policy under the model
- $K$ — number of simulations (the planning budget); estimates improve as $K \to \infty$
- $A_t$ — the single action actually executed in the real environment

After executing $A_t$ and observing $S_{t+1}$, the procedure repeats: a fresh estimate $\hat{q}(S_{t+1}, \cdot)$ is computed by planning from $S_{t+1}$.

For a [[Rollout Algorithm]], $\hat{q}(S_t, a)$ is a Monte Carlo average of rollout returns under a fixed rollout policy $b$ — by the policy improvement theorem, acting greedily on $\hat{q}_b$ is at least as good as $b$ itself. [[Monte Carlo Tree Search (MCTS)]] refines this by replacing the flat average with values backed up through an incrementally-grown search tree, using a [[Upper Confidence Bound|UCB]]-style tree policy to focus simulations.

## Key Properties / Variants

- **Focus on the present**: only the part of the state space reachable from $S_t$ is considered; compute is not "wasted" on irrelevant states.
- **Plan-then-discard**: values are computed to choose one action and usually not retained. MCTS is an exception — the subtree under the chosen child can be re-used as the next root.
- **Works with learned or given models**: AlphaGo uses the *given* rules of Go; in model-based RL the same scheme runs on a *learned* dynamics model $\hat{p}(s'\mid s,a)$.
- **Policy improvement, not optimality**: with enough simulations the selected action is at least as good as the base rollout policy, but generally *approximate* — not the true optimal policy.
- **Latency trade-off**: deliberation happens per decision, so it suits domains tolerant of per-step thinking time; [[Background Planning]] suits fast-reaction settings.
- **Main instances**: [[Rollout Algorithm]] (flat Monte Carlo from the current state) and [[Monte Carlo Tree Search (MCTS)]] (incremental tree, four phases); AlphaGo Zero augments MCTS with neural policy/value priors.

```pseudo
Algorithm: Decision-Time Planning (generic loop)
─────────────────────────────────────────────────
Given: model (learned or known) of dynamics/rewards
Initialize s_0
For each timestep t (real environment):
    # plan from the CURRENT state only
    For each planning step k = 1..K (within budget):
        Simulate a trajectory from s_t using the model
          (rollout policy, or tree policy + rollout for MCTS)
        Update value estimates q̂(s_t, ·) from the outcome
    a* ← argmax_a q̂(s_t, a)        # (or argmax visit count, for MCTS)
    Execute a* in the REAL environment, observe s_{t+1}
    # estimates for s_t are (largely) discarded; re-plan from s_{t+1}
```

## Connections

- Contrasted with: [[Background Planning]] (global policy ahead of acting, e.g. [[Dyna]]-Q)
- Instances: [[Rollout Algorithm]], [[Monte Carlo Tree Search (MCTS)]]
- Applied in: [[AlphaGo Zero]] (MCTS with neural priors over a given model)
- Tree-policy exploration: [[Upper Confidence Bound]]
- Umbrella topic: [[Model-Based Reinforcement Learning]]
- Underlying value backups use the same targets as [[Q-Learning]] / Monte Carlo returns

## Appears In

- [[RL-L12 - Model-Based RL]]
- [[RL-Book Ch8 - Planning and Learning]]
