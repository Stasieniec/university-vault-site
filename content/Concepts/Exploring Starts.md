---
type: concept
aliases: [exploring starts, ES, Monte Carlo ES]
course: [RL]
tags: [tabular-methods]
status: complete
---

# Exploring Starts

> [!definition] Exploring Starts
> The assumption that every episode begins with a **randomly chosen state-action pair** $(S_0, A_0)$, with every pair having non-zero probability. This guarantees that all state-action pairs will be visited infinitely often.

- Used in Monte Carlo ES (Exploring Starts) algorithm for MC control
- Ensures sufficient exploration for convergence to optimal policy
- **Unrealistic in practice** — can't always control the starting state (e.g., in real-world environments)
- Alternatives: [[Epsilon-Greedy Policy]] (on-policy), [[Importance Sampling]] (off-policy)

## Appears In

- [[RL-L03 - Monte Carlo Methods]], [[RL-Book Ch5 - Monte Carlo Methods]]
