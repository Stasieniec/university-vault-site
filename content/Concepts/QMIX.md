---
type: concept
aliases: [QMIX, monotonic value decomposition, Monotonic value decomposition, mixing network]
course: [AGAI]
tags: [multi-agent, marl, deep-rl, exam-topic]
status: complete
---

# QMIX

## Definition

> [!definition] QMIX
> A CTDE value-based method for common-reward games that combines individual utilities into the joint action-value with a learned **monotonic mixing function**:
> $$Q(h, z, a;\theta) = f_{\text{mix}}\big(Q(h_1, a_1;\theta_1), \dots, Q(h_n, a_n;\theta_n);\,\theta_{\text{mix}}\big), \qquad \frac{\partial Q(h, z, a;\theta)}{\partial Q(h_i, a_i;\theta_i)} > 0 \;\; \forall i, \forall a$$
> The mixing parameters come from a **hypernetwork** fed with centralised information: $\theta_{\text{mix}} = f_{\text{hyper}}(z;\theta_{\text{hyper}})$.

## Intuition

> [!intuition] Linearity was never the point
> [[Value Decomposition Networks (VDN)|VDN]] adds utilities so that every agent's greedy action adds up to the best joint action. But that only needs one thing: making any agent's utility bigger should never make the team value smaller. QMIX keeps that guarantee and drops the sum, so the team value can be any increasing function of the utilities, and the function can change with the state.

## Mathematical Formulation

Target and loss, from a shared buffer of $(h^k, z^k, a^k, r^k, h^{k+1}, z^{k+1})$:

$$y^k = r^k + \gamma\, f_{\text{mix}}\Big(\max_{a_1'}Q(h_1^{k+1}, a_1';\bar\theta_1), \dots, \max_{a_n'}Q(h_n^{k+1}, a_n';\bar\theta_n);\; f_{\text{hyper}}(z^{k+1};\theta_{\text{hyper}})\Big)$$

$$\mathcal{L}(\theta) = \frac{1}{B}\sum_k\Big(y^k - f_{\text{mix}}\big(Q(h_1^k, a_1^k;\theta_1), \dots;\; f_{\text{hyper}}(z^k;\theta_{\text{hyper}})\big)\Big)^2$$

- $f_{\text{mix}}$: the mixing network; $f_{\text{hyper}}$: a network that outputs $f_{\text{mix}}$'s weights from $z$ (typically the global state)
- taking each agent's max inside $f_{\text{mix}}$ is valid only because $f_{\text{mix}}$ is monotonic
- $\theta$ covers the utility networks and the hypernetwork, trained together

Monotonicity is enforced by making the mixing weights non-negative (in the original design the hypernetwork's weight outputs pass through an absolute value). Strictly, non-negative weights give $\partial Q / \partial Q_i \geq 0$, the form the original QMIX paper uses; the AGAI course writes $> 0$. Either way raising a utility never lowers the joint value, which is all IGM needs.

```
utilities:  h_1 → Q(h_1,·;θ_1)  ...  h_n → Q(h_n,·;θ_n)
                      ↘                ↙
z → f_hyper(·;θ_hyper) ──θ_mix──► f_mix(·;θ_mix) → Q(h, z, a; θ)
```

## Key Properties / Variants

- Satisfies the [[Individual-Global-Max Property|IGM property]]: decentralised greedy execution on each agent's utility.
- **Generalises VDN:** with the sum as mixing function it is VDN, so it can represent everything VDN can and more.
- Utilities need not look like returns: in the linear game (A,A) = 1, (A,B) = (B,A) = 5, (B,B) = 9, QMIX's utilities are about (0.19, 0.96) and (−0.21, 0.68) for (A, B), yet the mix reproduces all four values and both agents greedily pick B.
- Uses centralised information $z$ only through the hypernetwork, during training.
- **Limitation** (MARL book): a monotonic mix cannot represent games where one agent's best action flips with the other's choice (for instance a matching game with two optimal joint actions). Methods such as QTRAN go beyond monotonic decomposition.

## Connections

- Special case: [[Value Decomposition Networks (VDN)]]
- Criterion: [[Individual-Global-Max Property]]
- Base algorithm: [[Deep Q-Network (DQN)]]
- Training mode: [[Centralised Training with Decentralised Execution]]
- Problem addressed: [[Multi-Agent Credit Assignment]]

## Appears In

- [[AGAI-L05 - Deep MARL, Part II]]
