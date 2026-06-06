---
type: concept
aliases: [Long Short-Term Memory]
course: [RL, IR]
tags: [deep-rl, neural-ir, partial-observability, exam-topic]
status: complete
---

# LSTM

## Definition

> [!definition] LSTM
> A **Long Short-Term Memory (LSTM)** network is a recurrent neural network (RNN) architecture designed to process **sequences** while learning **long-range dependencies**. Its defining feature is a **cell state** $c_t$ — a memory vector carried across time steps with mostly additive (rather than repeatedly multiplicative) updates — regulated by three multiplicative **gates** (forget, input, output). The gates let the network *learn what to remember, what to write, and what to read* at each step, which prevents the vanishing/exploding gradients that cripple vanilla RNNs over long horizons.

## Intuition

A plain RNN updates its hidden state by $h_t = \tanh(W_h h_{t-1} + W_x x_t)$. Because $h_{t-1}$ is passed through a saturating nonlinearity and a weight matrix at **every** step, the gradient of an early input with respect to a late loss is a long product of Jacobians — it shrinks toward zero (vanishing) or blows up (exploding) exponentially in the sequence length. The network therefore cannot connect events separated by many steps.

The LSTM fixes this with a **constant error carousel**: the cell state $c_t$ has a near-identity recurrence, $c_t \approx f_t \odot c_{t-1} + \dots$, so gradient can flow back many steps almost undamped when the forget gate $f_t \approx 1$. The gates are themselves learned sigmoid functions of the input and previous hidden state, so the network decides *dynamically* how long to hold each piece of information.

In RL this is exactly what is needed under [[Partial Observability]]: the LSTM hidden state $h_t$ acts as a learned, recursively-updated **internal state** $s_t = f(H_t)$ summarising the whole history of observations, rather than only the last $k$ frames (frame stacking). This is the basis of [[Deep Recurrent Q-Learning]] (DRQN). In IR, the same sequence-modelling ability underlies early neural rankers and query/document encoders before [[Transformers]] became dominant.

## Mathematical Formulation

At each time step $t$, given input $x_t$, previous hidden state $h_{t-1}$, and previous cell state $c_{t-1}$, the LSTM computes:

$$
\begin{aligned}
f_t &= \sigma\!\big(W_f x_t + U_f h_{t-1} + b_f\big) && \text{(forget gate)}\\
i_t &= \sigma\!\big(W_i x_t + U_i h_{t-1} + b_i\big) && \text{(input gate)}\\
o_t &= \sigma\!\big(W_o x_t + U_o h_{t-1} + b_o\big) && \text{(output gate)}\\
\tilde{c}_t &= \tanh\!\big(W_c x_t + U_c h_{t-1} + b_c\big) && \text{(candidate cell)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{(cell state update)}\\
h_t &= o_t \odot \tanh(c_t) && \text{(hidden state / output)}
\end{aligned}
$$

where:
- $x_t$ — input at step $t$ (e.g. an embedded token, or CNN features of an observation)
- $h_t \in \mathbb{R}^d$ — hidden state, also the layer's output at step $t$
- $c_t \in \mathbb{R}^d$ — cell state (the long-term memory carried across steps)
- $f_t, i_t, o_t \in (0,1)^d$ — forget / input / output gates (element-wise sigmoid $\sigma$)
- $\tilde{c}_t$ — candidate values proposed for writing into the cell ($\tanh$, range $(-1,1)$)
- $W_\ast, U_\ast$ — input and recurrent weight matrices; $b_\ast$ — biases (each gate has its own set)
- $\odot$ — element-wise (Hadamard) product

> [!formula] Why the gradient survives
> The cell recurrence $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ gives $\partial c_t / \partial c_{t-1} = \operatorname{diag}(f_t)$. Backpropagating through $T$ steps multiplies these diagonal terms: $\prod_{k} f_k$. When the forget gate stays near $1$, this product stays near $1$, so error flows back across long spans without vanishing — unlike the dense Jacobian product $\prod_k W_h^\top \operatorname{diag}(\tanh')$ of a vanilla RNN.

## Key Properties / Variants

- **Gated additive memory**: the additive cell update (vs. the multiplicative hidden-state update of vanilla RNNs) is what tames vanishing gradients and enables long-range memory.
- **Parameter cost**: roughly $4(d^2 + d\,n + d)$ parameters per layer for hidden size $d$ and input size $n$ — four affine maps (three gates + candidate).
- **Trained with Backpropagation Through Time (BPTT)**: the network is *unrolled* over the sequence and gradients are summed across steps; long sequences are usually handled with **truncated BPTT**.
- **GRU (Gated Recurrent Unit)**: a lighter variant merging cell and hidden state and using only two gates (reset, update); fewer parameters, often comparable performance.
- **Bidirectional / stacked LSTMs**: read the sequence both directions and/or stack layers; common in IR encoders where the full sequence is available offline.
- **Largely superseded by [[Transformers]]** (self-attention) for both NLP and IR — attention gives direct, parallelisable access to all positions — but LSTMs remain relevant where streaming/online recurrence or low memory is needed, and in RL recurrent agents.

```pseudo
Algorithm: LSTM forward pass over a sequence
─────────────────────────────────────────────
Input: sequence x_1, ..., x_T ; params {W_*, U_*, b_*}
Initialize h_0 = 0,  c_0 = 0

for t = 1 .. T:
    f_t  = σ(W_f x_t + U_f h_{t-1} + b_f)      # forget: keep how much of c_{t-1}
    i_t  = σ(W_i x_t + U_i h_{t-1} + b_i)      # input:  how much to write
    o_t  = σ(W_o x_t + U_o h_{t-1} + b_o)      # output: how much of cell to expose
    c~_t = tanh(W_c x_t + U_c h_{t-1} + b_c)   # candidate content
    c_t  = f_t ⊙ c_{t-1} + i_t ⊙ c~_t          # update memory (additive)
    h_t  = o_t ⊙ tanh(c_t)                     # emit hidden state / output
return (h_1..h_T, c_1..c_T)
```

> [!warning] RNN training is finicky
> Even with gates, LSTMs can hit local optima, need gradient clipping for the exploding-gradient direction, and are slow to train because BPTT is inherently sequential (no within-sequence parallelism). In DRQN this shows up as sensitivity to the unrolling strategy (bootstrapped random-start vs. sequential whole-episode updates).

## Connections

- Provides the recurrent internal-state function in [[Deep Recurrent Q-Learning]] (DRQN), the deep approach to [[Partial Observability]] / [[POMDP]]
- Alternative to frame stacking as the state-update function $f(H_t)$ (see [[Belief State]], [[Predictive State Representation]])
- Builds on [[Deep Q-Network (DQN)]] (DRQN replaces DQN's first fully-connected layer with an LSTM) and [[Convolutional Neural Networks]] (for per-frame visual features)
- A type of [[Neural Networks]] / sequence model; largely superseded by [[Transformers]] and the [[Attention architecture]]
- Used as a sequence encoder in [[Neural Networks|neural]] [[Information Retrieval]] before BERT-era [[Transformer Model]] rankers

## Appears In

- [[RL-L13 - Partial Observability]]
