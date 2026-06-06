---
type: concept
aliases: [RNN]
course: [RecSys]
tags: [sequential-rec, exam-topic]
status: complete
---

# Recurrent Neural Network (RNN)

## Definition

> [!definition] Recurrent Neural Network
> A **Recurrent Neural Network (RNN)** is a neural network with **recurrent connections** that processes a sequence $\langle x_1, x_2, \dots, x_T \rangle$ **one step at a time**, maintaining a **hidden state** $h_t$ that is carried forward and updated at every step. The hidden state acts as a learned, compressed summary of *all* prior inputs, so the output at step $t$ depends on the entire history $x_1, \dots, x_t$ — not just the current input.
>
> In sequential recommendation, an RNN consumes the user's chronologically ordered interaction history and predicts the **next item**. This is the basis of [[GRU4Rec]] [Hidasi et al., 2015], among the first deep-learning models for [[Sequential Recommendation]].

## Intuition

> [!intuition] A loop that remembers
> A feedforward network maps one fixed-size input to one output. An RNN adds a **feedback loop**: it reuses the same weights at every timestep and threads a hidden state through them, so the network "remembers" what it has seen. Conceptually it is the neural analogue of a [[Markov Chain]] — but instead of a fixed first-order transition table, the hidden state can in principle encode dependencies of *arbitrary* length.
>
> This is exactly what plain [[Matrix Factorization]] lacks: MF treats a user's interactions as an **unordered set** and ignores order, recency, and transitions. An RNN reads the history *in order*, so it captures sequential signals (e.g. you buy a phone case *after* the phone, not before).

## Mathematical Formulation

The vanilla (Elman) RNN recurrence, applied at each timestep $t$:

$$h_t = \sigma\!\left( W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h \right), \qquad o_t = W_{ho}\, h_t + b_o$$

where:
- $x_t$ — input at step $t$ (in [[GRU4Rec]], the dense [[Embedding Layer|embedding]] of the current item, from its one-hot / 1-of-N code)
- $h_t$ — hidden state at step $t$; the running summary of the sequence so far, with $h_0 = \mathbf{0}$
- $h_{t-1}$ — previous hidden state (the recurrent input — this is the "recurrent connection")
- $W_{hh}, W_{xh}, W_{ho}$ — **shared** weight matrices, reused at every timestep (parameter tying across time)
- $b_h, b_o$ — bias vectors
- $\sigma(\cdot)$ — nonlinearity (e.g. $\tanh$)
- $o_t$ — output at step $t$; in recommendation this is mapped to **scores over candidate items** for next-item prediction

**Training** uses **Backpropagation Through Time (BPTT)**: the recurrence is unrolled across the $T$ steps into a deep feedforward graph and gradients are propagated back through every step. For a sequential recommender like [[GRU4Rec]], the resulting scores are fed into a pairwise [[Bayesian Personalized Ranking (BPR)|BPR]] loss over a true next item $i$ and sampled negatives $j$:

$$\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S} \sum_{j=1}^{N_S} \log \sigma\!\left( \hat{r}_{s,i} - \hat{r}_{s,j} \right)$$

where $N_S$ is the number of negative samples, $\hat{r}_{s,i}$ the score of the true next item, and $\hat{r}_{s,j}$ the score of negative $j$.

## Key Properties / Variants

- **Sequential, not parallel.** Step $t$ needs $h_{t-1}$, so computation is inherently serial — this makes RNNs **slow to train** versus the parallelizable [[Self-Attention]] of [[SASRec]].
- **Vanishing / exploding gradients.** BPTT through many steps repeatedly multiplies by $W_{hh}$; gradients shrink or blow up, so vanilla RNNs **struggle with long sequences**. This is the core motivation for gated variants.
- **[[Gated Recurrent Unit (GRU)]] (used in [[GRU4Rec]]).** A reset gate $r_t$ and update gate $z_t$ control how much past state is kept vs. overwritten, mitigating vanishing gradients with fewer parameters than LSTM:
  - $z_t = \sigma(W_z x_t + U_z h_{t-1})$ — update gate (how much new info to admit)
  - $r_t = \sigma(W_r x_t + U_r h_{t-1})$ — reset gate (how much past state to forget)
  - $\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}))$ — candidate state
  - $h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t$ — gated interpolation of old and new state
- **[[LSTM|Long Short-Term Memory (LSTM)]].** The other major gated variant; adds a separate cell state with input/forget/output gates.
- **Per-step generation in recsys.** [[GRU4Rec]] stacks one or more GRU layers between an embedding layer and feedforward output layers; GRU layers capture sequential dependencies from previous interactions and emit a score per candidate item.
- **When to use.** RNN/GRU recommenders allow **more complex modeling and longer sequences than [[FPMC]]** and beat it when more data is available; but on **sparse** data or tight compute budgets, the simpler FPMC can win. They are **left-to-right unidirectional** (predict next from past only), like [[SASRec]] and unlike the bidirectional [[BERT4Rec]].

Forward pass over a sequence (pseudo-code):

```pseudo
Algorithm: RNN forward pass for next-item prediction
─────────────────────────────────────────────────────
Input: item sequence ⟨x_1, ..., x_T⟩  (each x_t = item embedding)
Initialize h_0 ← 0
for t = 1 .. T:
    h_t ← σ(W_hh · h_{t-1} + W_xh · x_t + b_h)   # update hidden state
    o_t ← W_ho · h_t + b_o                        # scores over all items
return scores o_T for the next item (rank items by o_T)
# Train with BPTT: unroll over t and backprop the (e.g. BPR) loss
```

## Connections

- Type of: [[Neural Networks]] / [[Neural Network Function Approximation]] with feedback connections
- Gated variants: [[Gated Recurrent Unit (GRU)]], [[LSTM|Long Short-Term Memory]]
- Used by: [[GRU4Rec]] (RNN-based [[Sequential Recommendation]])
- Compared with: [[SASRec]] and [[BERT4Rec]] (replace recurrence with [[Self-Attention]] / [[Transformers]]); [[FPMC]] and [[Markov Chain]] (fixed-order alternatives)
- Trained with: [[Bayesian Personalized Ranking (BPR)]] loss; alternatives include [[Binary Independence Model|BCE]]-style and listwise losses
- Also appears in RL as: [[DRQN|Deep Recurrent Q-Network]] for [[Partial Observability]]

## Appears In

- [[RS-L03a - Sequential Recommendation Models]]
