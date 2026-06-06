---
type: concept
aliases: [GRU]
course: [RecSys]
tags: [sequential-rec, exam-topic]
status: complete
---

# Gated Recurrent Unit (GRU)

## Definition

> [!definition] Gated Recurrent Unit (GRU)
> A **GRU** is a gated [[Recurrent Neural Network (RNN)|RNN]] cell that maintains a hidden state $h_t$ summarizing a sequence and updates it at each step through two learned **gates** — a **reset gate** $r_t$ and an **update gate** $z_t$. The gates let the cell decide how much past information to keep versus how much new input to absorb, which mitigates the **vanishing-gradient** problem of vanilla RNNs and lets it capture longer-range sequential dependencies. In RecSys it is the core cell of **GRU4Rec** [Hidasi et al., 2015], one of the first deep models for [[Sequential Recommendation]] (originally for [[Session-based Recommendation|session-based]] settings), where it consumes a sequence of item embeddings and produces a representation used to score the next item.

## Intuition

> [!intuition] Gates as Read/Write Controls on Memory
> A plain RNN overwrites its entire hidden state every step, so old signal is multiplied away and gradients vanish over long sequences. A GRU instead **interpolates** between the old state and a freshly proposed state:
> - The **update gate** $z_t$ is a soft switch on a per-dimension basis: $z_t \approx 1$ means "keep the old memory" (carry information far forward, an identity-like path that preserves gradient), $z_t \approx 0$ means "overwrite with the new candidate".
> - The **reset gate** $r_t$ controls how much of the *past* state is allowed into the *candidate* computation: $r_t \approx 0$ lets the cell forget the past and treat the current input as a fresh start (useful at item-sequence boundaries).
>
> For recommendation, this means the model can carry a user's earlier interests forward while still reacting to the most recent click — a soft blend of long- and short-term intent that the order-agnostic [[Matrix Factorization|MF]] cannot represent.

## Mathematical Formulation

> [!formula] GRU Cell Recurrence
> Given input $x_t$ at step $t$ and previous hidden state $h_{t-1}$:
> $$
> \begin{aligned}
> z_t &= \sigma\!\left(W_z x_t + U_z h_{t-1} + b_z\right) && \text{(update gate)}\\
> r_t &= \sigma\!\left(W_r x_t + U_r h_{t-1} + b_r\right) && \text{(reset gate)}\\
> \tilde{h}_t &= \tanh\!\left(W_h x_t + U_h\,(r_t \odot h_{t-1}) + b_h\right) && \text{(candidate state)}\\
> h_t &= (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t && \text{(new state)}
> \end{aligned}
> $$
>
> where:
> - $x_t$ — input at step $t$ (in GRU4Rec, the embedding of the item interacted with at step $t$; the raw input is a 1-of-N / one-hot item code)
> - $h_t \in \mathbb{R}^d$ — hidden state (the running sequence summary); $h_0 = \mathbf{0}$
> - $z_t \in (0,1)^d$ — **update gate**: how much of the candidate to mix into the new state
> - $r_t \in (0,1)^d$ — **reset gate**: how much past state feeds the candidate
> - $\tilde{h}_t$ — **candidate** hidden state (proposed update)
> - $W_\ast, U_\ast, b_\ast$ — learned input weights, recurrent weights, and biases (one set per gate / candidate)
> - $\sigma(\cdot)$ — logistic sigmoid (squashes gates to $(0,1)$); $\tanh$ — squashes the candidate to $(-1,1)$
> - $\odot$ — element-wise (Hadamard) product

> [!formula] GRU4Rec: From Hidden State to Item Scores
> The final GRU layer output is passed through feedforward layer(s) to produce a score per candidate item; with item embedding/output parameters the score of item $i$ given session state $h_t$ is
> $$\hat{r}_{s,i} = f(h_t)^\top e_i$$
> where $e_i$ is the (output) embedding of item $i$ and $f(\cdot)$ are the top feedforward layers. Training uses the pairwise [[Bayesian Personalized Ranking (BPR)|BPR]] loss over a positive next item $i$ and $N_S$ sampled negatives $j$:
> $$\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S}\sum_{j=1}^{N_S}\log\sigma\!\left(\hat{r}_{s,i}-\hat{r}_{s,j}\right)$$
> which pushes the true next item's score above those of the negatives. (BPR is not GRU-specific; GRU4Rec can also use TOP1-max, binary cross-entropy, or full cross-entropy.)

## Key Properties / Variants

- **Two gates, no separate cell state.** Unlike the [[LSTM]] (input/forget/output gates + an explicit cell state $c_t$), a GRU merges memory into a single $h_t$ and uses only $z_t$ and $r_t$ — fewer parameters, often comparable performance, faster to train.
- **Update gate enables long-range gradient flow.** When $z_t \to 0$, $h_t \approx h_{t-1}$: an additive, identity-like carry that preserves gradients across many steps, addressing the vanishing gradient of vanilla RNNs.
- **Sequential (left-to-right) and causal.** A GRU processes one step at a time and only sees the past, so GRU4Rec is a unidirectional next-item predictor (contrast [[BERT4Rec]]'s bidirectional Cloze objective).
- **Strengths / limitations in RecSys (lecture):** GRU4Rec captures short temporal patterns within a session and outperforms [[FPMC]] when data is plentiful, allowing longer sequences and more complex modeling. But it is **slow to train** and **struggles with very long sequences**, and on sparse data the simpler FPMC can win. It is also slower / less parallelizable than the self-attention [[SASRec]].
- **Stacking.** GRU4Rec can stack multiple GRU layers; deeper recurrence with feedforward layers on top before the item-scoring output.

```pseudo
Algorithm: GRU4Rec forward pass (session next-item scoring)
───────────────────────────────────────────────────────────
Input: session item sequence (i_1, ..., i_T), item embeddings E
Initialize h_0 ← 0
for t = 1 .. T:
    x_t ← E[i_t]                      # embed current item (one-hot → dense)
    z_t ← σ(W_z x_t + U_z h_{t-1} + b_z)
    r_t ← σ(W_r x_t + U_r h_{t-1} + b_r)
    h~_t ← tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)
    h_t ← (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t
# score all candidate items from the last state
scores ← f(h_T) @ E_out^T            # f = top feed-forward layer(s)
return scores                         # argmax / top-K = recommended next items
```

## Connections

- Type of: [[Recurrent Neural Network (RNN)]] (gated cell)
- Contrasted with: [[LSTM]] (more gates + explicit cell state)
- Core cell of: [[GRU4Rec]] for [[Sequential Recommendation]] / [[Session-based Recommendation]]
- Trained with: [[Bayesian Personalized Ranking (BPR)]] loss (pairwise; alternatives include BCE and full CE)
- Improves on: [[FPMC]] (only first-order transitions) when data is abundant
- Superseded by (on accuracy/efficiency): [[SASRec]] (self-attention), [[BERT4Rec]] (bidirectional)
- Motivated by failure of: [[Matrix Factorization]] to model interaction order

## Appears In

- [[RS-L03a - Sequential Recommendation Models]]
