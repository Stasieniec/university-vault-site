---
type: lecture
course: RecSys
lecture: 3a
date: 2026-06-04
topics:
  - "[[Sequential Recommendation]]"
  - "[[Matrix Factorization]]"
  - "[[Collaborative Filtering]]"
  - "[[Markov Chain]]"
  - "[[Factorized Personalized Markov Chains (FPMC)]]"
  - "[[GRU4Rec]]"
  - "[[SASRec]]"
  - "[[BERT4Rec]]"
  - "[[Self-Attention]]"
  - "[[Gated Recurrent Unit (GRU)]]"
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Negative Sampling]]"
  - "[[Session-based Recommendation]]"
  - "[[Semantic IDs]]"
status: complete
---

# RS-L03a - Sequential Recommendation Models

> [!abstract] Overview
> Standard [[Matrix Factorization]] / [[Collaborative Filtering]] treats a user's history as an **unordered set** — it throws away temporal order, recency, repetition and item-to-item transitions. **[[Sequential Recommendation]]** fixes this: given a chronologically ordered interaction history $\langle i_1, \dots, i_n \rangle$, predict the **next item** $i_{n+1}$. This lecture traces the field from counting-based **[[Markov Chain|Markov Chains]]** → **[[Factorized Personalized Markov Chains (FPMC)|FPMC]]** (personalized MF + first-order transitions) → the deep-learning era: **[[GRU4Rec]]** ([[Recurrent Neural Network (RNN)|RNN]]/[[Gated Recurrent Unit (GRU)|GRU]]), **[[SASRec]]** (causal [[Self-Attention]]), and **[[BERT4Rec]]** (bidirectional self-attention via a Cloze task). A recurring theme: the **training loss and number of negatives** ([[Bayesian Personalized Ranking (BPR)|BPR]] / BCE / CE) can matter *more* than the architecture. We close with open challenges (million-item catalogs, cold start, very long histories).
>
> Lecturer: Xiaoyu Zhang (with Yuyue Zhao). Slides based on material by David Vos (RecSys 2025) and the ECIR 2024 Transformers-for-RecSys tutorial [Petrov & Macdonald, 2024]. **No textbook — slides are the only source.**

---

## 1. A Brief Recap: Recommendation as Matrix Completion

The classic recommendation task is framed as **completing a sparse user-item interaction matrix** (the [[User-Item Interaction Matrix]]). Two views of [[Collaborative Filtering]]:

- **[[User-based Collaborative Filtering|User-based CF]]:** "How do similar *users* to the target user like item $i$?" → find neighbours among users.
- **[[Item-based Collaborative Filtering|Item-based CF]]:** "How does the target user like *items* similar to $i$?" → find neighbours among items.
- **Item-based CF is usually preferred** because it is **more stable**: item-item relationships drift far less over time than user profiles.
- We use **[[Matrix Factorization]] (MF)** to model the task: factorize the CF matrix into smaller latent matrices, enabling **generalization and efficient inference**.

### 1.1 User-based CF (slide 8)

"How do similar users to user $u$ like item $i$?" — prediction uses *other users'* ratings on the **same** item $i$ (highlight the **column** $i$). Cells can be buys/clicks/views/rates/reviews. The target cell is `?`.

```
            it1  it2  it3  it4  it5  it6  it7
 User 1      .    .    1    3    .    2    .
 User 2      1    2    5    .    4    .    1
 User 3      4    .    .    3    5    .    4
 User u →    .    2  [ ? ]  .    5    4    .     ← target user u
 User 5      .    3    4    .    5    .    3
                      ↑
                   item i (target column, look at other users here)
```

### 1.2 Item-based CF (slide 9)

Same matrix, but now we highlight the **row** $u$ (the target user's interactions across items) and predict the `?` at column $i$. "How does user $u$ like items similar to item $i$?" — prediction uses the **same** user's ratings on **similar items** (the highlighted row).

### 1.3 Matrix Factorization: a 2D embedding example (slide 10)

MF learns low-dimensional latent vectors (embeddings) for users and items; predicted preference = their **dot product**. Embeddings often align with interpretable semantic axes.

```
              children's (−1)
                   ▲
   Triplets of     │   Harry Potter
   Belleville      │      (blockbuster + children's)
                   │
 arthouse (−1) ────┼──── blockbuster (+1)
                   │
        Memento    │   The Dark Knight Rises
     (arthouse,    │   (blockbuster, adult)
      adult)       ▼
              adult's (+1)
```

> [!example] How the prediction works
> Each user is a vector (e.g. ■ = arthouse↔blockbuster value, ▲ = children's↔adult's value); each movie is a vector in the same 2D space. The predicted score for a (user, movie) pair is the **dot product** of the two vectors. Observed likes (✓) constrain the embeddings; a missing entry `?` is filled by the dot product of the learned user and item latent vectors. Source: Google ML crash course.

### 1.4 When does Matrix Factorization fail? (slide 11)

> [!warning] MF ignores the ORDER of interactions
> Standard MF/CF treats a user's interactions as an **unordered set**. In many real scenarios the order is essential:
> - **Natural sequence patterns:** buy a phone case *after* the phone, not before.
> - **Series of items:** Star Wars IV → V → VI.
> - **Evolving interests:** a user listened to pop 10 years ago, now prefers rock.
> - **Geographical proximity:** Amsterdam → Almere → Zwolle → Groningen (route order matters).
> - **Repeating interactions:** a user buys a pack of coffee every month.
>
> Temporal order, recency, repetition and transitions are **lost** in plain MF — this motivates **[[Sequential Recommendation]]**.

### 1.5 The Sequential Recommendation goal (slide 12)

> [!definition] Sequential Recommendation
> Predict the **next item** in a **chronologically ordered** sequence of (historical) user-item interactions.
>
> *Example:* a reader finishes Harry Potter *Philosopher's Stone* → *Chamber of Secrets* → *Prisoner of Azkaban*; the system recommends the next book in the series, *Goblet of Fire*. This is **[[Next Item Prediction|next-item prediction]]** from a temporally ordered history.

### 1.6 Paradigms (slide 13)

- Instead of single items we can recommend **baskets, bundles, playlists**, etc.
- Two paradigms: **[[Session-based Recommendation|session-based]]** vs. **user-based** recommendations.
- This lecture **focuses on collaborative signals**, but item/user representations **can be augmented with content** (side information / content features).

---

## 2. Early Works: Markov Chains

### 2.1 Markov Chains for sequences (slide 15)

Given a sequence of $n$ interactions $\langle i_1, i_2, \dots, i_n \rangle$, we want the probability of the next interaction:

$$P(i_{n+1} \mid i_1, \dots, i_n)$$

If we condition only on the **last $k$ interactions**, this is a **[[Markov Chain|k-order Markov Chain]]**:

$$P(i_{n+1} \mid i_{n-k}, \dots, i_n)$$

- $i_{n+1}$ = next item to predict; $i_1,\dots,i_n$ = full history; $k$ = order (memory length).
- Early approaches use an **n-gram model** built by **counting observations** in the training data [Shani et al., 2005].

> [!warning] Data sparsity is the major problem
> Most item-to-item transitions are never observed. Mitigations:
> - **Skipping:** observing $\langle x_1, x_2, x_3 \rangle$ also lends likelihood to $\langle x_1, x_3 \rangle$ (allow gaps).
> - **Clustering:** treat $\langle x, y, z \rangle \approx \langle w, y, z \rangle$ (group similar items/contexts).
> - **Mixture modeling:** combine different orders $n$ (interpolate between Markov orders).

### 2.2 Next-basket data (slide 16)

Markov chains over baskets model item→item transition probabilities across consecutive baskets. **Figure 1** — four users, five items $\{a,b,c,d,e\}$; columns are time-ordered baskets $B_{t-3}, B_{t-2}, B_{t-1}, B_t$, and $B_t$ is the basket to predict (`?`):

| User | $B_{t-3}$ | $B_{t-2}$ | $B_{t-1}$ | $B_t$ |
|---|---|---|---|---|
| User 1 | $\{a,b,c\}$ | $\{b,c\}$ | $\{a,b\}$ | `?` |
| User 2 | — | $\{a\}$ | $\{a,c\}$ | `?` |
| User 3 | $\{d\}$ | $\{c,e\}$ | $\{e\}$ | `?` |
| User 4 | — | — | $\{c,e\}$ | `?` |

### 2.3 Global transition matrix (slide 17)

**Figure 2** — a **single global / aggregated** transition matrix estimated from the data of all four users. Entry $(i,j)$ estimates $P(\text{to } j \mid \text{from } i)$ from observed consecutive co-occurrences; the `#` column is the support (number of observed out-transitions).

| from \ to | a | b | c | d | e | # |
|---|---|---|---|---|---|---|
| **a** | 0.5 | 0.5 | 1 | 0 | 0 | 2 |
| **b** | 0.5 | 1 | 0.5 | 0 | 0 | 2 |
| **c** | 0.3 | 0.7 | 0.3 | 0 | 0.3 | 3 |
| **d** | 0 | 0 | 1 | 0 | 1 | 1 |
| **e** | 0 | 0 | 0 | 0 | 1 | 1 |

> [!note] Key limitation
> One **global** [[Transition Matrix]] is shared by **all users** — there is **no personalization**.

### 2.4 Personalized Markov Chains (slides 18-19)

> [!definition] Naive vs. Personalized Markov Chains
> - **Naive Markov Chain:** a single global transition matrix $T \in \mathbb{R}^{|I| \times |I|}$ → same transition behaviour for everyone.
> - **[[Factorized Personalized Markov Chains (FPMC)|FPMC]]** [Rendle et al., 2010]: a **per-user** transition matrix $T^{(u)} \in \mathbb{R}^{|I| \times |I|}$. Modelling these directly is **infeasible due to sparsity** ⇒ use **[[Matrix Factorization]]** to factorize the transition tensor/cube.

**Figure 3** — per-user transition matrices. Most entries are `?` because each individual user has very few observed transitions (extreme sparsity):

- **User 1:** $a\to(0,1,1,0,0)\,\#1$; $b\to(0.5,1,0.5,0,0)\,\#2$; $c\to(0.5,1,0.5,0,0)\,\#2$; rows $d,e$ all `?` $\#0$.
- **User 2:** $a\to(1,0,1,0,0)\,\#1$; rows $b$–$e$ all `?`.
- **User 3:** rows $a,b$ `?`; $c\to(0,0,0,0,1)\,\#1$; $d\to(0,0,1,0,1)\,\#1$; $e\to(0,0,0,0,1)\,\#1$.
- **User 4:** all entries `?` $\#0$ (only one basket ⇒ no transitions).

⇒ Per-user matrices are mostly unknown; **factorization** fills them in by **sharing parameters** across users and items.

### 2.5 FPMC: factorizing the transition cube (slide 20)

> [!formula] FPMC predicted score
> $$\hat{x}_{u,i,j} = P_u^\top Q_j + R_i^\top S_j$$
>
> - $P_u^\top Q_j$ = **long-term preference**: user $u$'s general affinity for target item $j$ (a standard MF term).
> - $R_i^\top S_j$ = **short-term transition**: likelihood of moving **from item $i$ to item $j$** (a factorized first-order transition).
> - $P_u$ = user latent vector; $Q_j$ = item-as-target latent vector (user term); $R_i$ = previous-item latent vector; $S_j$ = next-item latent vector (transition term).
>
> Trained with a **ranking loss via [[Stochastic Gradient Descent|SGD]]** — the original FPMC uses **S-BPR** ([[Bayesian Personalized Ranking (BPR)|Bayesian Personalized Ranking]]).

**Takeaway:** FPMC = personalized MF (long-term) + factorized first-order Markov transition (short-term), tied together. Factorization solves the sparsity of per-user transition cubes. Its limitation: it captures only **first-order** dependencies.

### 2.6 FPMC results (slide 21)

**Figure 4** — FPMC on an **online-shopping (sparse)** dataset. X-axis = embedding **dimensionality** (~10→128), Y-axis = **F-Measure @ top-5** (~0.018→0.046).

```
F-meas@5
 0.046 |                              ● SBPR-FPMC  (best, rises with dim)
       |                        ● ─ ─ ─ SBPR-FMC   (just below FPMC)
 0.038 |              ▲────────── SBPR-MF (rises then flattens)
       |   +  MC dense (~0.020, single point, does NOT scale)
 0.018 |   ×  most popular (~0.018, flat, lowest)
       +-------------------------------------------- dimensionality
          10                                   128
```

> [!example] Ranking of methods (sparse data)
> **FPMC > FMC > MF > MC dense > most-popular.** Performance grows with embedding dimensionality (more capacity). Combining personalized long-term (MF) **and** short-term (Markov transition) signals beats either alone, especially on sparse data. The non-parametric dense MC does not benefit from added dimensions.

---

## 3. The Deep-Learning Era

### 3.1 GRU4Rec — RNN-based sequential recommendation (slides 23-26)

**GRU4Rec** [Hidasi et al., 2015]:
- Among the **first deep-learning models** for sequential recommendation.
- Originally designed for **[[Session-based Recommendation|session-based]]** recommendations.
- Built on a **[[Gated Recurrent Unit (GRU)|GRU]]**, a type of **[[Recurrent Neural Network (RNN)|RNN]]**.

**Architecture (Figures 5-7), bottom → top:**

```
            scores on items          ← output: one score per candidate item
                  ▲
          Feedforward layer(s)
                  ▲
   ┌──── GRU layer ──┐  ↺            ← stacked GRU layers, recurrent
   │     GRU layer   │  ↺              feedback captures sequential
   └──── GRU layer ──┘  ↺              dependencies from past items
                  ▲
           Embedding layer            ← each item has its own dense embedding
                  ▲
   Input: 1-of-N (one-hot) coding of the current item
```

- **Input:** one-hot (1-of-N) coding of the current item.
- **Embedding layer:** maps the one-hot to a dense embedding (each item has its **own** dense embedding).
- **GRU layer(s):** one or more stacked GRUs with recurrent self-connections; **capture sequential dependencies** from previous interactions.
- **Feedforward + output:** a score per candidate item for next-item prediction.

> [!formula] GRU4Rec — pairwise BPR loss (slide 26)
> $$\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S} \sum_{j=1}^{N_S} \log \sigma\!\left(\hat{r}_{s,i} - \hat{r}_{s,j}\right)$$
>
> - $N_S$ = number of **negative samples** per positive instance.
> - $\hat{r}_{s,i}$ = score of the **true next item** $i$; $\hat{r}_{s,j}$ = score of **negative sample** $j$.
> - $\sigma(\cdot)$ = sigmoid.
> - **Interpretation:** push the score of the true next item *above* the scores of sampled negatives (a **pairwise** objective). See [[Negative Sampling]].

> [!note] When to use GRU4Rec? (slide 25)
> - **Outperforms FPMC**, especially when **more data is available**.
> - Allows **more complex modelling and longer sequences** than FPMC.
> - But when data is **sparse**, the model must stay simple or compute is limited ⇒ **FPMC can outperform GRU4Rec**.

### 3.2 SASRec — Self-Attentive Sequential Recommendation (slides 27-30)

**SASRec** [Kang & McAuley, 2018]:
- **First sequential recommender relying solely on [[Self-Attention]].**
- Input embedding = **item embedding + positional embedding**.
- Self-attention produces a **contextual representation** of the sequence.

**Architecture (Figures 8-9), input = training action sequence $s_1, s_2, s_3, s_4$:**

```
        Expected next item   (one prediction per position)
                ▲
        Prediction Layer
                ▲
   Point-Wise Feed-Forward Network (per position)
                ▲
   ┌── Self-Attention Layer ──┐
   │   (causal: position t     │  ↺ "Can Stack More" blocks
   │    attends to ≤ t only)   │
   └───────────────────────────┘
                ▲
        Embedding Layer  =  item embeddings  +  positional embeddings
                ▲
        s1   s2   s3   s4     (training action sequence)
```

> [!note] SASRec mechanics (slide 28)
> - Uses a **causal mask**: each position attends only to **itself and earlier positions** (no peeking at the future), making it **unidirectional / left-to-right**.
> - **Training:** scores **one positive** and **several negatives** per position (see [[Negative Sampling]]).
> - **Inference:** scores **all items** by multiplying the **last token's representation** with the item-embedding matrix.
> - Trained with **binary cross-entropy (BCE)** + negative sampling.

> [!formula] SASRec — Binary Cross-Entropy loss (slide 29)
> $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N_S} \sum_{i=1}^{N_S} \left[ y_{s,i} \log \hat{y}_{s,i} + (1 - y_{s,i}) \log(1 - \hat{y}_{s,i}) \right]$$
>
> - $N_S$ = number of **samples** (positive + negative) per sequence.
> - $y_{s,i} \in \{0,1\}$ = ground-truth label for sample $i$.
> - $\hat{y}_{s,i}$ = predicted score from SASRec's output.

**Results (Figure 3, slide 30)** — training efficiency on **MovieLens 1M (ML-1M)**. X-axis = wall-clock time (s, 0→7000); Y-axis = NDCG@10 (0.35→0.60). SASRec is roughly an **order of magnitude faster per epoch** than CNN/RNN baselines and converges to a higher score:

| Model | s/epoch | Final NDCG@10 |
|---|---|---|
| **SASRec** (cut 200) | **1.7** | **~0.59** (highest, fastest) |
| Caser (full, CNN) | 31.98 | ~0.55 |
| Caser (cut 200) | 19.1 | ~0.52–0.53 |
| GRU4Rec⁺ (full) | 46.9 | ~0.55 (slow rise) |
| GRU4Rec⁺ (cut 200) | 30.7 | ~0.52 |

⇒ SASRec reaches a **higher** NDCG@10 and does so **far faster** (lowest s/epoch, fastest wall-clock convergence) than Caser (CNN) and GRU4Rec (RNN).

### 3.3 BERT4Rec — Bidirectional Self-Attention (slides 31-35)

**BERT4Rec** [Sun et al., 2019]:
- Applies **bidirectional self-attention** to sequential recommendation.
- Rationale: **causal (unidirectional) attention may miss patterns in loosely ordered data.**
- Trained with a **Cloze / masked-item task** — randomly mask a percentage of items and predict them from **both** left and right context:

$$S = [v_1, \dots, v_n] \;\longrightarrow\; S_{\text{masked}} = [v_1, \dots, [\text{MASK}], \dots, v_n]$$

> [!note] "Mask at the end" at inference (slide 32)
> The Cloze task masks *interior* positions, but at inference we want the *next* item. So we **append a [MASK] to the user history** and predict it:
> $$S = [v_1, \dots, v_n] \;\longrightarrow\; S_{\text{masked}} = [v_1, \dots, v_n, [\text{MASK}]]$$
> Adding this **'mask-at-the-end' task as a second training stage increases** BERT4Rec's performance — it mitigates the train/inference mismatch between random masking and last-position prediction.

**Architecture comparison (Figure 10 / paper Fig. 1, slide 33):**

```
(a) Transformer layer "Trm":
    input → Multi-Head Attention → Add & Norm → Dropout
          → Position-wise Feed-Forward → Add & Norm → Dropout

(b) BERT4Rec  — BIDIRECTIONAL (every pos. attends to every pos.)
    emb = item v_i + positional p_i ;  L× stacked Trm ;
    [mask] at some position → Projection head → output v_t
    learns a bidirectional model via the Cloze task

(c) SASRec    — LEFT-TO-RIGHT (unidirectional)
    stacked Trm; output v_{t+1} predicted from v_1 … v_t

(d) RNN-based — LEFT-TO-RIGHT (unidirectional)
    chained GRU cells
```

**Caption:** BERT4Rec learns a **bidirectional** model via the Cloze task, while SASRec and RNN methods are **left-to-right unidirectional** next-item predictors.

> [!formula] BERT4Rec — Masked-LM / Cross-Entropy loss (slide 34)
> $$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P(v_i \mid S_{\text{masked}}; \theta)$$
>
> - $\mathcal{M}$ = set of **masked positions**; $v_i$ = the **true item** at position $i$.
> - $P(v_i \mid \cdot)$ = predicted probability from the **Transformer + softmax** (full-vocabulary softmax over items).
> - $\theta$ = model parameters.

### 3.4 Does bidirectionality actually win? — the loss-function caveat (slide 35)

[Sun et al., 2019] claim BERT4Rec **outperforms SASRec** thanks to its bidirectional objective. But **reproducibility studies show the LOSS FUNCTION has a large impact** — especially the **role of negatives**, which can cause **overconfidence**.

**Figure 11** — results from [Klenitskiy & Vasilev, 2023] on **ML-1M**:

| Dataset | Model | HR@10 | HR@100 | NDCG@10 | NDCG@100 |
|---|---|---|---|---|---|
| ML-1M | BPR-MF | 0.0762 | 0.3656 | 0.0383 | 0.0936 |
| | GRU4Rec (ours) | 0.2811 | 0.6359 | 0.1648 | 0.2367 |
| | BERT4Rec | 0.2843 | 0.6680 | 0.1537 | 0.2322 |
| | SASRec | 0.2500 | 0.6492 | 0.1341 | 0.2153 |
| | SASRec+ (Full CE Loss) | 0.3152 | 0.6743 | 0.1821 | 0.2555 |
| | **SASRec+ (BCE, 3000 negatives)** | **0.3159** | **0.6808** | **0.1857** | **0.2603** |

> [!important] "Turning Dross into Gold" — loss > architecture
> Vanilla SASRec (few negatives, BCE) **underperforms** BERT4Rec. But train SASRec with a **full cross-entropy loss** or **BCE with many (3000) negatives** ("SASRec+") and it **beats** BERT4Rec on *every* metric. The apparent superiority of BERT4Rec is largely due to the **choice of loss and number of negatives**, not the bidirectional architecture per se. Too few negatives ⇒ overconfidence.

---

## 4. Wrapping Up

### 4.1 Methods covered (slide 37)

- **Markov Chains:** naive Markov Chains; **FPMC**.
- **Deep Learning:** **GRU4Rec**; **SASRec**; **BERT4Rec**.

### 4.2 Architecture comparison (slide 38)

| Model | Strengths | Limitations |
|---|---|---|
| **FPMC** | Lightweight, interpretable; good for simple datasets with short sequences. | Captures only **first-order** dependencies. |
| **GRU4Rec** | Effective at short temporal patterns within sessions. | Lots of training time; struggles with **long sequences**. |
| **SASRec** | Balances complexity and efficiency; outperforms GRU4Rec. | Does **not** consider **bidirectional** context. |
| **BERT4Rec** | Leverages **bidirectional** context; outperforms SASRec on multiple datasets. | Slower to train; gains may vary (loss-dependent). |

### 4.3 Losses (slide 39)

We discussed **BPR, BCE and CE**. These losses are **not model-specific** — any of the discussed models can use any of them. Alternatives:
- **TOP1-max** (from the GRU4Rec paper [Hidasi et al., 2015]).
- **[[Listwise LTR|Listwise]] losses** such as **LambdaRank loss** [Li et al., 2021].
- **[[Contrastive Learning|Contrastive]] losses** such as **InfoNCE** [Zhou et al., 2020, S³-Rec].

### 4.4 Open challenges (slide 40)

- **Scaling to catalogs with millions of items** → efficient nearest-neighbour ([[ANN Search]]) or **sub-item IDs** ([[Semantic IDs]]).
- **User- and item-cold-start** → **generative models, like [[Large Language Models (LLM)|LLMs]]**.
- **Very long user histories** → architectural improvements or smart data pre-processing.
- **RecSys is uniquely tied to industry** — challenges are often driven by circumstance and **domain-specific requirements**.

---

## Key Takeaways

> [!tip] Exam focus
> - **Why sequential?** Plain MF/CF treats history as an **unordered set**, losing order, recency, repetition and transitions. Sequential recommendation predicts $i_{n+1}$ given an ordered history.
> - **Markov Chains:** $P(i_{n+1}\mid i_{n-k},\dots,i_n)$ for a $k$-order chain; built by counting (n-gram). A **single global** transition matrix ⇒ **no personalization**; sparse. Mitigate with skipping / clustering / mixture-of-orders.
> - **FPMC** $\hat{x}_{u,i,j} = P_u^\top Q_j + R_i^\top S_j$ = personalized MF (**long-term** $P_u^\top Q_j$) **+** factorized first-order transition (**short-term** $R_i^\top S_j$). Factorization fixes per-user sparsity; trained with **S-BPR**. Only first-order.
> - **GRU4Rec:** RNN/GRU, session-based, **BPR** (pairwise) loss; first deep model; longer sequences than FPMC but slow; FPMC can still win on **sparse** data.
> - **SASRec:** **self-attention** + item & positional embeddings, **causal mask** (left-to-right), **BCE** + negatives; faster & stronger than RNN/CNN; inference = last-token × item matrix.
> - **BERT4Rec:** **bidirectional** self-attention, trained via the **Cloze/MLM** task (CE over masked positions); add **'mask-at-the-end'** as a second stage to match inference.
> - **Loss can beat architecture (Klenitskiy & Vasilev 2023):** SASRec+ (full CE, or BCE with 3000 negatives) **beats** BERT4Rec on ML-1M. Too few negatives ⇒ **overconfidence**. BPR/BCE/CE are model-agnostic.
> - **Formulas to know:** BPR $-\frac{1}{N_S}\sum\log\sigma(\hat r_{s,i}-\hat r_{s,j})$; BCE; MLM/CE over masked positions.
> - **Open challenges:** million-item scaling (**semantic IDs**, ANN), **cold start** (LLMs), very long histories.

---

## Links

**Concepts**
- [[Sequential Recommendation]] · [[Session-based Recommendation]] · [[Next Item Prediction]]
- [[Matrix Factorization]] · [[Collaborative Filtering]] · [[User-based Collaborative Filtering]] · [[Item-based Collaborative Filtering]] · [[User-Item Interaction Matrix]]
- [[Markov Chain]] · [[Transition Matrix]] · [[Factorized Personalized Markov Chains (FPMC)]]
- [[GRU4Rec]] · [[Gated Recurrent Unit (GRU)]] · [[Recurrent Neural Network (RNN)]]
- [[SASRec]] · [[BERT4Rec]] · [[Self-Attention]] · [[Transformer Model]]
- [[Bayesian Personalized Ranking (BPR)]] · [[Negative Sampling]] · [[Contrastive Learning]] · [[Listwise LTR]]
- [[Semantic IDs]] · [[ANN Search]] · [[Large Language Models (LLM)]] · [[Cold Start]]

**Related RecSys lectures**
- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03b - From LLMs to LRMs]] — Part 2 of this lecture (LLM-based recommendation)
- [[RS-L04 - Generative Recommendation]]
