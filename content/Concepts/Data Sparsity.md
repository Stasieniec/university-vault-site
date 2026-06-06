---
type: concept
aliases: []
course: [RecSys]
tags: [collaborative-filtering, sequential-rec, exam-topic]
status: complete
---

# Data Sparsity

## Definition

> [!definition] Data Sparsity
> **Data sparsity** is the condition where the [[User-Item Interaction Matrix]] is almost entirely **unobserved**: each user interacts with only a tiny fraction of the catalog, so the vast majority of cells $r_{ui}$ are missing. With $n$ users and $m$ items the matrix has $n \times m$ entries, but the number of observed interactions $|\mathcal{O}|$ is orders of magnitude smaller, giving a density $\rho = |\mathcal{O}| / (n \cdot m) \ll 1$ (often well below 1%). Sparsity is the core obstacle for [[Neighborhood-based Collaborative Filtering]]: similarities and neighbor averages are estimated from few or zero co-observed items, making predictions noisy or undefined.

## Intuition

Imagine a 1-million-item music catalog. A single user might have listened to a few hundred tracks — a density on the order of $10^{-4}$. Two users may genuinely have identical taste yet share **zero** co-rated items, so any overlap-based similarity (cosine, Pearson) is computed over an empty or near-empty intersection and is essentially meaningless. The neighbor average in user-based CF,

$$\hat{r}_{ui} = \frac{1}{|\mathcal{N}_i(u)|} \sum_{v \in \mathcal{N}_i(u)} r_{vi},$$

degrades because $\mathcal{N}_i(u)$ — the neighbors of $u$ who actually rated item $i$ — is small or empty. When it is empty there is **no prediction at all**. Sparsity is the most extreme in its limit, the [[Cold Start]] problem: a brand-new user or item has no interactions whatsoever.

The deeper issue is statistical: with very few observations per user/item, the model has too little signal to estimate parameters reliably, so memory-based methods overfit noise and model-based methods overfit "with insufficient data."

## Mathematical Formulation

> [!formula] Sparsity and Its Mitigation by Low-Rank Factorization
> Let $R \in \mathbb{R}^{n \times m}$ be the rating matrix with observed index set $\mathcal{O} = \{(u,i) : r_{ui} \text{ observed}\}$. Sparsity is quantified by the **density**
> $$\rho = \frac{|\mathcal{O}|}{n \cdot m}, \qquad \text{(sparsity} = 1 - \rho).$$
>
> [[Matrix Factorization]] combats sparsity by fitting a **low-rank** model only on observed entries and generalizing to the unobserved ones:
> $$\min_{U, V} \sum_{(u,i) \in \mathcal{O}} \left( r_{ui} - \bar{u}_u^\top \bar{v}_i \right)^2 + \lambda \left( \lVert U \rVert_F^2 + \lVert V \rVert_F^2 \right)$$
>
> where:
> - $U \in \mathbb{R}^{n \times k}$, $V \in \mathbb{R}^{m \times k}$ — user and item latent factors, with rank $k \ll \min(n, m)$
> - $\bar{u}_u, \bar{v}_i$ — the $k$-dimensional factor rows for user $u$ and item $i$; predicted rating $\hat{r}_{ui} = \bar{u}_u^\top \bar{v}_i$
> - the sum runs **only over $\mathcal{O}$** — the loss ignores missing cells rather than treating them as zero
> - $\lambda$ — $L_2$ regularization weight; essential because sparse data otherwise overfits
>
> The low rank $k$ forces parameter sharing across users and items: every observed entry constrains the shared latent dimensions, so an unseen $(u,i)$ is reconstructed from patterns pooled across the whole matrix instead of from $u$'s and $i$'s own (scarce) data.

## Key Properties / Variants

- **Where it bites hardest — neighborhood CF.** Listed explicitly in RS-L01 as a drawback of memory-based CF (alongside noise and scalability): similarity and the neighbor mean are unreliable when co-observations are few.
- **Cold start is the limiting case.** Zero interactions for a new user/item; even factorization cannot help without side information ([[Content-Based Filtering]] / hybrid signals).
- **Implicit feedback worsens the count problem.** With [[Implicit Feedback]] only positives are seen; all unobserved entries are an ambiguous mix of "disliked" and "not yet seen," so [[Negative Sampling]] is used to make training tractable without labeling every missing cell.
- **Sparsity in sequential models — Markov chains.** A naive $k$-order [[Markov Chain]] estimates $P(i_{n+1} \mid i_{n-k} \dots i_n)$ by counting n-grams; long contexts are almost never observed, so estimates collapse. Mitigations from RS-L03a:
  - **Skipping** — $\langle x_1, x_2, x_3 \rangle$ lends likelihood to $\langle x_1, x_3 \rangle$ (allow gaps).
  - **Clustering** — treat $\langle x, y, z \rangle \approx \langle w, y, z \rangle$ (group similar contexts).
  - **Mixture modeling** — interpolate across Markov orders $n$.
- **Per-user transition matrices are catastrophically sparse.** Personalized Markov chains need a transition matrix $T^{(u)} \in \mathbb{R}^{|I| \times |I|}$ per user; direct estimation is infeasible. FPMC (Rendle et al., 2010) factorizes the transition cube instead:
  $$\hat{x}_{u,i,j} = P_u^\top Q_j + R_i^\top S_j$$
  (long-term preference + factorized item-to-item transition). Factorization shares parameters and fills in the unobserved transitions — directly analogous to how MF fills the rating matrix.
- **Density vs. method choice.** On very sparse data, simpler models win: GRU4Rec needs ample data, and "when data is sparse, FPMC can outperform GRU4Rec." More data shifts the advantage to higher-capacity deep models.

```pseudo
Mitigations for data sparsity (recipe):
────────────────────────────────────────
1. Low-rank generalization:  fit R ≈ U Vᵀ on observed cells only (MF, NCF)
                             → small k + regularization share signal
2. Side information:         add content features (text/audio/image) → hybrid
                             → covers cold-start users/items with no history
3. Sequence smoothing:       skipping / clustering / mixture-of-orders (Markov)
4. Implicit-feedback tricks: negative sampling instead of full missing-cell labels
5. Generative / LLM models:  exploit pretrained world knowledge for cold start
```

## Connections

- Manifests in: [[User-Item Interaction Matrix]], [[Collaborative Filtering]], [[Neighborhood-based Collaborative Filtering]]
- Limiting case: [[Cold Start]]
- Mitigated by: [[Matrix Factorization]] (low-rank generalization), [[Negative Sampling]], [[Content-Based Filtering]] / [[Hybrid Recommendation]]
- Sequential context: [[Markov Chain]], [[FPMC]], [[Sequential Recommendation]]
- Related feedback setting: [[Implicit Feedback]]
- Compounding factor: [[Long-Tail Distribution]] (sparsity concentrates on tail items)

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L03a - Sequential Recommendation Models]]
