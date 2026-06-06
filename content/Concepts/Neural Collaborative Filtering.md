---
type: concept
aliases: [NCF, Neural CF, Embedding Layer]
course: [RecSys]
tags: [collaborative-filtering, exam-topic]
status: complete
---

# Neural Collaborative Filtering

## Definition

> [!definition] Neural Collaborative Filtering (NCF)
> **Neural Collaborative Filtering** [He et al., 2017] is a neural framework for **top-n recommendation** that replaces the fixed inner-product interaction of [[Matrix Factorization]] with a learnable neural function. Each user $u$ and item $i$ is one-hot encoded, mapped through an **[[Embedding Layer]]** to dense latent vectors $p_u, q_i$, and the interaction $\hat{y}_{ui}$ is modeled by stacked **neural CF layers** with non-linear activations. NCF is trained on [[Implicit Feedback]] as a binary classification problem, and famously **subsumes [[Matrix Factorization|MF]] as a special case**.

## Intuition

> [!intuition] Why a neural function instead of a dot product
> Plain [[Matrix Factorization|MF]] fixes the user–item interaction to be the **inner product** $p_u \cdot q_i$ — a *linear*, symmetric combination of latent dimensions. This caps its expressiveness: two users can be close in the latent space yet have genuinely non-linear preference overlaps that a dot product cannot represent (the "MF limited to linear relationships" problem).
>
> NCF keeps the [[Embedding Layer]] idea (one-hot $\to$ dense latent vector) but lets a [[Multi-Layer Perceptron|MLP]] *learn* the interaction function from data. With non-linear activations it can model complex user–item patterns; because the embeddings and the interaction network are trained jointly, NCF can also fold in heterogeneous content and sequential signals. The punchline of the paper: if you cripple the neural layers down to element-wise multiplication with fixed unit weights and identity output, you recover MF exactly — so MF is *one point* in the NCF hypothesis space.

## Mathematical Formulation

The NCF predictor maps user/item one-hot vectors $v_u^U, v_i^I$ to a relevance score $\hat{y}_{ui} \in [0,1]$:

$$
p_u = P^{\top} v_u^U, \qquad q_i = Q^{\top} v_i^I, \qquad
\hat{y}_{ui} = \phi_{\text{out}}\Big( \phi_X\big( \cdots \phi_1([\,p_u ; q_i\,]) \cdots \big) \Big)
$$

where:
- $P \in \mathbb{R}^{M \times K}$, $Q \in \mathbb{R}^{N \times K}$ — user / item embedding matrices ($M$ users, $N$ items, $K$ latent dims)
- $p_u, q_i \in \mathbb{R}^{K}$ — dense user / item **latent vectors** (output of the [[Embedding Layer]])
- $[\,p_u ; q_i\,]$ — concatenation, the input to the first **neural CF layer**
- $\phi_1, \dots, \phi_X$ — the stacked neural CF layers (e.g. MLP with ReLU), giving the non-linearity
- $\phi_{\text{out}}$ — output layer mapping to a score, typically $\sigma(h^{\top} \phi_X(\cdot))$ with sigmoid $\sigma$
- $\hat{y}_{ui}$ — predicted score; $y_{ui}$ — target label

**Training as binary classification.** Treat $y_{ui}=1$ if $i$ is observed/relevant for $u$, else $0$. For [[Implicit Feedback]] the loss is **binary cross-entropy** over observed positives $\mathcal{Y}$ and sampled negatives $\mathcal{Y}^{-}$:

$$
L = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^{-}} \Big[\, y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log (1 - \hat{y}_{ui}) \,\Big]
$$

where:
- $\mathcal{Y}$ — set of observed (positive) interactions
- $\mathcal{Y}^{-}$ — set of unobserved instances drawn by [[Negative Sampling]] (reduces the cost of all unobserved pairs)
- For [[Explicit Feedback]], a **weighted square loss** $\sum w_{ui}(y_{ui} - \hat{y}_{ui})^2$ is used instead

**MF as a special case (GMF).** Replace the neural CF layers with **element-wise** multiplication and a fixed output, and NCF reduces exactly to the [[Matrix Factorization|MF]] dot product:

$$
\hat{y}_{ui} = a_{\text{out}}\big( h^{\top} (p_u \odot q_i) \big) \;\xrightarrow{\;a_{\text{out}}=\text{identity},\; h = J_{K\times 1}\;}\; \sum_{k=1}^{K} p_{u,k}\, q_{i,k} = p_u \cdot q_i
$$

where $\odot$ is element-wise product, $a_{\text{out}}$ the output activation, and $h = J_{K\times 1}$ a fixed unit (all-ones) weight vector. With a learnable $h$ and non-linear $a_{\text{out}}$ this becomes **Generalized Matrix Factorization (GMF)**.

## Key Properties / Variants

- **Generality.** NCF is a *framework*, not a single model: the interaction network is a design choice. MF is recovered by element-wise product + fixed unit output + identity activation (slide 46).
- **GMF** — Generalized Matrix Factorization: keeps element-wise product but learns $h$ and uses a non-linear output activation.
- **MLP** — feeds the *concatenation* $[p_u ; q_i]$ through a deep stack with ReLU; learns an arbitrary interaction function rather than a product.
- **NeuMF** — fuses GMF and MLP, each with its **own separate embeddings**, concatenating their last hidden layers before the output. Combines MF's linear modeling with the MLP's non-linearity.
- **Loss by feedback type.** Binary cross-entropy for [[Implicit Feedback]]; weighted square loss for [[Explicit Feedback]].
- **[[Negative Sampling]].** Unobserved pairs vastly outnumber observed ones; sample a few negatives per positive instead of using all unobserved instances.
- **Strengths over MF.** Non-linear interactions, can ingest heterogeneous content (text/image/audio), and is trained end-to-end. Empirically outperformed strong baselines on two public datasets in the original paper.
- **Caveat (Reproducibility).** Dacrema et al. (2019) found many neural recommenders, NCF among the era's wave, were hard to reproduce and often did not beat well-tuned simple baselines — always tune baselines and never tune on the test set.

Forward pass (NeuMF) in pseudo-code:

```pseudo
Algorithm: NeuMF forward pass
─────────────────────────────
Input: user id u, item id i
  # GMF branch (its own embeddings)
  p_u_gmf ← P_gmf[u];  q_i_gmf ← Q_gmf[i]
  z_gmf   ← p_u_gmf ⊙ q_i_gmf          # element-wise product (vector)

  # MLP branch (its own embeddings)
  p_u_mlp ← P_mlp[u];  q_i_mlp ← Q_mlp[i]
  z       ← concat(p_u_mlp, q_i_mlp)
  for layer ℓ = 1..X:
    z ← ReLU(W_ℓ · z + b_ℓ)             # neural CF layers
  z_mlp ← z

  # Fusion + output
  ŷ_ui ← σ( hᵀ · concat(z_gmf, z_mlp) ) # sigmoid → score in [0,1]
  return ŷ_ui
```

## Connections

- Generalizes: [[Matrix Factorization]] (recovered as the GMF special case with fixed unit output)
- Built on: [[Embedding Layer]], [[Multi-Layer Perceptron]], [[Neural Networks]]
- A model-based form of: [[Collaborative Filtering]]
- Trained with: [[Implicit Feedback]] / [[Explicit Feedback]], [[Negative Sampling]]
- Motivated by limits of linear: [[Matrix Factorization]]'s fixed inner product
- Successor paradigms for ranking: [[Sequential Recommendation]], [[Generative Recommendation]]
- Evaluation caveat: Reproducibility

## Appears In

- [[RS-L01 - Course Overview & Introduction]]
