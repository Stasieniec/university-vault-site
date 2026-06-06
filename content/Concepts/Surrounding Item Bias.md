---
type: concept
aliases: [context bias, distinctive item bias]
course: [IR]
tags: [bias, click-models, unbiased-ltr, exam-topic]
status: complete
---

# Surrounding Item Bias

## Definition

> [!definition] Surrounding Item Bias
> **Surrounding Item Bias** is the violation of the [[Examination Hypothesis]] in which the probability of examining (and clicking) the item at rank $k$ depends not only on its own position and relevance, but on the **features of the items displayed around it** (ranks $k-1, k, k+1, \dots$). The relative appearance of an item — how it contrasts with its neighbours — modulates the user's attention, so two identical items can receive different click rates purely because of who their neighbours are.

It is the **context-dependent generalisation** of [[Outlier Bias]]: instead of one distinctive item attracting clicks in absolute terms, the bias is framed as a function of the **local list context**, where any item that stands out from its immediate surroundings gains examination probability and any item that blends in loses it.

## Intuition

The classic [[Position-Based Click Model]] assumes examination is a property of the **rank slot alone**: $P(\text{Exam}_{d,k}) = P(\text{Exam}_k)$, identical for every document placed there. Surrounding item bias says this is false — the eye is drawn by *contrast*, not by absolute position.

```
Same item, two contexts (item X is identical in both):

Context A (X blends in):        Context B (X stands out):
[ Blue $20 ]                    [ Blue $20 ]
[ Blue $22 ]                    [ Blue $22 ]
[ Blue $21 ]  ← X, $21, blue    [ Red  $90 ]  ← X' neighbours differ
[ Blue $19 ]                    [ Blue $19 ]

X here looks ordinary           ...but if X is the Red $90 outlier,
→ baseline examination          its neighbours' uniformity makes
                                it pop → examination boosted
```

Two mechanisms are at play, mirroring the two factors of the [[Examination Hypothesis]]:

1. **Attention contrast (examination)**: an item that is visually or feature-wise different from its neighbours is examined more — the [[Outlier Bias]] effect, but defined *relative to the local window*.
2. **Comparative relevance (context bias / [[Trust Bias]]-like)**: an item's *perceived* relevance shifts when it is shown next to much better or much worse alternatives (a contrast/anchoring effect), violating $P(\text{Rel}_d \mid q, \text{context}) = P(\text{Rel}_d \mid q)$.

Because examination now depends on neighbours, clicks are no longer a clean signal of position $\times$ relevance, and naive [[Inverse Propensity Weighting]] (which assumes position-only propensities) becomes biased.

## Mathematical Formulation

The PBM factorises a click as examination times relevance. Surrounding item bias replaces the **position-only** examination term with a **context-conditioned** one:

$$P(\text{Click}_{d,k} \mid \mathcal{C}_k) = \underbrace{P\big(\text{Exam}_k \mid \mathcal{C}_k\big)}_{\text{context-dependent}} \cdot \; P(\text{Rel}_d \mid q)$$

where:
- $\mathcal{C}_k = (x_{k-w}, \dots, x_{k-1}, x_k, x_{k+1}, \dots, x_{k+w})$ — the **context window** of feature vectors of the items surrounding rank $k$ (window half-width $w$)
- $x_j$ — feature vector (price, rating, colour, visual descriptors, $\dots$) of the item at rank $j$
- $P(\text{Exam}_k \mid \mathcal{C}_k)$ — examination probability **modulated by the surrounding items**, no longer a constant for slot $k$
- $P(\text{Rel}_d \mid q)$ — true (context-free) relevance of document $d$

A common additive parameterisation isolates a baseline position term plus a **distinctiveness** bonus:

$$P(\text{Exam}_k \mid \mathcal{C}_k) = \min\Big(1,\; \beta_k + \alpha \cdot s_{\text{dist}}(x_k, \mathcal{C}_k)\Big)$$

where:
- $\beta_k$ — the standard position-bias examination probability for rank $k$ (the PBM term)
- $\alpha$ — sensitivity of attention to local distinctiveness ($\alpha = 0$ recovers the PBM)
- $s_{\text{dist}}(x_k, \mathcal{C}_k)$ — a **distinctiveness score** of item $k$ relative to its window, e.g. a normalised deviation from the local neighbours:

$$s_{\text{dist}}(x_k, \mathcal{C}_k) = \max_{f \in \text{features}} \frac{\big| x_k^{(f)} - \operatorname{median}_{j \in \mathcal{C}_k} x_j^{(f)} \big|}{\operatorname{std}_{j \in \mathcal{C}_k}\, x_j^{(f)} + \epsilon}$$

In neural click models the whole context map is learned directly, the window of features being fed to a network whose output is the examination logit:

$$P(\text{Exam}_k \mid \mathcal{C}_k) = \sigma\big(f_\theta(x_{k-w}, \dots, x_k, \dots, x_{k+w})\big)$$

where $\sigma$ is the sigmoid and $f_\theta$ a learned function (e.g. an MLP or attention layer over the window) that captures arbitrary neighbour interactions.

> [!formula] Debiased Listwise Objective
> To learn true relevance from context-biased clicks, weight each click by the **inverse context-dependent propensity**:
> $$\hat{\mathcal{L}}_{\text{ULTR}} = \sum_{d} \frac{c_d}{\;P(\text{Exam}_{k_d} \mid \mathcal{C}_{k_d})\;}\,\ell\big(f(x_d), d\big)$$
> where:
> - $c_d \in \{0,1\}$ — observed click on document $d$
> - $P(\text{Exam}_{k_d} \mid \mathcal{C}_{k_d})$ — the **context-aware propensity** (not the position-only $\beta_{k_d}$ of standard IPS)
> - $\ell$ — the per-document ranking loss; $f(x_d)$ — the scoring model being trained

## Key Properties / Variants

- **Generalises Outlier Bias**: a single outlier is the special case where one item's $s_{\text{dist}}$ is large; surrounding item bias treats distinctiveness as a continuous, *window-relative* quantity for every item.
- **Two violated assumptions**: it breaks both PBM assumptions at once — examination is no longer position-only (attention contrast) and relevance is no longer context-free (comparative/anchoring contrast).
- **Symmetric effect**: an item gains examination by standing out *and* loses examination by blending into a uniform neighbourhood, so the bias can both over- and under-expose items.
- **Identifiability cost**: adding context terms enlarges the parameter space, worsening the identifiability problem already noted for the [[Examination Hypothesis]] — diverse logging policies (rank/context randomisation) are needed to separate $P(\text{Exam} \mid \mathcal{C})$ from $P(\text{Rel})$.
- **Display-side mitigation**: because the bias is relative, presenting a *diverse* result page (so every item is roughly equally distinctive) flattens $s_{\text{dist}}$ and reduces the bias without changing the ranking model.

Estimation procedure (EM over a context-aware click model):

```pseudo
Algorithm: EM for Context-Aware (Surrounding Item) Click Model
──────────────────────────────────────────────────────────────
Input: click logs {(query q, ranked list L, clicks c)},
       context window half-width w
Initialize examination network θ (params of f_θ over windows)
Initialize relevance estimates rel_d for all (q, d)

Repeat until convergence:
  E-step:  for each impression and each rank k:
             build context C_k = features(L[k-w : k+w])
             exam_k ← σ( f_θ(C_k) )                 # context-dependent
             infer P(Exam_k = 1 | c_k, exam_k, rel_d) via Bayes
  M-step:  update θ to maximise click log-likelihood
             given inferred examination events
           update rel_d from clicks de-weighted by exam_k

Output: relevance estimates rel_d (debiased for surrounding context)
        examination model f_θ (context propensities for IPS)
```

## Connections

- Special case / sibling of: [[Outlier Bias]] (single distinctive item; window-free)
- Violation of: [[Examination Hypothesis]], [[Position-Based Click Model]] (position-only examination)
- Related biases: [[Position Bias]], [[Trust Bias]], [[Cascading Position Bias]]
- Affects estimation: [[Inverse Propensity Weighting]] needs context-aware propensities; [[Doubly Robust Estimation]] for added robustness
- Lives within: [[Click Models]], [[Unbiased Learning to Rank]]

## Appears In

- [[Examination Hypothesis]]
- [[Outlier Bias]]
- [[Unbiased Learning to Rank]]
- [[Click Models]]
- [[IR-L11 - Unbiased Learning to Rank]]
