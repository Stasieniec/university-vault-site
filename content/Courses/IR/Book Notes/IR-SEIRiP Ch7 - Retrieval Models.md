---
type: book-chapter
course: IR
book: "Search Engines: Information Retrieval in Practice"
chapter: 7
sections: ["7.1", "7.2", "7.3"]
topics:
  - "[[BM25]]"
  - "[[TF-IDF]]"
  - "[[Vector Space Model]]"
  - "[[Query Likelihood Model]]"
  - "[[Language Model for IR]]"
  - "[[Boolean Retrieval]]"
  - "[[Binary Independence Model]]"
status: complete
---

# Chapter 7: Retrieval Models

## 7.1 Overview of Retrieval Models
Retrieval models provide a mathematical framework to formalize the process of deciding if a piece of text is relevant to an information need. Good models produce rankings that correlate with human relevance decisions, leading to high effectiveness.

> [!definition] Relevance
> - **Topical Relevance**: Whether a document and query are "about the same thing."
> - **User Relevance**: A broader concept incorporating factors like age, language, novelty, and target audience.
> - **Binary vs. Multi-valued**: While relevance is often multi-level in reality, many models assume binary relevance (relevant/not relevant) for simplicity, often calculating a probability of relevance to represent uncertainty.

### 7.1.1 Boolean Retrieval
The oldest model, also known as **exact-match retrieval**. Documents are retrieved only if they exactly match the query specification.

- **Outcome**: Binary (Matches or doesn't). No inherent ranking among retrieved documents.
- **Operators**: `AND`, `OR`, `NOT`, proximity operators, wildcards.
- **Pros**: Predictable, easy to explain, efficient, allows complex metadata filtering.
- **Cons**: Effectiveness depends entirely on the user; "searching by numbers" (too many or too few results); lacks term weighting.

### 7.1.2 The Vector Space Model (VSM)
The [[Vector Space Model]] represents documents and queries as vectors in a $t$-dimensional space, where $t$ is the number of index terms.

> [!formula] Cosine Similarity
> The most successful similarity measure for ranking in VSM:
> $$Cosine(D_i, Q) = \frac{\sum_{j=1}^{t} d_{ij} \cdot q_j}{\sqrt{\sum_{j=1}^{t} d_{ij}^2} \cdot \sqrt{\sum_{j=1}^{t} q_j^2}}$$
> Where $d_{ij}$ is the weight of term $j$ in document $i$, and $q_j$ is the weight in the query.

**Term Weighting ([[TF-IDF]]):**
1. **Term Frequency (tf)**: Reflects term importance within a document.
   - Standard: $tf_{ik} = \frac{f_{ik}}{\sum f_{ij}}$
   - Log-normalized (to reduce impact of frequent terms): $\log(f_{ik}) + 1$
2. **Inverse Document Frequency (idf)**: Reflects term discriminative power in the collection.
   - $$idfk = \log \frac{N}{n_k}$$
   - Where $N$ is total documents and $n_k$ is the number of documents containing term $k$.

> [!intuition] TF-IDF
> A term is important if it occurs frequently in a specific document but rarely across the rest of the collection.

**Rocchio Algorithm (Relevance Feedback):**
Used to modify the query vector $Q$ into $Q'$ based on relevant and non-relevant sets:
$$q_j' = \alpha \cdot q_j + \beta \cdot \frac{1}{|Rel|} \sum_{D_i \in Rel} d_{ij} - \gamma \cdot \frac{1}{|Nonrel|} \sum_{D_i \in Nonrel} d_{ij}$$

---

## 7.2 Probabilistic Models
The dominant paradigm today, based on representing the uncertainty inherent in [[Information Retrieval]].

### 7.2.1 Probability Ranking Principle (PRP)
> [!info] Principle
> If a system ranks documents in order of decreasing probability of relevance, the overall effectiveness of the system will be the best obtainable.

### 7.2.2 Binary Independence Model (BIM)
Treats IR as a classification problem. Documents are viewed as binary vectors (presence/absence of terms).

**Assumptions:**
- Binary Relevance.
- **Naïve Bayes Assumption**: Terms occur independently in relevant and non-relevant sets.

> [!formula] BIM Scoring Function
> Derived from the likelihood ratio and Bayes' Rule:
> $$Score = \sum_{i: d_i = 1} \log \frac{p_i (1 - s_i)}{s_i (1 - p_i)}$$
> Where $p_i = P(d_i = 1 | R)$ and $s_i = P(d_i = 1 | NR)$.

In the absence of relevance information, the weight simplifies to an IDF-like variant: $\log \frac{N - n_i}{n_i}$.

### 7.2.3 The BM25 Ranking Algorithm
[[BM25]] (Best Match 25) is a robust and widely used ranking algorithm that extends [[Binary Independence Model]] by adding term frequency and length normalization.

> [!formula] BM25 Score
> $$BM25(Q, D) = \sum_{i \in Q} \log \frac{(r_i + 0.5)/(R - r_i + 0.5)}{(n_i - r_i + 0.5)/(N - n_i - R + r_i + 0.5)} \cdot \frac{(k_1 + 1)f_i}{K + f_i} \cdot \frac{(k_2 + 1)qf_i}{k_2 + qf_i}$$
> Where:
> - $K = k_1 ((1 - b) + b \cdot \frac{dl}{avdl})$
> - $f_i$: term frequency in document.
> - $qf_i$: term frequency in query.
> - $dl$: document length; $avdl$: average document length.
> - $k_1, k_2, b$: empirical parameters (typical: $k_1=1.2, b=0.75$).

> [!warning] Parameter k1
> $k_1$ controls **tf saturation**. As $f_i$ increases, the marginal contribution of additional occurrences decreases.

---

## 7.3 Ranking Based on Language Models
Treats documents as being generated from a [[Language Model for IR]] (a probability distribution over words).

### 7.3.1 Query Likelihood Model
Ranks documents by the probability of generating the query text from the document's language model $M_D$.
$$P(Q|D) = \prod_{i=1}^n P(q_i | M_D)$$

### 7.3.2 Smoothing
Critical to avoid zero probabilities for missing query terms and to improve estimation.

> [!formula] Jelinek-Mercer Smoothing
> A linear interpolation between the document MLE and the collection model:
> $$P(q_i | D) = (1 - \lambda) \frac{f_{q_i, D}}{|D|} + \lambda \frac{c_{q_i}}{|C|}$$

> [!formula] Dirichlet Smoothing
> Uses a document-length dependent weighting, generally more effective for short queries:
> $$P(q_i | D) = \frac{f_{q_i, D} + \mu \frac{c_{q_i}}{|C|}}{|D| + \mu}$$
> Best results in TREC usually seen with $\mu \approx 1000-2000$.

### 7.3.3 Relevance Models and KL-Divergence
Generalizes the [[Query Likelihood Model]] by comparing two probability distributions: the Relevance Model ($R$) and the Document Model ($D$).

> [!formula] KL-Divergence Ranking
> $$Score \propto \sum_{w \in V} P(w | R) \log P(w | D)$$
> This framework provides a formal basis for pseudo-relevance feedback and query expansion.

---

## Comparison Summary
| Model | Term Weighting | Length Norm | Relevance Assumption |
| :--- | :--- | :--- | :--- |
| **Boolean** | None | None | Exact match |
| **VSM** | TF-IDF (Cosine) | Cosine Norm | Similarity in Vector Space |
| **BIM** | Probabilistic (IDF-like) | None | Binary / Naïve Bayes |
| **BM25** | Adv. TF-IDF | $avdl$ based | PRP optimized |
| **LM** | Smoothing-based | Implicit in Smoothing | Query Generation Probability |
