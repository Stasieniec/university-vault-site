---
type: lecture
course: IR
week: 2
lecture: 3
topics:
  - "[[TF-IDF]]"
  - "[[BM25]]"
  - "[[Query Likelihood Model]]"
  - "[[Language Model for IR]]"
  - "[[Vector Space Model]]"
  - "[[Smoothing]]"
status: complete
---

# IR Lecture 3: Retrieval Models

## 1. Introduction to Retrieval Models

Retrieval models provide a mathematical framework for defining query-document matching. They include assumptions about relevance and serve as the basis for ranking algorithms.

Retrieval models are generally categorized into two paradigms:
1. **Lexical Matching**: Matching based on word occurrences (terms).
    - **Vector Space Model (VSM)**: TF-IDF weighting.
    - **Probabilistic Models**: BM25.
    - **Language Models**: Query Likelihood.
2. **Semantic Matching**: Matching based on meaning/representations.
    - **Distributed Representations**: Neural models (e.g., word embeddings, BERT).

---

## 2. Vector Space Model & Term Weighting

In the Vector Space Model (VSM), documents and queries are represented as vectors in a high-dimensional space where each dimension corresponds to a term in the vocabulary.

### 2.1 Lexical Incidence Matrix
The simplest representation is binary, indicating presence (1) or absence (0) of a term.
Similarity is often measured using **Cosine Similarity**:

$$ \text{score}(q, d) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \cdot |\vec{d}|} = \frac{\sum_{i=1}^{|V|} q_i d_i}{\sqrt{\sum q_i^2} \sqrt{\sum d_i^2}} $$

### 2.2 Term Frequency (TF)
Raw frequency counts are better than binary indicators but have diminishing returns.
> [!info] Retrieval Axioms: Term Frequency
> - **TFC1**: Higher score for documents with more query term occurrences.
> - **TFC2 (Saturation)**: The increase in score due to TF should be sub-linear (the difference between 1 and 2 occurrences is more significant than the difference between 100 and 101).
> - **TFC3**: If total occurrences are equal, documents covering more *distinct* query terms should be preferred.

Common sub-linear transformation (logarithmic):
$$ w_{t,d} = \begin{cases} 1 + \log_{10} \text{tf}_{t,d} & \text{if } \text{tf}_{t,d} > 0 \\ 0 & \text{otherwise} \end{cases} $$

### 2.3 Inverse Document Frequency (IDF)
According to **Zipf's Law**, a small set of words appear very frequently. These words (e.g., "the", "and") are poor discriminators.
> [!info] Term Discrimination Constraint (TDC)
> Terms popular across the entire collection should be penalized.

$$ \text{idf}_t = \log \frac{N}{\text{df}_t} $$
where:
- $N$ — total number of documents in collection.
- $\text{df}_t$ — number of documents containing term $t$.

### 2.4 TF-IDF Weighting
The weight of a term $t$ in document $d$ is:
$$ w_{t,d} = \text{tf}_{t,d} \times \log \frac{N}{\text{df}_t} $$

---

## 3. Probabilistic Models: BM25

**BM25 (Best Matching 25)** is the most widely used weighting function in IR. It effectively balances TF, IDF, and document length.

### 3.1 The BM25 Formula
For a query $Q$ containing terms $q_1, \dots, q_n$, the score for document $D$ is:

$$ \text{score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})} $$

where:
- $f(q_i, D)$ — term frequency of $q_i$ in document $D$.
- $|D|$ — length of document $D$ (number of tokens).
- $\text{avgdl}$ — average document length in the collection.
- $k_1$ — term frequency saturation parameter (typically $1.2$ to $2.0$). Controls how quickly TF effect saturates.
- $b$ — length normalization parameter (typically $0.75$). $b=1$ is full normalization, $b=0$ is no normalization.

### 3.2 Intuition
- **Saturation**: As $f(q_i, D)$ increases, the score approaches a limit.
- **Length Normalization**: Longer documents are expected to have higher TFs naturally; we normalize to avoid bias toward long documents that aren't necessarily more relevant.

---

## 4. Language Models for IR (LMIR)

Instead of matching vectors, we treat retrieval as a generative process. We estimate a **Language Model** $\theta_d$ for each document and ask: "What is the probability that this document model generated the query?"

### 4.1 Query Likelihood Model
The documents are ranked by the probability $P(q|d)$. Using Bayes' Rule:
$$ P(d|q) \propto P(q|d) P(d) $$
Assuming a uniform prior $P(d)$, we rank by $P(q|d)$.

Under the **Multinomial Assumption** (terms generated independently):
$$ P(q|\theta_d) = \prod_{w \in q} P(w|\theta_d)^{c(w,q)} $$
In log-space (to avoid underflow):
$$ \log P(q|\theta_d) = \sum_{w \in q} c(w,q) \log P(w|\theta_d) $$

### 4.2 Estimation and Smoothing
The Maximum Likelihood Estimate (MLE) for a term in a document is:
$$ P_{ML}(w|\theta_d) = \frac{c(w,d)}{|D|} $$

**Problem**: If a query term $w$ is missing from document $d$, $P(w|\theta_d) = 0$, making the entire product 0.
**Solution**: **Smoothing** — adjusting estimates to avoid zero probabilities and incorporate background knowledge (collection model $C$).

#### Smoothing Methods
1. **Jelinek-Mercer (Linear Interpolation)**:
   $$ P(w|\theta_d) = (1 - \lambda) \frac{c(w,d)}{|D|} + \lambda P(w|C) $$
   - Small $\lambda$ (e.g., 0.1) $\to$ highlights document-specific content (precision).
   - Large $\lambda$ (e.g., 0.7) $\to$ better for long queries (recall).

2. **Dirichlet Prior Smoothing**:
   $$ P(w|\theta_d) = \frac{c(w,d) + \mu P(w|C)}{|D| + \mu} $$
   - $\mu$ is the smoothing parameter (often $\approx 2000$).
   - It performs "Bayesian" smoothing where the amount of smoothing depends on the document length $|D|$.

3. **Absolute Discounting**:
   Subtracts a constant $d$ from seen counts and redistributes it to unseen terms proportional to $P(w|C)$.

---

## 5. Model Comparison Summary

| Model | Foundation | Key Components |
| :--- | :--- | :--- |
| **VSM (TF-IDF)** | Geometry | TF, IDF, Cosine Sim |
| **BM25** | Probabilistic (BIM) | Saturation-TF, IDF, Length Norm |
| **LMIR** | Probability/Generative | Term distribution, Smoothing |

> [!tip] Choice of Model
> BM25 and LMIR with Dirichlet smoothing are generally state-of-the-art for lexical retrieval and perform similarly in practice. BM25 is easier to tune (parameters $k_1, b$), while LMIR is more theoretically grounded for extensions (e.g., translation models).

---

## 6. Optimization: Skip Pointers

To speed up query processing, inverted lists contain **skip pointers**.
- Allow jumping over large portions of the list if the current document is smaller than the document being evaluated on another list.
- **Trade-off**: More skips $\to$ smaller data read, but more overhead in pointer storage. Optimal skip distance is typically around 100 bytes.

---

## 7. Update Strategies
- **Index Merging**: Create small new index and merge with old one periodically.
- **Geometric Partitioning**: Maintain multiple indexes of increasing size ($I_0, I_1, \dots$). When $I_n$ reaches limit, merge into $I_{n+1}$.
