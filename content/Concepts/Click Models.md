---
type: concept
aliases: [User Click Models, Implicit Feedback Models]
course: [IR]
tags: [evaluation]
status: complete
---

# Click Models

> [!definition] Click Models
> **Click Models** are probabilistic models that describe how users interact with search engine result pages (SERPs). They are used to interpret user clicks as noisy, implicit signals of document relevance while accounting for various biases.

> [!intuition] Clicks are not Relevance
> A user might click a link just because it's the first result (**position bias**), even if it's not the best one. Conversely, they might not click a perfect result because they already found what they needed in a previous link. Click models attempt to "de-bias" this data to recover the true underlying relevance.

## Common Models

- **Random Click Model (RCM)**: Assumes users click on results with a fixed probability, regardless of position or relevance. A very simple baseline.
- **Position Bias Model (PBM)**: Assumes horizontal browsing; the probability of a click depends on both the relevance of the document and its rank (position) on the page.
- **Cascade Model**: Assumes users scan results from top to bottom. They examine a result, and if it's relevant, they click and stop. If not, they move to the next.
- **Examination Hypothesis**: A click $C_i$ occurs only if the user **examines** the result $E_i$ and the result is **relevant** $R_i$: $P(C_i=1) = P(E_i=1)P(R_i=1)$.

## Key Biases Addressed

| Bias | Description |
|------|-------------|
| **Position Bias** | Higher-ranked items get more clicks regardless of quality. |
| **Trust Bias** | Users trust the search engine to put good results at the top. |
| **Presentation Bias** | The way a snippet looks (bolding, length) affects click-through rate. |

## Connections

- Used for: [[Evaluation|Learning from Implicit Feedback]], [[Offline Evaluation]]
- Related to: [[Discounted Cumulative Gain (DCG)]] (which also uses a position-based decay).

## Appears In

- [[IR-L04 - Evaluation]]
