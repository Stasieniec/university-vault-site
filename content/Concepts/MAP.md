---
type: concept
aliases: [MAP, mean average precision, average precision]
course: [IR]
tags: [evaluation, key-formula, exam-topic]
status: complete
---

# Mean Average Precision (MAP)

> [!definition] MAP
> **MAP** is the mean of the Average Precision (AP) scores across all queries. AP summarizes a precision-recall curve into a single number for a ranked list.

> [!formula] Average Precision
> $$\text{AP} = \frac{1}{|R|} \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)$$
> 
> where:
> - $|R|$ — total number of relevant documents
> - $P(k)$ — precision at position $k$
> - $\text{rel}(k)$ — 1 if document at position $k$ is relevant, 0 otherwise
> - Sum only over positions where a relevant document appears

> [!formula] MAP
> $$\text{MAP} = \frac{1}{|Q|} \sum_{j=1}^{|Q|} \text{AP}(q_j)$$

> [!intuition] What AP Captures
> AP rewards placing relevant documents **early** in the ranking. A relevant document at position 1 contributes $P(1) = 1.0$, while one at position 100 contributes much less. Unlike P@k, AP considers **all** relevant documents.

## Appears In

- [[IR-L04 - Evaluation]]
