---
type: concept
aliases: [MRR, mean reciprocal rank, reciprocal rank]
course: [IR]
tags: [evaluation, key-formula]
status: complete
---

# Mean Reciprocal Rank (MRR)

> [!formula] MRR
> $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
> 
> where $\text{rank}_i$ is the rank of the **first** relevant document for query $i$.

Useful when you care about how quickly the user finds **any** relevant result (e.g., question answering, navigational queries).

## Appears In

- [[IR-L04 - Evaluation]]
