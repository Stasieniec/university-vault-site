---
type: concept
aliases: [P@K, precision at k, P@10]
course: [IR]
tags: [evaluation, key-formula]
status: complete
---

# Precision at K

> [!formula] P@K
> $$P@K = \frac{|\text{relevant documents in top } K|}{K}$$

Simple, intuitive: "of the top K results, how many are relevant?" Ignores documents below rank K and doesn't consider the order within the top K.

## Appears In

- [[IR-L04 - Evaluation]]
