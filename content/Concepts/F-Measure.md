---
type: concept
aliases: [F-Measure, F1-Score, F-score]
course: [IR]
tags: [evaluation, key-formula, exam-topic]
status: complete
---

# F-Measure

> [!definition] F-Measure
> The **F-Measure** is the weighted harmonic mean of precision and recall. It provides a single score that balances the trade-off between the two metrics.

> [!formula] F-Measure Formula
> $$F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{(\beta^2 \cdot P) + R}$$
> 
> where:
> - $P$ — [[Precision and Recall|Precision]]
> - $R$ — [[Precision and Recall|Recall]]
> - $\beta$ — Parameter indicating the relative importance of recall vs precision.

## The F1 Score
The most common version is the **F1 Score**, where $\beta = 1$ (equal weight to precision and recall).
$$F_1 = \frac{2PR}{P+R} = \frac{2}{\frac{1}{P} + \frac{1}{R}}$$

> [!intuition] Why Harmonic Mean?
> Unlike the arithmetic mean, the harmonic mean is sensitive to very low values. If either Precision or Recall is 0, the F1 score becomes 0. It penalizes extreme imbalances, forcing the system to perform well on both.

## The Beta Parameter ($\beta$)
- **$\beta = 1$**: Equal weight.
- **$\beta > 1$**: Weights **Recall** higher than Precision (e.g., $F_2$). Used when missing a relevant document is more costly than a false alarm.
- **$\beta < 1$**: Weights **Precision** higher than Recall (e.g., $F_{0.5}$). Used when false alarms are very costly.

## Connections

- Components: [[Precision and Recall]]
- Context: Used throughout Machine Learning and [[Evaluation]] in IR.

## Appears In

- [[IR-L04 - Evaluation]]
