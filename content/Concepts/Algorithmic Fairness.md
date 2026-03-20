---
type: concept
aliases: [algorithmic fairness, fairness in ranking, fair ranking]
course: [IR]
tags: [ir-society]
status: complete
---

# Algorithmic Fairness

> [!definition] Algorithmic Fairness
> **Algorithmic fairness** is the study of how automated decision systems can be designed to avoid systematic discrimination against individuals or groups based on protected characteristics or other morally relevant factors. In IR, it addresses how ranking systems allocate the scarce resource of user attention.

> [!intuition] Why It Matters
> Rankings are not neutral—they determine which job candidates get interviews, which news stories shape public opinion, which products get purchased, and which voices are heard. Fairness research asks: who benefits and who is harmed by these ranking decisions?

## Key Properties / Variants

### Types of Harms

- **Allocative harms**: Unfair distribution of resources or opportunities through ranking decisions (e.g., job candidates from certain demographics systematically ranked lower)
- **Representational harms**: Harms to how groups are portrayed or perceived, independent of direct resource allocation (e.g., image search for "CEO" returning predominantly white male faces)

### Fairness Definitions

| Fairness Type | Definition | Focus |
|---------------|------------|-------|
| **Individual fairness** | Similar items should be treated similarly | If two candidates have equivalent qualifications, they should receive similar exposure |
| **Group fairness** | Protected groups should receive proportional representation | Subgroups defined by protected attributes should not be systematically disadvantaged |
| **Meritocratic fairness** | Exposure should be proportional to relevance | An item twice as relevant should receive approximately twice the exposure |

### Treatment vs Outcome Fairness

- **Treatment fairness**: The process/algorithm treats individuals equally regardless of protected attributes
- **Outcome fairness**: The results/rankings achieve equitable distributions across groups

> [!warning] Incompatibility
> These fairness definitions can conflict with each other and with relevance optimization. There is no single "fair" ranking—fairness requires explicit value choices about what matters.

## The Fairness-Relevance Trade-off

A fundamental tension exists between:
- **User utility**: Users want the most relevant results first
- **Item provider fairness**: Providers want fair opportunity for exposure
- **Representational balance**: Society may want diverse representation

Research explores this through:
- **Fairness-constrained optimization**: Maximize relevance subject to fairness constraints
- **Re-ranking approaches**: Post-process rankings to improve fairness
- **Stochastic ranking**: Introduce controlled randomization to balance fairness and relevance

## Connections

- Applied as: [[Exposure Fairness]] (specific fairness criterion in IR)
- Contrasted with: [[Emancipatory IR]] (critical approach questions whether fairness fixes are sufficient)
- Related to: [[Explainability]] (understanding why rankings may be unfair)
- Evaluation: Requires extending [[NDCG]], [[MAP]] with fairness-aware metrics

## Appears In

- [[IR-L12 - IR and Society]]
