---
type: concept
aliases: [explainability, explainable IR, XAI, explainable AI]
course: [IR]
tags: [ir-society]
status: complete
---

# Explainability

> [!definition] The Transparency Spectrum
> - **Transparency**: Openness about how a system works, what data it uses, and what decisions it makes. A *system property* that enables external scrutiny.
> - **Interpretability**: The degree to which a human can understand the cause of a decision. Focuses on the model's *internal logic* being comprehensible.
> - **Explainability**: The ability to provide reasons for specific decisions in terms a human can understand. Often involves *post-hoc explanations* of black-box models.

> [!intuition] The Key Distinction
> Transparency is about *openness*, interpretability is about *model comprehensibility*, and explainability is about *justifying specific outputs*. A neural ranker might be transparent (open-source) but not interpretable (too complex), yet still provide explanations ("ranked high because of keyword match").

## Comparison

| Concept | Focus | Question Answered | Example |
|---------|-------|-------------------|---------|
| **Transparency** | System openness | "What does the system do?" | Publishing ranking algorithm details |
| **Interpretability** | Model comprehension | "Why does the model work this way?" | Decision tree with clear rules |
| **Explainability** | Decision justification | "Why this specific output?" | "This result ranked high because..." |

## Why Explainability Matters in IR

### For Users
- Understanding why results appear helps assess reliability
- Explanations build appropriate trust (or distrust)
- Users can provide better feedback if they understand the system

### For Item Providers
- Knowing how rankings work enables fair competition
- Can identify and contest unfair treatment
- Reduces arbitrary power of platforms

### For Regulators and Society
- Enables accountability for harms
- Allows democratic oversight of consequential systems
- Supports informed public debate

## Types of Explanations in IR

| Explanation Type | Description | Example |
|-----------------|-------------|---------|
| **Feature-based** | Highlights important input features | "Ranked high because query terms appear in title" |
| **Example-based** | References similar cases | "Users who clicked X also clicked this" |
| **Contrastive** | Compares to alternatives | "Ranked higher than Y because of factor Z" |
| **Counterfactual** | Describes what would change outcome | "Would rank lower without keyword match" |

## Faithfulness Concerns

> [!warning] Explanation ≠ Truth
> Post-hoc explanations may not accurately reflect how the model actually made decisions. They are *rationalizations*, not mechanistic accounts.

Key concerns:
- **Faithfulness**: Does the explanation reflect the actual decision process?
- **Plausibility**: Is the explanation believable to humans (even if unfaithful)?
- **Completeness**: Does the explanation capture all relevant factors?

### Additional Tensions

- **Gaming risk**: Detailed explanations enable adversarial manipulation of rankings
- **Complexity**: Neural ranking models may be fundamentally difficult to explain faithfully
- **Stakeholder variance**: Different users need different types of explanations

## Connections

- Supports: [[Algorithmic Fairness]] (understanding unfair rankings)
- Related to: [[Misinformation]] (explaining content moderation decisions)
- Critiqued by: [[Emancipatory IR]] (questions power dynamics in who explains to whom)
- Technical context: [[Neural Reranking]], [[Cross-Encoder]], [[Dense Retrieval]] are hard to explain

## Appears In

- [[IR-L12 - IR and Society]]
