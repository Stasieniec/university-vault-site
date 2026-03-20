---
type: concept
aliases: [misinformation, disinformation, malinformation, fake news, information disorder]
course: [IR]
tags: [ir-society]
status: complete
---

# Misinformation

> [!definition] Information Disorder Taxonomy
> - **Misinformation**: False or inaccurate information, regardless of intent. The spreader may genuinely believe it to be true.
> - **Disinformation**: Deliberately false information spread with intent to deceive. Includes propaganda, hoaxes, and strategic manipulation.
> - **Malinformation**: Genuine information shared with malicious intent, such as leaking private information to cause harm.

> [!intuition] The Key Distinction
> The difference lies in *intent* and *truth value*: misinformation is accidentally wrong, disinformation is deliberately wrong, and malinformation is deliberately harmful even when true.

## Key Properties / Variants

| Type | Intent | Truth Value | Example |
|------|--------|-------------|---------|
| **Misinformation** | Unintentional | False | Sharing outdated medical advice |
| **Disinformation** | Deliberate deception | False | State-sponsored propaganda |
| **Malinformation** | Malicious | True | Doxxing, revenge porn |

## IR's Role in the Information Ecosystem

### Amplification Risks

IR systems may inadvertently promote false content because:
- **Engagement optimization**: Sensational content generates more clicks
- **Controversy bias**: Controversial topics drive engagement
- **Filter bubbles**: Personalization reinforces existing beliefs
- **Feedback loops**: Popular content becomes more visible, amplifying initial spread

### Mitigation Opportunities

IR systems can potentially:
- **Downrank** unreliable sources in search results
- **Promote** authoritative information for sensitive queries
- **Label** content with credibility indicators
- **Diversify** results to break filter bubbles

## Technical Approaches

### Source Credibility Assessment
- Domain-level reliability scores
- Author expertise evaluation
- Citation and linking patterns
- Historical accuracy tracking

### Content-Based Detection
- Claim verification against knowledge bases
- Stylistic indicators of unreliability
- Contradiction detection across sources
- Multimodal analysis (text, images, metadata)

### User-Facing Interventions
- Warning labels on disputed content
- Related articles providing context
- "Read before sharing" friction
- Fact-check panels

> [!warning] Significant Challenges
> - Misinformation evolves to evade detection
> - "Ground truth" is contested for many claims
> - Interventions may backfire (reactance, distrust)
> - Cultural and linguistic variation in what constitutes misinformation
> - Who decides what is "true"? (epistemological challenge)

## Connections

- Related to: [[Algorithmic Fairness]] (who is harmed by misinformation spread)
- Requires: [[Explainability]] (explaining why content was flagged)
- Critiqued by: [[Emancipatory IR]] (questions who controls truth-determination)
- Technical approaches use: [[BERT for IR]], [[Cross-Encoder]] for claim verification

## Appears In

- [[IR-L12 - IR and Society]]
