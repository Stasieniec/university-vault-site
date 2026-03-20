---
type: lecture
course: IR
week: 6
lecture: 12
topics:
  - "[[Algorithmic Fairness]]"
  - "[[Exposure Fairness]]"
  - "[[Misinformation]]"
  - "[[Explainability]]"
  - "[[Emancipatory IR]]"
  - "[[Critical Information Theory]]"
status: complete
---

# IR and Society

**Guest Lecturer:** Bhaskar Mitra, Microsoft Research
**Course:** Information Retrieval, University of Amsterdam

## Overview & Motivation

Information Retrieval systems are no longer just tools for finding documents—they have become critical infrastructure that shapes public discourse, economic opportunity, and social reality. As [[Information Retrieval]] systems increasingly mediate access to information, jobs, news, and opportunities, researchers must grapple with their broader societal implications.

This lecture introduces **two distinct frames** for thinking about IR's societal impact:

1. **Liberal Approach**: Focuses on technical solutions within existing systems—algorithmic fairness, misinformation detection, and explainability
2. **Critical Approach**: Questions the fundamental power structures underlying IR systems and proposes emancipatory alternatives

Both approaches acknowledge that ranking systems concentrate power and that this power must be examined. The lecture argues that a complete picture requires engaging with both perspectives.

---

## Part 1: Two Frames for IR & Society

### The Core Tension

> [!intuition]
> **Rankings concentrate power.** When a system decides what information appears first, it shapes attention, opportunity, and ultimately reality. The question is not *whether* to engage with this power, but *how*.

The field of "IR & Society" has emerged to address questions about:
- **Fairness**: Who benefits and who is harmed by ranking decisions?
- **Misinformation**: How do retrieval systems amplify or suppress false information?
- **Transparency**: Can users and stakeholders understand how rankings are produced?
- **Power**: Who controls these systems, and whose interests do they serve?

### Liberal Approach

The **liberal approach** works within existing institutional frameworks to make IR systems fairer, more accurate, and more transparent. It accepts the fundamental premise of ranking systems but seeks to optimize them for better societal outcomes.

**Key characteristics:**
- **Technical focus**: Develops algorithms, metrics, and evaluation frameworks
- **Reformist**: Seeks incremental improvement within existing structures
- **Empirical**: Emphasizes measurement, benchmarks, and quantifiable outcomes
- **Industry-aligned**: Often developed in collaboration with technology companies

**Core research areas:**
1. [[Algorithmic Fairness]] in ranking
2. [[Misinformation]] and disinformation detection
3. [[Explainability]], interpretability, and transparency

### Critical Approach

The **critical approach** questions the fundamental assumptions underlying IR systems and the power structures they reinforce. Drawing on critical theory, it asks who benefits from current arrangements and whether alternative configurations are possible.

**Key characteristics:**
- **Structural focus**: Examines power relations, institutions, and political economy
- **Transformative**: Seeks fundamental change rather than optimization
- **Normative**: Explicitly engages with values and visions of a better society
- **Skeptical of industry**: Questions whether corporate interests align with public good

**Core research areas:**
1. [[Emancipatory IR]] frameworks
2. [[Critical Information Theory]]
3. Regulatory capture and Big Tech power
4. Alternative information infrastructures

---

## Part 2: Algorithmic Fairness in Ranking (Liberal Approach)

### Why Fairness Matters in IR

> [!definition]
> **Algorithmic Fairness**: The study of how automated decision systems can be designed to avoid systematic discrimination against individuals or groups based on protected characteristics or other morally relevant factors.

Rankings are not neutral—they allocate a scarce resource: **user attention**. The order in which items appear determines:
- Which job candidates get interviews
- Which news stories shape public opinion
- Which products get purchased
- Which voices are heard in public discourse

### Two Types of Harms

> [!definition]
> **Allocative Harms**: Unfair distribution of resources or opportunities through ranking decisions. The system directly withholds or provides something of value.

Examples of allocative harms:
- Job candidates from certain demographics systematically ranked lower
- Businesses in minority neighborhoods receiving less visibility
- Loan applicants unfairly denied based on proxy variables

> [!definition]
> **Representational Harms**: Harms to how groups are portrayed or perceived, independent of direct resource allocation. These shape stereotypes, norms, and social understanding.

Examples of representational harms:
- Image search for "CEO" returning predominantly white male faces
- Search results reinforcing gender stereotypes for professions
- Autocompletion suggesting negative associations for minority groups

| Harm Type | Nature | Example | Intervention Focus |
|-----------|--------|---------|-------------------|
| **Allocative** | Resource distribution | Job candidates ranked unfairly | Equal opportunity in exposure |
| **Representational** | Perception/stereotypes | Biased image search results | Diverse representation |

### Exposure Fairness

> [!definition]
> **Exposure Fairness**: The principle that items (or their providers) should receive attention/visibility proportional to their relevance or merit, not distorted by position bias or systematic discrimination.

The challenge: position bias means top-ranked items receive disproportionate attention regardless of merit. Combined with any systematic bias in the ranking algorithm, this creates compounding unfairness.

**Key insight from Biega et al. (2018)**: Fairness should be measured not just at a single point in time but across repeated exposures. An item might be fairly ranked in expectation but experience high variance that harms its provider.

### Fairness Metrics

Several quantitative approaches have been proposed:

**Individual Fairness**: Similar items should be treated similarly. If two job candidates have equivalent qualifications, they should receive similar exposure.

**Group Fairness**: Protected groups should receive proportional representation or exposure. Subgroups defined by race, gender, or other protected attributes should not be systematically disadvantaged.

**Meritocratic Fairness**: Exposure should be proportional to relevance. An item that is twice as relevant should receive (approximately) twice the exposure.

> [!warning]
> These fairness definitions can conflict with each other and with relevance optimization. There is no single "fair" ranking—fairness requires explicit value choices about what matters.

### Tension: Fairness vs. Relevance

A fundamental tension exists between:
- **User utility**: Users want the most relevant results first
- **Item provider fairness**: Providers want fair opportunity for exposure
- **Representational balance**: Society may want diverse representation

Research explores this trade-off through:
- **Fairness-constrained optimization**: Maximize relevance subject to fairness constraints
- **Re-ranking approaches**: Post-process rankings to improve fairness
- **Stochastic ranking**: Introduce controlled randomization to balance fairness and relevance

---

## Part 3: Misinformation and Disinformation (Liberal Approach)

### Definitions

> [!definition]
> **Misinformation**: False or inaccurate information, regardless of intent. The spreader may genuinely believe it to be true.

> [!definition]
> **Disinformation**: Deliberately false information spread with intent to deceive. This includes propaganda, hoaxes, and strategic manipulation.

> [!definition]
> **Malinformation**: Genuine information shared with malicious intent, such as leaking private information to cause harm.

| Type | Intent | Truth Value | Example |
|------|--------|-------------|---------|
| **Misinformation** | Unintentional | False | Sharing outdated medical advice |
| **Disinformation** | Deliberate deception | False | State-sponsored propaganda |
| **Malinformation** | Malicious | True | Doxxing, revenge porn |

### IR's Role in the Misinformation Ecosystem

Search engines and recommendation systems play a dual role:

**Amplification risk**: Algorithms may inadvertently promote engaging but false content because:
- Sensational content generates more clicks
- Controversy drives engagement
- Filter bubbles reinforce existing beliefs

**Mitigation opportunity**: IR systems can potentially:
- Downrank or label unreliable sources
- Promote authoritative information
- Provide fact-checking context

### Technical Approaches

**Source credibility assessment**:
- Domain-level reliability scores
- Author expertise evaluation
- Citation and linking patterns

**Content-based detection**:
- Claim verification against knowledge bases
- Stylistic indicators of unreliability
- Contradiction detection across sources

**User-facing interventions**:
- Warning labels on disputed content
- Related articles providing context
- "Read before sharing" prompts

> [!warning]
> **Challenges remain significant:**
> - Misinformation evolves to evade detection
> - "Ground truth" is contested for many claims
> - Interventions may backfire (reactance, distrust)
> - Cultural and linguistic variation in what constitutes misinformation

---

## Part 4: Explainability, Interpretability, and Transparency (Liberal Approach)

### Distinguishing the Concepts

These terms are often used interchangeably but have distinct meanings:

> [!definition]
> **Transparency**: Openness about how a system works, what data it uses, and what decisions it makes. A system property that enables external scrutiny.

> [!definition]
> **Interpretability**: The degree to which a human can understand the cause of a decision. Focuses on the model's internal logic being comprehensible.

> [!definition]
> **Explainability**: The ability to provide reasons for specific decisions in terms a human can understand. Often involves post-hoc explanations of black-box models.

| Concept | Focus | Question Answered | Example |
|---------|-------|-------------------|---------|
| **Transparency** | System openness | "What does the system do?" | Publishing ranking algorithm details |
| **Interpretability** | Model comprehension | "Why does the model work this way?" | Decision tree with clear rules |
| **Explainability** | Decision justification | "Why this specific output?" | "This result ranked high because..." |

### Why These Matter for IR

**For users**:
- Understanding why results appear helps assess reliability
- Explanations build appropriate trust (or distrust)
- Users can provide better feedback if they understand the system

**For item providers**:
- Knowing how rankings work enables fair competition
- Can identify and contest unfair treatment
- Reduces arbitrary power of platforms

**For regulators and society**:
- Enables accountability for harms
- Allows democratic oversight of consequential systems
- Supports informed public debate

### Types of Explanations in IR

**Feature-based explanations**: "This document ranked highly because it contains the query terms in the title."

**Example-based explanations**: "Users who clicked on X also clicked on this result."

**Contrastive explanations**: "This result ranked higher than that one because of factor Y."

**Counterfactual explanations**: "If the document had property Z, it would have ranked lower."

### Tensions and Limitations

> [!warning]
> **Explanation ≠ Truth**: Post-hoc explanations may not accurately reflect how the model actually made decisions. They are rationalizations, not mechanistic accounts.

**Gaming risk**: Detailed explanations enable adversarial manipulation of rankings.

**Complexity**: Neural ranking models may be fundamentally difficult to explain faithfully.

**Whose understanding?**: Different stakeholders need different types of explanations.

---

## Part 5: Big Tech and Regulatory Capture (Critical Approach)

### The Concentration of Power

The critical approach begins by examining who controls IR systems and whose interests they serve:

**Market concentration**: A small number of companies dominate search, social media, and recommendation:
- Google controls ~92% of global search
- Meta dominates social media in many markets
- Amazon dominates product search and e-commerce

**Information gatekeeping**: These platforms determine what information billions of people see, creating unprecedented private power over public discourse.

### Regulatory Capture

> [!definition]
> **Regulatory Capture**: When regulatory agencies, created to act in the public interest, instead advance the commercial or political interests of the industries they are supposed to regulate.

**Mechanisms of capture in tech**:
- **Revolving door**: Personnel move between industry and regulatory positions
- **Information asymmetry**: Regulators depend on industry for technical expertise
- **Lobbying**: Massive spending to influence policy
- **Funding of research**: Industry shapes academic agendas through grants

**Examples**:
- Industry-funded "fairness" research that focuses on technical fixes rather than structural change
- Self-regulatory frameworks that forestall stricter government regulation
- Standards bodies dominated by industry representatives

### The Critique of Liberal Approaches

The critical perspective argues that liberal approaches to fairness, misinformation, and explainability may:

1. **Legitimate existing power structures**: By proposing technical fixes, they suggest the fundamental system is sound
2. **Divert attention from structural issues**: Focus on algorithmic bias rather than concentrated ownership
3. **Enable "ethics washing"**: Companies can claim to address concerns without meaningful change
4. **Reinforce technocratic control**: Solutions remain in the hands of technical experts, not affected communities

> [!intuition]
> **The critical question**: Are we making a harmful system slightly less harmful, or should we be building fundamentally different systems?

---

## Part 6: Emancipatory IR (Critical Approach)

### The Emancipatory Framework

> [!definition]
> **Emancipatory IR**: An approach to information retrieval that explicitly aims to reduce domination, increase human autonomy, and serve the interests of marginalized communities rather than powerful institutions.

This framework draws on critical theory traditions including:
- **Frankfurt School critical theory**: Questioning whose interests systems serve
- **Feminist epistemology**: Centering marginalized perspectives
- **Postcolonial theory**: Examining how IR systems encode Western/colonial assumptions
- **Participatory design**: Involving affected communities in system design

### Core Principles

**1. Question the Neutral Stance**

Traditional IR claims to neutrally retrieve "relevant" information. Emancipatory IR asks:
- Relevant to whom?
- Who defined relevance?
- Whose knowledge counts as information?

**2. Center Marginalized Perspectives**

Rather than treating "fairness" as a constraint on optimization, center the needs of those most likely to be harmed:
- Design with, not for, affected communities
- Prioritize reducing harm over maximizing engagement
- Recognize that "users" are not a homogeneous group

**3. Examine Political Economy**

Analyze how IR systems relate to:
- Labor conditions (content moderation, data labeling)
- Economic concentration (platform monopolies)
- Surveillance capitalism (data extraction)
- Global inequalities (whose languages, whose knowledge)

**4. Prefigurative Design**

Build systems that embody desired social relations:
- Community-controlled information infrastructure
- Cooperative ownership models
- Transparent and accountable governance

### Alternative Visions

**Community-controlled search**: Search engines governed by and accountable to user communities rather than shareholders.

**Federated systems**: Decentralized infrastructure that prevents concentration of power.

**Solidarity-based design**: Systems explicitly designed to support mutual aid and collective action.

**Epistemic justice**: IR systems that recognize and value diverse ways of knowing.

---

## Part 7: Synthesis and Future Directions

### Complementary Perspectives

While liberal and critical approaches differ fundamentally, they can inform each other:

| Aspect | Liberal Contribution | Critical Contribution |
|--------|---------------------|----------------------|
| **Fairness** | Quantitative metrics, algorithms | Questions whose definition of fairness |
| **Misinformation** | Detection techniques | Asks who decides what's "true" |
| **Explainability** | Technical methods | Examines power in explanation |
| **Change theory** | Incremental improvement | Structural transformation |

### Research Agenda

**Near-term (liberal frame)**:
- Better fairness metrics that account for intersectionality
- More robust misinformation detection across languages and cultures
- Explanations that are faithful to model behavior
- Auditing tools for external accountability

**Long-term (critical frame)**:
- Alternative ownership and governance models for information infrastructure
- Participatory design methodologies for IR
- Non-Western and indigenous approaches to information organization
- Decentralized and community-controlled systems

### Responsibilities of IR Researchers

Regardless of which frame resonates, IR researchers should:

1. **Acknowledge positionality**: Recognize that technical choices embody values
2. **Engage with affected communities**: Research "on" people vs. research "with" people
3. **Consider structural context**: Individual algorithmic fixes exist within larger systems
4. **Be humble about solutions**: Unintended consequences are common
5. **Support accountability**: Enable external scrutiny of systems

---

## Key Takeaways & Summary

> [!summary]
>
> ### Two Frames for IR & Society
>
> **Liberal Approach:**
> - Works within existing systems to make them fairer and more transparent
> - Focuses on algorithmic fairness, misinformation detection, and explainability
> - Emphasizes technical solutions, metrics, and empirical research
> - Seeks incremental improvement through industry collaboration
>
> **Critical Approach:**
> - Questions fundamental power structures underlying IR systems
> - Examines regulatory capture, corporate concentration, and whose interests are served
> - Proposes emancipatory alternatives centered on marginalized communities
> - Seeks structural transformation rather than optimization
>
> ### Key Concepts
>
> **Fairness:**
> - **Allocative harms**: Unfair distribution of resources/opportunities
> - **Representational harms**: Harms to how groups are portrayed
> - **Exposure fairness**: Attention proportional to merit
>
> **Information Integrity:**
> - **Misinformation**: False information (unintentional)
> - **Disinformation**: Deliberately false information
> - **Malinformation**: True information shared maliciously
>
> **Transparency Spectrum:**
> - **Transparency**: System openness
> - **Interpretability**: Model comprehensibility
> - **Explainability**: Decision justification
>
> **Critical Concepts:**
> - **Regulatory capture**: Industry influencing its own regulation
> - **Emancipatory IR**: IR aimed at reducing domination and increasing autonomy
> - **Prefigurative design**: Building systems that embody desired social relations
>
> ### Core Insight
>
> Rankings concentrate power. The question is not whether to engage with this power but how. Both liberal (reformist) and critical (transformative) approaches offer valuable perspectives, and a complete picture requires engaging with both.

---

## References

### Key Papers and Researchers

**Algorithmic Fairness:**
- Biega, A. J., Gummadi, K. P., & Weikum, G. (2018). Equity of attention: Amortizing individual fairness in rankings. *SIGIR*.
- Singh, A., & Joachims, T. (2018). Fairness of exposure in rankings. *KDD*.
- Zehlike, M., Bonchi, F., Castillo, C., Hajian, S., Megahed, M., & Baeza-Yates, R. (2017). FA*IR: A fair top-k ranking algorithm. *CIKM*.

**Misinformation:**
- Lazer, D. M., et al. (2018). The science of fake news. *Science*, 359(6380), 1094-1096.
- Wardle, C., & Derakhshan, H. (2017). Information disorder: Toward an interdisciplinary framework for research and policymaking. *Council of Europe Report*.

**Explainability:**
- Lipton, Z. C. (2018). The mythos of model interpretability. *Queue*, 16(3), 31-57.
- Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1-38.

**Critical Perspectives:**
- Noble, S. U. (2018). *Algorithms of oppression: How search engines reinforce racism*. NYU Press.
- Zuboff, S. (2019). *The age of surveillance capitalism*. Public Affairs.
- Eubanks, V. (2018). *Automating inequality*. St. Martin's Press.

**Emancipatory Approaches:**
- Costanza-Chock, S. (2020). *Design justice: Community-led practices to build the worlds we need*. MIT Press.
- D'Ignazio, C., & Klein, L. F. (2020). *Data feminism*. MIT Press.

### Lecture Source

This lecture is based on the guest lecture by **Bhaskar Mitra** (Microsoft Research) on IR and Society, covering both liberal and critical approaches to understanding the societal implications of information retrieval systems.
