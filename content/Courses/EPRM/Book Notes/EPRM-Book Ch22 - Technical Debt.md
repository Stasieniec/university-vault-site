---
type: book-chapter
course: 5204EPRM6Y
book: "Machine Learning in Production: From Models to Products"
chapter: 22
sections: ["Deliberate and Prudent Technical Debt", "Technical Debt in Machine-Learning Projects", "Managing Technical Debt"]
topics:
  - "[[MLOps]]"
  - "[[Clean Code]]"
status: complete
---

# EPRM-Book Ch22: Technical Debt

> [!abstract] In short
> Technical debt is the metaphor for taking a shortcut now (a faster release, more features) and paying for it later in rework, slower development or operating costs, with interest. Debt can be deliberate or inadvertent, prudent or reckless; the goal is debt that is deliberate and prudent, then tracked and repaid.
>
> ML makes it very easy to pile up debt without noticing: using ML where it is not needed, and skipping the engineering practices that keep data, infrastructure, versions, monitoring, code, architecture and requirements under control. Christian Kästner's advice: decide about debt consciously, make the right thing the easy thing, and track every debt you take on.
>
> Last chapter of the book part "Process and Teams", assigned in week 1 with [[EPRM-Book Ch20 - Data Science and Software Engineering Process Models]] and [[EPRM-Book Ch21 - Interdisciplinary Teams]].

## The metaphor

A loan gives you money now, repaid later with interest. In software, some actions give short-term benefits (features, earlier releases) at the price of lower productivity, rework or operating costs later. The Dagstuhl 16162 definition:

> [!definition] Technical debt
> "Design or implementation constructs that are expedient in the short term, but set up a technical context that can make a future change more costly or impossible."

Like financial debt, it accumulates, and it can suffocate a project when nothing new can be built before old decisions are fixed. Its strength is that non-developers understand it: it explains to managers and customers why time on infrastructure and maintenance now is worth more than one more feature. Kästner calls it "management-compatible language for software developers".

The chapter's running example: **sidewalk delivery robots**, suitcase-sized robots that carry restaurant food to homes, navigating from cameras and GPS among pedestrians, with human operators who can take over remotely.

## Fowler's technical debt quadrants

Martin Fowler's technical debt quadrants classify debt on two dimensions.

**Deliberate or inadvertent.** Deliberate debt is a conscious shortcut. The robot team deploys an obstacle-detection model by copying data by hand, running a notebook and copying the model to the robot, knowing this will slow future updates, make experiments in production harder and invite subtle faults, because a first prototype matters more right now. Inadvertent debt comes from not understanding the consequences of a decision or not knowing better practices: a team unaware of pipeline automation only discovers later that hours go into copying files and remembering which scripts to run.

**Prudent or reckless.** Prudent debt is an investment. In a hot market with many competitors, deploying many robots early with weak navigation models, paying heavily for remote human operators, may win investors and customers. Companies in markets with strong network effects (social networks, video conferencing, dating sites, app stores) take similar shortcuts to grow their user base fast. Reckless debt is credit-card impulse spending: skipping safety testing of the perception models saves time but risks accidents, bad publicity, liability and losing the licence to operate.

| | **Reckless** | **Prudent** |
|---|---|---|
| **Deliberate** | "We don't have time to plan for mistakes" | "We must release now before the competition and deal with slow manual releases for now" |
| **Inadvertent** | "What's data drift?" | "We didn't know about data versioning, and in retrospect it's good we didn't waste effort on it early on" |

Ideally all debt is deliberate and prudent, and deliberate-but-reckless decisions are rare. Inadvertent debt can turn out prudent by luck: weak early models that forced heavy human intervention may have produced more training data and driven a competitor out. In practice much debt is inadvertent and reckless, from inexperience or pressure to deliver.

## Where ML projects run up debt

Sculley and other Google engineers framed ML's extra maintenance costs as debt in an influential paper first titled "Machine Learning: The High Interest Credit Card of Technical Debt" (published as "Hidden Technical Debt in Machine Learning Systems", 2015, often seen as a starting point of the MLOps movement).

**Using ML where it is not needed** is debt in itself. Hype makes ML look like a quick fix, but production-quality ML components are expensive to engineer and maintain. Using ML as a gimmick, or where heuristics or established non-ML solutions suffice, without anticipating the later costs, can be inadvertent, reckless debt.

**Skipping good engineering practices** is the other source. Hazard analysis, architecture design, robust pipelines, testing in production, fairness evaluation, threat modelling and provenance tracking all cost effort up front. Skip them and you get:

- **Data debt.** Data creates dependencies on whoever produces it: the robot relies on third-party map and weather data, labelled training data from other researchers, and cameras that are hopefully maintained and not tampered with. Without schemas, distribution-shift monitoring and provenance tracking, changes in data go unnoticed until customer complaints pile up.
- **Infrastructure debt.** Copying a notebook-trained model into production is easy but brittle. Without automated deployment the team updates robots at most monthly, and some robots run old models because a failed update went unnoticed.
- **Versioning debt.** Ad hoc notebook work makes results hard to reproduce and changes hard to trace. Without data versioning you cannot find out whether (perhaps malicious) changes in training data made the robot stop recognising dogs as obstacles.
- **Observability debt.** Offline evaluation is easy; monitoring model quality in production needs infrastructure early. Without it, the team guesses at fixes and spends days watching a few robots by hand.
- **Code quality debt.** The usual problems (postponed cleanup, duplication, poor readability and structure, missing tests) plus ML-specific ones: dead experimental code, glued-together scripts, feature extraction implemented twice, poorly named variables. Developers cannot find every place a feature extraction patch must go, or which scripts to run in what order. The fix is an explicit cleanup when the pipeline is ready for release.
- **Architectural and test debt.** Products often chain models, some consuming others' outputs. Without model and integration testing, nobody can tell which of five recently updated models makes the robot drift into bike lanes.
- **Requirements debt.** Requirements analysis can catch harmful feedback loops before they ship. A robot optimised for delivery speed may drive aggressively; pedestrians step aside, the robot goes faster still, and the speed gets rewarded, while the neighbourhood gets less safe.

All of these are the long-term price of skipping responsible engineering. Skipping steps to build a prototype fast can be prudent, but the debt should be repaid. Framing good practice as debt helps convince managers to fund requirements analysis, testing and monitoring, or to hire for it.

## Managing debt

- **Make it a decision.** Debt is not bad in itself. Teams should weigh the long-term cost of, say, skipping data quality tests or monitoring against the short-term gain and decide deliberately. That requires knowing the good practices and tools in the first place.
- **Know which debt is hard to repay.** Requirements and design debt is much harder to fix than debt from low code quality or missing automation. Skipping risk analysis may leave a fundamentally flawed design (thousands of robots ordered before realising they needed redundant sensors). Missing monitoring can be added later without redesigning the system.
- **Make the right thing easy.** Education prevents much inadvertent debt. If version control is team culture, newcomers will not run up versioning debt; if risk analysis is a normal process step, or someone always asks about fairness in meetings, people will not skip it without reason. Mature organisations can provide or mandate shared infrastructure that versions data and tracks dependencies and provenance automatically.
- **Track deliberate debt.** At minimum, put the deferred work in the issue tracker, as a backlog item or on the product roadmap. Ideally assign a person responsible, monitor the extra cost the debt causes, and set goals for when to repay it. Some organisations hold "fix-it" days or weeks dedicated to the debt backlog.

The tension between shipping fast and building for maintainability never goes away. The metaphor gives teams the vocabulary to argue for cleanup, maintenance and infrastructure time.

## What to be able to do

- Define technical debt and place a decision in Fowler's quadrants, with a reason.
- Name the ML-specific kinds of debt and give an example of each from your own project.
- Explain why requirements and design debt is harder to repay than code or automation debt.
- When you take a shortcut in the course project, record it as tracked debt and say when you would repay it.

## Flashcards

> [!card]- What is technical debt?
> Shortcuts that are expedient now but make future changes more costly or impossible.

> [!card]- Why is the technical debt metaphor useful with managers?
> It explains in familiar terms why maintenance and infrastructure work now avoids larger costs later.

> [!card]- What are the two dimensions of Fowler's technical debt quadrants?
> Deliberate versus inadvertent, and reckless versus prudent.

> [!card]- What makes technical debt inadvertent?
> Not understanding the consequences of a decision or not knowing better practices.

> [!card]- Give an example of prudent technical debt for delivery robots.
> Deploying early with weak navigation models, paying for remote human operators, to win a competitive market.

> [!card]- Give an example of reckless technical debt for delivery robots.
> Skipping safety testing of perception models, risking accidents, liability and lost licences.

> [!card]- Which quadrant is ideal for technical debt?
> Deliberate and prudent.

> [!card]- Why is using ML where it is not needed a form of technical debt?
> Production ML is expensive to engineer and maintain, unlike heuristics or established non-ML solutions.

> [!card]- What is data debt?
> Unchecked dependencies on external and internal data whose quality can change unnoticed without schemas, monitoring and provenance.

> [!card]- What is infrastructure debt in ML?
> Manual, brittle model deployment, making updates rare and leaving machines on inconsistent model versions.

> [!card]- What does versioning debt prevent you from doing?
> Reproducing results and tracing what changed in data or training code when performance drops.

> [!card]- Why does observability debt slow down fixing production problems?
> Without production monitoring of model quality, the team guesses at fixes and spends days watching robots by hand.

> [!card]- Name ML-specific sources of code quality debt.
> Dead experimental code, glued-together scripts, duplicate feature extraction, poorly named variables.

> [!card]- Why does chaining models create architectural and test debt?
> Without model and integration tests you cannot tell which model causes a new system behaviour.

> [!card]- How can skipped requirements analysis create debt, in the delivery robot example?
> A speed-optimised robot drives aggressively, pedestrians yield, speed is rewarded: a harmful feedback loop.

> [!card]- Which is harder to repay: requirements and design debt, or code quality and automation debt?
> Requirements and design debt: it can leave a fundamentally flawed design, while monitoring or automation can be added later.

> [!card]- What is the minimum for managing deliberately taken debt?
> Track the deferred work in an issue tracker, backlog or roadmap.

## Links

- **Course:** [[EPRM - Overview|Course overview]]
- **Lecture:** [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]] (clean code, unit tests and the course's example pipeline)
- **Same book part:** [[EPRM-Book Ch20 - Data Science and Software Engineering Process Models]] · [[EPRM-Book Ch21 - Interdisciplinary Teams]]
- **Concepts:** [[MLOps]] · [[Clean Code]]
- **Source:** Christian Kästner, *Machine Learning in Production: From Models to Products* (MIT Press), Chapter 22 "Technical Debt", online edition, https://mlip-cmu.github.io/book/22-technical-debt.html (CC BY-NC-ND 4.0).
- **Further (cited in the chapter):** Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015). Martin Fowler, "Technical Debt Quadrant" (blog post).
