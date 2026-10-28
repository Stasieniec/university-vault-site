---
type: book-chapter
course: 5204EPRM6Y
book: "Machine Learning in Production: From Models to Products"
chapter: 21
sections: ["Unicorns Are Not Enough", "Conflicts Within and Between Teams are Common", "Coordination Costs", "Conflicting Goals and T-Shaped People", "Groupthink", "Team Structure and Allocating Experts", "Learning from DevOps and MLOps Culture"]
topics:
  - "[[MLOps]]"
status: complete
---

# EPRM-Book Ch21: Interdisciplinary Teams

> [!abstract] In short
> Christian Kästner's chapter asks why building ML products needs teams of people with different expertise, why those teams so often clash, and what helps. No single person (the "unicorn") has all the skills, so teams are unavoidable, and they bring coordination costs, conflicting goals and groupthink. ML makes these harder, because ML projects need more diverse expertise and model interfaces are unstable and hard to document. The remedies: deliberate team structure, T-shaped people who understand each other's concerns, a culture of constructive conflict, and the DevOps and MLOps model of joint goals supported by joint tools.
>
> It is the chapter the course assigns under Kästner's book part "Process and Teams", and it backs the week 1 argument in [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]] that handovers between siloed roles kill ML projects.

The chapter runs one scenario throughout: a large short-video platform wants to predict whether a user is developing signs of depression from their behaviour and posts, then intervene (filtering harmful content, showing information, even informing parents of young users). The project needs software engineers, data scientists, clinical experts, social scientists, designers, operators, security and legal experts: dozens of people, some outside the company.

## Why a unicorn is not enough

A **unicorn** is the mythical developer who combines statistics, engineering skills and domain expertise (and, depending on who is asking, business sense, people skills and data collection too). Such people are rare, expensive, or do not exist. Even with one you would still need a team, for two reasons:

- **Division of labour:** most products are too big for one person to build.
- **Division of expertise:** nobody, a unicorn included, has the capacity to be a deep expert in everything relevant.

So teams are inevitable, and they have to be interdisciplinary.

## Where conflicts come from

Conflicts arise **within teams** (members with conflicting goals tied to their roles, or who simply do not understand each other) and, more easily, **between teams**: when responsibilities are unclear, when teams are siloed by role (a software team using models thrown over the wall by a separate data science team), when incentives conflict, or when one team sets priorities without listening. A common frustration: experts in security, UI, privacy, safety, fairness and law, but also data scientists and software engineers, want to shape the project early and are only consulted at the end.

Two anonymised real organisations illustrate it:

- **Organisation A:** four data scientists build a model for an external customer, whose own engineers (with little ML experience) integrate it. The data scientists cannot shape data collection or requirements, spend their time educating the customer about data quality and fighting unrealistic expectations, struggle to reach the customer's domain experts, and run complex infrastructure themselves when they would rather do less engineering. Nobody gave them latency, fairness or robustness requirements, so they did not feel responsible for those qualities.
- **Organisation B:** one team of eight, mostly software engineers plus a single data scientist. The team splits into cliques; the data scientist feels isolated and does not know how the model fits the system; the engineers struggle to deploy ML code and feel they must teach others about code quality; everyone focuses on their own part, with little integration testing or joint evaluation of the product; the data scientist is absent when the team lead talks to the client, so unrealistic requests are not pushed back early. The client ends up happy with the functionality and unhappy with the neglected user interface.

## Coordination costs

Communication grows faster than team size. A group of $n$ people has

$$\frac{n(n-1)}{2}$$

possible communication links. Seven people in one group have 21. Split them into teams of 3 and 4 that talk to each other through one representative each, and the count drops to $3 + 6 + 1 = 10$. Larger projects therefore need internal structure with smaller teams.

**Information hiding** is the traditional technical way to need less communication: a component is represented by its interface, and others call the interface without knowing the implementation. Teams then only coordinate when they agree on the interface and when they change it. For the depression project, a stable interface for the prediction model and its explanations would let the UI team design interventions without constantly talking to the model team.

**Conway's law** (also called socio-technical congruence) says system structure and team structure tend to align, mostly because people design components to fit the existing teams: give a compiler to three groups and you get a three-pass compiler. Studies associate a mismatch, one component maintained by people from different teams, with slower development and more defects. So assign each component to one team.

### Why divide and conquer is harder with ML

- **Interfaces are rarely stable.** Data science is exploratory. In the depression project the team may spend months testing whether detection is feasible at all before anyone can write a model interface, and may only late on discover what explanations it can offer, which affects the UI designers.
- **Interface descriptions are incomplete.** Delegating "data collection and cleaning" to a separate team sounds clean, but specifying it well needs knowledge of social media, depression, the modellers' needs, data protection law and what telemetry the UI can collect. A poorly defined task gives the illusion of progress in isolation. And we lack good ways to document what a model can do and where it fails.
- **Many system qualities cannot be localised in a component.** Safety, security, usability, fairness and operating cost emerge from how components interact in their environment. Each team may assume someone else handles them (diffusion of responsibility). A fairness audit of the model alone misses how teen girls' usage patterns combine with telemetry and reporting choices.
- **Expertise does not line up with modules.** The model team needs a software engineer; the UI team benefits from ML and operations knowledge to anticipate mistakes. So build an interdisciplinary team per component instead of sorting people into teams by specialty (siloing).

### Awareness

The complementary strategy to hiding information is **broadcasting it** to everyone who may be affected: cross-team meetings, mailing lists, chat announcements, following another team's repository activity or issue tracker. Broadcasting everything causes information overload, so it works best targeted: warning affected teams about an upcoming change in data quality, automated notifications for model updates, a data scientist telling every team using the model that a new architecture will change latency. Personal contacts break when people leave, so formalise them (a documented list of whom to ask about how user interactions are logged, for instance). Awareness works worse across company boundaries, where formal documentation and channels do better.

Kästner's judgement: information hiding seems less effective in ML projects than in traditional ones, so ML teams should invest either in stabilising and documenting interfaces or in deliberate awareness mechanisms, usually some of both.

## Conflicting goals and T-shaped people

Team members optimise for different things. In the depression project, data scientists and lawyers disagree about which data may be analysed; operators want small, low-latency models; UX designers value reliable confidence estimates and explanations over a higher average accuracy, because those let them design mitigations for wrong predictions. Performance reviews reinforce the split: engineers measured by features shipped, operators by downtime, data scientists by accuracy gains, while lawyers and fairness advocates struggle to show their value because it shows up as problems that never happened.

Resolving this starts with **making goals visible**: say what you are trying to achieve and how you measure success, and try to understand why others' goals matter to them. That needs some grasp of their field.

> [!definition] T-shaped person
> Deep expertise in one area combined with broad general knowledge of the others the project involves, enough to understand and talk to their specialists. (Recruiters also speak of π-shaped people, deep in two areas.)

A T-shaped data scientist in the depression project might be an expert in multi-modal sentiment analysis, learn about clinical depression, have basic software engineering skills, know the challenges of running distributed systems, and know how to talk to a privacy lawyer. With such people a team can agree project-level goals: which privacy rules are hard constraints, when an explainability request outweighs accuracy, which latency targets are realistic. Contributions should then be judged against the project's goals, so a data scientist who trades some accuracy for explainability gets credit for it. Conflict, made explicit, is a useful part of deciding.

## Groupthink

**Groupthink** is a group's tendency to minimise conflict by not exploring alternatives, suppressing dissent and isolating itself from outside views. The group goes with the first suggestion or the most senior person's. Symptoms: overconfidence, closed minds, pressure to conform, self-censorship.

In software it shows up in hype-driven technology choices: the depression team prompts a well-known commercial LLM to analyse video text, without considering other techniques, because a manager suggested it and it seems modern, while members who think a cheaper, more explainable, equally accurate technique would do better stay quiet. It also shows up in dismissed risks: fairness or safety concerns are not raised because the leads always brush them off. Causes include homogeneous teams (data scientists often share educational backgrounds), cultures that isolate decision makers, and pressure from competitors, recent failures or public scrutiny.

Interventions: hire more diverse teams, involve outside experts, ask everyone for their opinion, let high-status members speak last, moderate meetings actively, always explore a second solution, and use a devil's advocate (someone who argues the contrary view for the sake of debate).

## Team structure and scarce experts

- **Matrix organisation:** experts belong to departments (engineering, data science, operations, legal) and are lent to projects, sometimes several at once. It is flexible, spreads knowledge and builds deep expertise within departments, but people answer to department and project at once, which clashes near deadlines.
- **Project organisation:** people belong to one project only. They can focus, but learn less from peers in their field, may rebuild what exists elsewhere, and a project cannot justify a full-time specialist (a privacy lawyer) for a small share of work.
- **Hybrids** are common: mostly project teams, with rare experts (privacy, security) in departments and cross-team communities for exchange (Spotify's "chapters" and "guilds").

**The security lesson.** Software engineering went through this with security about two decades earlier. Teaching every developer to be a security expert failed: training felt like a compliance chore. What worked was light security literacy, enough to recognise when security matters and whom to call, plus tools and process checkpoints (secure libraries, automated checks, expert sign-off on designs). Kästner proposes the same for fairness, privacy, UX and operations in ML teams: T-shaped people who know when to bring in the expert.

## What DevOps and MLOps teach about collaboration

Developers historically wanted new features fast; operators wanted fewer outages and lower server costs. Operators complained about code that lacked dependencies or was too slow for production; developers complained that operators were slow and conservative.

**DevOps** replaced "us versus them" with a **culture of collaboration**: a joint goal (deliver value frequently and learn from rapid feedback), shared vocabulary, and joint tools where the two roles' work meets: containers, test automation, versioning, A/B testing infrastructure, telemetry and monitoring. Both sides gain:

- **Developers** take on extra work (automated tests, telemetry, containerising their code) and in return see their code live within minutes, make their own deployment decisions and get real user feedback.
- **Operators** stop installing dependencies by hand and rolling back failed updates from backups at 3 a.m., and invest in automated orchestration, canary releases and experiment infrastructure.

**MLOps** applies the same deal between data scientists and operators: data scientists learn to package models in a production-ready format (with a library such as BentoML) and gain the ability to experiment in production. Shared tools for pipelines, model versioning and packaging force each side to learn a bit of the other's role, which is exactly what makes people T-shaped. See [[MLOps]].

Changing culture is hard: inertia ("this is how we always did it"), the learning cost of the tooling, managers afraid of losing importance, people afraid of automating away their jobs. Partial adoption can make things worse: release automation without automated tests, or orchestration without monitoring, adds risk. What helps: supportive management, education, experienced hires, and a small pilot team that shows the benefits on one component, starting from current pain points such as automated model deployment.

**Beyond MLOps.** Kästner speculates about a "DevML" culture between software engineers and data scientists, built around the whole product: negotiating model requirements together, planning for mistakes, documenting and testing data and model quality, integration testing ML and non-ML components, versioning them together. Shared documentation such as model cards would act as the joint tool: asked to fill in a full template, data scientists consider qualities engineers care about.

## What to be able to do

- Explain why teams are unavoidable even if a unicorn exists, and use the communication-links formula to argue for team structure.
- Apply information hiding and Conway's law to your own project: which component belongs to whom, and what each interface promises.
- Name the reasons ML makes divide and conquer harder, with an example of each.
- Explain T-shaped people, groupthink and its interventions, and matrix versus project organisation.
- Explain how DevOps and MLOps turn conflicting roles into collaboration through joint goals and joint tools, and what each side gains.

## Flashcards

> [!card]- What is a "unicorn" in discussions of ML teams?
> A rare developer combining statistics, engineering and domain expertise in one person.

> [!card]- Give the two reasons Kästner says teams are needed even if you hire a unicorn.
> Division of labour (systems are too big for one person) and division of expertise.

> [!card]- How many communication links does a group of n people have?
> n(n-1)/2.

> [!card]- Seven people split into teams of 3 and 4, linked by one representative each: how many communication links?
> 10 (3 + 6 + 1), down from 21 in one group.

> [!card]- What is information hiding?
> Others use a component only through its interface, without knowing its implementation, so teams coordinate only on the interface.

> [!card]- What does Conway's law (socio-technical congruence) say?
> System structure tends to mirror team structure, so components should align with team boundaries.

> [!card]- Why are ML component interfaces rarely stable?
> Data science is exploratory: what the model can do is only discovered as the project evolves.

> [!card]- Why can't fairness be assigned to the model team alone?
> Fairness emerges from how all components interact in their environment, so no single component can assess it.

> [!card]- Why does Kästner recommend an interdisciplinary team per component over teams sorted by specialty?
> Expertise does not line up with modules: a model team needs engineering skills, a UI team ML knowledge.

> [!card]- What is awareness as a coordination strategy?
> Actively broadcasting changes and decisions to everyone who might be affected.

> [!card]- Why must awareness mechanisms be targeted?
> Broadcasting everything causes information overload, and important news gets lost in the noise.

> [!card]- What is a T-shaped person?
> Deep expertise in one area plus broad basic knowledge of the others, enough to talk to their specialists.

> [!card]- How do performance reviews deepen goal conflicts in ML teams?
> Each role is measured on its own metric (features, downtime, accuracy) instead of the project's goals.

> [!card]- What is groupthink?
> Minimising conflict by not exploring alternatives, suppressing dissent and isolating the group from outside views.

> [!card]- Name five interventions against groupthink.
> Diverse teams, outside experts, high-status members speaking last, devil's advocate, exploring a second solution.

> [!card]- What is the difference between a matrix and a project organisation?
> Matrix: experts belong to departments and are lent to projects. Project: people belong to one project only.

> [!card]- What did software security teach about spreading rare expertise across teams?
> Light literacy, so developers recognise when to call an expert, works better than making everyone an expert.

> [!card]- How does DevOps turn developers and operators from rivals into collaborators?
> Joint goal of frequent delivery with rapid feedback, shared vocabulary, and joint tools such as containers and monitoring.

> [!card]- What do data scientists gain from MLOps in exchange for learning model packaging?
> The ability to experiment rapidly in production.

## Links

- **Course:** [[EPRM - Overview|Course overview]]
- **Lecture:** [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]] (data roles, handovers and the full-stack ML engineer)
- **Concepts:** [[MLOps]]
- **Source:** Christian Kästner, *Machine Learning in Production: From Models to Products* (MIT Press), Chapter 21 "Interdisciplinary Teams", online edition last updated 2024-06-17, https://mlip-cmu.github.io/book/21-interdisciplinary-teams.html (CC BY-NC-ND 4.0).
- **Same book part "Process and Teams":** [[EPRM-Book Ch20 - Data Science and Software Engineering Process Models]] (exploratory data science versus planned software processes, model-first versus system-first development) · [[EPRM-Book Ch22 - Technical Debt]]
- **Further (cited in the chapter):** Nahar, Zhou, Lewis and Kästner, "Collaboration Challenges in Building ML-Enabled Systems" (ICSE 2022). Conway, "How Do Committees Invent?" (1968). Parnas, "On the Criteria to Be Used in Decomposing Systems into Modules" (1972). Kniberg and Ivarsson, "Scaling Agile @ Spotify" (2012).
