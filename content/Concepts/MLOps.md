---
type: concept
aliases: [MLOps lifecycle, ML lifecycle]
course: [EPRM]
tags: [mlops, software-engineering]
status: complete
---

# MLOps

## Definition

> [!definition] MLOps
> **MLOps** is the practice of building, deploying and operating machine-learning systems in production as one continuous lifecycle. It joins the **model development** loop (data and ML) to the **DevOps** loop (software development and operations):
> $$\text{MLOps} = \text{DataML} + \text{DevOps}$$

## Intuition

> [!intuition] A brilliant model is not a brilliant outcome
> A model that hits its target metric in a notebook delivers nothing until it runs reliably for real users, and keeps running as data and requirements change. Many ML/AI projects fail for exactly this reason: teams and roles cannot collaborate with each other or with stakeholders, and the models never get deployed. MLOps treats the model as one part of a software system that must be built, tested, released, monitored and fixed like any other, with the extra complication that its behaviour depends on data that keeps moving.

## The lifecycle loops

**DevOps loop** (software), drawn as a figure-eight with a development half and an operations half:

```text
  Dev:  plan -> code -> build -> test
  Ops:  release -> deploy -> operate -> monitor -> (back to plan)
```

**Model development loop** (DataML), the same figure-eight shape with a data half and an ML half:

```text
  Data: collect -> curate -> transform -> validate
  ML:   explore -> train -> evaluate -> formulate -> (back to collect)
```

- **Data half**: gather raw data, curate it (select and clean what is worth keeping), transform it into usable features, and validate its quality.
- **ML half**: explore the data, train models, evaluate them; formulate is the step that closes the ML half and feeds the next round.

**The joined lifecycle** (drawn as an "eternal knot"): the two figure-eights are linked into one knot of four loops. Data feeds the ML loop; the ML loop hands over to software development where formulate meets plan and code; development releases into operations; and what monitoring reveals in production feeds back into planning and, through it, into new data and models. There is no final step: the system keeps cycling.

## Key Properties / Variants

- **Dev, Acc, Prod environments.** Pressing a button to make changes live is a bad idea. Modern processes use layered environments with **the same tech stack deployed in each**:
  - **Development**: free experimentation.
  - **Acceptance**: integrated services and testing, a production-like rehearsal.
  - **Production**: the only "live" environment, and it is **protected**.
- **Productionization and operations** are the two jobs an ML engineer adds on top of modelling. Productionization turns a prototype into a live system; operations keep it working as expected: maintenance, monitoring and firefighting.
- **Roles along the journey from raw data to ML/AI product**: data engineers (data that is high-quality, in the right format, available and useful), data analysts (turn data into information for decisions), data scientists (build models that solve the business problem), ML engineers (productionize and operate), backend engineers, and infrastructure/platform teams (scaling, authentication, back-ups, security, governance, data access, so application teams move faster). Titles and responsibilities differ between organisations.
- **Full-stack ML engineer.** Hand-overs between data scientists and software engineers caused many failures. Combining data science and ML engineering in one person who has adequate skills in every area lets one engineer own the problem and solution end to end.
- **Code quality is part of operations.** Code that will run in production must be readable and tested: see [[Clean Code]] and [[Unit Testing]].

## Connections

- Engineering practices it relies on: [[Clean Code]], [[Unit Testing]]
- Example production ML systems: [[Recommender System]], [[Retrieval-Augmented Generation]] pipelines, [[Large Language Models (LLM)]]
- Evaluation before and after deployment: [[Online and Offline Evaluation]]

## Appears In

- [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]]
