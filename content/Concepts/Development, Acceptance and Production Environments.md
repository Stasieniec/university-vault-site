---
type: concept
aliases: [DTAP, Dev/Acc/Prod, "Dev, Acc, Prod", layered environments, staging environment, acceptance environment, production environment]
course: [EPRM]
tags: [software-engineering, process, mlops, deployment]
status: complete
---

# Development, Acceptance and Production Environments

## Definition

> [!definition] Layered environments
> Modern software processes deploy the same tech stack in several **environments**: development (Dev, also test), acceptance (Acc, also staging) and production (Prod). Changes move from one to the next; **only production is live**, and production is **protected**. DTAP (development, test, acceptance, production) is the four-stage name for the same idea.

## Intuition

> [!intuition] Mistakes happen, just not in production
> In June 2021 an HBO Max intern's empty "Integration Test Email #1" reached real subscribers. Blaming the intern misses the point: mistakes are inevitable, and the process must make sure they cannot reach the live system. Rehearse every change in an environment that behaves like production but has no real users or data.

## The environments

| | Local | Dev / Test | Acc / Staging | Prod |
|---|---|---|---|---|
| **What** | Your laptop | Sandbox | Replica of Prod | Live systems |
| **Systems** | Yours | Yours only | All, integrated | All, integrated |

- **Local:** unrestricted and personal; the easiest place to develop and debug.
- **Dev:** the same tools as production, or more, with access to just your own systems; a free-for-all for developing against the tech stack and experimenting, where nothing is guaranteed and things are temporary.
- **Acc:** the same tech stack with every system active and integrated, so the whole product can be tested. It differs from production only in scale and data: low scale, no production data.
- **Prod:** live, at live scale, serving users: the danger zone.

## Key Properties / Variants

- **Same tech stack everywhere**, so what passes in acceptance behaves the same in production. [[Containerization|Containers]] help by running the same image in every environment.
- **No manual deployment** to Acc or Prod: merges into protected branches trigger continuous deployment (CD). With [[GitFlow]], develop points to acceptance and main to production.
- **Acceptance is where the whole product is tested**, with all systems active and integrated.

## Connections

- Part of: [[MLOps]] (release and deploy stages)
- Branching model: [[GitFlow]]
- Tests run there: [[Integration Testing]]

## Appears In

- [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]]
- [[EPRM-L03 - Containerization and Software Engineering Processes]]
