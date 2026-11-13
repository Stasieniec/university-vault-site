---
type: concept
aliases: [CAES, "Commitments, Abilities, Expectations, Susceptibilities", commitments abilities expectations susceptibilities]
course: [AGAI]
tags: [agents, architectures]
status: complete
---

# CAES Model

## Definition

> [!definition] Commitments, Abilities, Expectations, Susceptibilities (CAES model; Boer and Sileno, "Anchoring Distributed Agency")
> - **Commitments:** what the agent *wants to achieve* in the world (intent selection).
> - **Abilities:** what the agent *can achieve* in the world, and *how* (plan selection).
> - **Expectations:** what the agent *knows about* the world (belief revision).
> - **Susceptibilities:** what the agent *is looking at* in the world, and *how* it responds to it (attention allocation).

Giovanni Sileno presents it as a cleaner template than the [[BDI Model]], whose full picture (desires to intent, plans to a selected plan, percepts to beliefs, with beliefs and desires feeding every step) "rapidly becomes very messy".

## Intuition

> [!intuition] Two sides of the agent-world boundary
> On the acting side, commitments (what to achieve) and abilities (how) meet in the action that goes out to the world. On the perceiving side, what comes in is filtered by susceptibilities (what the agent attends and reacts to) and integrated into expectations (what it takes the world to be).

```
      C                    E
      | intent             | belief
      | selection          | revision
 A ---*                    *--- S
 plan |                    ^  attention
 selection                 |  allocation
------v-------WORLD--------|------
    acts               perceives
```

## Key Properties / Variants

- **Relation to BDI.** Commitments (structural, contingent) take over desires and intent; abilities (skills, competences) take over plans and plan selection; expectations (structural, contingent) take over percepts and beliefs. Susceptibilities are new. The course material does not define "structural" and "contingent"; a plausible reading (not from the course material) is standing commitments or expectations against situation-dependent ones.
- **Applied to a coding agent harness (Sileno):**
  - Commitments: task input ("fix the bug"), constraint input ("preserve the API"), guardrail checks ("don't edit unrelated files"), completion criteria ("the bug is fixed and relevant tests pass").
  - Abilities: SKILL.md files exposing tool usage, the tools, the system running them.
  - Expectations: working state or context, memory, code repository, retrieved evidence (or better, their models).
  - Susceptibilities: watchers and triggers, tool-result handlers, interrupts, escalation routing, including their functional connection to the other components.
- **A problem-solving cycle.** A test fails: the failure is received and routed (S), the context is updated (E), the problem is identified and repairing the bug becomes the commitment (C), an available debugging skill is chosen (A), and the cycle repeats.
- **Current limit.** The mapping refers either to low-level constructs (no sufficiently interoperable declarative language exists yet) or to higher-level directives (we don't know how the LLM will interpret them).
- **Claim.** "If we are seeking to create 'rational' agents, we cannot but expect to proceed towards CAES-like frameworks."

## Connections

- Reworks: [[BDI Model]]
- Mental-state vocabulary from: [[Intentional Stance]]
- Applied to: [[LLM Agent]] harnesses
- Is a design for an: [[Intelligent Agent]]

## Appears In

- [[AGAI-L06b - What Would Make a Chatbot an Agent]]
