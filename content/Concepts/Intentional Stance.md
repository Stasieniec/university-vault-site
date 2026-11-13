---
type: concept
aliases: [intentional stance, design stance, physical stance, Dennett's stances, stances]
course: [AGAI]
tags: [agents, philosophy]
status: complete
---

# Intentional Stance

## Definition

> [!definition] Intentional stance (Dennett, *The Intentional Stance*, 1987)
> Interpreting entities as **agents**, ascribing them **beliefs, desires, intentions** and *enough rationality* to do what they *ought to do* given those beliefs and desires.

It is the top of Daniel Dennett's three stances, three ways of interpreting an entity:

- **Physical stance:** interpreting using physical laws.
- **Design stance:** interpretation related to what the entity is supposed to do, what it has been designed to do.
- **Intentional stance:** interpreting it as an agent, as above.

## Intuition

> [!intuition] A billiard ball, a clock and a poker player
> A billiard ball is predicted from physics: where and how hard the cue hits it. A clock is predicted from its design: you read the time without knowing the mechanism. That works only while it works; when the clock breaks, only the physical stance explains what it does. A poker player, a cat that looks scared, or a country in a trade dispute is predicted by asking what it believes and wants and assuming it acts sensibly on that.

The stance is a choice made by an observer. Nothing is an agent "in itself": Giovanni Sileno's ontology makes this explicit by defining **agents as actors ascribed with intentions**, where objects are passive (they undergo changes) and actors are active (they produce changes). This matches Russell and Norvig's remark that "agent" is a tool for analysing systems (see [[Intelligent Agent]]). (Outside the course material: Dennett justifies the intentional stance by its predictive success. Predicting a chess program's next move from its code is hopeless; assuming it wants to win and knows the rules works well.)

## Key Properties / Variants

- **Layered.** Each stance sits on top of the one below; when a higher stance's predictions fail (a broken clock, an agent acting irrationally), we fall back to a lower one.
- **Ascribed by an observer.** The question "is X an agent?" becomes "does taking the intentional stance towards X help?" A thermostat can be described as "wanting" 20 °C, but this adds as little as calling a calculator an agent.
- **Rationality assumption.** The stance assumes the entity does what it ought to given its beliefs and desires; this is what makes its predictions possible.
- **Source of the BDI vocabulary.** The mental states ascribed (beliefs, desires, intentions) are the ones the [[BDI Model]] turns into an architecture.

## Connections

- Explains when to call something an: [[Intelligent Agent]]
- Mental states turned into an architecture: [[BDI Model]], [[CAES Model]]
- Criteria-based alternative: the four characteristics of weak agency (reactivity, proactivity, autonomy, social ability), in [[AGAI-L06b - What Would Make a Chatbot an Agent]]
- Applied to modern systems: [[LLM Agent]]

## Appears In

- [[AGAI-L06b - What Would Make a Chatbot an Agent]]
