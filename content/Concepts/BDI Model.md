---
type: concept
aliases: [BDI, Belief-Desire-Intention, belief-desire-intention, BDI agent, BDI agents, BDI architecture, beliefs desires intentions]
course: [AGAI]
tags: [agents, architectures]
status: complete
---

# BDI Model

## Definition

> [!definition] Beliefs, desires, intentions (BDI)
> In (symbolic) AI, multi-agent systems and philosophy of action, beliefs, desires and intentions map onto three distinct mental states and practical levels of reasoning:
> - **Informational level:** what the agent **believes** to be the case in the world.
> - **Volitional level:** what the agent **desires** to be the case in the world.
> - **Deliberative level:** what the agent is **committed to** in the world.

A **BDI agent** is an agent built around these three states. BDI agents are programmed in agent-oriented languages such as AgentSpeak (with the Jason platform).

## Intuition

> [!intuition] Toothache
> Belief: "my tooth has a cavity." Desire: "I want my tooth cured." Intention: "I am going to my dentist." The first two describe the world as it is and as you would like it; only the third settles what you will do.

Why intentions are needed on top of beliefs and desires is Michael Bratman's point (*Intention, Plans, and Practical Reason*, 1987). A fleeting craving for chocolate can make the desire for dessert predominant for a moment, and you may even judge that it will lead you to act. Then you reflect, recall your dieting plans and skip dessert. A desire-belief account says you had a fleeting intention to eat chocolate; Bratman says you had none, because you were never **settled** on it. Intention is a commitment, a mental state of its own.

## Key Properties / Variants

- **Intention includes side effects.** Committing to a conduct brings more consequences than the desired outcome (the dentist also means the drill and the bill), so deliberating about intentions means weighing those too.
- **Architectural claim (Sileno).** If an architecture misses the function of intention deliberation, it misses performing, offline or online, an evaluation and weighing of possible choices.
- **Intent selection and plan selection (Georgeff and Rao, 1991):**
  - **Intent selection:** among several, possibly conflicting desires, give a priority on which goal to advance.
  - **Plan selection:** among several plans for an intent, give a priority on which plan to commit to.
- **Intention = intent + selected plan.** Desires set preferences about both the intent and the plans; beliefs about the current world determine which plans are possible and how the intent translates concretely.
- **Belief side.** Percepts may conflict and must be resolved into consistent beliefs (belief revision, data fusion).
- **Categorial leap.** A goal may be about a situation (a state), while a plan is always a conduct (a composite action); reading goals as *achieving* a situation narrows the gap.
- **Perceive-reason-act loop.** Practitioner descriptions of agentic AI (for example AWS's) place a reasoning core between perception and action, working with goals, a plan library, beliefs and intentions: the BDI structure.
- **Successor.** Because the full picture becomes messy, Giovanni Sileno proposes the [[CAES Model]], which regroups BDI into commitments, abilities and expectations and adds susceptibilities.

## Connections

- Vocabulary from: [[Intentional Stance]]
- Reworked as: [[CAES Model]]
- Used in simulation as agent-oriented programming: [[Agent-Based Modelling]]
- Compare: the internal state of a model-based agent ([[Agent Program]]), and an agent's probability distribution over states ([[Belief State]])
- Relevant to: [[LLM Agent]] harnesses that lack an explicit deliberation step

## Appears In

- [[AGAI-L06b - What Would Make a Chatbot an Agent]]
- [[AGAI-L01 - Agents, Multiagent Systems and Agentic AI]] (history: BDI agents, 1993 to 1995)
