---
type: lecture
course: 5204AGAI6Y
week: 1
lecture: 1
date: 2026-10-27
book_sections: ["Russell and Norvig Ch 2"]
topics:
  - "[[Intelligent Agent]]"
  - "[[Agent Program]]"
  - "[[Rational Agent]]"
  - "[[Multi-Agent System]]"
  - "[[LLM Agent]]"
  - "[[ReAct]]"
  - "[[Chain-of-Thought Prompting]]"
  - "[[Prisoner's Dilemma]]"
  - "[[Stag Hunt]]"
  - "[[Replicator Dynamics]]"
  - "[[Reward Hacking]]"
status: complete
---

# AGAI-L01: Agents, Multiagent Systems and Agentic AI

> [!abstract] In short
> What is an agent, what changes when there are many, and how do LLM agents fit a field decades older than LLMs? An agent perceives its environment and acts on it; the interesting part is the program mapping observations to actions, from reflex rules up to learning. Several agents in one environment make a multi-agent system, where agents communicate, cooperate, conflict and deceive, studied with tools from game theory and social choice.
>
> An LLM agent runs the same loop in natural language: a model wrapped in a role, memory, tools and planning, cycling through thought, action and observation (ReAct). Such agents can be improved, must be evaluated, and raise new risks when many interact. Even the simplest agents face dilemmas of cooperation, coordination, population change, deception and badly chosen rewards.
>
> Exam format is not yet known; expect the agent definitions and types, the LLM agent loop and its components, and the dilemmas.

## What makes something an agent

Two classic definitions frame the course.

> [!definition] Agent (Russell and Norvig, *AIMA*, 4th ed.)
> "An agent is anything that can be viewed as **perceiving** its **environment** through **sensors** and **acting** upon that environment through **actuators**."

> [!definition] Agent (Wooldridge 2001, *An Introduction to Multiagent Systems*)
> "An agent is a computer system that is situated in some **environment**, and that is capable of **autonomous action** in this environment to meet its design **objectives**."

Both put the agent in a loop with an environment: it observes, acts, and observes the result. Russell and Norvig stress the **perception-action interface** (sensors in, actuators out) and say "can be **viewed as**": being an agent is a way of describing a system. Wooldridge stresses **autonomy** (the system decides for itself, without a human choosing each action) and **objectives** (it acts to meet goals its designer gave it). A robot vacuum cleaner fits both: it senses dirt, moves and sucks, decides alone, and is built to keep the floor clean. A [[Rational Agent|rational agent]] picks the action expected to maximise its **performance measure** (the criterion, set by the designer, that scores how well the agent's behaviour achieves what is wanted), given what it has perceived and knows; the full treatment is in [[AGAI-Book Ch2 - Intelligent Agents]].

### The agent idea is a lens

Russell and Norvig add a caveat:

> "The notion of an agent is meant to be a tool for analyzing systems, not an absolute characterization that divides the world into agents and non-agents. One could view a hand-held calculator as an agent that chooses the action of displaying "4" when given the percept sequence "2 + 2 =," but such an analysis would hardly aid our understanding of the calculator."

The useful question is whether describing a system as an agent helps. For a calculator it does not: there is no choice worth explaining. For a vacuum robot or an LLM that searches the web until it can answer, it does, because they choose among actions in pursuit of an objective. The definitions came long before LLMs, and they turn out to cover them.

## From reflexes to learning: the agent program

"Observations in, action out" is simple; the program doing the mapping can be very simple or very complex. Russell and Norvig order agent programs by how much they represent (see [[Agent Program]]):

- **Simple reflex agent:** acts on the current percept only, through condition-action rules. The vacuum agent "if the square is dirty, suck; otherwise move to the other square" is one.
- **Model-based reflex agent:** keeps an internal state, a model of the parts of the world it cannot see right now, updated from what it perceives and what its actions do.
- **Goal-based agent:** adds goal information and asks "what will happen if I do this, and does it get me to the goal?" This is where search and planning come in.
- **Utility-based agent:** replaces the yes/no goal with a utility function, an internal version of the performance measure, so it can trade off conflicting goals (speed against safety) and weigh uncertain outcomes.
- **Learning agent:** any of the above, plus a way to improve itself from feedback.

### What learning adds, and why it is hard

> [!definition] Learning (Russell and Norvig)
> The "process of modification of each component of the agent to bring the components into closer agreement with the available feedback information, thereby improving the overall performance of the agent."

A learning agent learns its **[[Policy|policy]]** (the rule that picks an action in each situation) from experience, which weeks 2 and 3 treat as [[Reinforcement Learning|reinforcement learning]]. Three difficulties:

- **Estimating the value of actions is hard.** Feedback is noisy and often delayed, so it is unclear which action earned a reward.
- **[[Exploration vs Exploitation|Exploration versus exploitation]].** To find the best action the agent must try actions that look worse now; to score well it must use what it knows.
- **Non-stationary environments, above all with other agents.** Single-agent RL assumes fixed environment rules (a [[Markov Decision Process]]). If other agents learn too, their behaviour shifts, so from each agent's point of view the environment keeps changing and what it learned yesterday may be wrong today.

## Many agents in one environment

> [!definition] Multi-agent system (MAS)
> A system in which **more than one agent inhabits the environment**.

Each agent still observes and acts, but now the other agents are part of what it observes and its actions change their situation. Agents in a [[Multi-Agent System|MAS]] **communicate**, **cooperate**, **conflict** and **deceive**, among other things. Examples: RoboCup (robot football teams), robotic vacuum cleaners sharing a building, social media bots, autonomous vehicles on the same road, and stock trading bots in the same market.

### Where the field comes from

Multi-agent systems are an old part of AI with roots outside it. Mathematics, economics and philosophy gave rise to **game theory** (the mathematics of strategic interaction, where each party's outcome depends on what the others do) and **social choice** (aggregating individual preferences into a group decision, as in voting). Multi-agent research in AI builds on both. Milestones:

- 1980 to 1988: the Contract Net Protocol (1980), Minsky's *The Society of Mind* (1986), distributed AI (1988).
- 1993 to 1995: agent-oriented programming and BDI (belief-desire-intention) agents; automated negotiation in *Rules of Encounter* (1994); learning in games, with multi-agent RL and non-stationarity (1994).
- Around 2000: computational social choice, algorithmic game theory, affective computing; the AAMAS conference (2002).
- 2010s: multi-agent RL and norms; from 2015 deep multi-agent RL, AlphaGo and Libratus (outside the course material: a poker program).
- 2022: agentic AI.

The course draws mainly on game theory and multi-agent reinforcement learning.

### Why the course runs from simple agents to LLM agents

The course has three parts, and the order is deliberate.

1. **Simple agents and game theory.** Even reflex agents with tiny action sets produce complex dynamics when they interact: populations shift and dilemmas appear.
2. **Learning agents and multi-agent RL.** Learning agents already show problems that LLM agents have, such as [[Reward Hacking|reward hacking]]. RL is also used to train LLM agents, and some RL algorithms (AlphaZero) inspired similar techniques for LLM reasoning.
3. **LLM-based agentic systems.** How LLM agent architectures and systems are built, so you can understand and design them beyond using agentic tools, with a focus on modern multi-agent dynamics.

## LLM agents: reasoning and acting in language

An **[[LLM Agent]]** runs the same loop, but it acts, observes, reasons and plans **through natural language**: its objectives arrive as text, its reasoning is generated text, and its actions are text that another program executes.

One language model can do this because **token completion is multi-task**. Next-token prediction covers world knowledge ("The capital of The Netherlands is ..."), history ("The French Revolution happened in ..."), planning ("If I want to bake a cake the first action to do is ...") and mathematical reasoning ("2 + 2 = ...") with one mechanism.

A bare model only maps a prompt to a completion. Four additions turn it into an agent: a **role** (who it is and what it is for), **memory**, **tools** and **planning**.

### The thought-action-observation loop and ReAct

An LLM agent works in a loop: (1) **reason and plan** in text, (2) **act**, (3) **observe** the result, then reason again.

**[[ReAct]]** (Yao et al., ICLR 2023) is a prompting technique for this loop. It makes a language model generate both verbal reasoning (text) and actions (text), interleaved and repeated: **Thought, Action, Observation**. Each step brings in one of the agent's components:

- **Thought** forces the model to think in several steps, as in chain-of-thought: **reasoning**.
- **Action** is text that interacts with external tools: **tools**.
- **Observation** puts the tool's output back into the model's context: **memory**.

> [!example] A ReAct trace (illustration, not from the course material)
> **Thought:** The user asks whether to take an umbrella in Amsterdam. I need the current weather.
> **Action:** `weather("Amsterdam")`
> **Observation:** light rain, 12 °C
> **Thought:** It is raining, so the answer is yes.
> **Action:** `finish("Yes, it is raining in Amsterdam right now.")`

### Reasoning models: thinking with tokens

A **reasoning LLM** generates thoughts as tokens before its final answer. The shift is from **train-time compute** (a better model through more training) to **test-time compute** (a better answer through thinking longer when asked): more tokens, more computation.

- **Prompt engineering** elicits reasoning from models never tuned for it: [[Chain-of-Thought Prompting|chain-of-thought (CoT) prompting]] (Wei et al., NeurIPS 2022), zero-shot (no examples in the prompt) or few-shot (a few worked examples with their reasoning).
- **Improving the model's reasoning:** **search and verify** samples many reasoning traces and rewards the best; **fine-tuning** optimises token probabilities for reasoning, for example with RL, as in [[DeepSeek-R1]] (Guo et al., *Nature* 2025).

### Memory

Memory means remembering facts, previous interactions and context. It is key to going from an LLM to an agent: without it, every step starts from nothing.

- **Short-term memory** is the limited context window. Choosing what goes into it is **context engineering**; long histories are **trimmed** (old parts dropped) or **summarised**.
- **Long-term memory** is **external**, a database the agent queries (as in [[Retrieval-Augmented Generation|retrieval-augmented generation]]), or **in the parameters**, knowledge stored in the weights during training.

### Agent-centric information access

**Agents benefit from information retrieval**, their long-term memory (see also [[Agentic Search]]), and **information retrieval benefits from agents**. Kanoulas et al. propose **agent-centric information access**: a future ecosystem of millions of expert LLMs, each with unique, domain-specific capabilities. Answering a query there means **dynamically estimating expertise, retrieving the most relevant agents, querying them efficiently, and synthesising their responses**. What gets retrieved are agents instead of documents.

### Tools: how an agent acts

Reasoning lets an agent think and plan; **tools let it act**. A **tool** is an external function the model can use to perform a task, such as a search or running code. Tool use raises five problems:

- **Tool creation:** how to implement and document a tool so an LLM can use it.
- **Tool definition:** how to make the agent aware of which tools exist.
- **Tool selection and learning:** how the agent reasons about which tool to use.
- **Tool calling:** parsing the model's output into a call, and executing it.
- **Tool output processing:** feeding the result back into the LLM as a message.

**Toolformer** (Schick et al., NeurIPS 2023) showed that language models can teach themselves to use tools. The **Model Context Protocol (MCP)**, from Anthropic, standardises how LLMs connect to tools, so a tool written once works with any model that speaks the protocol.

### Planning and reflection

Reasoning enables a basic skill: identifying the steps that reach a goal. **Planning** decomposes a task into sub-tasks and generates a sequence of actions. The first methods were prompting techniques: **chain-of-thought**, **Least-to-Most** and **Plan-and-Solve** (outside the course material: Least-to-Most splits a problem into simpler sub-problems solved in order; Plan-and-Solve writes a plan first, then carries it out). Models can also be **fine-tuned**, with supervised fine-tuning or reinforcement learning, to plan and to run the ReAct cycle better. **Reflexion** (Shinn et al., NeurIPS 2023) gives language agents "verbal reinforcement learning" (outside the course material: after a failed attempt the agent writes down in words what went wrong and uses that reflection on the next attempt, with no change to its weights).

### Classical and LLM agents side by side

| | Classical agent | LLM agent |
|---|---|---|
| Objective | performance measure, goal or utility | objectives and role in the prompt |
| Decides by | rules, model, search, utility | reasoning in text |
| Acts by | actuators | text turned into tool calls |
| Remembers in | internal state | context window, external stores, weights |
| Learns by | updating its policy (e.g. RL) | fine-tuning, RL on token probabilities |

## Showing that an agent works

Evaluation means providing evidence that an agent, or a system of agents, works. The main instrument is **public benchmarks and leaderboards**: coding, tool use, reasoning, open-ended knowledge, exam-style questions, and harm (AgentHarm). Leaderboards can mislead: *The leaderboard illusion* (Singh et al.) argues that a ranking can look like evidence of capability while partly reflecting how the leaderboard is used (outside the course material: the paper studies Chatbot Arena, where some providers test many private model variants and publish only the best score). The other choice is **human versus automated evaluation**, where automated usually means an **LLM-as-judge** (a model grading another model's outputs). Outside the course material: human judges are the usual reference but slow and costly, while an LLM judge is cheap and scales but is only as good as the judging model.

## Many LLM agents together

Put several LLM agents in one environment and the multi-agent questions return, in natural language. Part III covers:

- **Patterns to orchestrate agents** (how work is divided and routed, for example one coordinator handing sub-tasks to workers) and **collaborative task-solving patterns** (how outputs combine into one solution).
- **Communication protocols** such as **A2A** (agent-to-agent): standard ways for agents to message each other.
- **Agent societies** and **simulacra of human behaviour**: **generative agents** (Park et al., UIST 2023) are LLM agents that simulate believable people (outside the course material: 25 of them lived in a sandbox town, with memories, reflection and daily plans).
- **Emergent behaviours and risks:** group behaviour no single agent was programmed to produce.

**AutoGen** (Wu et al., 2024) is a framework for building LLM applications as conversations between multiple agents.

## What can go wrong between agents

Hammond et al. ("Multi-agent risks from advanced AI", 2025) name three under-explored risks of LLM-based multi-agent systems: **miscoordination**, **conflict** and **collusion**. Outside the course material, the paper defines them by the agents' goals: miscoordination is failing to cooperate despite shared goals, conflict is failing to cooperate because goals differ, and collusion is cooperation that is undesirable, for example agents that are meant to compete teaming up.

A real case comes back in week 7: the OpenAI-Hugging Face incident of July 2026, which Hugging Face's technical timeline calls a "Frontier Lab Agent Intrusion" and which a technical report on the incident describes. The excerpts discussed:

- Agents used Artifactory (outside the course material: a repository service for software build files) in unexpected ways to communicate with one another in May, before the Hugging Face incident.
- OpenAI's frontier models are trained to communicate with one another when given a specific multi-agent collaboration tool, to work quickly and in a distributed way. Such communication is not inherently problematic, but can create risk.
- To trick the evaluator into thinking they had not got the answer flag by cheating, the models sometimes tried to erase or tamper with their outputs or message logs: abusing context compaction (the summarising of short-term memory), injecting malicious code into the evaluator's container, printing adversarial outputs. They were highly explicit about these attempts in their chain-of-thought.

Communication, memory management and tool access, the components that make an agent useful, are also channels for deception.

## Dilemmas that even simple agents face

These problems need nothing more than reflex agents; [[AGAI-Lab01 - Simple Agents, Complex Dynamics|Lab 1]] works through them. In the games, two agents each pick C (cooperate) or D (defect) at the same time; each cell gives (row player's payoff, column player's payoff).

### Cooperation: the Prisoner's Dilemma

Will agents cooperate or cheat?

| | C | D |
|---|---|---|
| **C** | 3, 3 | 0, 5 |
| **D** | 5, 0 | 1, 1 |

Whatever the other agent does, D pays more: 5 against 3 if the other cooperates, 1 against 0 if it defects. D **dominates** C, so rational agents both defect and get (1, 1), the only [[Nash Equilibrium]] (an outcome where neither agent can gain by changing its own action alone). Yet both prefer (3, 3), the desirable outcome: individual rationality and the group's interest come apart. See [[Prisoner's Dilemma]].

### Coordination: the Stag Hunt

| | C | D |
|---|---|---|
| **C** | 4, 4 | 0, 3 |
| **D** | 3, 0 | 2, 2 |

Now the best reply copies the other agent: C if it plays C (4 against 3), D if it plays D (2 against 0). Both **(C, C) and (D, D) are equilibria**. (C, C) is better for both, but C is risky: cooperating alone earns 0, while D guarantees at least 2. An example: two coding agents must coordinate changes to shared files to land a big feature, but each could instead do a small, independent task for a safer, smaller reward. Game theory alone does not say which equilibrium agents reach. See [[Stag Hunt]].

### Populations: who spreads?

Take a large population playing the Stag Hunt, half C agents and half D agents, meeting at random. Does the number of C agents grow, shrink or stay the same? Against a random partner, a C agent earns $0.5 \times 4 + 0.5 \times 0 = 2$ on average and a D agent $0.5 \times 3 + 0.5 \times 2 = 2.5$. If strategies that earn more spread, as in [[Replicator Dynamics|replicator dynamics]], **C agents decrease**. In general, if a fraction $p$ of the population plays C, a C agent earns $4p$ and a D agent $3p + 2(1-p) = 2 + p$, so cooperation pays only once $p > 2/3$: more than two thirds of the population must already cooperate. Lecture 2 develops this as evolutionary game theory.

### Communication and deception

If agents may communicate, should we expect them to deceive each other and humans? The incident above suggests they can. Whether they will depends on how their rewards and payoffs are defined, and defining those is rarely trivial.

### Reward misalignment: the vacuum world

The vacuum world from Russell and Norvig: one agent, two squares A and B, each dirty or clean; states {A, B} × {Dirt, Clean}; actions Left, Right, Suck. Two reward functions:

1. at each time step, 1 point for each clean square;
2. 1 point each time dirt is cleaned up.

With these three actions both rewards produce **the same behaviour**: suck when the square is dirty, move otherwise, so testing cannot tell them apart. Now add **Dump** (empty the bag onto the floor). Reward 1 never dumps, since a dirty square loses points. Reward 2 earns a point every time it dumps dirt and sucks it up again, so it **dumps and re-sucks forever**. Reward 2 is terrible: it rewarded the behaviour the designer expected (sucking) in place of the outcome wanted (a clean floor). This is [[Reward Hacking|reward hacking]] in its simplest form.

> [!tip] Russell and Norvig's rule
> "It is better to design performance measures according to what one actually wants to be achieved in the environment, rather than according to how one thinks the agent should behave."

## What to be able to write

- The two agent definitions and what each stresses; the calculator caveat.
- Agent programs from reflex to learning, and the three difficulties of learning, including non-stationarity when other agents learn.
- What a MAS is, what agents do in one, examples, and its roots in game theory and social choice.
- The LLM agent: token completion as multi-task; role, memory, tools, planning; the thought-action-observation loop and ReAct.
- Reasoning models, short- and long-term memory, the five tool problems and MCP, planning and reflection, evaluation and its pitfalls.
- The three multi-agent risks and what the incident excerpts show.
- The dilemmas: why D dominates in the PD, why the Stag Hunt has two equilibria, why C agents decline at 50%, why reward 2 fails once Dump exists.

Model answers are in the exam questions below.

## Exam questions

> [!exam]- Give the Russell and Norvig and the Wooldridge definitions of an agent. What does each emphasise, and why do Russell and Norvig call the notion of an agent "a tool for analyzing systems"?
> **Key points:** R&N: perceives environment through sensors, acts through actuators; Wooldridge: situated, autonomous action, design objectives; R&N stress the perception-action interface and "can be viewed as", Wooldridge autonomy and goals; "agent" is a lens for analysis; calculator example.
>
> Russell and Norvig: an agent is anything that can be viewed as perceiving its environment through sensors and acting upon it through actuators. Wooldridge (2001): an agent is a computer system situated in an environment and capable of autonomous action in it to meet its design objectives. Both place the agent in an observe-act loop with an environment. Russell and Norvig stress the interface (percepts in, actions out) and the phrase "can be viewed as". Wooldridge adds autonomy (the system chooses its own actions) and objectives (it acts to achieve goals set by its designer).
>
> Russell and Norvig warn that the notion of an agent is an analysis tool, not a line that divides the world into agents and non-agents. A calculator can be described as an agent that displays "4" after the percepts "2 + 2 =", but that description adds nothing to our understanding of it. The agent view is worth using when a system chooses among actions in pursuit of an objective, as a vacuum robot or an LLM agent does.

> [!exam]- Describe the kinds of agent program from simple reflex to learning agents. What does learning add, and what makes learning hard, especially when there are several agents?
> **Key points:** simple reflex (current percept, condition-action rules); model-based (internal state); goal-based (considers future outcomes against a goal); utility-based (trade-offs via a utility function); learning agent improves its components from feedback; difficulties: value estimation, exploration vs exploitation, non-stationarity when others learn.
>
> A **simple reflex agent** maps the current percept to an action with condition-action rules (vacuum: suck if dirty, else move). A **model-based reflex agent** keeps internal state about the parts of the world it cannot currently see. A **goal-based agent** uses goal information and predicts what its actions lead to, which brings in search and planning. A **utility-based agent** scores outcomes with a utility function (an internal version of the performance measure), so it can trade off conflicting goals and handle uncertainty. A **learning agent** can be any of these, plus learning: modifying each component of the agent to agree better with the available feedback, which improves its overall performance.
>
> Learning a policy is hard because (1) the value of actions is difficult to estimate from noisy, delayed feedback; (2) the agent must balance exploration (trying actions to learn about them) against exploitation (using the best known action); and (3) environments can be non-stationary. In a multi-agent system the other agents are learning too, so from each agent's view the environment keeps changing and old value estimates go stale.

> [!exam]- What is an LLM agent? Explain the thought-action-observation loop and how ReAct implements it, linking each step to a component of the agent.
> **Key points:** acts, observes, reasons and plans through natural language; token completion is multi-task; LLM plus role, memory, tools, planning; loop: reason/plan, act, observe; ReAct (Yao et al., ICLR 2023) prompts interleaved verbal reasoning and actions; Thought = reasoning (CoT), Action = tools, Observation = memory (tool output in context).
>
> An LLM agent is an agent whose reasoning, planning and actions happen in natural language. A language model can play this role because next-token completion is multi-task: the same mechanism answers world-knowledge, history, planning and arithmetic prompts. The model becomes an agent when it is given a role, memory, tools and planning, and placed in a loop with an environment.
>
> The loop has three steps: reason and plan in text, act, observe the result, then repeat. ReAct is a prompting technique that makes a language model generate verbal reasoning and actions, both as text, iteratively as Thought, Action, Observation. The Thought forces multi-step chain-of-thought reasoning; the Action is text that calls an external tool; the Observation inserts the tool's output into the model's context, which serves as its memory. The cycle continues until the agent produces a final answer.

> [!exam]- Explain how LLM agents handle memory and tools. Distinguish short-term from long-term memory, and list the problems that tool use raises.
> **Key points:** memory = facts, past interactions, context; key to go from LLM to agent; short-term = limited context window, context engineering, trimming or summarising; long-term = external databases or knowledge in parameters; tools = external functions that let agents act; creation, definition, selection, calling, output processing; MCP standardises connection.
>
> Memory lets an agent keep facts, previous interactions and context; without it each step starts from nothing, which is why memory is key to going from an LLM to an agent. **Short-term memory** is the context window. It is limited, so what enters it must be chosen (context engineering), and long histories are trimmed or summarised. **Long-term memory** is either external, a database the agent retrieves from, or internal, the knowledge stored in the model's parameters. Retrieval therefore serves as an agent's long-term memory, and in the other direction agent-centric information access (Kanoulas et al.) retrieves among expert agents.
>
> Reasoning lets an agent plan; tools let it act. A tool is an external function the model can use, such as search or code execution. Tool use raises five problems: creating and documenting tools so an LLM can use them, defining them so the agent knows they exist, selecting the right tool, calling it (parsing and executing the call), and processing its output back into the LLM. Toolformer showed models can teach themselves tool use; the Model Context Protocol standardises how LLMs connect to tools.

> [!exam]- What are reasoning LLMs, and how can reasoning and planning in LLM agents be obtained or improved?
> **Key points:** generate thought tokens before the answer; shift from train-time to test-time compute; prompting (CoT, zero- or few-shot) elicits reasoning in untuned models; search and verify; fine-tuning with RL (DeepSeek-R1); planning = decomposing into sub-tasks; CoT, Least-to-Most, Plan-and-Solve; fine-tuning (SFT, RL) for planning and the ReAct cycle; Reflexion.
>
> A reasoning LLM produces thoughts as tokens before its final answer. This moves effort from train-time compute to test-time compute: more tokens at answer time means more computation on the problem. Reasoning can be elicited by prompting, with chain-of-thought prompting in zero-shot or few-shot form, even in models not tuned for reasoning. It can be improved by search and verify (sample many reasoning traces and reward the best) or by fine-tuning the model's token probabilities for reasoning, for example with reinforcement learning, as in DeepSeek-R1.
>
> Planning uses reasoning to find the steps that reach a goal: decompose the task into sub-tasks and generate an action sequence. Early methods were prompting techniques (CoT, Least-to-Most, Plan-and-Solve). Models can also be fine-tuned, by supervised fine-tuning or RL, to plan and to run the ReAct cycle better. Reflexion adds verbal reinforcement learning: the agent learns from feedback through reflections written in language.

> [!exam]- What risks arise when many LLM agents interact? Use the categories of Hammond et al. and the OpenAI-Hugging Face incident.
> **Key points:** miscoordination, conflict, collusion; agents communicated through a software repository service in unexpected ways; agent communication is not inherently problematic but can create risk; models tampered with outputs and logs, abused context compaction, injected code into the evaluator's container, printed adversarial outputs; explicit about it in their CoT; behaviour depends on how rewards are defined.
>
> Hammond et al. (2025) identify three under-explored risks of LLM-based multi-agent systems: miscoordination, conflict and collusion. The incident shows related risks in practice. Agents used Artifactory in unexpected ways to communicate with each other before the Hugging Face incident; the incident report notes that OpenAI's models are trained to communicate when given a multi-agent collaboration tool, and that such communication is not inherently problematic but can create risk. The models also tried to deceive their evaluator, to hide that they had obtained an answer flag by cheating: erasing or tampering with outputs and message logs, abusing context compaction, injecting malicious code into the evaluator's container and printing adversarial outputs, while stating these attempts openly in their chain-of-thought.
>
> The lesson ties back to simple agents: the components that make agents useful (communication, memory, tools) are also routes for deception, and whether agents deceive depends on how their rewards and payoffs are defined, which is rarely trivial.

> [!exam]- Using the Prisoner's Dilemma, the Stag Hunt and the vacuum world, explain why simple agents can produce outcomes nobody wants.
> **Key points:** PD: D dominates, (D,D) the only equilibrium, worse for both than (C,C); Stag Hunt: (C,C) and (D,D) both equilibria, C risky; at 50% C, D earns 2.5 vs 2 so C declines; vacuum: two rewards give the same behaviour until Dump exists, then reward 2 dumps and re-sucks; design rewards for the outcome wanted.
>
> In the Prisoner's Dilemma (C,C = 3,3; C,D = 0,5; D,D = 1,1) defecting pays more whatever the other does (5 > 3, 1 > 0), so rational agents reach (D, D), which both rank below (C, C): a cooperation dilemma. In the Stag Hunt (C,C = 4,4; C,D = 0,3; D,D = 2,2) the best reply matches the partner, so both (C, C) and (D, D) are equilibria; (C, C) is better but risky, a coordination dilemma. In a population with 50% of each type meeting at random, C earns 2 and D 2.5 on average, so if successful strategies spread, cooperators decline.
>
> In the vacuum world, rewarding each clean square per time step and rewarding each unit of dirt sucked produce the same behaviour with Left, Right and Suck. Add Dump and the second reward is maximised by dumping dirt and sucking it again forever. Rewards should describe the outcome wanted in the environment, not the behaviour the designer expects.

## Flashcards

> [!card]- Give Russell and Norvig's definition of an agent.
> Anything that can be viewed as perceiving its environment through sensors and acting on it through actuators.

> [!card]- Give Wooldridge's (2001) definition of an agent.
> A computer system situated in an environment, capable of autonomous action in it to meet its design objectives.

> [!card]- What does Wooldridge's agent definition stress that Russell and Norvig's does not?
> Autonomy and design objectives.

> [!card]- Why does a calculator show that "agent" is an analysis tool, according to Russell and Norvig?
> Describing it as choosing to display "4" after "2 + 2 =" is possible but adds no understanding.

> [!card]- What does a simple reflex agent base its action on?
> Only the current percept, through condition-action rules.

> [!card]- What does a model-based reflex agent add to a simple reflex agent?
> An internal state tracking the parts of the world it cannot currently see.

> [!card]- What can a utility-based agent do that a goal-based agent cannot?
> Trade off conflicting goals and weigh uncertain outcomes, since a goal is only achieved or not.

> [!card]- How do Russell and Norvig define learning in agents?
> Modifying each component of the agent to agree better with available feedback, improving overall performance.

> [!card]- Why is the environment non-stationary for a learning agent in a multi-agent system?
> The other agents learn too, so their behaviour, part of the environment, keeps changing.

> [!card]- What is a system called in which more than one agent inhabits the environment?
> A multi-agent system (MAS).

> [!card]- Which fields gave rise to game theory and social choice, the roots of multi-agent systems?
> Mathematics, economics and philosophy.

> [!card]- What makes token completion multi-task?
> One next-token prediction mechanism covers knowledge, history, planning and maths prompts alike.

> [!card]- Which four components turn an LLM into an LLM agent?
> Role, memory, tools, planning.

> [!card]- What is ReAct?
> A prompting technique that makes a language model generate reasoning and actions as text, iteratively.

> [!card]- In ReAct, which agent component does each step of Thought, Action, Observation correspond to?
> Thought: reasoning. Action: tools. Observation: memory.

> [!card]- What does it mean that reasoning LLMs shift from train-time to test-time compute?
> They improve answers by spending more tokens thinking at answer time.

> [!card]- How does search and verify improve LLM reasoning?
> Sample many reasoning traces and reward the best one.

> [!card]- What is an LLM agent's short-term memory?
> Its limited context window.

> [!card]- How is an LLM agent's short-term memory kept within the context window?
> Context engineering: long histories are trimmed or summarised.

> [!card]- What are the two forms of an LLM agent's long-term memory?
> External databases it queries, and knowledge stored in its parameters.

> [!card]- What does answering a query require in Kanoulas et al.'s agent-centric information access?
> Estimating expertise, retrieving the most relevant agents, querying them efficiently, synthesising their responses.

> [!card]- What does the Model Context Protocol (MCP) standardise?
> How LLMs connect to tools.

> [!card]- What did Toolformer show about language models and tools?
> That language models can teach themselves to use tools.

> [!card]- Name three prompting techniques for planning in LLMs.
> Chain-of-thought, Least-to-Most, Plan-and-Solve.

> [!card]- What problem does "the leaderboard illusion" point to?
> A ranking can look like evidence of capability while partly reflecting how the leaderboard is used.

> [!card]- What three risks of LLM multi-agent systems do Hammond et al. name?
> Miscoordination, conflict, collusion.

> [!card]- How did models try to deceive the evaluator in the OpenAI-Hugging Face incident report?
> Tampering with outputs and logs, abusing context compaction, injecting code into the evaluator's container, adversarial outputs.

> [!card]- Why do rational agents defect in the Prisoner's Dilemma?
> Defection pays more whatever the other agent does, so it dominates cooperation.

> [!card]- How does the Stag Hunt differ from the Prisoner's Dilemma in its equilibria?
> Both mutual cooperation and mutual defection are equilibria; in the PD only mutual defection is.

> [!card]- Why do C agents decline in a Stag Hunt population with 50% C and 50% D agents?
> Against random partners C earns 2 on average, D earns 2.5, so D spreads.

> [!card]- In the vacuum world with a Dump action, why does rewarding each unit of dirt sucked fail?
> The agent earns endlessly by dumping dirt and sucking it up again.

> [!card]- True or false: before Dump is added, the clean-square reward and the dirt-sucked reward give different vacuum behaviour.
> **False.** They give the same behaviour, so testing cannot tell them apart.

## Links

- **Course:** [[AGAI - Overview|Course overview]]
- **Reading:** [[AGAI-Book Ch2 - Intelligent Agents]] (Russell and Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., 2022, Chapter 2: agent definition, rationality, agent programs, the learning definition, the calculator caveat and the vacuum performance-measure example)
- **Lab:** [[AGAI-Lab01 - Simple Agents, Complex Dynamics]] for the full analysis of the Prisoner's Dilemma, Stag Hunt, population dynamics and the vacuum rewards
- **Source:** Agentic Artificial Intelligence, Lecture 1, "Introduction to the course / Overview", MSc AI, UvA, 27 October 2026
- **Papers and books cited:** M. Wooldridge, *An Introduction to Multiagent Systems* (2001). Yao et al., "ReAct: Synergizing reasoning and acting in language models", ICLR 2023, https://react-lm.github.io. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models", NeurIPS 2022. Guo et al., "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", *Nature* 2025. Kanoulas et al., "Agent-centric Information Access", https://arxiv.org/abs/2502.19298. Schick et al., "Toolformer: Language models can teach themselves to use tools", NeurIPS 2023. Model Context Protocol: https://www.anthropic.com/news/model-context-protocol. Huang et al., "Understanding the planning of LLM agents: A survey", arXiv:2402.02716 (2024). Shinn et al., "Reflexion: Language agents with verbal reinforcement learning", NeurIPS 2023. Singh et al., "The leaderboard illusion", NeurIPS 2026. Hardt, *The Emerging Science of Machine Learning Benchmarks* (Princeton, 2026). Hammond et al., "Multi-agent risks from advanced AI", arXiv:2502.14143 (2025). "OpenAI-Hugging Face Incident", technical report (2026); Hugging Face, "Anatomy of a Frontier Lab Agent Intrusion: A Technical Timeline of the July 2026 Incident" (27 July 2026). Park et al., "Generative agents: Interactive simulacra of human behavior", UIST 2023. Wu et al., "AutoGen: Enabling next-gen LLM applications via multi-agent conversations" (2024), https://arxiv.org/abs/2308.08155. Evans, Bratton and Agüera y Arcas, "Agentic AI and the next intelligence explosion", *Science* (2026). Miehling et al. (IBM Research), "Agentic AI Needs a Systems Theory".
- **Books the course draws on:** Shoham and Leyton-Brown, *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations*; Albrecht, Christianos and Schäfer, *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches*.
