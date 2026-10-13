---
type: lecture
course: 5354PHSC6Y
week: 7
lecture: 7
date: 2026-10-13
lecturer: Enrico Cinti
book_sections: ["Chen, Laws of Physics, sections 1-2.4, 3.1-3.2, 4.1-4.3"]
status: complete
topics:
  - Where the idea of laws of nature came from
  - Laws versus accidental generalisations
  - Counterfactuals, possible worlds and causation
  - Ontology, state space and nomological possibility and necessity
  - Determinism, strong determinism and chance
  - Maudlin's fundamental laws of temporal evolution and the objections
  - Humean versus non-Humean accounts
  - Chen's minimal primitivism and explanation by constraint
  - The Dretske-Tooley-Armstrong account (universals)
  - Dispositionalism
  - Lewis's Best Systems Account and the Humean mosaic
---

# PhilSci-L07: Laws of Nature

> [!abstract] In short
> What is a law of nature? "All Fs are Gs", true everywhere and always, is not enough: "all gold spheres are less than a mile across" is true and is no law. A law seems to *force* its truth, and that shows in what laws do: they back counterfactuals, fix what is physically possible, and decide whether the world is deterministic.
>
> **Non-Humeans** hold that a real, irreducible necessity stands behind the laws: in primitive laws that constrain the world (Chen), in relations between universals (Dretske, Tooley, Armstrong), or in dispositions. **Humeans** (Lewis) hold that laws are the axioms of the best summary of what actually happens, so the world fixes the laws.
>
> No mock exam question is on laws, but it is syllabus topic 6, so it is examinable. The practice questions below follow the mock's style.

## Where the idea of a law came from

Before modern science, God was thought to direct the world directly. Then modern science arrived with Galileo and Newton, and with it **equations of motion** that predict the future from the present, **deterministically**: give the positions and velocities now, and the equations fix everything after. At first sight this leaves God nothing to do.

The response was to call the equations **laws of nature** and to say that laws **govern** how the world evolves. God fixes the laws, the laws run the world, and God is still in control even if the evolution is deterministic.

That raised a new question: what *are* laws? Empiricists (who hold that all knowledge of the world comes from experience, see [[PhilSci-L01 - Introduction and Logical Empiricism|L01]]) were sceptical. David Hume's slogan is "no necessary connections between distinct entities": we see one event followed by another, never a necessity binding them. Against Hume and the **Humeans** stand the **non-Humeans**, who think the necessity in laws is real. Humeans still accept laws; what they deny is that laws carry a necessity of their own.

**Scope.** The debate here is about the **fundamental laws of physics**, not laws of the special sciences (biology, economics), which usually hold only *ceteris paribus*, "other things being equal" (optional Carroll reading). We have no candidate theory that supplies the fundamental laws (string theory is a hope at best), and we do not know there is one. So the approach is **suppositional**: given current science and the concept of a law, what might fundamental laws look like, knowing future science may overturn it.

## The puzzle: laws versus accidents

The obvious first answer is that a law is a **true universal generalisation**: a true statement of the form "all Fs are Gs". It fails, because of **accidental generalisations**: universal statements that are true by accident.

> [!example] The gold spheres
> "All gold spheres are less than one mile in diameter" is true: there is not that much gold gathered anywhere. But nothing, and in particular no law of nature, forces it to be true. It just happens to be the case.
>
> Compare (optional Carroll reading): "all uranium spheres are less than one mile in diameter". Same form, also true, but plausibly a law: uranium's critical mass rules out such a sphere. Nothing in the form of the two sentences tells them apart.

So an account of laws has to say what laws have that accidents lack: the sense in which laws **make** something true, where the gold-sphere statement just happens to be true.

## What laws do

Laws sit at the centre of a web of concepts, so taking a view on laws commits you on all of them.

### Laws back counterfactuals

A **counterfactual** is a statement "if A were to happen, B would happen", usually about an A that did not happen. Counterfactuals are **modal** statements: about possibility and necessity. They talk about what would happen in situations that did not occur, so they are claims about **possible worlds**: complete ways the world could have been.

How would you check whether one is true? A first try: in **every** possible world where A holds, B holds. Too strong: there are possible worlds where the match is struck and does not light (it is wet, there is no oxygen, chemistry works differently), yet "if this match had been struck, it would have lit" still seems true. A better try, from David Lewis and Robert Stalnaker (two philosophers who built the logic of counterfactuals in the 1970s): the worlds where A and B both hold are **more similar to ours** than the worlds where A holds without B. Only the closest A-worlds count.

> [!definition] Laws and counterfactuals
> Laws **support counterfactuals** because they fix a **crucial respect of similarity** between worlds: a world that keeps our laws is much closer to ours than one that breaks them. So the closest worlds where A holds obey our laws from A onwards. (Outside the course material: for a deterministic world, Lewis allows a small break in the laws just before A.) If our laws take A to B, then B holds in those worlds, and "if A were the case, B would be" is true. In short: the counterfactual is true when "A, then B" is an instance of a law.

> [!example] The match and the coin
> "If this match had been struck, it would have lit": true, because "struck matches (in normal conditions) light" is a law.
>
> Suppose every coin in my pocket happens to be silver. "If this copper coin were in my pocket, it would be silver" is false: in the closest worlds where I put it there, it stays copper. "All coins in my pocket are silver" is accidental, so it supports no counterfactual.

**Causation.** Hume, and later Lewis, used counterfactuals to give an account of causation: two events are causally related when there is **counterfactual dependence** between them. My striking the match caused it to light because, had I not struck it, it would not have lit. (Hume is better known for the regularity view, causation as constant conjunction, in [[PhilSci-L04 - Scientific Explanation and Understanding|L04]]; his definition of cause contains both. Eddy Keming Chen, philosopher of physics and author of this week's reading, *Laws of Physics*, notes that counterexamples have led many to drop the counterfactual analysis.)

### Laws pick out what is physically possible

If you are a scientific realist ([[PhilSci-L05 - Scientific Realism and its Critiques|L05]]), you take a theory at its word about what exists. Then a theory specifies two things:

- an **ontology**: the entities there are, which correspond to the theory's variables;
- **dynamics** for those variables, saying how they evolve. **These are the laws.**

Picture this with a **state space**: a space where every point is one possible state of all the theory's entities. For $N$ particles, a point lists all $N$ positions (and momenta). As the world changes, its state moves, so a **trajectory** through state space is a **history** of the entities.

Without laws, any curve through state space is a possible history. **Laws are rules that select only some histories as allowed.** Newton's $F = ma$ (force equals mass times acceleration) plus his law of gravitation, for instance, allow exactly the curves that solve their equations. Treat each history as a possible world, and laws select the allowed worlds.

> [!definition] Nomological possibility and necessity
> Something true in **some** of the law-allowed worlds is **nomologically possible**; something true in **all** of them is **nomologically necessary**. This is **nomological modality**: the possibility and necessity that come with the laws of nature ("nomological" from Greek *nomos*, law).

This restates the puzzle. A law is nomologically necessary: it holds in every law-allowed world. The gold-sphere statement is true in ours, but some law-allowed worlds contain a mile-wide gold sphere.

### Laws decide determinism

> [!definition] Determinism
> Roughly: if you know the state of the universe at one time and you know the laws, you know its state at any other time.
>
> In possible-worlds language: given a segment of a nomologically possible world, there is **only one** full nomologically possible world with that segment as part.

Picture it in state space: under deterministic laws the allowed trajectories never cross or branch, so a single point fixes the whole curve. So whether the world is deterministic is a fact about its laws.

- **Newtonian mechanics** is the classic deterministic theory.
- **General relativity** is the interesting case. Only its **globally hyperbolic** models are deterministic: those where the state at one time plus the Einstein field equations (which relate spacetime's curvature to the matter in it) fix the state at all times. Not every model of the theory is globally hyperbolic. So whether general relativity is deterministic depends on whether the other models count as **physically significant**.
- **Quantum mechanics**, on collapse interpretations (on measurement the quantum state jumps at random to one outcome), is **not deterministic**.

**Chance.** Enrico Cinti, the lecturer, draws a strong moral: a theory that fails to be deterministic involves **chance in an ineliminable way**. Probabilities are then part of what the laws say, and there is no way to get rid of them. Chen is more careful. An indeterministic theory *may* posit chances, but it does not have to. And chances can also sit on top of deterministic laws: statistical mechanics puts a probability distribution over the possible initial states of a deterministic world.

**Strong determinism.** Chen, following the physicist Roger Penrose, discusses a stronger notion: **only one world** is compatible with the laws. Ordinary determinism fixes the world once you give a segment; strong determinism fixes it with no input at all. So the theory itself says there is only one physically possible world. There are no real examples (string theory was hoped to have the property, but does not on any reasonable reading).

### Must laws push the world forward in time?

Tim Maudlin (philosopher of physics) argues that laws need a **direction of time** and work by **producing later states of the world from earlier ones**. Laws, for him, are **Fundamental Laws of Temporal Evolution (FLOTEs)**. $F = ma$ and the Schrödinger equation (quantum mechanics' equation for how the state changes) are FLOTEs.

Two objections:

1. **Most laws of physics are time-symmetric**: run any allowed history backwards and you get another allowed history. The direction of time is **emergent**, for example from thermodynamics (entropy, roughly disorder, increases in one direction). For Chen, this asymmetry comes from the **Past Hypothesis**: a law about the universe's boundary (its starting state), saying that at one end of time the universe was in a low-entropy state. That law is itself not a FLOTE.
2. **Not all laws describe evolution.** Some, like the Einstein field equations, are **constraints**: they say which whole spacetimes are allowed, without producing one moment from the last. Chen adds Gauss's law (a purely spatial law linking electric charge to the field around it) and principles of least action (which constrain a whole history between two times).

## The great divide: Humeans and non-Humeans

> [!definition] The two camps
> - **Non-Humeans:** there is an **irreducible** form of nomological modality in the world. Most non-Humeans add that the laws themselves **govern** the world; dispositionalists place the modality in dispositions instead (below).
> - **Humeans** deny both: laws **do not govern**, and nomological modality is **reducible to non-modal notions** (facts about what actually happens).

Non-Humean views come in more varieties than Humean ones. Three matter here, each with Chen's name for it:

- **Primitivism**: Maudlin's view, and Chen's **minimal primitivism**, a variant of it;
- **Universals**: the Dretske-Tooley-Armstrong account, which Chen calls **Platonic reductionism** (universals are like Plato's forms);
- **Dispositionalism**, which Chen calls **Aristotelian reductionism** (Aristotle grounded change in the powers of things).

> [!warning] Two kinds of reduction
> DTA reduces laws to **necessitation relations between universals**, dispositionalism to **dispositions**. The necessitation and the dispositions are still modal, so the necessity stays irreducible and the views stay non-Humean. Humeans **reduce the modality itself** to non-modal facts. So "reductive about laws" and "irreducible modality" are compatible.

## Non-Humean 1: minimal primitivism (Chen)

"Primitive" means we do **not analyse** a notion in terms of anything else. A primitive notion is conceptual rock-bottom. Saying laws are primitive gives up any chance of analysing them, in exchange for avoiding the complicated metaphysics any analysis brings.

Maudlin's primitive laws govern by **producing** later states from earlier ones (FLOTEs). Chen's govern differently.

**How Chen's primitive laws govern: by constraining.** Laws restrict the space of possibilities admissible for the theory's entities. The analogy is **mathematical axioms**: axioms govern mathematics by fixing what is allowed and what is not, without directing any evolution. $F = ma$ governs by limiting the physical possibilities to those compatible with $F = ma$ and requiring the actual world to be one of them.

> [!definition] Minimal primitivism (Chen, with Sheldon Goldstein)
> Fundamental laws of nature are certain **primitive facts** about the world. There is **no restriction on the form** of the fundamental laws. They **govern** the behaviour of material objects **by constraining the physical possibilities**.

**Why it is non-Humean:** the restriction comes from the laws and is irreducible, so laws really govern.

**Why "minimal":** compared with heavier primitivism like Maudlin's, it commits to the least irreducible nomological modality you can have and still be non-Humean. There are only the entities and a constraint on how they may combine. It sets no substantial conditions on what laws look like, unlike Maudlin's demand that they be FLOTEs. So time-symmetric laws and constraint laws like the Einstein equations fit without trouble.

**How primitive laws explain.** For Chen, laws explain by constraining; they need not produce one state from the last. Explanations are especially illuminating when a rich, surprising pattern follows from a simple rule. His toy case is a world whose matter is laid out like the Mandelbrot set: an intricate fractal that follows from one short rule (a number $c$ is in the set if repeatedly applying $z \mapsto z^2 + c$, starting from 0, stays bounded). Finding that rule would be a deep discovery, and it would be that world's law, although it says nothing about time. Because the law is a fact over and above the pattern, it can explain the pattern; a Humean law only summarises it.

Chen adds two points. Simplicity and informativeness are only **clues** that help us find the laws. For Humeans they are what makes something a law (below); for Chen they are not. And to the **mystery objection** (who imposes the laws?) he answers with the axiom analogy: $2 + 3 = 5$ constrains every possible world, and no agent imposes it. Chen grants that this does not remove every mystery. It answers only the worry that governing needs someone to do the governing.

## Non-Humean 2: universals (Dretske, Tooley, Armstrong)

A **universal** is, roughly, the abstract entity that encodes a property in full generality. Redness is a universal, and every red thing is an instance of it. The one universal is **wholly present in each red thing**, so universals are **multilocated**.

On the **DTA account** (after Fred Dretske, Michael Tooley and David Armstrong), laws are **relations of necessitation between universals**.

> [!definition] The DTA account
> A law has the form "**necessarily, if F then G**", where F and G are universals. Written $N(F, G)$: the universal F-ness **necessitates** the universal G-ness, and this makes it the case that all Fs are Gs.

$F = ma$ becomes, roughly: necessarily, if the universal force takes some value, the universals mass and acceleration take matching values. Insofar as the modality coming from the laws is irreducible, the account is non-Humean.

**Advantages.** It fits simple laws of the form "all Fs are Gs" naturally. And it is **reductive**: it analyses laws in terms of something more fundamental (universals). So it is **more explanatory than primitivism**: it says what laws are, where primitivism declines to.

**Problems.**

- It is unclear whether **all laws fit the schema**. What about laws that behave like constraints, such as the Einstein equations, or **path integrals** (a formulation of quantum mechanics that sums over all possible histories between two points)?
- Chen adds that the necessitation relation is **put in by hand**, simply postulated. Lewis objected that it is unclear why a relation $N(F,G)$ between universals should guarantee that all Fs are Gs: calling it "necessitation" does not make it necessitate.

## Non-Humean 3: dispositions

**Dispositionalism** analyses laws in terms of **causation** instead of universals, using a specific notion of causation built on **dispositions**.

> [!definition] Disposition
> A property that shows itself only under certain conditions. **Fragility** is the disposition to break when dropped or struck. The triggering condition is the **stimulus** (being dropped); what then happens is the **manifestation** (breaking). Fragility, solubility and electric charge are standard examples.

**Causation from dispositions:** certain events have a disposition to happen when certain other events happen, and the latter are their causes. The stone caused the window to break because windows have a disposition to break when hit by stones.

**Laws from causation:** exploit the link between causation and laws. For instance, Heather Demarest (a dispositionalist philosopher): laws are the **best possible summary of certain patterns of causal relations**. This sounds Humean, but what is summarised is causal relations grounded in modal dispositions, not a non-modal mosaic.

**Is it non-Humean?** Yes, insofar as dispositions are **irreducible and modal**.

- **Modal:** a disposition encodes a **tendency**: under certain conditions, something happens. It is there **before** anything happens (unstruck glass is still fragile), so it is about what would happen: it is **intrinsically modal**.
- **Irreducible:** less obvious. Dispositions are **not analysed further**. One could try to explain glass's fragility by some property of the glass that is not itself a disposition, such as how its molecules are arranged (a standard example from outside the course material). Friends of dispositions deny that such explanations are available at the fundamental level.

Keep one nuance apart: Chen says that on this view **laws themselves do not govern**. The necessity lives in the dispositions, and laws are derived from them. On this view, Chen notes, the laws are also metaphysically necessary: given the dispositions, they could not have been otherwise.

**Assessment.** It is **reductionist about laws** (laws analysed in terms of dispositions), so it has the **same advantage over primitivism** as DTA. It also has **similar problems**: irreducible dispositions are as mysterious as universals and primitive laws. And **modern physical theories may not fit the dispositionalist mould**. From a list of problems the dispositionalist Alexander Bird himself admits (reported by Chen): conservation laws say a total stays fixed, and nothing responds to a stimulus. Chen adds that a manifestation must follow its stimulus, which builds in the fundamental direction of time the first objection to Maudlin said laws need not have.

## Humeanism: the Best Systems Account

The main alternative to all three is **Humeanism**, today chiefly David Lewis's **Best Systems Account (BSA)**, also called the **Mill-Ramsey-Lewis** account (after John Stuart Mill and Frank Ramsey).

> [!definition] Best Systems Account
> Laws of nature are **the axioms of a system that provides the best balance of strength and simplicity** when accounting for the **mosaic of spatiotemporally related local matters of fact** of our world.

- **The Humean mosaic** is the collection of local matters of fact, each about a single point or small region, arranged in space and time: things like "a particle of mass $m$ is here". It is the **bedrock**: the ontological structure that determines the laws. Nothing in it is modal.
- **A system** is a set of true sentences about the mosaic, organised as axioms and their consequences.
- **Strength** is how much the system tells you about the mosaic; **simplicity** is how short and elegant it is. They pull against each other.
- **Natural properties.** Simplicity depends on vocabulary. Invent a predicate F true of exactly the things in the actual world: "everything is F" is maximally simple and says everything, so it would win. To block this, Lewis lets competing systems use only predicates for **natural properties**, such as mass, charge and spacetime location (Chen's illustration).

> [!example] Choosing the best system (Chen's illustration)
> Take a world of charged particles and an electromagnetic field.
> - **System 1** lists every fact, point by point. Maximally strong, hopelessly complicated.
> - **System 2** says "Things exist." Maximally simple, says almost nothing.
> - **System 3** is Maxwell's equations plus the Lorentz force law (the law for the force a field exerts on a charge). Fairly simple and very informative: the best balance.
>
> So, on the BSA, Maxwell's equations and the Lorentz force law are this world's laws.

This handles the puzzle: laws are distinguished from mere universal generalisations by belonging to the best balance of strength and simplicity. "All gold spheres are under a mile" would add a little strength at a real cost in simplicity, so it stays out. (Optional Carroll reading: in Lewis's own statement, theorems of the best system are laws too, as well as its axioms; the axioms are the fundamental laws. The uranium generalisation follows from the nuclear physics in the system, so it is a law.)

It also **reduces** laws, and with them nomological modality, to something **non-modal**: the mosaic.

> [!definition] The hallmark of Humeanism
> The relation between laws and world is **reversed**. For non-Humeans the laws govern, constrain or determine the world. For Humeans **the world determines the laws**.

**Advantages.**

1. **Nothing mysterious**: no universals, dispositions or primitive laws, and no governance. Laws are **effective summaries** of familiar local matters of fact.
2. **A natural link with science.** The BSA's balancing of competing summaries is an idealised version of what science actually does. Non-Humean accounts have no such link: nothing stops the laws that govern from **differing from the equations** our theories discover, so we would have no guarantee that science finds the real laws. (Chen's reply: we are never given the mosaic either. In modern physics it is as theoretical as the laws, and non-Humeans can use the same simplicity clues to find them.)
3. **Flexible.** Chen grants that, like minimal primitivism, it puts **no restriction on the form of laws**: any simple, informative axiom can win.

**Problems.**

- Not all physical theories fit the template of laws fixed by a mosaic of **local** facts. **Quantum entanglement** is the famous case: entangled particles have joint properties that are not fixed by the properties of each particle at its own location.
- **Empty spacetime** (Chen, after Maudlin): physics allows an empty, flat spacetime governed by general relativity's Einstein equations. On the BSA, the best summary of that empty mosaic is just special relativity, so the Einstein equations could not be its law. Scientific practice lets laws differ while the mosaic stays the same; the BSA rules that out.
- Chen adds that the patterns in the mosaic are **ultimately unexplained**, since the laws only summarise them. And what counts as "best" may depend on us, making laws mind-dependent.

## The accounts side by side

| | **Laws** | **Gain** | **Cost** |
|---|---|---|---|
| **Minimal primitivism** | Constrain possibilities | Fits any form of law | Unanalysed |
| **DTA** | Link universals | Says what laws are | Mysterious; constraint laws |
| **Dispositions** | Come from dispositions | Says what laws are | Mysterious; conservation laws |
| **Humean BSA** | Summarise the mosaic | No mystery; any form | Entanglement; empty spacetime |

## What to be able to write

- Laws versus accidents: gold spheres, and how laws make counterfactuals true.
- Nomological possibility and necessity; determinism, chance (Cinti and Chen), strong determinism.
- Maudlin's FLOTEs and the two objections.
- The two camps, and the two kinds of reduction.
- Each non-Humean account: core claim, why non-Humean, advantage, problem.
- The BSA: definition, natural properties, the reversed relation, advantages, entanglement, empty spacetime.

Model answers for these are in the practice questions below.

## Exam questions

> [!exam]- Practice question: Why is it not enough to say that a law of nature is a true universal generalisation? Use an example, and explain how counterfactuals help draw the line.
> **Key points:** Accidental generalisations are true universal statements (gold spheres); laws force truth, accidents happen to be true; counterfactuals are judged by the closest worlds (Lewis, Stalnaker); laws fix similarity, so the closest A-worlds keep our laws and the counterfactual holds; accidents support no counterfactual (copper coin).
>
> - **The proposal:** a law is a true statement "all Fs are Gs".
> - **Counterexample:** **accidental generalisations**, true universal statements whose truth is accidental. "All gold spheres are less than one mile in diameter" is true, but no law of nature forces it; it just happens to be the case. (Contrast, from the optional Carroll reading, "all uranium spheres are less than a mile": plausibly a law, because of critical mass.)
> - So an account of laws must say what laws have that accidents lack: laws **make** things true.
> - **Counterfactuals** ("if A were to happen, B would happen") are modal: claims about possible worlds. On the Lewis-Stalnaker account, one is true when worlds where A and B hold are more similar to ours than worlds where A holds without B.
> - **Laws support counterfactuals** because they fix a crucial respect of similarity: the closest worlds where the match is struck keep our laws, and in them it lights, so "if this match had been struck, it would have lit" is true. If every coin in my pocket happens to be silver, "if this copper coin were in my pocket, it would be silver" is still false: in the closest worlds it stays copper, since the generalisation is accidental.
> - So a generalisation that supports counterfactuals behaves like a law; one that does not is an accident.

> [!exam]- Practice question: Explain the contrast between Humean and non-Humean accounts of laws of nature, illustrating it with Lewis's Best Systems Account and one non-Humean account of your choice. Which do you find more convincing?
> **Key points:** Non-Humeans: irreducible nomological modality, usually laws that govern; Humeans: laws do not govern, modality reducible to the mosaic; BSA: axioms of the best balance of strength and simplicity, so the world determines the laws; a non-Humean account with its advantage and problem; argued verdict.
>
> - **Definitions:** non-Humeans hold that there is an **irreducible** nomological modality, and most add that laws **govern** the world (on dispositionalism, the dispositions carry the modality instead). Humeans deny both and reduce nomological modality to **non-modal** facts. Background: Hume's "no necessary connections between distinct entities".
> - **BSA (Lewis; Mill-Ramsey-Lewis):** laws are the axioms of the system with the **best balance of strength and simplicity** in accounting for the **Humean mosaic**, the spatiotemporal arrangement of local matters of fact ("a particle of mass $m$ is here"), using only natural properties. Example: a full list of facts is too complex, "things exist" too weak, Maxwell's equations plus the Lorentz force law strike the balance. Hallmark: **the world determines the laws**.
> - **Non-Humean example, minimal primitivism (Chen):** laws are primitive facts with no restriction on their form; they govern by **constraining** the possibilities, as axioms govern mathematics. They explain by revealing simple constraints beneath complex patterns (the Mandelbrot world).
> - **For Humeanism:** no mysterious entity or governance; mirrors the balancing science actually does; flexible about the form of laws.
> - **Against:** entanglement does not fit a mosaic of local facts; an empty spacetime governed by the Einstein equations is impossible on the BSA; patterns in the mosaic are left unexplained. Against primitivism: it says nothing about what laws are, and Chen grants governing keeps some mystery.
> - **Verdict (one option):** primitivism, since only a law over and above the pattern can explain it; the price is leaving "governing" unanalysed.

> [!exam]- Practice question: Explain the Dretske-Tooley-Armstrong account of laws and the dispositionalist account. In what sense are both reductive, and what advantage and problems do they share compared with primitivism?
> **Key points:** DTA: laws are necessitation relations between multilocated universals, N(F,G); dispositionalism: laws derived from dispositions via causation (window, Demarest); both reduce laws to something that is still modal (necessitation between universals, dispositions), so they stay non-Humean; reductive, so more explanatory than primitivism; shared problems: mysterious entities, laws of modern physics that do not fit.
>
> - **Universal:** the abstract entity encoding a property in full generality; redness is wholly present in every red thing, so universals are multilocated.
> - **DTA:** laws are **relations of necessitation between universals**: "necessarily, if F then G", $N(F,G)$. $F = ma$: necessarily, if force takes some value, mass and acceleration take matching values. Non-Humean insofar as this modality is irreducible.
> - **Dispositionalism (Chen: Aristotelian reductionism):** a disposition is a property that shows itself under a stimulus (fragility: breaking when dropped). Causation via dispositions: the stone broke the window because windows are disposed to break when hit. Laws recovered from causation, e.g. as the best summary of patterns of causal relations (Demarest). Non-Humean insofar as dispositions are **modal** (a tendency present before anything happens) and **irreducible** (not explained by non-dispositional properties such as molecular arrangement, at the fundamental level). On this view laws themselves do not govern; the dispositions carry the necessity.
> - **In what sense reductive:** both **reduce laws** to something more fundamental (necessitation between universals; dispositions), but the necessitation and the dispositions are modal, so the modality stays irreducible. Humeans, by contrast, reduce the modality itself.
> - **Shared advantage:** both say what laws are, which primitivism declines to do, so they are more explanatory.
> - **Shared problems:** the posited entities are as mysterious as primitive laws; modern physics may not fit (constraint laws like the Einstein equations and path integrals for DTA; conservation laws, a problem Bird admits, for dispositions). Also: the necessitation relation is put in by hand (Chen), and Lewis asked why $N(F,G)$ should guarantee that all Fs are Gs.

> [!exam]- Practice question: How are laws of nature connected to physical possibility and to determinism? Explain nomological necessity, determinism and strong determinism, with examples from physics.
> **Key points:** Laws select the allowed histories in state space; nomologically possible is true in some allowed worlds, necessary in all; determinism: a segment fixes the whole world; Newton yes, GR only globally hyperbolic models, collapse QM no, with Cinti and Chen on chance; strong determinism: only one physically possible world, no real examples.
>
> - A theory specifies an **ontology** (entities matching its variables) and **dynamics** (the laws). In **state space** each point is a possible state, and each trajectory a history.
> - **Laws select which histories are allowed.** Treating histories as possible worlds: true in some allowed worlds is **nomologically possible**, true in all is **nomologically necessary** (nomological modality). Laws are nomologically necessary; the gold-sphere generalisation is not.
> - **Determinism:** the state at one time plus the laws fixes the state at all times; in modal terms, a segment of a nomologically possible world belongs to only one such world.
> - **Examples:** Newtonian mechanics is deterministic. In general relativity only **globally hyperbolic** models are (state at one time plus the Einstein equations fix all times), so its determinism depends on whether other models are physically significant. Quantum mechanics on collapse interpretations is not deterministic. For Cinti, chance then enters the laws ineliminably; Chen: indeterminism may, but need not, bring chances, and chances can sit on deterministic laws (statistical mechanics).
> - **Strong determinism** (Chen, after Penrose): only **one** world is compatible with the laws, so there is a single physically possible world. No real examples (string theory does not qualify), but a live conceptual option since we do not know the fundamental laws.

> [!exam]- Practice question: Maudlin holds that laws of nature are Fundamental Laws of Temporal Evolution. Explain this view and the objections to it, and show how Chen's minimal primitivism avoids them.
> **Key points:** FLOTEs: laws produce later states from earlier ones, with a direction of time; objection: laws mostly time-symmetric, asymmetry emergent; objection: some laws are constraints (Einstein equations); minimal primitivism: laws govern and explain by constraining possibilities, with no restriction on form.
>
> - **Maudlin:** laws need a direction of time and **produce later states of the world from earlier ones**. Laws are **FLOTEs** ($F = ma$, Schrödinger's equation). This is a heavyweight primitivism: laws are primitive, and they govern by producing.
> - **Objection 1:** most laws of physics are **time-symmetric** (a reversed history is also allowed). Temporal asymmetry is **emergent**, e.g. from thermodynamics; for Chen, from the low-entropy Past Hypothesis, itself a boundary law that is no FLOTE.
> - **Objection 2:** not all laws involve dynamical evolution. The **Einstein field equations** are **constraints** on whole spacetimes (also Gauss's law, least-action principles).
> - **Chen's minimal primitivism:** laws are primitive facts that govern **by constraining** the space of possibilities, as axioms govern mathematics by fixing what is allowed. No restriction on their form, no direction of time presupposed. So time-symmetric and constraint laws fit, and laws explain by simple constraint (the Mandelbrot world has a law that says nothing about time).
> - It stays **non-Humean**: the constraint comes from the laws and is irreducible. And it is **minimal**: the least irreducible modality needed to be non-Humean, with no FLOTE requirement.

## Flashcards

> [!card]- What is Hume's slogan against the reality of laws?
> "**No necessary connections between distinct entities.**"

> [!card]- What is an accidental generalisation?
> A true universal generalisation whose truth is **completely accidental**: nothing forces it.

> [!card]- Give the standard example of an accidental generalisation.
> "All **gold spheres** are less than one mile in diameter."

> [!card]- What must an account of laws capture, beyond universal truth?
> The sense in which laws **force** something to be true, where accidents just happen to be true.

> [!card]- What is a counterfactual?
> A statement "**if A were to happen, B would happen**".

> [!card]- When is "if A were to happen, B would happen" true, for Lewis and Stalnaker?
> When **A-and-B worlds** are **more similar to ours** than A-without-B worlds.

> [!card]- Why do laws support counterfactuals?
> They fix a **crucial respect of similarity**: the closest A-worlds keep our laws, so the law's B follows.

> [!card]- Why is "if this copper coin were in my pocket, it would be silver" false, even if every coin in my pocket is silver?
> "All coins in my pocket are silver" is **accidental**: in the closest worlds the coin stays copper.

> [!card]- What is the counterfactual account of causation?
> Two events are causally related if there is **counterfactual dependence** between them.

> [!card]- What two things does a scientific theory specify, on a realist reading?
> An **ontology** (entities matching its variables) and **dynamics** for them: the **laws**.

> [!card]- What do laws do to the histories in state space?
> **Select only some histories as allowed.**

> [!card]- When is something nomologically possible?
> When it is true in **some** law-allowed possible world.

> [!card]- When is something nomologically necessary?
> When it is true in **all** law-allowed possible worlds.

> [!card]- What is nomological modality?
> The **possibility and necessity associated with laws of nature**.

> [!card]- Why is the gold-sphere generalisation not nomologically necessary?
> Some **law-allowed worlds** contain a mile-wide gold sphere.

> [!card]- What is determinism, roughly?
> Knowing the **state at one time** and **the laws**, you know the state at **any other time**.

> [!card]- Why is it open whether general relativity is deterministic?
> It depends on whether **non-globally hyperbolic models** count as **physically significant**.

> [!card]- On Cinti's view, what follows if a theory fails to be deterministic?
> It involves **chance ineliminably**: probabilities are part of the laws' content.

> [!card]- Must an indeterministic theory posit chances, according to Chen?
> **No**: it **may** posit chances, but need not.

> [!card]- What is strong determinism (Chen, after Penrose)?
> **Only one world** is compatible with the laws: a single physically possible world.

> [!card]- How is strong determinism stronger than ordinary determinism?
> Determinism needs a segment as input; strong determinism fixes the world with **no input**.

> [!card]- What does Maudlin take laws of nature to be?
> **FLOTEs**: laws that **produce later states from earlier ones**, with a direction of time.

> [!card]- Why does time symmetry count against Maudlin's view of laws?
> FLOTEs build in a direction of time, but most laws are **time-symmetric**; the direction is **emergent**.

> [!card]- Why do the Einstein field equations count against Maudlin's view of laws?
> They are **constraints** on whole spacetimes, not laws of evolution.

> [!card]- What do non-Humeans hold about laws?
> There is **irreducible nomological modality**; most add that **laws govern** (dispositionalists: dispositions carry it).

> [!card]- What do Humeans hold about laws?
> Laws **do not govern**; nomological modality **reduces to non-modal** facts.

> [!card]- How can DTA be reductive about laws yet non-Humean?
> It reduces **laws** to universals, but their necessitation stays **modal and irreducible**.

> [!card]- What do Humeans reduce that non-Humeans do not?
> The **modality itself**, to non-modal facts about the mosaic.

> [!card]- What does it mean to call laws primitive?
> They are **not analysed** in terms of anything else: conceptual **rock-bottom**.

> [!card]- How do laws govern on Chen's minimal primitivism?
> By **constraining** the possibilities admissible for the theory's entities.

> [!card]- What analogy explains governing by constraint?
> **Mathematical axioms**: they fix what is allowed, without directing any evolution.

> [!card]- Why is Chen's primitivism called minimal?
> It commits to the **least irreducible modality** needed to be non-Humean.

> [!card]- What is the difference between Chen's and Maudlin's primitivism?
> Maudlin's laws govern by **producing** (FLOTEs); Chen's by **constraining**, with no condition on form.

> [!card]- How do primitive laws explain, for Chen?
> By showing a rich pattern follows from a **simple constraint**, as in the Mandelbrot world.

> [!card]- Why can a primitive law explain a pattern where a Humean law cannot?
> It is a fact **over and above** the pattern; a Humean law only **summarises** it.

> [!card]- How does Chen answer "who imposes the laws?"
> Like axioms: **2 + 3 = 5** constrains every world, and no agent imposes it.

> [!card]- What is a universal?
> The **abstract entity** encoding a property **in full generality**, e.g. redness.

> [!card]- Which account makes laws relations of necessitation between universals?
> The **Dretske-Tooley-Armstrong** (DTA) account.

> [!card]- Why is the DTA account more explanatory than primitivism?
> It is **reductive**: it analyses laws in terms of **universals**, so says what laws are.

> [!card]- Which laws are hard to fit into the DTA schema?
> **Constraint** laws like the **Einstein equations**, and **path integrals**.

> [!card]- What was Lewis's objection to DTA's necessitation relation?
> Calling it "necessitation" does not explain why **N(F,G) guarantees all Fs are Gs**.

> [!card]- What is a disposition?
> A property that shows itself only under a **stimulus**, e.g. **fragility**.

> [!card]- How does dispositionalism explain why a stone broke a window?
> **Windows have a disposition to break when hit by stones.**

> [!card]- How does Demarest recover laws from dispositions?
> Laws are the **best summary of patterns of causal relations**.

> [!card]- Why are dispositions intrinsically modal?
> A disposition is present **before** anything happens: it is about what **would** happen.

> [!card]- In what sense are dispositions irreducible, for dispositionalists?
> At the fundamental level they are **not explained** by any **non-dispositional** property.

> [!card]- True or false: on dispositionalism, the laws themselves govern the world.
> **False.** The necessity lives in the **dispositions**; laws are derived from them (Chen).

> [!card]- Why do conservation laws trouble dispositionalism?
> A conserved total **responds to no stimulus**, so it is no manifestation (Bird's own list).

> [!card]- What is Lewis's Best Systems Account of laws?
> Laws are the **axioms** of the system with the **best balance of strength and simplicity** for the Humean mosaic.

> [!card]- What is the Humean mosaic?
> The **spatiotemporally related local matters of fact**: the bedrock that determines the laws.

> [!card]- What does strength measure in the Best Systems Account?
> How much the system **tells you about the mosaic**.

> [!card]- Why does Lewis restrict best systems to natural properties?
> Otherwise "**everything is F**", with F true of all actual things, would win.

> [!card]- What is the hallmark of the Humean account of laws?
> The relation is **reversed**: the **world determines the laws**.

> [!card]- How does Chen answer the Humean claim to a better link with science?
> We are never given the mosaic: it is **as theoretical as the laws**.

> [!card]- What is the most famous problem for Humeanism?
> **Quantum entanglement**: not all theories fit laws fixed by a mosaic of **local** facts.

> [!card]- What is the empty-spacetime argument against the Best Systems Account?
> An empty spacetime under the **Einstein equations** is possible, but its best summary is only **special relativity**.

## Links

- **Course:** [[PhilSci - Overview|Course overview]] · [[PhilSci - Tutorials|Tutorials]]
- **Earlier lectures:** [[PhilSci-L01 - Introduction and Logical Empiricism]] (empiricism, the tradition Hume belongs to) · [[PhilSci-L01b - Popper and Lakatos]] (Hume's problem of induction, and why no observations can verify a universal law) · [[PhilSci-L04 - Scientific Explanation and Understanding]] (Hempel's covering law model, which needs a general law in every explanation and so needs the law/accident distinction; Hume's regularity and counterfactual views of causation) · [[PhilSci-L05 - Scientific Realism and its Critiques]] (scientific realism, behind reading a theory as ontology plus laws) · [[PhilSci-L06 - Scientific Realism and the Pessimistic Meta-Induction]] (the previous lecture)
- **Source:** "Lecture 7: Laws of Nature", Enrico Cinti, 13 October 2026 (Canvas file `7 Laws of Nature`, `Assets/7 Laws of Nature.pdf`).
- **Reading:** Eddy Keming Chen, *Laws of Physics* (Cambridge Elements in Philosophy of Physics, 2024), only sections 1-2.4, 3.1-3.2, 4.1-4.3. Not on Canvas (library); used here from the arXiv preprint 2309.03484 (September 2023), whose section numbers match: introduction and the Great Divide section 1, ontology, nomology and possibility 2.1, counterfactuals and causation 2.2, dynamic production and the direction of time (FLOTEs, Past Hypothesis) 2.3, determinism, chance and strong determinism 2.4, minimal primitivism and the mystery objection 3.1, explanation by simple constraint (Mandelbrot world) 3.2, Humean reductionism (the three systems, natural properties, empty spacetime, the reply on epistemic access, unexplained patterns) 4.1, Platonic reductionism (DTA, Lewis's objection) 4.2, Aristotelian reductionism (dispositions, Demarest, Bird's problems) 4.3.
- **Optional reading:** John W. Carroll, "Laws of Nature", *Stanford Encyclopedia of Philosophy* (revised 2020): gold and uranium spheres section 1, systems and theorems as laws section 2, universals and Lewis's objection section 3, special-science and ceteris paribus laws section 10.
