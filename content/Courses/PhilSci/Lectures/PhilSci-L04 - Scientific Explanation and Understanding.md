---
type: lecture
course: 5354PHSC6Y
week: 4
lecture: 4
date: 2026-09-22
lecturer: Enrico Cinti
status: complete
topics:
  - Hempel's deductive-nomological model
  - Inductive-statistical explanation
  - Asymmetry, irrelevance and correlation objections
  - Causal-mechanical explanation (Salmon)
  - Unification (Friedman, Kitcher)
  - Van Fraassen's pragmatic account of explanation
  - Godfrey-Smith's contextualism about explanation
  - Eliminativism and reductivism about understanding
  - Pragmatic theories of understanding
  - De Regt and Dieks, CUP and CIT
  - Intelligibility and visualizability (Anschaulichkeit)
---

# PhilSci-L04: Scientific Explanation and Scientific Understanding

> [!abstract] Overview
> Why is the shadow of the flagpole 12 metres long? Because the pole is this tall and the sun is at that angle. Fine. Now: why is the flagpole this tall? Because its shadow is 12 metres long and the sun is at that angle. The second answer is absurd, and the most influential theory of explanation of the twentieth century cannot tell you why.
>
> The lecture has two halves. **Part A** is about *explanation*: Hempel's deductive-nomological model, which says to explain is to deduce from laws, the objections that killed it, and the two big replacements (causation and unification), plus van Fraassen's claim that explanation is not part of science at all. **Part B** is about *understanding*: whether "understanding" is anything more than having an explanation, and De Regt and Dieks's answer that it is a *skill*, the ability to use a theory, measured by whether you can see what it implies without calculating.
>
> Part A is examined directly: mock exam question 7 is Hempel's model plus one objection. Part B has no mock question, but it takes up half the deck and the whole second half of tutorial 4, and the week's optional reading (De Jong and De Haro on technological understanding) is by the course coordinator, so do not skip it.

## Part A: Scientific explanation

## 1. How explanation became a philosophical topic

**Before 1948**, the idea that science *explains* phenomena was not a topic the logical empiricists took seriously. The reasons are worth knowing, because the whole D-N model is shaped by them.

- **Aristotle's** theory of the four causes is really a theory about the *structure of explanations* (the reading comes from Moravcsik). So "explanation" arrived in philosophy already tangled up with causes and essences.
- **Duhem** associated explanation with metaphysics. To explain is to claim to know what is really behind the appearances, and that "gives hostages to fortune": it makes science dependent on metaphysics. "Atoms exist" and "there is an aether" are, for him, metaphysical claims that science can neither confirm nor refute. So explanation is not the job of physics. (This is the same Duhem as in [[PhilSci-L03 - Under-determination]].)
- **Carnap** rejects "metaphysical causes" and metaphysical why-questions of the kind he finds in Hegel: these are pseudo-explanations, in the same way metaphysical statements are pseudo-statements (see [[PhilSci-L01 - Introduction and Logical Empiricism]]). But he admits a respectable **"empiricist explanation"**.
- For the **logical empiricists**, that respectable kind of explanation is closely linked with **prediction from laws**.

**1948:** Carl Hempel and Paul Oppenheim publish the **covering law model**. The lecturer calls it a "philosophically light" (epistemic) account of explanation: it says nothing about hidden causes or essences, only about the logical relation between statements.

> [!intuition] Why "light" was the point
> Godfrey-Smith's reading puts it well: the positivists "made peace with the idea that science explains" by construing explanation "in a low-key way that fitted into their empiricist picture". An explanation, on this view, is just an argument. Nothing metaphysical is smuggled in. That is exactly what makes it attractive to an empiricist, and, as section 3 shows, exactly what makes it fail.

## 2. Hempel's deductive-nomological model

### The definition

> [!definition] Deductive-nomological (D-N) explanation
> An **explanation** is a deductive argument in which the **explanandum** (the phenomenon or law to be explained) is deduced from premises containing **general laws** and **particular facts and initial/boundary conditions** (together, the **explanans**).

Terminology, since the exam uses it: if we ask "why $X$?", $X$ is the **explanandum**. If we answer "because $Y$", $Y$ is the **explanans**.

> [!formula] The D-N schema
> $$\begin{array}{lll} \textit{Explanans:} & C_1, C_2, \dots, C_k & \text{particular facts / conditions} \\ & L_1, L_2, \dots, L_r & \text{general laws} \\ \hline \textit{Explanandum:} & E & \text{phenomenon} \end{array}$$

where:
- $C_1, \dots, C_k$: statements of particular facts, initial conditions or boundary conditions
- $L_1, \dots, L_r$: statements of general laws
- the horizontal line: logical deduction
- $E$: the statement describing the phenomenon to be explained

The name: **deductive** because we go from premises to a conclusion that follows logically; **nomological** because it uses laws of nature (Greek *nomos*, law). This is the deductive variant of the more general **covering law model**: the phenomenon is "covered" by a law.

### The four conditions of adequacy

For the argument to count as an explanation:

1. The explanandum must be a **logical consequence** of the explanans.
2. The explanans must contain a **general law**, and must use it in an **essential** way (it is indispensable: remove it and the deduction fails).
3. The explanans must have **empirical content**: it must be capable, at least in principle, of test by experiment or observation.
4. The explanans must be **true**.

Conditions 1 to 3 are logical. Condition 4 is empirical. Godfrey-Smith's gloss: the first task is to say what *sort* of statements would explain if true; truth is then needed for the explanation to be good "in the fullest sense".

### The inductive-statistical variant

The covering law model also has an **inductive-statistical (I-S)** version, for when at least one of the laws is **probabilistic**. The explanandum is then a true singular fact whose **high probability** is shown by the explanans. The argument is not deductively valid, but it makes the explanandum highly expected.

### Worked example: the front door

The lecturer's example. *Lately I have difficulty opening and closing my front door. Why?*

> [!example] A D-N explanation
> $$\begin{array}{ll} C_1: & \text{My front door is made of wood.} \\ L_1: & \text{Wooden objects expand if humidity increases.} \\ C_2: & \text{Humidity has increased this autumn.} \\ \hline E: & \text{My front door has expanded this autumn.} \end{array}$$

Everything the model asks for is there: a law ($L_1$) used essentially, particular conditions ($C_1, C_2$), empirical content, a valid deduction.

The point the slide draws from it: **Hempel holds that explanation and prediction are symmetric. They have the same logical structure.** Had I known in summer that the door is wooden and that humidity would rise, I could have *predicted* the sticking door with exactly the same argument. The only difference between explaining and predicting is whether you already know the conclusion is true.

### Worked example: chemistry from the periodic table

Given the periodic table and the laws governing electrons in atoms, plus additional empirical assumptions about the composition of materials, we can deduce properties of chemical substances:

- the **low chemical reactivity of the noble gases** (a full outer shell),
- the **high electrical conductivity of metals** (loosely bound outer electrons).

The slide shows the standard 18-column periodic table (periods 1 to 7, lanthanides and actinides below) as the "law" doing the work. This is D-N explanation of *regularities*, not just of single events: a law (or a pattern) can itself be an explanandum. Godfrey-Smith's example of the same kind is Newton explaining Kepler's laws from the laws of mechanics plus facts about the solar system.

### Comments on Hempel's model

- **Laws do not need to be causal.** Functional laws are admitted: $PV = nRT$ explains without saying that pressure *causes* volume or the reverse.
- **The model is normative.** It is not meant to *describe* how scientists actually explain, but to state the ideal of a **good scientific explanation**. The analogy is the logician's concept of **proof**: mathematicians rarely write fully formal proofs, and that does not make the formal concept of proof useless.
- **Elliptically formulated explanations.** Scientists often give **incomplete** explanations, leaving laws or conditions implicit. Hempel counts these as abbreviations of a full D-N argument, rather than counterexamples to it.

## 3. Objections to the D-N model

Three objections, all of which appear in the mock exam's model answer.

### Objection 1: asymmetry

> [!warning] Asymmetry of explanations, but (sometimes) symmetry of deductions
> Deductions can often be run in both directions. Explanations cannot. So the D-N model, which is *only* about deduction, lets in explanations that are obviously backwards.

**The flagpole.** The slide's figure shows a flagpole with the sun behind it: a dashed ray from the sun grazes the top of the pole and hits the ground at the tip of the shadow, at an elevation angle of about $30°$, with the shadow along the ground marked in feet. Using trigonometry, the same laws support two deductions:

```
   Laws of optics                  Laws of optics
   Laws of geometry                Laws of geometry
   Position of the sun             Position of the sun
   Length of the flagpole          Length of the shadow
   ----------------------          ----------------------
   Length of the shadow            Length of the flagpole

   explains  (good)                predicts, but does not explain
```

In symbols, with $h$ the pole's height, $s$ the shadow's length and $\theta$ the sun's elevation: $s = h / \tan\theta$ and equally $h = s \tan\theta$. Both are valid D-N arguments. Both satisfy all four conditions of adequacy. Only the left one is an explanation: the pole explains the shadow, the shadow does not explain the pole. We can interchange flagpole and shadow **to predict, but not (always) to explain**.

Godfrey-Smith calls this "something close to a knockdown argument" and "the killer". The objection is due to Sylvain Bromberger (1966), with a slightly different example. Two further points from the reading:

- **Symptoms.** If only disease $D$ produces symptom $S$, you can infer $D$ from $S$ with a law. But a symptom never explains its disease. Prediction runs both ways; explanation only from $D$ to $S$.
- **Hempel's reply** was to bite the bullet: if his theory lets an explanation run both ways, both directions must be fine. That is defensible for some cases in physics where the direction is genuinely unclear, and hopeless for the flagpole. (The one exception Godfrey-Smith allows: a very unusual flagpole designed to regulate its own height to cast a shadow of a particular length. Then the shadow, as a goal, does explain the height.)

### Objection 2: irrelevance

An argument can satisfy the D-N conditions while containing **irrelevant information that robs it of explanatory power**.

> [!example] Hexed salt (Salmon, Achinstein)
> $$\begin{array}{ll} L: & \text{Every sample of sodium chloride hexed by a magician dissolves in water.} \\ C_1: & \text{This sample was hexed by a magician.} \\ C_2: & \text{This sample was placed in water.} \\ \hline E: & \text{This sample dissolved.} \end{array}$$
>
> The law is true (every sample of salt dissolves, hexed or not). The deduction is valid. But the hexing explains nothing. The D-N model has no way to rule out premises that are true and used, but explanatorily irrelevant.

A second standard example from the same literature, not on the slides: "Every man who regularly takes birth-control pills fails to get pregnant. John takes them. So John did not get pregnant." Valid, lawlike, true and irrelevant.

### Objection 3: correlation is not explanation

> [!example] The barometer
> $$\begin{array}{ll} L: & \text{If the barometer drops, a storm will occur.} \\ C: & \text{The barometer drops.} \\ \hline E: & \text{A storm will occur.} \end{array}$$
>
> This is an excellent *prediction*. It is not an explanation: the barometer does not cause the storm. Both are effects of a common cause, the drop in atmospheric pressure. **Correlation $\neq$ explanation.** Showing that something was *to be expected* is not the same as showing *why* it happened.

The mock answer phrases this as "apparently missing causation (correlation or expectation $\neq$ explanation)". It also lists a fourth objection that follows from the same line of thought: **explanations do not always involve laws**. Citing a causally relevant fact can be enough ("the window broke because the ball hit it"), with no law in sight.

The reading adds a problem for the I-S version: good explanations need not confer high probability. Paresis is explained by untreated syphilis, even though only a minority of untreated syphilis cases develop paresis. So "showing the explanandum was highly probable" is neither necessary nor sufficient.

> [!summary] Where all three objections point
> In every case, what is missing is **causation**. The pole causes the shadow, not the reverse. The hex causes nothing. The barometer and the storm share a cause. That is the lead the first alternative theory follows.

## 4. Alternative theories of explanation

The slide lists four directions:

1. Explanation related to **causation**: good explanations show what caused the phenomena.
2. Explanation related to **unification**: good explanations show how the phenomena fit into a broader pattern.
3. Explanation is **pragmatic**: it depends on the *context* whether answers to why-questions are satisfactory explanations.
4. Distinguish **scientific explanation** (which requires a scientific theory) from explanation in daily life.

### 4.1 Causal theories of explanation

> [!definition] Causal theory of explanation
> **Explanation = uncovering the causes of phenomena.** "Causal processes, causal interactions, and causal laws provide the **mechanisms** by which the world works; to understand why certain things happen, we need to see how they are produced by these mechanisms." (Wesley Salmon, 1984)

This solves the flagpole at once: sunlight hitting the pole causes the shadow, so the explanation runs pole to shadow.

The price is that you now need to say **what causality is**, which empiricists since Hume have regarded as suspect. Three families of answer:

| Theory | Causation is... | Names on the slide |
|---|---|---|
| **Regularity** | **Constant conjunction**: $C$ is regularly followed by $E$. This produces an **expectation** in us, not a necessary connection in nature | Hume |
| **Counterfactual** | Based on the intuition that **if the cause $C$ hadn't occurred, the effect $E$ wouldn't have occurred** | Hume (who states the idea in passing), J. Woodward |
| **Process** | One can identify **causal processes and interactions** in nature (think of a chain of falling dominoes, the slide's picture) | W. Salmon |

Note that the regularity theory brings the barometer problem straight back: barometer drops are constantly conjoined with storms.

### 4.2 Causal-mechanical explanation (Salmon)

- An explanation **situates the explanandum in a network or nexus of causal relations**, the causal structure of the world. To explain is to **systematically causally relate** the explanandum to other items.
- Example: "The water evaporated because heat was applied to it, and the van der Waals bonds between the molecules were broken." This **exhibits the causal structure**.
- Giving an explanation is **answering a why-question with "because..."**, hence citing the causes of the explanandum.

Two problems the lecturer raises:

1. **How do we distinguish causation from correlation?** This is **Hume's problem**, and the causal theory inherits it whole.
2. **Too narrow?** Perhaps not all explanations need to invoke causation (a point De Regt and Dieks press, see Part B). Explanations in quantum theory, or explanations of a law by a more general law, are hard to phrase causally.

Godfrey-Smith adds a refinement worth having: the idealised **complete** causal explanation of anything (Railton 1981) would contain its entire causal history in total detail. Nobody wants it or knows it. In practice, context determines which **relevant pieces** of the causal structure a good explanation needs to describe.

### 4.3 Unificationist theories of explanation

> [!definition] Unification
> Phenomena are explained by **fitting them into a broader pattern**. "Science advances our understanding of nature by showing us how to derive descriptions of many phenomena, using the same patterns of derivation again and again, and, in demonstrating this, it teaches us how to **reduce the number of types of facts we have to accept as ultimate** (or brute)." (Philip Kitcher, 1989)

- The idea was developed by **Michael Friedman (1974)** and **Kitcher (1981, 1989)**. Godfrey-Smith notes it was an "unofficial" theory inside logical empiricism all along, and "a good deal better than the official theory".
- Kitcher on understanding: it is "not simply a matter of reducing the 'fundamental incomprehensibilities' but of seeing **connections, common patterns**, in what initially appeared to be different situations".
- An explanation is a certain type of reasoning, an **argumentative pattern**: an argument that proceeds from some simple, unified principles to a multiplicity of (possibly miscellaneous) events.
- **Advantage:** it accounts for **non-causal explanations**, for example explanations in quantum theory.
- The flagpole, on Kitcher's view: our causal talk is a loose summary of deeper asymmetries in unification. Deriving shadows from poles belongs to a pattern that covers vastly more phenomena than deriving poles from shadows.
- Historical support: Darwin's theory of evolution and Newton's later work on matter were compelling to scientists before they made many specific new predictions, because of their **explanatory promise**, the ability to unify a great range of phenomena with a few principles.

The slide's two figures are both about physics unifying forces and theories.

**Figure 1, "How gauge theory unifies the fundamental forces of nature"**, a merging-lines diagram:

```mermaid
flowchart LR
  E[electricity] --> U1["U(1): Maxwell"]
  M[magnetism] --> U1
  U1 --> EW["SU(2)⊗U(1): Weinberg-Salam"]
  W[weak force] --> EW
  EW --> GUT["SU(5)⊗O(10)? gauge-unified theory"]
  S["strong force, SU(3)"] --> GUT
  GUT --> SS["superstring? Osp(N/4)"]
  G["gravitation, GL(4)⊗O(3,1)?"] --> SS
```

The figure also labels the electroweak step "Yang-Mills-Shaw", after the gauge theory it is built on. Each merge is a unification: electricity and magnetism into Maxwell's electromagnetism, U(1); that with the weak force into the electroweak theory of Weinberg and Salam, SU(2)⊗U(1); adding the strong force, SU(3), gives a hypothetical grand unified theory; adding gravity gives a hypothetical superstring theory. The question marks are on the slide: the last two steps are speculative.

**Figure 2, M-theory**, drawn as a six-pointed star with "M-theory" in the middle and the six known limits at its points: 11D supergravity, $E_8 \times E_8$ heterotic, $SO(32)$ heterotic, Type I, Type IIB, Type IIA. Five superstring theories and one supergravity theory, previously thought distinct, are presented as limits of one underlying theory. Unification as explanation in its purest form.

### 4.4 Causation or unification? Godfrey-Smith's contextualism

From the week's reading (Godfrey-Smith 2003, ch. 13), and the subject of tutorial 4:

- The two proposals have been treated as competitors ("does causation win or does unification win?"). That is a mistake. Much of the time explaining means describing causal mechanisms or histories; sometimes there are clear explanatory relations between patterns or principles where causal language is hard to apply, and unification does the work. Salmon eventually accepted unification as part of the story; Kitcher eventually accepted causation.
- That familiar **pluralism** is a step in the right direction, but Godfrey-Smith goes further. The mistake is to think there is **one** special explanatory relation, or a fixed short list of two or three.
- His view is **contextualism**: the standards for good explanation are **partially dependent on the scientific context**. Different fields, and the same field at different times, establish their own criteria. The standards in field A need not suffice in field B.
- This is Kuhn's view (1977), with Kuhn's example: did Newton's gravity explain falling bodies, given that it offered a mathematical law and no mechanism? Some said no; over time it became part of Newtonianism that the right kind of mathematical law **counts** as an explanation. (This is the same history as the gravitation case study in section 8.)
- It is not "anything goes". A conception of explanation can embed a **factual error**: if good explanations must cite God's will and there is no God, that conception is mistaken.
- So the covering law theory is dead **as a general account**, but some explanations really do have roughly its form. The mistake was applying it to every case.

### 4.5 Bas van Fraassen: explanation is pragmatic

Van Fraassen's position, from *The Scientific Image* (1980):

- Explanation and understanding belong to the **pragmatic dimension** of science. They are **contextual**: they depend on our aims and preferences.
- Explanation is part of the **reasons we may have to *accept* a theory** as useful for particular purposes (making predictions, answering questions).
- Explanations **do not add to our *beliefs* about the relation between the theory and the world**.
- So **to explain is not an epistemic aim of science**. It is a pragmatic dimension of **theory acceptance**.

The long quotation on the slide, in three moves:

1. **Duhem** argued that explanation is not an aim of science, and in retrospect **fostered the very explanation-mysticism he attacked**, by arguing that only metaphysical theories explain and that metaphysics is foreign to science. Fifty years later, once **Quine** had argued there is **no demarcation between science and philosophy**, and the ametaphysical stance of the positivists had run into trouble, a return to metaphysics became tempting: one noticed that **scientific activity does involve explanation**, and Duhem's argument was "deftly reversed" (science explains, so science involves metaphysics).
2. "Once you decide that explanation is something irreducible and special, the door is opened to elaboration by means of further concepts pertaining thereto, all equally irreducible and special." **Not everyone has joined this return to essentialism or neo-Aristotelian realism, but some eminent realists have.**
3. "The discussion of explanation went wrong at the very beginning when explanation was conceived of as a relationship like description: a relation between theory and fact. Really it is a **three-term relation, between theory, fact, and context**. An explanation is an answer... So **scientific explanation is not (pure) science but an application of science. It is a use of science to satisfy certain of our desires.**"

> [!warning] Two "pragmatic, contextual" views that are opposites
> Godfrey-Smith and van Fraassen both say explanation varies with context, and that is where the agreement ends. For **van Fraassen**, explanation is **external** to science: something people do *with* a theory to answer questions from outside scientific discussion, adding nothing to what we believe about the world. For **Godfrey-Smith**, explanation is **thoroughly internal** to science, and assessments of explanatory power are an important part of scientific reasoning, but different fields use different standards. Tutorial 4 asks exactly this ("is explanation a crucial notion for the inner workings of science, or rather something external to it?").

Why this matters later: van Fraassen's constructive empiricism (week 5) needs explanation to be non-epistemic, because otherwise **inference to the best explanation** would push him to believe in unobservables. Mock exam question 8 is Musgrave's reply to exactly this argument.

## Part B: Scientific understanding

## 5. Background views: is understanding anything over and above explanation?

### Hempel's eliminativism about understanding

> "Such expressions as 'realm of understanding' and 'comprehensible' do not belong to the vocabulary of logic, for they refer to the psychological and pragmatic aspects of explanation." (Hempel)

- **Explanation** is an **objective** relation between a theory $T$ and a phenomenon $P$.
- **Understanding** by a subject $S$ is **epistemically irrelevant**.
- "Pragmatic" is used **synonymously with "subjective"**.
- An investigation of understanding gives us knowledge of people's preferences and interests, not of the topic. It is **of interest to psychology, not philosophy**.

### Reductivism about understanding

The milder position: **explanation is understanding enough**, so there is no need for a separate theory of understanding.

- **Khalifa (2012):** understanding is a **form of knowledge**, reducible to explanation and cognate notions. Nevertheless **philosophically interesting**, which separates him from Hempel.
- **Lipton (2004):** "understanding is not some sort of super-knowledge, but simply more knowledge: knowledge of causes."
- **Trout (2002):** understanding as a *Eureka!* experience, the feeling of understanding, is the result of **cognitive biases**, for example overconfidence and hindsight. So the feeling is no guide to anything epistemic.

### Carnap (1939, 1947): three senses of "understanding"

"Understanding" is a vague word that requires explication. Three senses:

| Sense | What it is | Carnap's verdict |
|---|---|---|
| **1. Pragmatic** | **Capability of use** of a theory for the description and prediction of facts | Legitimate (1939) |
| **2. Subjective / metaphysical** | **Intuitive understanding**, the feeling of grasping | Rejected (1939) |
| **3. Epistemic** | To understand a language system is to **know its semantic rules / truth conditions**. Epistemic because it says what one must **know** to count as understanding | Legitimate (1947) |

Note that sense 1 is already close to where Part B ends up: understanding as the ability to *use* a theory. Even a logical empiricist allowed it.

### Two meanings of "pragmatic"

**Friedman (1974)** and **Woodward and Ross (2021)** point to an **equivocation** on "pragmatic" in Hempel and van Fraassen. Distinguish:

1. **"Subjective"**: varying with an individual's psychology.
2. **"Use(able) for certain aims"**: this is **compatible with an objective (inter-subjective) notion** of understanding.

On the second notion, **understanding can be an epistemic aim of science, achieved by adequate explanations**. Hempel's move from "pragmatic" to "epistemically irrelevant" only works if you read "pragmatic" in the first sense.

This leaves the question the rest of the lecture answers: **what is the relation between explanation and understanding?**

> [!tip] The quote the tutorial built a question around
> "Contra Hempel, van Fraassen, and Trout, we hold that the pragmatic nature of understanding is not inconsistent with it being epistemically relevant." (De Regt and Dieks, p. 141). This sentence is the whole of section 5 compressed: the three people named are the eliminativist, the pragmatist about explanation, and the bias reductivist, and the move that answers all three is the distinction between the two senses of "pragmatic".

## 6. Pragmatic theories of understanding

Requirements for understanding common to the various theories:

1. **Explanation**: there is a scientific explanation of the phenomenon, often in Hempel's sense. A bridge between phenomena and theory: a deduction, argument or model.
2. **Requirements of adequacy**: theoretical virtues or epistemic values, usually **internal consistency** and **empirical confirmation** (also: approximate truth).
3. **Useability**: it should be **possible to construct** adequate explanations.

The dividing line:

- Authors who **reduce** understanding to explanation accept only (1) and (2): **understanding = adequate explanation = knowledge**.
- Authors for whom understanding is **"more than"** explanation add (3).

So the discussion focusses on (3), and (3) is **not about *knowledge* but about the *ability to act or do***: it involves **skills and judgement**.

### Requirement (3) and objectivity: three levels

The worry about (3) is that skills belong to individuals, which makes understanding subjective again. To emphasise that (3) is objective, **De Regt and Dieks (2005)** distinguish three levels of analysis of the scientific community:

> "The **macro-level** of science as a whole; the **meso-level** of the scientific communities; and the **micro-level** of individual scientists... The three-level distinction reconciles the existence of **universal aims of science** with the existence of **variation** in the precise specification and/or application of these general aims."

```
MACRO   science as a whole          understanding is a universal aim of science
  │
MESO    scientific communities      standards of intelligibility are set HERE,
  │     (a discipline, a period)    and vary between communities
  │
MICRO   individual scientists       who have or lack the skills
```

- **Understanding is a macro-level (universal) aim** of science.
- **Standards of intelligibility** are not universally fixed for all of science: they vary across scientific communities, at the **meso-level** of a discipline.
- They are **objective, but relative to the level of progress** of a given discipline, that is, contextual. Not a matter of individual taste.

## 7. De Regt's contextual theory of understanding

### The grammar of understanding

> [!definition] The basic form
> **Scientist $S$ (in context $C$) understands phenomenon $P$ on the basis of theory $T$.**

Compare Hempel, for whom explanation is a two-place relation between $T$ and $P$. Understanding is at least four-place.

- Understanding is **pragmatic**: it involves a relation to a subject and a context.
- **Contextuality**: variation is possible. The same $T$ can yield understanding for one community and not another.

### Model-based explanation

Explanation in real science rarely goes straight from theory to phenomenon. It goes through a model:

$$T \sim M \sim P$$

- $T$: the theory
- $M$: a model that **represents $P$ such that $T$ can be applied to it**
- $P$: the phenomenon

There are **no algorithms and no strict rules for building models**. Instead: **approximation, idealisation and pragmatic decisions**. So $S$ needs **skills** for constructing $M$ to explain $P$.

> [!definition] Understanding
> **Understanding = the skill to use a theory for building models to explain phenomena.**

### Explaining phenomena requires intelligible theories

If $S$ wants to explain a phenomenon on the basis of $T$, she needs appropriate skills to use $T$. So **$T$ should be *intelligible* to $S$.**

> [!definition] Intelligibility
> **Intelligibility** = the value that scientists attribute to the qualities of a theory $T$ that facilitate the use of the theory.
> - Not an intrinsic property of theories, but a **context-dependent value related to scientists' skills**.
> - Example: **visualizability**.

### The two criteria

> [!formula] CUP: Criterion for Understanding Phenomena
> A phenomenon $P$ is **understood** scientifically **iff** there is an explanation of $P$ that is based on an **intelligible theory** $T$ and conforms to the basic epistemic values of **empirical adequacy** and **internal consistency**.
>
> Scientists often understand phenomena by constructing models of them (including simulations).

> [!formula] CIT: Criterion (test) for the Intelligibility of Theories
> A scientific theory $T$ (in one or more of its representations) is **intelligible** for scientists (in context $C$) if they can **recognise qualitatively characteristic consequences of $T$ without performing exact calculations** (or fully explicit theoretical argumentation).
>
> The idea is that scientists have an "insight" into the workings of the theory, and are accordingly able to use it to construct models of the phenomena that satisfy the basic values of empirical adequacy and internal consistency.

How the pieces fit: **CUP** says understanding a phenomenon needs an intelligible theory. **CIT** gives a test for when a theory is intelligible. The test is a skill test, and it is sensitive to $S$ and $C$.

> [!example] Passing the CIT
> The standard illustration is the kinetic theory of gases. A physicist who has the theory can say without calculating anything that heating a gas in a closed container will raise its pressure: the molecules move faster, hit the walls harder and more often. That qualitative, calculation-free prediction is what "intelligible" means here. Someone who can only get there by solving the equations has the theory but, by the CIT, does not find it intelligible.

> [!question] Does CIT make understanding subjective? (tutorial 4)
> No, and the three-level picture is the reason. The criterion is relative to scientists **in a context**, but the context is the **meso-level community** with its shared standards and trained skills, not one person's feelings. Whether a physicist can recognise qualitative consequences without calculating is a public, testable fact about her competence, which is exactly what Trout's "feeling of understanding" is not. The honest concession: it makes intelligibility **contextual** (relative to a community and its level of progress), and critics can press whether "contextual" collapses into "subjective" when communities disagree, as in the 1926 quantum case below.

Kinds of conceptual toolkit that help with qualitative reasoning, and the level they operate at (tutorial 4 asks for these): visualisation and diagrams (Feynman diagrams, the Bohr picture of the atom), causal-mechanical stories, analogies, thought experiments, toy models, and simulations. They are taught and shared within a discipline, which puts them at the **meso-level**; individuals at the micro-level have mastered them to different degrees.

## 8. Historically differing standards of intelligibility

Two case studies, making two different points:

| Case | Standards of intelligibility differ... |
|---|---|
| **Theories of gravitation**, Newton (1687) to Einstein (1915) | **Diachronically**: over time, historically |
| **Quantum theory** around 1926 | **Synchronically**: at the same time, between different scientists |

### Case 1: gravitation

**Newton's theory of gravitation (1687).** The slide's figure shows two masses $m_1$ and $m_2$ a distance $r$ apart, with forces $F_1$ and $F_2$ pointing towards each other. The force acts across empty space instantly: **action at a distance**.

$$F = G\,\frac{m_1 m_2}{r^2}$$

**Christiaan Huygens** found this unintelligible:

> "I look for an *understandable* cause of gravitation, because it seems to me that to say that bodies fall down because of some gravitational attraction, of earth or of those bodies, is to **say nothing**."

For Huygens, a Cartesian mechanist, the standard of intelligibility was **contact action**: bodies push bodies. A force across empty space was an occult quality.

**Around 1800**, the standard had flipped: **action at a distance becomes the ideal of understanding.** The example is **Coulomb's law**:

$$F = k\,\frac{q_1 q_2}{r^2}$$

The slide makes the point with a meme: Coulomb, in an exam hall, copying Newton's answer sheet. Same form, charges in place of masses. A theory was intelligible if it looked like Newton's.

**After 1850**, it flipped again. **Action by contact** (through a field) becomes acceptable again with **Maxwell**, and then **Einstein's theories of relativity (1905 and 1915)**, in which gravity is curvature of space-time rather than a force at a distance.

Same phenomenon, three standards of intelligibility in two and a half centuries. None of the theories changed their empirical content to cause this; the community's standards changed.

### Case 2: quantum mechanics around 1926

**Around 1926**, two competing theories of the atom:

- **Matrix mechanics** (Heisenberg, Pauli): **abstract**.
- **Wave mechanics** (Schrödinger): **visualizable**.

The **Schrödinger versus Pauli and Heisenberg debate** was about when a theory is visualizable, *anschaulich*.

**Background: 1920 to 1925, the loss of visualizability.**

- **Wave-particle duality**, of light (Einstein, 1905) and of matter (De Broglie, 1923): there is **no unambiguous visualization** and **no particle trajectories**.
- The **reality of electron orbits was disputed**. **Pauli's fourth quantum number** (spin, 1925) had no picture in the Bohr model, so **atoms were now completely non-visualizable**.
- Hence the search for a radically new **"quantum mechanics"** (the term is Born's).

The slide's figure is the **double-slit experiment**. A source of electrons or photons fires at a wall with two slits (1 and 2); behind it a backstop with a detector records where they land. With only one slit open you get single-humped distributions $P_1 = |\phi_1|^2$ and $P_2 = |\phi_2|^2$. With both open you do **not** get $P_1 + P_2$; you get an interference pattern with many fringes:

$$P_{12} = |\phi_1 + \phi_2|^2 \neq P_1 + P_2$$

```
source ──▷   wall with     backstop     pattern on the backstop
             slits 1, 2    + detector
                                        slit 1 only:  one broad hump   P1 = |φ1|²
                                        slit 2 only:  one broad hump   P2 = |φ2|²
                                        both open:    many fringes     P12 = |φ1 + φ2|²
                                                      (not P1 + P2)
```

No picture of a particle on a trajectory through one slit produces fringes. That is the loss of visualizability in one figure.

**Schrödinger on Heisenberg's matrix mechanics:**

> "I naturally knew about his theory, but was discouraged, if not repelled, by what appeared to me as very difficult methods of transcendental algebra, and by the lack of *Anschaulichkeit*."

*Anschaulichkeit*: visualisability, intelligibility. And physicists did find Schrödinger's theory easier to deal with, so it was more widely used.

**Schrödinger (1926) on scientific understanding:**

> "We cannot really alter our manner of thinking in space and time, and what we cannot comprehend within it we cannot understand at all."

**Pauli (1924), on the other side:**

> "... our good friend Kramers and his colorful picture books, 'and the children, they love to listen.' Even though the demand of these children for visualizability (*Anschaulichkeit*) is partly legitimate and healthy, this should never count as an argument for the retention of fixed conceptual systems in physics. Once the new conceptual systems are settled, then also these will be *anschaulich*."

Note what Pauli is saying: visualizability is not a fixed standard. New theories *become* intelligible once people have the skills to use them. That is De Regt's contextual theory in a 1924 letter.

**Visualizing the hydrogen atom.** The slide reproduces "Fig. 24, Modes of hydrogen atom" from C.G. Darwin, *The New Conception of Matter* (1931): greyscale blobs showing simple solutions of the hydrogen wave function, labelled by mode, for example (0,0,0) a small dot, (1,0,0) a dot with a ring, (2,0,0) a dot with two rings, (0,1,0) two lobes one above the other, (1,1,0) stacked lobes, (0,2,0) a lobe above and below with a band around the middle. The caption says the diagrams show the **intensity of vibration** at each place, and so indicate the **probability of finding the electron** there; each is to be rotated about a vertical axis, so (0,2,0) is a ring round the equator plus two lumps at the poles.

- **Schrödinger's realistic interpretation** of these as **charge densities** fails.
- Instead they are to be interpreted as **probability densities** (Born).

**Outcome of the debate:**

- Schrödinger's visualization was **problematic**.
- **Heisenberg went on to use visualizable concepts** (his 1927 uncertainty paper is literally titled after the *anschaulich* content of quantum kinematics: he redefined what visualizable should mean).
- Result: a new **quantum mechanics** that combined both theories, with other ingredients too: **Dirac's mathematical unification** and **Born's interpretation**.
- **Visualizability**, in this historical context, was valued as a property that **increases the intelligibility** of theories.
- **Context-dependence**: visualizability is **not necessary** for understanding. It is one tool, valued by some communities at some times.

## 9. Summary

- Hempel's **deductive-nomological model** of explanation: closely connected with **prediction from laws**. Objections: **asymmetry**, **irrelevance**, **expectation/correlation**.
- Alternative models: **causal-mechanistic** explanation, and **unification**.
- **Understanding** as knowledge or explanation (eliminativism, reductivism), and as **more than** explanation.
- **Pragmatic theories: explanation + use/abilities.** Scientific understanding of phenomena requires **intelligible theories**.
- **Intelligibility** is the value that scientists $S$, in a context $C$, ascribe to the properties of a theory $T$ that facilitate its use. It is **contextual**, as illustrated by the visualizability (*Anschaulichkeit*) of quantum mechanics.

## Key Takeaways

> [!tip] Exam Focus
> **This lecture is examined directly.** Mock exam question 7:
>
> > Carl Hempel's "deductive-nomological model" for a long time was the "received view" of scientific explanation. Briefly summarize Hempel's model, and give at least one argument *against* the model's being a satisfactory account of scientific explanation.
>
> A model answer, at the half-page the rubric asks for:
>
> > On Hempel's D-N model, an explanation is a deductive argument in which the explanandum (the phenomenon to be explained) is deduced from an explanans consisting of general laws together with particular facts and initial or boundary conditions. The explanans must contain at least one law, used essentially, must have empirical content, and must be true. On this model explanation and prediction have the same logical structure: to explain is to show that the phenomenon was to be expected given the laws.
> >
> > Objection (asymmetry): deductions run both ways, explanations do not. From the laws of optics, the sun's position and the height of a flagpole we can deduce the length of its shadow, which explains the shadow. But from the same laws and the shadow's length we can equally deduce the height of the pole, and the shadow does not explain the pole. Both arguments satisfy the D-N conditions, so the model cannot be sufficient. Further objections: irrelevance (salt hexed by a magician dissolves in water: valid, lawlike, and the hex explains nothing); correlation is not explanation (a falling barometer predicts a storm but does not explain it, since both have a common cause); and explanations need not cite laws at all, since citing a causally relevant factor can be enough.
>
> One objection argued well gets the marks; the asymmetry is the strongest one to lead with. Naming the barometer example is in the official answer, so have it ready as the second.
>
> **Also feeds question 8** (van Fraassen and Musgrave on explanation, a week 5 question). The part from this lecture: van Fraassen holds that explanation is pragmatic, a three-term relation between theory, fact and context, and so **not an epistemic aim of science** but an application of it.

> [!tip] What to be able to state cold for Part B
> 1. The **grammar**: $S$ in context $C$ understands $P$ on the basis of $T$.
> 2. **CUP** and **CIT**, near-verbatim. CIT's key phrase: *recognise qualitatively characteristic consequences of $T$ without performing exact calculations*.
> 3. **Understanding = skill** to use a theory to build models of phenomena ($T \sim M \sim P$).
> 4. The **two senses of "pragmatic"** (subjective versus useable for aims), and why the second lets understanding be epistemic.
> 5. One case: Huygens versus Newton, or Schrödinger versus Heisenberg and Pauli, as intelligibility standards varying **diachronically** or **synchronically**.

> [!warning] The distinction people blur
> **Explanation** is a relation between a theory and a phenomenon (two places for Hempel, three for van Fraassen: theory, fact, context). **Understanding**, for De Regt and Dieks, is an **achievement of a subject**, with four places ($S$, $C$, $P$, $T$). "Pragmatic" means "contextual and use-related" for De Regt and Dieks and "subjective, not epistemic" for Hempel. Using the word without saying which sense is the fastest way to lose the point.

## Flashcards

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- Mock exam Q7: "Carl Hempel's "deductive-nomological model" for a long time was the "received view" of scientific explanation. Briefly summarize Hempel's model, and give at least one argument *against* the model's being a satisfactory account of scientific explanation." (half a page)
> Follow the official key:
> 1. **The model.** An explanation is a **deductive argument**: the **explanandum** (the phenomenon to be explained) is deduced from an **explanans** of **general laws** plus **particular facts and initial or boundary conditions**.
> 2. **Conditions of adequacy.** The explanandum is a logical consequence of the explanans; the explanans contains a general law used **essentially**; it has **empirical content**; it is **true**.
> 3. **Explanation and prediction are symmetric:** same logical structure. To explain is to show the phenomenon was to be expected given the laws.
> 4. **Objections** (the key lists all of these; one argued well is enough):
>    - **Asymmetry:** flagpole height, sun's position and optics yield the shadow's length (explains); shadow length and the same laws yield the pole's height (predicts, does not explain). Both satisfy every D-N condition, so the model is not sufficient.
>    - **Irrelevance:** salt hexed by a magician dissolves in water. Valid, lawlike, true, and the hex explains nothing.
>    - **Apparently missing causation** (correlation or expectation $\neq$ explanation): a falling barometer predicts a storm but does not explain it, since both have a common cause (the drop in atmospheric pressure).
>    - **Explanations do not always involve laws:** citing a causally relevant fact can be enough ("the window broke because the ball hit it").
>
> Lead with the asymmetry (the strongest), and have the barometer ready as the second, since it is named in the official answer.

> [!exam]- Causation or unification: which gives the better account of scientific explanation? Use the flagpole, Salmon, Kitcher and Godfrey-Smith.
> - **Causal (Salmon 1984):** explanation situates the explanandum in the **causal nexus**, showing the mechanisms that produce it. Solves the flagpole at once: the pole causes the shadow. Costs: it inherits **Hume's problem** (telling causation from correlation), and it may be **too narrow**, since explanations in quantum theory or of a law by a more general law are hard to phrase causally.
> - **Unification (Friedman 1974, Kitcher 1981, 1989):** explanation fits phenomena into a broader pattern, using the same patterns of derivation again and again and reducing the number of brute facts. Handles **non-causal explanation**. On the flagpole, Kitcher says causal talk summarises deeper asymmetries in unification: deriving shadows from poles belongs to a far wider pattern.
> - **Godfrey-Smith:** treating them as competitors is the mistake. Salmon came to accept unification, Kitcher came to accept causation. He goes past this pluralism to **contextualism**: there is no single special explanatory relation (or fixed short list), and standards of good explanation partly depend on the scientific field and period (following Kuhn). Not "anything goes": a standard can embed a factual error.

> [!exam]- Is explanation internal to science or external to it? Contrast van Fraassen with Godfrey-Smith.
> - **Shared ground:** both say explanation varies with context.
> - **Van Fraassen** (*The Scientific Image*, 1980): explanation is **pragmatic** and **external**. It is a **three-term relation between theory, fact and context**; an explanation is an answer to a why-question. It counts among reasons to **accept** a theory for some purpose but adds nothing to our **beliefs** about how theory relates to world. So it is **not an epistemic aim of science**: "scientific explanation is not (pure) science but an application of science", "a use of science to satisfy certain of our desires".
> - **Godfrey-Smith:** explanation is **thoroughly internal**. Assessing explanatory power is an important part of scientific reasoning, but each field (and period) sets its own standards.
> - **Why it matters:** van Fraassen's constructive empiricism needs explanation to be non-epistemic, or inference to the best explanation would push him to believe in unobservables (Musgrave's reply targets exactly this).

> [!exam]- Is scientific understanding anything more than having an explanation? Answer with Hempel, the reductivists, and De Regt and Dieks.
> - **Hempel (eliminativist):** understanding belongs to the psychological, "pragmatic" side of explanation, with pragmatic meaning subjective. It is **epistemically irrelevant**, a topic for psychology.
> - **Reductivists:** explanation is understanding enough. **Khalifa:** understanding is a form of knowledge reducible to explanation (still philosophically interesting). **Lipton:** "simply more knowledge: knowledge of causes". **Trout:** the feeling of understanding is a product of cognitive biases.
> - **De Regt and Dieks:** understanding adds a third requirement, **useability**: the **skill to use a theory to build models** ($T \sim M \sim P$) that explain phenomena. This is an ability, not knowledge.
> - **Against the subjectivity charge:** separate "pragmatic" as **subjective** from "pragmatic" as **useable for aims**; the second is objective (inter-subjective). Standards of intelligibility are set at the **meso-level** of scientific communities. **CUP** and **CIT** make it testable: can scientists recognise qualitatively characteristic consequences of $T$ without exact calculation?
> - **Evidence:** standards of intelligibility vary historically (Huygens on Newton's action at a distance) and between contemporaries (Schrödinger against Heisenberg and Pauli, 1926).
>
> Losing marks: using "pragmatic" without saying which sense.

> [!card]- Why did the logical empiricists long avoid treating explanation as an aim of science? Cover Aristotle, Duhem and Carnap.
> - **Aristotle:** his four causes are really a theory of the structure of explanations (Moravcsik's reading), so explanation arrived tangled up with causes and essences.
> - **Duhem:** to explain is to claim to know what lies behind appearances, which makes science hostage to **metaphysics** ("atoms exist", "there is an aether"). So explanation is not the job of physics.
> - **Carnap:** rejects metaphysical causes and why-questions (Hegel) as **pseudo-explanations**, but admits a respectable "**empiricist explanation**".
>
> For the logical empiricists that respectable kind is tied to **prediction from laws**. Hempel and Oppenheim's **covering law model (1948)** is "philosophically light": only logical relations between statements, nothing about hidden causes. Godfrey-Smith: the positivists "made peace with the idea that science explains" by construing it "in a low-key way".

> [!card]- Define D-N explanation, the explanandum and the explanans, write the schema, and explain the name.
> An explanation is a **deductive argument** in which the **explanandum** (phenomenon or law to be explained, $E$) is deduced from the **explanans**: general laws $L_1, \dots, L_r$ plus statements of particular facts, initial or boundary conditions $C_1, \dots, C_k$.
> $$\frac{C_1, \dots, C_k \quad L_1, \dots, L_r}{E}$$
> The line is logical deduction. "Why $X$?": $X$ is the explanandum; "because $Y$": $Y$ is the explanans.
> **Deductive:** the conclusion follows logically. **Nomological:** it uses laws of nature (*nomos*, law). It is the deductive variant of the **covering law model**: the phenomenon is "covered" by a law.

> [!card]- State Hempel's four conditions of adequacy for D-N explanation and say which are logical and which empirical.
> 1. The explanandum is a **logical consequence** of the explanans.
> 2. The explanans contains a **general law** used **essentially** (remove it and the deduction fails).
> 3. The explanans has **empirical content** (testable at least in principle).
> 4. The explanans is **true**.
>
> 1 to 3 are **logical**, 4 is **empirical**. Godfrey-Smith's gloss: first say what sort of statements would explain if true; truth is then needed for an explanation that is good "in the fullest sense".

> [!card]- What is inductive-statistical (I-S) explanation, and what does the paresis case show against it?
> The covering law variant for when at least one law is **probabilistic**: the explanans shows that a true singular explanandum had **high probability**. Not deductively valid, but it makes the explanandum highly expected.
> **Paresis:** it is explained by untreated syphilis, although only a minority of untreated syphilis cases develop paresis. So conferring high probability is **neither necessary nor sufficient** for explaining.

> [!card]- Give the front-door D-N example and the thesis Hempel draws from it about explanation and prediction.
> $C_1$: my front door is made of wood. $L_1$: wooden objects expand if humidity increases. $C_2$: humidity has increased this autumn. $E$: my front door has expanded this autumn (so it sticks).
> **Thesis: explanation and prediction are symmetric**, with the same logical structure. Knowing $C_1$, $C_2$ and $L_1$ in summer, the same argument would have predicted the sticking door. The only difference is whether you already know the conclusion is true.

> [!card]- How can the D-N model explain regularities as well as single events? Give two examples.
> A law or pattern can itself be the explanandum. The **periodic table** plus the laws governing electrons in atoms (and empirical assumptions about the composition of materials) yields the **low reactivity of noble gases** (full outer shell) and the **high conductivity of metals** (loosely bound outer electrons). Godfrey-Smith's example: **Newton explaining Kepler's laws** from mechanics plus facts about the solar system.

> [!card]- Give three clarifying comments on Hempel's model: on causal laws, on its status, and on incomplete explanations.
> 1. **Laws need not be causal.** Functional laws count: $PV = nRT$ explains without saying pressure causes volume or the reverse.
> 2. **It is normative.** It states the ideal of a good scientific explanation, not how scientists actually explain. Analogy: the logician's concept of **proof**, still useful though mathematicians rarely write formal proofs.
> 3. **Elliptical explanations.** Scientists often leave laws or conditions implicit; Hempel treats these as **abbreviations** of a full D-N argument, not counterexamples.

> [!card]- State the asymmetry objection to the D-N model with the flagpole, including the formulas, who raised it and how strong it is taken to be.
> Deductions often run both ways; explanations do not. With $h$ the pole's height, $s$ the shadow's length and $\theta$ the sun's elevation: $s = h/\tan\theta$ and $h = s\tan\theta$. From optics, geometry, the sun's position and the pole's height we deduce the shadow (an explanation); from the same laws and the shadow's length we deduce the pole's height (a prediction, no explanation). Both meet all four conditions, so the D-N conditions are **not sufficient**.
> Due to **Sylvain Bromberger (1966)**, with a slightly different example. Godfrey-Smith: "something close to a knockdown argument", "the killer".

> [!card]- Explain the symptom version of the asymmetry objection, Hempel's reply, and the one flagpole case where the shadow does explain.
> - **Symptoms:** if only disease $D$ produces symptom $S$, a law lets you infer $D$ from $S$, but a symptom never explains its disease. Prediction runs both ways, explanation only $D \to S$.
> - **Hempel's reply:** bite the bullet; if the theory allows both directions, both are fine. Defensible for some physics cases where direction is genuinely unclear, hopeless for the flagpole.
> - **Exception (Godfrey-Smith):** a flagpole designed to regulate its own height to cast a shadow of a particular length. Then the shadow, as a goal, explains the height.

> [!card]- State the irrelevance objection to the D-N model with the hexed-salt example. Who is it associated with?
> (Salmon, Achinstein.) $L$: every sample of sodium chloride hexed by a magician dissolves in water. $C_1$: this sample was hexed. $C_2$: it was placed in water. $E$: it dissolved.
> The law is true (all salt dissolves, hexed or not), the deduction valid, the law used. But the hex explains nothing. The D-N model cannot rule out premises that are **true and used but explanatorily irrelevant**.

> [!card]- State the correlation objection with the barometer, the further objection about laws, and what all the D-N counterexamples have in common.
> **Barometer:** "if the barometer drops, a storm will occur; it drops; so a storm will occur." An excellent prediction, no explanation: barometer and storm are effects of a **common cause** (falling atmospheric pressure). Showing something was **to be expected** is not showing **why** it happened.
> **No laws needed:** "the window broke because the ball hit it" explains by citing a causally relevant fact, with no law.
> **Common thread:** what is missing is **causation**. The pole causes the shadow, the hex causes nothing, barometer and storm share a cause.

> [!card]- List the four directions taken as alternatives to the D-N model.
> 1. **Causation:** good explanations show what caused the phenomena.
> 2. **Unification:** good explanations show how phenomena fit a broader pattern.
> 3. **Pragmatic:** whether answers to why-questions are satisfactory depends on **context**.
> 4. Distinguish **scientific explanation** (which requires a scientific theory) from explanation in daily life.

> [!card]- State Salmon's causal theory of explanation (with his 1984 formulation), and explain what it solves and what it costs.
> Explanation = **uncovering the causes of phenomena**. "Causal processes, causal interactions, and causal laws provide the mechanisms by which the world works; to understand why certain things happen, we need to see how they are produced by these mechanisms."
> **Solves** the flagpole: sunlight hitting the pole causes the shadow.
> **Cost:** you must say **what causality is**, which empiricists since Hume have found suspect.

> [!card]- Name the three theories of causation, what each says causation is, and who holds each.
> - **Regularity (Hume):** **constant conjunction**, $C$ regularly followed by $E$, which produces an **expectation** in us, not a necessary connection in nature. It brings the barometer problem straight back.
> - **Counterfactual (Hume in passing, J. Woodward):** if $C$ hadn't occurred, $E$ wouldn't have occurred.
> - **Process (W. Salmon):** we can identify **causal processes and interactions** in nature, like a chain of falling dominoes.

> [!card]- What is causal-mechanical explanation in Salmon's sense? Give the example, the two problems, and Railton's point about complete causal explanation.
> An explanation **situates the explanandum in a network (nexus) of causal relations**, systematically relating it causally to other items; it answers a why-question with "because..." by citing causes. Example: "the water evaporated because heat was applied and the van der Waals bonds between molecules were broken".
> **Problems:** (1) distinguishing causation from correlation (**Hume's problem**); (2) **too narrow**: quantum explanations and explanations of laws by more general laws are hard to phrase causally (a point De Regt and Dieks press).
> **Railton (1981):** the complete causal explanation would contain the entire causal history in full detail; nobody has or wants it. Context picks the **relevant pieces** of causal structure.

> [!card]- State the unificationist theory of explanation: who developed it, Kitcher's formulation, what an explanation is on it, and its main advantage.
> **Friedman (1974)** and **Kitcher (1981, 1989)**. Phenomena are explained by **fitting them into a broader pattern**. Kitcher: science shows "how to derive descriptions of many phenomena, using the same patterns of derivation again and again", teaching us to "**reduce the number of types of facts we have to accept as ultimate** (or brute)". Understanding is "seeing connections, common patterns" in apparently different situations.
> An explanation is an **argumentative pattern**: from a few simple, unified principles to many (possibly miscellaneous) events.
> **Advantage:** covers **non-causal explanations**, such as those in quantum theory. Godfrey-Smith: an "unofficial" theory within logical empiricism all along, "a good deal better than the official theory".

> [!card]- How does Kitcher handle the flagpole, and what historical and physical cases support unification as explanation?
> **Flagpole:** causal talk is a loose summary of deeper asymmetries in unification; deriving shadows from poles belongs to a pattern covering far more phenomena than deriving poles from shadows.
> **History:** Darwin's evolution and Newton's later work on matter convinced scientists before yielding many new predictions, through their **explanatory promise** (unifying many phenomena with few principles).
> **Physics:** gauge theory merges forces step by step: electricity and magnetism into Maxwell's electromagnetism, U(1); plus the weak force into Weinberg-Salam electroweak theory, SU(2)⊗U(1); plus the strong force, SU(3), into a hypothetical grand unified theory; plus gravity into a hypothetical superstring theory (the last two steps speculative). **M-theory** presents five superstring theories and 11D supergravity as limits of one theory.

> [!card]- What is Godfrey-Smith's contextualism about explanation, how does it go beyond pluralism, and what is Kuhn's example?
> Pluralism (causation sometimes, unification sometimes) is a step forward, but the real mistake is thinking there is **one** special explanatory relation or a fixed short list. **Contextualism:** standards of good explanation are **partially dependent on the scientific context**; different fields, and one field at different times, set their own criteria.
> **Kuhn (1977):** did Newton's gravity explain falling bodies, offering a mathematical law and no mechanism? Some said no; over time it became part of Newtonianism that the right kind of mathematical law **counts** as explanation.
> **Limits:** not anything goes, since a conception can embed a factual error (explanations must cite God's will, and there is no God). The covering law theory is dead **as a general account**, though some explanations do roughly have its form.

> [!card]- Summarise van Fraassen's account of explanation in four claims.
> 1. Explanation and understanding belong to the **pragmatic dimension** of science and are **contextual**, depending on our aims and preferences.
> 2. Explanation is among the reasons to **accept** a theory as useful for purposes (predicting, answering questions).
> 3. Explanations **add nothing to our beliefs** about the relation between theory and world.
> 4. So explaining is **not an epistemic aim** of science but a pragmatic dimension of **theory acceptance**.

> [!card]- Give van Fraassen's three-move argument about the history of explanation, ending in his own definition.
> 1. **Duhem** denied that science explains, and so **fostered the explanation-mysticism he attacked** (only metaphysics explains). Once **Quine** denied any demarcation between science and philosophy and positivism's ametaphysical stance faltered, people noticed science does explain, and Duhem's argument was "deftly reversed": science explains, so science involves metaphysics.
> 2. "Once you decide that explanation is something irreducible and special, the door is opened" to further irreducible concepts; some eminent realists joined this return to **essentialism or neo-Aristotelian realism**.
> 3. The error was conceiving explanation as a theory-fact relation like description. It is a **three-term relation: theory, fact and context**. An explanation is an answer, so scientific explanation is "**an application of science**", "a use of science to satisfy certain of our desires".

> [!card]- State Hempel's eliminativism about understanding, with his quotation.
> "Such expressions as 'realm of understanding' and 'comprehensible' do not belong to the vocabulary of logic, for they refer to the psychological and pragmatic aspects of explanation."
> - **Explanation** is an **objective** relation between theory $T$ and phenomenon $P$.
> - **Understanding** by a subject $S$ is **epistemically irrelevant**.
> - "Pragmatic" is used as a synonym for "**subjective**".
> - Studying understanding tells us about people's preferences, so it belongs to **psychology, not philosophy**.

> [!card]- What is reductivism about understanding? Give the positions of Khalifa, Lipton and Trout.
> The milder view: **explanation is understanding enough**, so no separate theory of understanding is needed.
> - **Khalifa (2012):** understanding is a **form of knowledge**, reducible to explanation and cognate notions, yet philosophically interesting (which separates him from Hempel).
> - **Lipton (2004):** "understanding is not some sort of super-knowledge, but simply more knowledge: knowledge of causes."
> - **Trout (2002):** the *Eureka!* feeling of understanding results from **cognitive biases** (overconfidence, hindsight), so it is no epistemic guide.

> [!card]- What are Carnap's three senses of "understanding", and which did he accept?
> 1. **Pragmatic:** capability of **use** of a theory for describing and predicting facts. Legitimate (1939).
> 2. **Subjective / metaphysical:** intuitive understanding, the feeling of grasping. Rejected (1939).
> 3. **Epistemic:** understanding a language system is **knowing its semantic rules / truth conditions**. Legitimate (1947).
>
> Sense 1 already anticipates De Regt and Dieks: understanding as the ability to use a theory, allowed even by a logical empiricist.

> [!card]- What equivocation on "pragmatic" do Friedman and Woodward and Ross find in Hempel and van Fraassen, and what follows?
> Two meanings: (1) **subjective**, varying with individual psychology; (2) **useable for certain aims**, which is compatible with an **objective (inter-subjective)** notion of understanding.
> On (2), understanding can be an **epistemic aim of science**, achieved by adequate explanations. Hempel's step from "pragmatic" to "epistemically irrelevant" works only on reading (1).
> De Regt and Dieks: "Contra Hempel, van Fraassen, and Trout, we hold that the pragmatic nature of understanding is not inconsistent with it being epistemically relevant." (Hempel the eliminativist, van Fraassen the pragmatist about explanation, Trout the bias reductivist.)

> [!card]- What three requirements for understanding do pragmatic theories share, and where is the dividing line between reductivists and "more than explanation" theorists?
> 1. **Explanation:** a scientific explanation of the phenomenon (often Hempelian), a bridge from theory to phenomena: deduction, argument or model.
> 2. **Requirements of adequacy:** epistemic values, usually **internal consistency** and **empirical confirmation** (also approximate truth).
> 3. **Useability:** it must be possible to **construct** adequate explanations.
>
> Reductivists accept only 1 and 2: understanding = adequate explanation = knowledge. "More than" theorists add 3, which is about an **ability to act**, involving **skills and judgement**, rather than knowledge.

> [!card]- What are De Regt and Dieks's three levels of analysis, and how do they keep understanding objective?
> **Macro:** science as a whole. **Meso:** scientific communities (a discipline, a period). **Micro:** individual scientists.
> They reconcile "universal aims of science" with "variation in the precise specification and/or application of these general aims". **Understanding is a macro-level, universal aim.** **Standards of intelligibility** are set at the **meso-level** and vary between communities. Individuals at the micro-level have or lack the skills. So the standards are **objective but contextual**: relative to a discipline's level of progress, not individual taste.

> [!card]- Give De Regt's basic form of understanding, how many places it has compared with Hempel's explanation, and what follows from it.
> **Scientist $S$ (in context $C$) understands phenomenon $P$ on the basis of theory $T$.**
> At least **four places** ($S$, $C$, $P$, $T$); Hempel's explanation is a **two-place** relation between $T$ and $P$ (van Fraassen's is three: theory, fact, context).
> Understanding is **pragmatic** (relative to a subject and a context) and **contextual**: the same $T$ can give understanding to one community and not another. Explanation is a relation; understanding is an **achievement of a subject**.

> [!card]- Explain model-based explanation ($T \sim M \sim P$) and De Regt's resulting definition of understanding.
> Explanation usually runs from theory $T$ through a model $M$ that **represents $P$ so that $T$ can be applied to it**, to phenomenon $P$. There are **no algorithms or strict rules** for building models: approximation, idealisation and pragmatic decisions. So $S$ needs **skills** to construct $M$.
> **Understanding = the skill to use a theory for building models to explain phenomena.**

> [!card]- Define intelligibility in De Regt's sense, and say why explaining with a theory requires it.
> To explain $P$ with $T$, $S$ needs the skills to use $T$, so $T$ must be **intelligible** to $S$.
> **Intelligibility** = the value scientists attribute to the qualities of a theory that **facilitate its use**. It is **not an intrinsic property** of theories but a **context-dependent value related to scientists' skills**. Example: **visualizability**.

> [!card]- State CUP, the Criterion for Understanding Phenomena, as close to verbatim as possible.
> A phenomenon $P$ is understood scientifically **iff** there is an explanation of $P$ that is based on an **intelligible theory** $T$ and conforms to the basic epistemic values of **empirical adequacy** and **internal consistency**.
> Scientists often understand phenomena by constructing models of them (including simulations).

> [!card]- State CIT, the Criterion for the Intelligibility of Theories, and illustrate it with the kinetic theory of gases.
> A scientific theory $T$ (in one or more of its representations) is **intelligible** for scientists (in context $C$) if they can **recognise qualitatively characteristic consequences of $T$ without performing exact calculations** (or fully explicit theoretical argumentation). They have an "insight" into the theory's workings that lets them build models meeting empirical adequacy and internal consistency.
> **Kinetic theory:** a physicist can say without calculating that heating a gas in a closed container raises its pressure (molecules move faster, hit the walls harder and more often). Someone who gets there only by solving the equations has the theory but, by CIT, does not find it intelligible.
> CUP says understanding needs an intelligible theory; CIT tests when a theory is intelligible.

> [!card]- Does CIT make understanding subjective? Give the answer and its honest concession.
> No. CIT is relative to scientists **in a context**, but the context is the **meso-level community** with shared standards and trained skills. Whether someone can recognise qualitative consequences without calculating is a **public, testable fact** about competence, unlike Trout's feeling of understanding.
> **Concession:** it makes intelligibility **contextual** (relative to a community and its level of progress), and critics can ask whether contextual collapses into subjective when communities disagree, as in quantum mechanics around 1926.

> [!card]- Name conceptual toolkits that support qualitative reasoning with a theory, and the level at which they operate.
> Visualisation and diagrams (Feynman diagrams, the Bohr picture of the atom), causal-mechanical stories, analogies, thought experiments, toy models and simulations. They are taught and shared within a discipline, so they sit at the **meso-level**; individual scientists (micro-level) master them to different degrees.

> [!card]- Trace how standards of intelligibility for gravitation changed from Newton to Einstein. Is this variation diachronic or synchronic?
> **Diachronic** (over time).
> - **Newton (1687):** $F = G\,m_1 m_2 / r^2$, action at a distance across empty space. **Huygens**, a Cartesian mechanist whose standard was **contact action**, found it unintelligible: attributing falling to gravitational attraction is to "say nothing".
> - **Around 1800:** action at a distance becomes the **ideal**. **Coulomb's law** $F = k\,q_1 q_2 / r^2$ copies Newton's form; a theory was intelligible if it looked like Newton's.
> - **After 1850:** contact action through a field is acceptable again with **Maxwell**, then **Einstein's relativity (1905, 1915)**, where gravity is space-time curvature.
>
> The empirical content did not drive this; the community's standards changed.

> [!card]- Why had atoms become non-visualizable by 1925? Include the double-slit formula.
> - **Wave-particle duality** of light (Einstein 1905) and matter (De Broglie 1923): no unambiguous picture, **no particle trajectories**.
> - The reality of **electron orbits** was disputed, and **Pauli's fourth quantum number** (spin, 1925) had no picture in the Bohr model, so atoms became completely non-visualizable. Hence the search for a new "quantum mechanics" (Born's term).
> - **Double slit:** with one slit open, $P_1 = |\phi_1|^2$ and $P_2 = |\phi_2|^2$; with both open, $P_{12} = |\phi_1 + \phi_2|^2 \neq P_1 + P_2$, an interference pattern no trajectory through one slit can produce.

> [!card]- Describe the 1926 dispute between Schrödinger and Heisenberg and Pauli over *Anschaulichkeit*, with each side's key quotation.
> **Synchronic** variation: **matrix mechanics** (Heisenberg, Pauli) was **abstract**; **wave mechanics** (Schrödinger) was **visualizable**, and physicists found it easier, so it was more widely used.
> - **Schrödinger** on matrix mechanics: "repelled" by "very difficult methods of transcendental algebra, and by the lack of *Anschaulichkeit*". And (1926): "We cannot really alter our manner of thinking in space and time, and what we cannot comprehend within it we cannot understand at all."
> - **Pauli (1924)** mocks Kramers's "colorful picture books": the demand for visualizability is "partly legitimate and healthy" but should never justify retaining fixed conceptual systems; "Once the new conceptual systems are settled, then also these will be *anschaulich*."
>
> Pauli's point is De Regt's: intelligibility is not fixed, and new theories become intelligible once people have the skills.

> [!card]- How did the visualizability debate in quantum mechanics end, and what lesson do De Regt and Dieks draw?
> - Pictures of hydrogen wave-function modes (C.G. Darwin, 1931) show intensity of vibration; Schrödinger's realistic reading as **charge densities** failed, and they were read as **probability densities** (Born).
> - Schrödinger's visualization proved problematic; **Heisenberg went on to use visualizable concepts**, redefining what *anschaulich* means (his 1927 uncertainty paper).
> - The resulting quantum mechanics combined both theories plus **Dirac's mathematical unification** and **Born's interpretation**.
>
> **Lesson:** visualizability was valued in that context because it **increases intelligibility**, but it is **not necessary** for understanding: one tool, valued by some communities at some times.

## Links

- **Course:** [[PhilSci - Overview|Course overview]] · [[PhilSci - Tutorials|Tutorials]] (tutorial 4 is on this lecture's two readings)
- **Previous:** [[PhilSci-L01 - Introduction and Logical Empiricism]] (Carnap, the logical empiricist background) · [[PhilSci-L02 - Kuhn on Scientific Practice]] (Kuhn's paradigm-relative standards, which Godfrey-Smith's contextualism builds on) · [[PhilSci-L03 - Under-determination]] (Duhem)
- **Source:** `4 Scientific Explanation and Scientific Understanding.pdf` on Canvas (Enrico Cinti, 22 September 2026), uploaded 2026-09-24. Readings: Godfrey-Smith (2003), *Theory and Reality*, ch. 13 "Explanation" (on Canvas); De Regt and Dieks (2005), "A Contextual Approach to Scientific Understanding", *Synthese* 144 (library, not read for this note: its content here comes from the deck and the tutorial slides).
- **Further reading, from the deck:** Woodward and Ross, "Scientific Explanation", *Stanford Encyclopedia of Philosophy* · Reutlinger et al. (2018), "Understanding (with) Toy Models", *BJPS* · De Regt (2017), *Understanding Scientific Understanding*, Oxford University Press
