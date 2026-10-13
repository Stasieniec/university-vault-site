---
type: lecture
course: 5354PHSC6Y
week: 3
lecture: 3
date: 2026-09-15
format: flipped classroom (pre-recorded video plus slides on Canvas)
status: complete
topics:
  - Duhem's under-determination thesis
  - Crucial experiments and good sense
  - Quine's Two Dogmas of Empiricism
  - The web of belief and the Duhem-Quine thesis
  - Critiques of Quine (Stein, Laudan)
  - Laudan's non-uniqueness and egalitarian theses
  - Empirical under-determination and Quine's individuation of theories
  - Poincaré's conventionalism
---

# PhilSci-L03: Under-determination

> [!abstract] In short
> An experiment contradicts your theory. Which of your beliefs is wrong?
>
> **Duhem:** logic cannot tell you. A prediction always comes from a bundle of hypotheses, so a failed prediction only shows that something in the bundle is false. Scientists choose what to give up by **good sense**, which is not a formal rule.
>
> **Quine:** the bundle is everything we believe, logic included, so any statement can be kept if you change enough elsewhere. **Laudan:** logically possible is not the same as rational, and Quine never shows the rules of theory choice are too weak to decide.
>
> Quine later sharpens this into rival theories that fit every possible observation, though real cases are scarce. Poincaré's sphere world shows the idea for geometry: which geometry space has is a convention.
>
> On the mock exam: Q6 (Duhem's thesis with an example), and the second half of Q3 (Popper on Einstein's "dear Lord").

## The problem: which theory did the observation refute?

Suppose an astronomical theory predicts where a planet will be, and the telescope shows it somewhere else. Which theory has just failed: the astronomy, or the optics that says the telescope shows things where they are? The observation does not say.

This matters for Popper ([[PhilSci-L01b - Popper and Lakatos|L01b]]), whose account of science rests on observations refuting theories. Popper's answer: no objective facts force the choice, not even the choice of which observation reports to accept. Scientists accept an **empirical basis** (the observation reports that tests rest on) by **intersubjective agreement**, a consensus among investigators. So accepting a report that falsifies a theory is, in the end, **a human decision**.

Duhem's thesis opens a second gap: even once the report is accepted, logic does not say which theory it refutes.

## Duhem: falsification is ambiguous

Pierre Duhem was a French physicist and historian of science.

No hypothesis predicts anything on its own, so a failed prediction says only that at least one member of a group is false.

> [!example] The telescope, step by step
> - $P_1$: the astronomical theory (the laws of planetary motion).
> - $P_2$: the optics of the telescope (it shows objects where they are).
> - $P_3$: the initial conditions (last month's position and speed).
> - Together these entail $O$: "the planet appears at position $x$ tonight".
> - Observation gives $O'$: "the planet appears at position $y$". $O'$ entails that $O$ is false.
> - Conclusion: $P_1 \wedge P_2 \wedge P_3$ is false. The astronomy, the telescope or last month's data: logic does not say which.

In symbols, where $P_1, \dots, P_n$ are the hypothesis under test plus every auxiliary assumption the prediction needs, and $O$ is the predicted observation:

$$\frac{(P_1 \wedge \dots \wedge P_n) \rightarrow O \qquad \neg O}{\therefore\ \neg(P_1 \wedge \dots \wedge P_n)}$$

The conclusion says only "at least one premise is false". This is Popper's modus tollens (from "$T$ implies $O$" and "not $O$", infer "not $T$") with the auxiliaries written in. Duhem's version, in the form the answer key uses:

> [!definition] Duhem's thesis (under-determination)
> Given that a set of premises $\{P_1, P_2, \dots, P_n\}$ deductively entails a statement $O$ describing a possible observation, and that from experimental observation one concludes that the statement $O'$ is true, where $O'$ entails that $O$ is false, it follows that **the conjunction of the premises is false**. It does **not** follow that any particular premise is false.
>
> In short: **falsification is ambiguous**.

"Under-determination" names exactly this: the evidence does not determine which belief has to go.

### No crucial experiment

A **crucial experiment** (*experimentum crucis*) is supposed to settle a contest between two hypotheses. Duhem describes it as "an irrefutable procedure for transforming one of the two hypotheses before us into a demonstrated truth". $H_1$ predicts one outcome, $H_2$ the other; run the experiment, the loser is refuted, the winner proven.

Duhem says this is impossible:

1. **Auxiliary assumptions.** Hypotheses are only ever tested in combination with auxiliary assumptions, such as the theory of the experimental apparatus. The experiment refutes a bundle, never $H_1$ alone, so $H_1$ can be kept by blaming an auxiliary.
2. **Unconceived alternatives.** Even if $H_1$ really were refuted, that would prove $H_2$ only if $H_1$ and $H_2$ were the only options. There may be hypotheses nobody has thought of yet.

### Good sense

What decides which premise to give up? Duhem's answer is the scientist's **good sense** (*bon sens*): "certain opinions which do not fall under the hammer of the principle of contradiction are in any case perfectly unreasonable". You could keep the astronomy and blame the telescope every single time without contradicting yourself, and it would still be unreasonable.

So Duhem does **not** say theory choice is arbitrary. He says it is not settled by **formal** logic alone. "Not formal" and "arbitrary" are different claims, and the difference returns with Laudan.

Good sense is deliberately not a rule. Duhem asks the scientist to be "an impartial and faithful judge", which leaves two questions open: what good sense actually is, and how a scientist manages to be impartial.

### What this means for Popper

Eddington's 1919 eclipse expedition looked for the bending of starlight predicted by general relativity. Asked what he would have said had the shift not been found, Einstein replied: "Then I am very sorry for the dear Lord, for the theory is correct." If the measurement was right, Popper must say the theory is falsified and has to be rejected, and Einstein's reply is dogmatic. But the prediction came from the theory plus auxiliaries: working telescopes, unwarped photographic plates, correctly identified stars. So Einstein could logically blame an auxiliary, which is the Duhem-Quine thesis (below). Popper grants this, but accepts blaming the measurement only with **independent reasons** to reject it, reasons other than "otherwise my theory would be refuted". Without them the rescue is **ad hoc** (made only to absorb the refuting evidence) and lowers the theory's scientific status. Einstein gives no such reason, so for Popper his reply is not acceptable.

## Quine: the bundle is everything we believe

W.V.O. Quine was an American logician and philosopher. In "Two Dogmas of Empiricism" (1951) he attacks two unargued assumptions of the logical positivists: the Vienna Circle philosophers of [[PhilSci-L01 - Introduction and Logical Empiricism|L01]], for whom a statement is meaningful only if it is analytic (true by meaning alone) or can be verified by experience.

### The two dogmas

**Dogma 1: reductionism.** The belief that every meaningful scientific statement can be translated into an equivalent statement in a special empiricist language: either the subjective language of immediate experience or an objective language of physical things. This is the **verification theory of meaning**: each statement has its own set of experiences that would confirm it.

Quine's objection is Duhem's point, generalised: **you cannot test isolated statements.** No single statement has a private stock of empirical consequences; only the whole body of our statements meets experience.

**Dogma 2: the analytic-synthetic distinction.** An **analytic** statement is true by meaning alone ("all bachelors are unmarried"); a **synthetic** one is true because of how the world is ("most bachelors in Amsterdam own a bike"). The positivists needed a sharp line, because it let them call logic and mathematics analytic: true by meaning, so known without experience.

Quine denies the sharp line: the difference between conceptual schemes (truths we hold because of what our words mean) and facts "is of degree only". Some statements are simply held more firmly than others.

**At bottom the two dogmas are identical.** If each statement had its own range of confirming experiences (dogma 1), there could be a **limiting case**: a statement with no empirical content, confirmed by every possible experience, however the world turns out. Those would be the analytic statements (dogma 2). Both dogmas assume statements can be assessed one at a time. Quine denies that, so no statement is confirmed by experience whatever happens, and both dogmas fall together. (This is compatible with his claim below that any statement can be *held* true come what may: that is our choice to protect a statement by changing other beliefs, and experience does not confirm it.)

Behind both attacks sits Quine's scepticism about **meanings** as such, which undercuts any non-empirical knowledge. If no statement is true by meaning alone, even logic and mathematics belong to the body of belief that faces experience as a whole.

### The web of belief

Quine's replacement picture of knowledge, and the passage where he calls it under-determined:

> [!quote] Quine, Two Dogmas of Empiricism
> The totality of our so-called knowledge or beliefs, from the most casual matters of geography and history to the profoundest laws of atomic physics or even of pure mathematics and logic, is a man-made fabric which impinges on experience **only along the edges**. ... A conflict with experience at the periphery occasions readjustments in the interior of the field.
>
> But the total field is so **underdetermined** by its boundary conditions, experience, that there is much latitude of choice as to what statements to reevaluate in the light of any single contrary experience.

Picture a spider's web inside a circle whose rim is observation.

- **The rim** touches experience. Beliefs near it (geography, history, particular facts) are the first candidates for revision when an observation goes wrong.
- **The centre** holds logic and mathematics. They are revisable too, but revising them forces the largest rearrangement of everything else. They sit deep because they are expensive to change, and for no other reason.

### How Quine strengthens Duhem

This view is called **holism**: beliefs face experience only as a whole, never one at a time. The **Duhem-Quine thesis** is holism read as a claim about testing: a disconfirming observation only hits the web at its edges, and adjusting auxiliary hypotheses inside can absorb it. Quine goes beyond Duhem in two ways.

1. **The bundle grows.** For Duhem, a hypothesis in physics is tested together with a group of other hypotheses: the rest of the theory and the theory of the instruments. For Quine, it is tested together with *everything* we believe, logic and mathematics included.
2. **Anything can be kept, and anything can be revised.** "Any statement can be held true come what may, if we make drastic enough adjustments elsewhere in the system", even by "pleading hallucination or by amending certain statements of the kind called logical laws". And in the other direction, no statement is immune to revision.

This is why holism seems to support under-determination. If a contrary observation can be absorbed by revising any of many beliefs in the web, there is always more than one way to repair the system, so it seems there must always be more than one system of beliefs that fits all the data. Laudan's reply comes below.

Popper accepted Duhem's logical point that a failed prediction refutes only the conjunction. He did not accept Quine's stronger holism.

## Objections to Quine: Stein and Laudan

### Stein: Quine dismisses meanings without argument

Howard Stein, a philosopher of physics, locates the trouble in Quine's **scepticism about meanings**, which leads Quine to reject theories of meaning outright. Gottlob Frege (founder of modern logic) and Rudolf Carnap (the logical positivist of L01) distinguish two kinds of meaning:

- **Sense** (or intension): the linguistic meaning of the words. "The morning star" and "the evening star" differ in sense.
- **Reference** (or extension): the actual object the words pick out. Both phrases refer to Venus.

In possible-world semantics (a possible world is a complete way things could have been), the intension of a sentence $S$ is the set of all possible worlds in which $S$ is true: a precise, workable notion of meaning.

Quine does not engage with this. He sets Leibniz's possible-worlds formulation of analyticity aside as "picturesque", and treats intensions as leftovers of Aristotle's essences. Stein's complaint: Quine **assumes** an empiricist must be sceptical about meanings, when intensions in the Frege-Carnap sense merited real engagement.

### Laudan: logically possible is not the same as rational

Larry Laudan, an American philosopher of science, complains that under-determination (UD) is stated sloppily and then used to support relativist conclusions (that no theory is objectively better supported than another). His diagnosis:

> [!quote] Laudan, Demystifying Underdetermination
> Too many of the discussions of UD in the last quarter century have proceeded in an **evaluative vacuum**. They imagine that if a course of action is logically possible, then one need not attend to the question of its rationality.

His reply to the argument from holism, step by step:

1. Holism shows that any hypothesis *can* be kept, whatever the evidence, by adjusting something else (pleading hallucination, if need be). That is a claim about **logical possibility**.
2. The under-determination that would threaten methodology is a claim about **rational warrant**: that the evidence supports rival theories *equally*. Rationality includes **epistemic warrant** (justification), above all the empirical evidence.
3. To get from 1 to 2, Quine would have to examine the **ampliative rules of theory choice** (rules beyond deductive logic for weighing evidence and preferring one theory to another) and show that they are so ambiguous that, faced with any pair of theories and any body of evidence, they could never yield a decision. Quine never examines them.
4. So holism does not show that theory choice is arbitrary.

### Non-uniqueness and the egalitarian thesis

Laudan separates two claims:

> [!definition] Laudan's two under-determination theses
> 1. **Non-uniqueness thesis.** For any theory $T$ and any body of evidence supporting $T$, there is **at least one** rival to $T$ that is as well supported as $T$.
> 2. **Egalitarian thesis** (stronger). **Every** theory is as well supported by the evidence as any of its rivals.
>
> Only (2), and not (1), is a threat to epistemology and methodology.

**Why only the egalitarian thesis is a threat.** Non-uniqueness says only that at least one rival is as well supported as $T$. That is compatible with the rules of theory choice rejecting most other theories, so methodology can still narrow the field sharply even if it cannot always name a single winner. The egalitarian thesis says every theory is as well supported as every rival, so the rules could never reject anything: theory choice would be arbitrary and methodology idle.

## Empirical under-determination

Quine's "On Empirically Equivalent Systems of the World" (1975) states a sharper thesis.

### Theories, formulations and observation conditionals

- An **observation conditional** is a sentence of the form "if such-and-such conditions hold, you will observe so-and-so".
- The logical relation between theory and data runs **one way**: a theory, as a whole, implies observation conditionals; the data never imply the theory. Single sentences of the theory usually have no observation consequences of their own (holism again).
- A **theory formulation** is a sentence: in practice the conjunction of a theory's axioms.
- Quine's **principle of individuation**: two formulations express **the same theory** when they are empirically equivalent **and** one can be turned into a logical equivalent of the other by **reconstrual of predicates**: systematically swapping predicates. Quine's example: a physical theory and the same theory with "electron" and "molecule" exchanged throughout are one theory.

Now the thesis. Imagine two theory formulations that are:

1. **Empirically equivalent**: they imply the same observation sentences, and
2. **Theoretically inequivalent**: they differ in their theoretical sentences, in a way that no reconstrual of predicates removes.

Then the theory is **under-determined by the empirical data**: the data cannot decide between them. Without the principle of individuation the thesis would be trivial: every mere rewording would count as a rival that fits the same data.

This is seen as a problem for **scientific realism** (the view that our best theories are approximately true about the unobservable world; [[PhilSci-L05 - Scientific Realism and its Critiques|L05]]): if two incompatible theories fit every possible observation, which one are you a realist about?

### Kinds of under-determination

- **Semantic** under-determination concerns meaning and interpretation, regardless of the evidence: it is unsettled what the theories say, or whether two formulations say different things at all, so more evidence would not help.
- **Epistemic** under-determination concerns what the empirical evidence can settle.

**Dualities** in current physics are pairs of very differently formulated theories that agree on everything empirical. De Haro, the lecturer, treats them as potential cases of **semantic** under-determination: perhaps two descriptions of one world. On his view they do not threaten a **cautious scientific realism**: one that claims approximate truth for well-confirmed theories within their domain, and is not committed to differences of formulation that make no difference there.

Under-determination is also **transient** when current evidence does not decide between theories but later evidence might, and **permanent** when no possible evidence decides. Transient under-determination is very common in scientific practice. Quine's thesis is the permanent kind: it says our system of the world would be under-determined even by *all possible* observations, so that everything the world could ever show us would fit irreconcilable theories (whether real cases exist he leaves open). Tutorial 3 asks why Quine calls under-determination a thesis about the world. The likely answer: it concerns what the world leaves open, and not the limits of one experiment. This reason is a reconstruction, not Quine's own wording.

Quine also holds that there is no "extra-theoretic" truth: we judge truth only from inside our own best theory, since there is no standpoint outside all theories. So if an irreconcilable empirically equivalent rival turned up, "which one is really true?" would have no answer from outside. On one of the positions Quine took, we would go on calling our own theory true.

### Does it ever actually happen?

Real examples are few, and suspiciously of one type.

- **Laudan and Jarrett Leplin** (Laudan's co-author): "contrived examples alleging empirical equivalence always involve the **relativity of motion**; it is the impossibility of distinguishing apparent from absolute motion to which they owe their plausibility". They add that the two leading historical cases have the same problem: Ptolemy against Copernicus (Earth-centred against Sun-centred astronomy), the case that gave rise to the idea of empirical equivalence, and Einstein against Lorentz (special relativity against Lorentz's theory of a stationary ether, a space-filling medium).
- **Quine (1975)** himself calls it an "open question" whether any real cases exist.
- **Kyle Stanford** (a present-day philosopher of science, 2006): critics of under-determination "have been well within their rights to demand that serious, nonsceptical, and genuinely distinct empirical equivalents to a theory actually be produced" before they withhold belief in a theory.

So the strong thesis is widely asserted and thinly evidenced.

**Which of Laudan's theses does Quine's empirical under-determination support?** There is no settled answer; a defensible one is: at most **non-uniqueness**. It says a theory has an empirically equivalent rival (and even that need not mean an equally supported one), and nothing about *every* rival being equally supported. The claim sometimes read as egalitarian is the Two Dogmas slogan, "held true come what may", which is what Laudan's "evaluative vacuum" objection targets.

### The common core: where under-determination really lurks

Quine himself says what the thesis requires. If a theory has finitely many observation conditionals, it can be written as their conjunction, a single sentence that "contains its observation conditionals without remainder". Then nothing is left to be under-determined: that conjunction is implied by every empirically equivalent rival and conflicts with none, so it is a **common core** the rivals share.

> [!quote] Quine, On Empirically Equivalent Systems of the World
> Under-determination lurks where there are two irreconcilable formulations each of which implies exactly the desired set of observation conditionals plus extraneous theoretical matter, and where no formulation affords a tighter fit.

Three things must hold at once:

1. two **irreconcilable** formulations (no reconstrual makes them agree),
2. each carrying **extraneous theoretical matter** beyond the observation conditionals,
3. **no formulation that fits more tightly** than they do.

That is a demanding requirement, which is one way to see why real examples are so scarce.


## Poincaré: geometry as convention

Henri Poincaré, French mathematician and physicist, gave an early worked example of under-determination in *Science and Hypothesis* (1902).

### The question: what kind of knowledge is geometry?

Immanuel Kant held that geometry and Newton's three laws are **synthetic a priori**: *a priori* (known independently of experience), yet *synthetic* (they say something about the world beyond what their terms mean, unlike mere logical truths).

In the nineteenth century Gauss and Riemann showed that **non-Euclidean geometries** (geometries of curved space, where a triangle's angles need not add up to 180 degrees) are mathematically consistent. So which geometry our space has, and how we would know, became a real question.

Poincaré's answer: geometry is **neither** purely a priori **nor** a posteriori. It is **conventional**. We choose the geometry that describes experience most simply, but we could take another.

### First argument: perception does not give us one geometry

We do not perceive space directly as three-dimensional. The visual field is at first sight two-dimensional, and only derivatively 3D; "tactile space" (space as felt through touch and movement) could be multi-dimensional. By experiment we discover a structure for space, but that structure is not unique. So perception does not settle which geometry space has.

### Second argument: the sphere world

> [!example] Poincaré's sphere
> A world inside a sphere of radius $R$ (drawn as a disk in two dimensions), where at distance $r$ from the centre the temperature is
> $$T = R^2 - r^2$$
> highest at the centre ($T = R^2$), zero at the boundary. Every body's size is proportional to the local temperature, by the same law for every material.
>
> **Description 1, from outside:** a finite Euclidean disk in which every body shrinks towards zero size near the boundary.
>
> **Description 2, from inside:** beings whose bodies and rulers shrink by the same law cannot detect the shrinking, since a ruler shrinks as fast as what it measures. Walking outwards, their steps get smaller, so they never reach the edge. They experience space as **infinite and curved**, and the geometry they develop will be **non-Euclidean**.

### The conclusion

For us, whose senses are used to Euclidean space, it is easier to describe this world as a Euclidean disk plus a law of physics saying that objects expand with temperature. But there is **no fact of the matter** about which description is right. Beings just like us who happened to live inside the sphere would describe it with non-Euclidean geometry.

**No experiment can decide**, because every experimental set-up can itself be interpreted in either picture. To measure the geometry you need measuring instruments, and what they do depends on the physics you assume. Change the physics (add or remove the temperature law) and you can keep either geometry. (In Poincaré's own version, beyond the lecture, light is also bent, because the refractive index of the medium changes with distance from the centre, so light cannot expose the shrinking either.) The choice is a **convention**, like measuring distance in metres or in yards.

This is Duhem's thesis again: laws are tested only indirectly, and geometry faces experience bundled with physics.

## Duhem, Quine and Laudan side by side

| | **Duhem** | **Quine** | **Laudan** |
|---|---|---|---|
| What faces a test | Hypothesis plus auxiliaries | The whole web | Same as Quine (granted) |
| What settles the choice | Good sense | Nothing forces one choice | Evidence, ampliative rules |
| Main point | Logic cannot pick the culprit | Any statement can be kept | Possible is not rational |

## What to be able to write

- Duhem's thesis in the answer key's form: premises entail $O$, experiment establishes $O'$ incompatible with $O$, so the **conjunction** is false, and it does not follow that any particular premise is.
- One worked example: the telescope, or Poincaré's sphere (geometry as convention).
- No crucial experiment: auxiliary assumptions, unconceived alternatives.
- Quine's strengthening: the whole web, logic included; any statement held true come what may.
- Q3: Duhem-Quine lets Einstein blame the measurement; Popper allows that only with **independent reasons**.
- Holism's case for under-determination, and Laudan's reply: possible is not rational; only the egalitarian thesis threatens methodology.
- Empirical under-determination, reconstrual of predicates, scarce real cases.

Full model answers for mock Q6 and Q3 are the first two exam questions below.

## Exam questions

> [!exam]- Mock exam Q6 (10 points, half a page): The under-determination thesis was first formulated by Pierre Duhem, and it was later on strengthened by W.V.O. Quine. State this thesis in Duhem's own version, and explain it with an example.
> **Key points:** premises entail $O$, experiment gives $O'$ incompatible with $O$; the conjunction is false, and no particular premise follows as false; no crucial experiment; hypotheses tested together with auxiliary assumptions, including the apparatus; an example (telescope or Poincaré's sphere).
>
> Follow the official key: the formal version, plus no crucial experiment because hypotheses are only tested with auxiliary assumptions.
> 1. **The formal version.** If premises $\{P_1, \dots, P_n\}$ deductively entail a statement $O$ describing a possible observation, and experiment establishes $O'$, where $O'$ entails that $O$ is false, then the **conjunction** of the premises is false. It does **not** follow that any particular premise is false: falsification is ambiguous.
> 2. **The consequence.** There is no crucial experiment that irrefutably decides between two hypotheses, because a hypothesis is only ever tested together with auxiliary assumptions, such as the theory of the experimental apparatus (and there may be unconceived alternatives). Logic does not say which premise to give up; for Duhem the scientist's good sense does.
> 3. **An example.** An astronomical theory, together with the optics of the telescope and the initial conditions, predicts the planet at position $x$; the telescope shows it at $y$. The conflict does not say whether the astronomy or the optics of the telescope (or the initial data) is at fault. Poincaré's sphere is the richer alternative: geometry and physics face experience together, so no experiment tests the geometry alone.
> 4. Optionally, one line on Quine: he widens the bundle to the whole web of belief, logic included, so any statement can be held true come what may.
>
> Losing marks: stating the slogan ("theory is under-determined by data") instead of the premises-and-conjunction version, or giving no example.

> [!exam]- Mock exam Q3 (10 points, half a page): Popper praised Albert Einstein's theory of general relativity as a truly scientific theory, citing the example of Eddington's expedition. This expedition set out to measure the shift of a distant star during a solar eclipse. Einstein's theory states that large masses, like the sun, curve space-time. Thus, incoming light rays will follow a curved path around the sun and will appear in a different spot as seen from the earth. The theory makes a very precise prediction about this shift. In 1953, however, Einstein was asked how he would have responded to a measurement outcome that did *not* show this shift: "Then I am very sorry for the dear Lord [Eddington], for the theory is correct." Would, and-or could, this be acceptable for Popper? Explain your answer.
> **Key points:** No, if the measurement was correct: theory falsified, Einstein dogmatic; perhaps the measurement was wrong; Duhem-Quine: blame can always go elsewhere; acceptable only with independent reasons for rejecting the measurement (no ad hoc hypothesis).
>
> Follow the official key:
> 1. **No**, assuming the measurement was correct: the theory is falsified and has to be rejected. Einstein shows a **dogmatic attitude**.
> 2. **However**, perhaps the measurement was not correct. The prediction comes from the theory plus auxiliary assumptions (telescopes, photographic plates, star identification). Given the **Duhem-Quine thesis**, scientists can always put the blame elsewhere and retain the theory.
> 3. For Popper this is acceptable **only if there are independent reasons for rejecting the measurement result**, reasons other than that the theory would otherwise be refuted. Otherwise it is an **ad hoc** hypothesis, which lowers the theory's scientific status.
> 4. Einstein as quoted gives no independent reason, only confidence in the theory, so the reply is not acceptable.
>
> Losing marks: stopping at "no", or letting Duhem-Quine excuse Einstein without the independent-reasons condition.

> [!exam]- Does under-determination show that theory choice is arbitrary? Answer using Duhem, Quine and Laudan.
> **Key points:** Duhem: logic does not decide but good sense does, so choice is not formal, which differs from arbitrary; Quine: any statement can be held true come what may; Laudan: logically possible is not rational, evaluative vacuum, ampliative rules never examined; only the egalitarian thesis would make choice arbitrary.
>
> - **Duhem:** no. Logic alone does not settle which premise to give up, but **good sense** does: some opinions are "perfectly unreasonable" even though no contradiction forces you to drop them. Choice is not *formal*, which is different from arbitrary.
> - **Quine:** pushes hardest. Our beliefs face experience as one web, and any statement can be held true "come what may" if you make drastic enough adjustments elsewhere (pleading hallucination, even amending logical laws).
> - **Laudan:** logically possible is not rational. Quine argues in an "**evaluative vacuum**": he never examines the ampliative rules of theory choice to show they are too weak to decide between any two theories on any evidence. Rationality includes epistemic warrant, above all the evidence.
> - Laudan's two theses: non-uniqueness (at least one rival as well supported) is harmless, since it is compatible with the rules still rejecting most rivals; only the **egalitarian** thesis (every theory as well supported as any rival) would make choice arbitrary, and nothing in Quine establishes it.
> - Verdict: holism shows a theory *can* be saved, not that it is *reasonable* to save it.

> [!exam]- Explain under-determination using Poincaré's sphere, and say what conclusion Poincaré draws.
> **Key points:** Euclidean disk with $T = R^2 - r^2$, bodies shrink towards the edge; inhabitants find infinite non-Euclidean space; no experiment separates the two; geometry is conventional, no fact of the matter; geometry is tested only together with physics (Duhem).
>
> - A world inside a sphere of radius $R$. **Description 1:** a Euclidean disk with temperature $T = R^2 - r^2$ (highest at the centre, zero at the edge), where every body's size is proportional to $T$, so everything shrinks towards the edge. **Description 2:** inhabitants, whose rulers shrink with them, never reach the edge and find space **infinite and non-Euclidean**.
> - No experiment separates the two, because every experimental set-up can itself be interpreted in either picture: measuring instruments behave according to the physics you assume.
> - Conclusion: geometry is **conventional**, neither a priori (against Kant's synthetic a priori) nor a posteriori. We pick the description that is simplest for us, as we choose metres or yards. There is no fact of the matter.
> - Link to Duhem: laws are tested only indirectly; geometry faces experience together with physics.

## Flashcards

> [!card]- For Popper, how does science come to accept its empirical basis?
> By **intersubjective agreement** among investigators: no objective facts force it.

> [!card]- For Popper, what is accepting an observation report that falsifies a theory, in the end?
> A **human decision**.

> [!card]- Duhem's thesis: if premises $P_1, \dots, P_n$ entail $O$ and experiment establishes $O'$ incompatible with $O$, what follows?
> The **conjunction** of the premises is false. It does **not** follow that any particular premise is false.

> [!card]- Which thesis is summed up as "falsification is ambiguous"?
> **Duhem's under-determination thesis**: modus tollens refutes the conjunction, never a named premise.

> [!card]- What is a crucial experiment (*experimentum crucis*)?
> "An irrefutable procedure for transforming one of the two hypotheses before us into a **demonstrated truth**."

> [!card]- Why, for Duhem, can an experiment never refute one hypothesis on its own?
> A hypothesis is only ever tested **together with auxiliary assumptions**, including the theory of the experimental apparatus.

> [!card]- Why would refuting one of two rival hypotheses still not prove the other, according to Duhem?
> There may be **unconceived alternatives**: hypotheses nobody has thought of yet.

> [!card]- What is the telescope example of Duhem's thesis?
> An astronomical theory conflicts with telescope observations: is the **astronomy or the optics** of the telescope at fault?

> [!card]- According to Duhem, what decides which belief to give up when logic does not?
> **Good sense** (*bon sens*): some opinions no contradiction refutes are still "perfectly unreasonable".

> [!card]- What is the difference between Duhem's conclusion and saying theory choice is arbitrary?
> Duhem says choice is **not formal**: logic does not decide, but good sense still does.

> [!card]- What questions does Duhem's appeal to good sense leave open?
> What good sense actually **is**, and how a scientist can be "an **impartial and faithful judge**".

> [!card]- What are Quine's two dogmas of empiricism?
> **Reductionism** and the **analytic-synthetic distinction**.

> [!card]- What does the dogma of reductionism claim, according to Quine?
> Every meaningful scientific statement is equivalent to one in a special empiricist language: the **verification theory of meaning**.

> [!card]- What is Quine's objection to reductionism?
> You **cannot test isolated statements**: no statement has its own private stock of empirical consequences.

> [!card]- What is Quine's objection to the analytic-synthetic distinction?
> The difference between conceptual schemes and facts is "**of degree only**".

> [!card]- What would an analytic statement be, if reductionism were true?
> The **limiting case**: a statement confirmed by every possible experience. Quine denies there are any.

> [!card]- Why does Quine say the two dogmas are "at bottom identical"?
> Both assume statements can be assessed **one at a time** against experience. Deny that and both fall together.

> [!card]- Why does Quine's scepticism about meanings undercut non-empirical knowledge?
> If nothing is **true by meaning alone**, even logic and mathematics belong to the web that faces experience.

> [!card]- Where does Quine's web of belief touch experience?
> **Only along the edges**. A conflict at the periphery forces readjustments in the interior.

> [!card]- Why do logic and mathematics sit at the centre of Quine's web of belief?
> Revising them forces the **largest rearrangement**. They are revisable, just expensive.

> [!card]- How does Quine strengthen Duhem's thesis?
> The bundle tested becomes **everything we believe**, logic and mathematics included.

> [!card]- What is holism, in Quine's sense?
> Beliefs face experience **only as a whole**, never one statement at a time.

> [!card]- What does the Duhem-Quine thesis say about a disconfirming observation?
> It hits the web only at its edges, so adjusting **auxiliary hypotheses** inside can absorb it.

> [!card]- Why does holism seem to support under-determination, according to Quine?
> There is always **more than one way to repair** the web, so more than one system seems to fit all the data.

> [!card]- What can be held true "come what may", according to Quine?
> **Any statement**, given drastic enough adjustments elsewhere, such as pleading hallucination or amending logical laws.

> [!card]- How far does Popper go along with the Duhem-Quine thesis?
> He grants **Duhem's logical point** but did not accept **Quine's stronger holism**.

> [!card]- When does Popper accept blaming a measurement instead of the theory it falsifies?
> Only with **independent reasons** to reject the measurement. Otherwise the rescue is **ad hoc**.

> [!card]- What does Stein's criticism of Quine target?
> Quine **assumes** an empiricist must be sceptical about meanings, dismissing Frege-Carnap intensions instead of engaging them.

> [!card]- What is the difference between sense and reference (Frege, Carnap)?
> **Sense** (intension) is the linguistic meaning of the words. **Reference** (extension) is the object they pick out.

> [!card]- In possible-world semantics, what is the intension of a sentence?
> The **set of all possible worlds** where it is true.

> [!card]- What does Laudan say discussions of under-determination have proceeded in?
> An "**evaluative vacuum**": assuming that if a move is logically possible, its rationality need not be asked about.

> [!card]- According to Laudan, what would Quine need to show to support "held true come what may", and never does?
> That the **ampliative rules of theory choice** are always too ambiguous to decide between any two theories.

> [!card]- Why, for Laudan, is saving a theory by pleading hallucination possible but not reasonable?
> Rationality includes **epistemic warrant**, chiefly the empirical evidence. Logically possible is not rational.

> [!card]- What is Laudan's non-uniqueness thesis of under-determination?
> For any theory and evidence supporting it, there is **at least one** rival as well supported.

> [!card]- What is Laudan's egalitarian thesis?
> **Every** theory is as well supported by the evidence as any of its rivals.

> [!card]- Which of Laudan's two under-determination theses is a threat to epistemology and methodology?
> Only the **egalitarian** thesis.

> [!card]- Why does the non-uniqueness thesis of under-determination not threaten methodology?
> It is **compatible with the rules rejecting most rivals**; it only denies that a unique winner is guaranteed.

> [!card]- Why would the egalitarian thesis make methodology idle?
> Every theory is as well supported as every rival, so the rules could **never reject anything**.

> [!card]- What two conditions define Quine's empirical under-determination (1975)?
> Two theory formulations that are **empirically equivalent** but **theoretically inequivalent**.

> [!card]- When do two theory formulations express the same theory, for Quine?
> When they are empirically equivalent and one becomes the other by **reconstrual of predicates**.

> [!card]- What is an observation conditional?
> A sentence of the form "if these conditions hold, you will **observe** so-and-so".

> [!card]- Which way does implication run between a theory and the data, for Quine?
> The **theory as a whole implies** observation conditionals; the data never imply the theory.

> [!card]- Why is empirical under-determination a problem for scientific realism?
> If the data cannot decide between rivals, which one are you a **realist about**?

> [!card]- What is the difference between semantic and epistemic under-determination?
> **Semantic** concerns meaning and interpretation, regardless of evidence. **Epistemic** concerns what the evidence can settle.

> [!card]- What is the difference between transient and permanent under-determination?
> **Transient**: current evidence does not decide, later evidence might. **Permanent**: no possible evidence decides.

> [!card]- Which kind of under-determination is very common in scientific practice?
> **Transient** under-determination.

> [!card]- Is Quine's empirical under-determination transient or permanent?
> **Permanent**: it concerns all possible observations.

> [!card]- Why does Quine's denial of "extra-theoretic" truth matter for under-determination?
> With no standpoint outside all theories, "which rival is **really true**?" has no outside answer.

> [!card]- Dualities are potential cases of which kind of under-determination?
> **Semantic** under-determination.

> [!card]- On De Haro's view, do dualities threaten scientific realism?
> No, not a **cautious scientific realism**.

> [!card]- According to Laudan and Leplin, what do contrived examples of empirical equivalence always involve?
> The **relativity of motion**: the impossibility of distinguishing apparent from absolute motion.

> [!card]- What are the leading historical cases of empirical equivalence, according to Laudan and Leplin?
> **Ptolemy vs Copernicus** and **Einstein vs Lorentz**.

> [!card]- What does Quine (1975) say about whether real cases of empirical under-determination exist?
> It is an "**open question**".

> [!card]- What does Stanford say critics of under-determination may demand before they withhold belief in a theory?
> That serious, genuinely distinct empirical equivalents **actually be produced**, not presumed.

> [!card]- Why does the conjunction of a theory's finitely many observation conditionals not create under-determination, for Quine?
> It is implied by every empirically equivalent rival and conflicts with none: a **common core**.

> [!card]- What three conditions must hold at once for under-determination to "lurk", according to Quine?
> - Two **irreconcilable** formulations
> - each with **extraneous theoretical matter**
> - **no formulation** fitting more tightly

> [!card]- What was Kant's view of the status of geometry and Newton's three laws?
> **Synthetic a priori**: known independently of experience, yet saying something beyond what their terms mean.

> [!card]- Which nineteenth-century result put pressure on Kant's view of geometry?
> **Non-Euclidean geometries** were shown to be mathematically consistent (Gauss, Riemann).

> [!card]- What is Poincaré's view of the status of geometry?
> **Conventional**: neither a priori nor a posteriori. We choose the geometry that describes experience most simply.

> [!card]- What is Poincaré's perceptual argument that space is not given as three-dimensional?
> The visual field is at first sight **two-dimensional**, and tactile space could be multi-dimensional.

> [!card]- In Poincaré's sphere, what is the outside (Euclidean) description?
> A Euclidean disk with temperature $T = R^2 - r^2$; every body's size is proportional to $T$, so everything **shrinks towards the edge**.

> [!card]- In Poincaré's sphere, what geometry do the inhabitants find?
> **Infinite, curved, non-Euclidean** space: their rulers shrink with what they measure, so they never reach the boundary.

> [!card]- Why can no experiment decide between the two descriptions of Poincaré's sphere?
> **Every experimental set-up** can itself be interpreted in either picture.

> [!card]- Why do we find it easier to call Poincaré's sphere world Euclidean?
> Our senses are **used to Euclidean space**, so a Euclidean disk plus a temperature law is simpler for us.

> [!card]- What analogy does Poincaré use for the choice of geometry?
> Measuring distance in **metres or yards**: a convention, with no fact of the matter.

> [!card]- How does Poincaré's conventionalism connect to Duhem's thesis?
> Laws are tested only indirectly: **geometry faces experience together with physics**.

## Links

- **Course:** [[PhilSci - Overview|Course overview]]
- **Before:** [[PhilSci-L01 - Introduction and Logical Empiricism]] for the logical positivists whose two dogmas Quine attacks; [[PhilSci-L01b - Popper and Lakatos]] for falsification, the independent-reasons condition and Lakatos's protective belt; [[PhilSci-L02 - Kuhn on Scientific Practice]]
- **Later:** [[PhilSci-L05 - Scientific Realism and its Critiques]] and [[PhilSci-L06 - Scientific Realism and the Pessimistic Meta-Induction]] for under-determination as an argument against scientific realism
- **Second half of the same session:** [[PhilSci-L03b - GenAI, Writing and Philosophical Learning]]
- **Tutorial:** [[PhilSci - Tutorials]], tutorial 3 (17 September), Quine and Laudan
- **Source:** lecture slides `3a Under-determination_online.pdf` and the flipped-classroom video (15 September). Quotations: Duhem on crucial experiments and good sense; Quine, "Two Dogmas of Empiricism" (1951); Laudan, "Demystifying Underdetermination", pp. 289, 293, 297, 298 (Curd & Cover, pp. 288-320); Quine, "On Empirically Equivalent Systems of the World" (1975), pp. 323-324; Laudan and Leplin on empirical equivalence; Stanford (2006); Poincaré, *Science and Hypothesis* (1902)
- **Readings:** Quine (1975), pp. 324 (last paragraph) to 326 (first paragraph) excluded; Laudan, Curd & Cover pp. 288-320; optional: Duhem, "Physical Theory and Experiment", Curd & Cover pp. 227-249
- **Further reading:** Curd & Cover, commentary to chapter 3; De Haro (2021), "The Empirical Under-Determination Argument against Scientific Realism for Dual Theories"; Le Bihan and Read (2018), "Duality and Ontology"
