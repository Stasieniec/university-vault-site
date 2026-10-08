---
type: lecture
course: 5354PHSC6Y
week: 3
lecture: 3
date: 2026-09-15
format: flipped classroom (pre-recorded video plus slides on Canvas)
status: complete
topics:
  - Duhem's under-determination
  - Quine's Two Dogmas of Empiricism
  - The web of belief
  - Critiques of Quine (Stein, Laudan)
  - Empirical under-determination
  - Poincare's conventionalism
---

# PhilSci-L03: Under-determination

> [!abstract] Overview
> An experiment contradicts your theory. Which belief do you give up?
>
> That question sounds trivial and is not. Duhem's answer is that logic alone cannot tell you, because you never test a hypothesis on its own, only a hypothesis bundled with a pile of auxiliary assumptions. Quine pushes this until the bundle is *everything you believe*, at which point almost any statement can be kept true if you are willing to pay elsewhere. Laudan then points out that "logically possible" and "reasonable" are different words.
>
> This is the lecture behind exam question 6, and it is also the escape hatch in question 3.

## 1. The problem: what exactly does an observation refute?

Suppose an astronomical theory conflicts with what you see through a telescope.

**Which theory just died: the astronomical one, or the theory of optics that says your telescope works?** Nothing in the observation itself answers that.

Popper's own answer is more concessive than his reputation suggests. There are no objective facts that force the choice. Science has to *accept an empirical basis by intersubjective agreement*. Accepting an observation report that falsifies a theory is, in the end, **a human decision**.

That admission is the crack Duhem widens into a doctrine.

## 2. Duhem's under-determination

### The formal statement

This is the version the exam asks for, so learn it in this shape.

> [!formula] Duhem's thesis
> Given that a set of premises $\{P_1, P_2, \dots, P_n\}$ deductively entails a statement $O$ describing a possible observation, and that from experimental observation one concludes that $O'$ is true, where $O'$ entails that $O$ is false, it follows that **the conjunction of the premises is false**. It does **not** follow that any particular premise is false.

In one line: *falsification is ambiguous*. Modus tollens kills the conjunction, not any named member of it.

```
   P1 ∧ P2 ∧ ... ∧ Pn  ⊨  O
   observation says     ¬O
   ─────────────────────────
   therefore            ¬(P1 ∧ P2 ∧ ... ∧ Pn)

   which is only:  "at least one of them is false"
   and never:      "P3 is false"
```

### No crucial experiment

A **crucial experiment** (`experimentum crucis`, an idea going back to Bacon, Hooke and Newton) is supposed to be "an irrefutable procedure for transforming one of the two hypotheses before us into a demonstrated truth."

Duhem says there is no such thing. Hypotheses are only ever tested **in combination with auxiliary assumptions**, the experimental apparatus among them. And there may be **unconceived alternatives**: hypotheses nobody has thought of yet, which the experiment therefore cannot rule out (this line is developed later by Stanford).

### Good sense

If logic does not decide, what does? Duhem's answer is **good sense** (`bon sens`): "certain opinions which do not fall under the hammer of the principle of contradiction are in any case perfectly unreasonable."

This is deliberately not an algorithm, and De Haro flags the obvious questions:

- What *is* good sense?
- How can a scientist be "an impartial and faithful judge"?

Note where this leaves the argument. Duhem does not conclude that theory choice is arbitrary. He concludes that it is not *formal*. Those are different claims, and conflating them is the mistake Laudan attacks in section 4.

## 3. Quine: Two Dogmas of Empiricism (1951)

Quine attacks two commitments of the logical-positivist picture from [[PhilSci-L01 - Introduction and Logical Empiricism|Lecture 1]].

| Dogma | What it claims | Quine's objection |
|---|---|---|
| **Reductionism** | Every scientific statement is logically equivalent to some statement in a special empiricist language, whether the subjective language of immediate experience or an objective physical-thing language. This is the verification theory of meaning | You **cannot test isolated statements**. Nothing has its own private stock of empirical consequences |
| **Analytic-synthetic distinction** | There is a sharp line between statements true by meaning alone and statements true by how the world is | The difference between conceptual schemes and facts "is of degree only". Analytic statements are just the limiting case of true statements: no empirical content, confirmed no matter how the world turns out |

Quine's sharpest move: **at bottom the two dogmas are identical.** Reductionism says each statement has its own confirming experiences; the analytic-synthetic distinction says some statements have none. Both assume statements can be assessed one at a time. Deny that, and both fall together.

### The web of belief

> [!quote] Quine, Two Dogmas
> The totality of our so-called knowledge or beliefs, from the most casual matters of geography and history to the profoundest laws of atomic physics or even of pure mathematics and logic, is a man-made fabric which impinges on experience **only along the edges**. Or, to change the figure, total science is like a field of force whose boundary conditions are experience. A conflict with experience at the periphery occasions readjustments in the interior of the field.
>
> But the total field is so **under-determined** by its boundary conditions, experience, that there is much latitude of choice as to what statements to reevaluate in the light of any single contrary experience.

```
                    ·  ·  ·  ·  ·  ·  ·  ·  ·          EXPERIENCE
                 ·                             ·       only touches
              ·      geography, history           ·    the edge
            ·                                      ·
           ·        physics, chemistry              ·
           ·                                        ·
            ·      mathematics, logic              ·   revision here is
              ·         (centre)                 ·     possible but costly
                 ·                            ·
                    ·  ·  ·  ·  ·  ·  ·  ·  ·
```

The slide's own figure ("Quine's web of belief") is a spider's web drawn inside a circle, with its outer rim formed by two dark, branch-like arcs, each labelled **OBSERVATIONS**. The web touches experience only at that rim; the strands run inwards to a dense centre. The ASCII sketch above adds the layering from the quotation (geography and history near the rim, logic and mathematics at the centre), which the figure itself leaves unlabelled.

The centre is not immune, only expensive. Logic and mathematics sit deepest because revising them forces the largest rearrangement, not because they are a different kind of truth.

**The Duhem-Quine thesis** is the web read as a claim about theory choice: a disconfirming observation only touches the edge, and auxiliary hypotheses absorb the blow. Laudan catalogues the available moves as Quine's "four stratagems".

## 4. Critiques of Quine

### Stein: the scepticism about meanings is unearned

The trouble sits at the heart of the argument: Quine is sceptical about **meanings** as such, and rejects theories of meaning outright rather than engaging them.

Frege and Carnap distinguish two kinds:

| Term | Also called | What it is | Classic example |
|---|---|---|---|
| **Sense** | intension | The linguistic meaning of the words | "morning star" and "evening star" differ in sense |
| **Reference** | extension | The actual object the words pick out | both refer to Venus |

The standard possible-worlds account gives the intension of a sentence $S$ as the maximal set of possible worlds where $S$ is true, or equivalently a map from possible worlds to truth values.

Quine attributes analyticity to Leibniz and writes it off as "picturesque", then spends a long discussion treating intensions as leftovers of Aristotelian essences. Stein's complaint: this **assumes** that an empiricist must be sceptical about meanings, and intensions in the Frege-Carnap sense would have merited real engagement rather than dismissal.

### Laudan: logically possible is not the same as rational

Laudan's objection is the one worth carrying into the exam, because it is the standard reply to anyone waving under-determination around.

> [!quote] Laudan, p. 289
> Sloppy formulations of the thesis of UD have encouraged authors to use it to support whatever relativist conclusions they fancy.

> [!quote] Laudan, p. 298
> Too many of the discussions of UD in the last quarter century have proceeded in an **evaluative vacuum**. They imagine that if a course of action is logically possible, then one need not attend to the question of its rationality.

Set the two side by side:

| Quine (p. 297) | Laudan (p. 293) |
|---|---|
| Any statement can be held true come what may, if we make drastic enough adjustments elsewhere in the system. Even a statement very close to the periphery can be held true in the face of recalcitrant experience by pleading hallucination, or by amending certain statements of the kind called logical laws | What grounds does Quine have for asserting this? One might expect he could establish its plausibility by examining the relevant rules of rational theory choice and showing that those rules were always so ambiguous that, confronted with any pair of theories and any body of evidence, they could never yield a decision. But Quine nowhere engages in a general examination of ampliative rules of theory choice |

Laudan's point: rationality includes **epistemic warrant**, above all the empirical evidence. Yes, you *can* save any theory by pleading hallucination. That does not make it a reasonable thing to do, and Quine never argues that the rules of theory choice are too weak to condemn it.

This is the same distinction Duhem was making with good sense, restated with more teeth.

### Laudan's two theses, and which one is the threat

> [!definition] Added to the deck on 2026-09-15, under "Additional questions"
> Laudan distinguishes two claims that get run together:
>
> 1. **Non-uniqueness UD thesis.** For any theory $T$ and any body of evidence supporting $T$, there is **at least one** rival to $T$ that is as well supported as $T$.
> 2. **Egalitarian thesis** (stronger). **Every** theory is as well supported by the evidence as any of its rivals.
>
> Laudan holds that **only (2), not (1), is a threat to epistemology and methodology.**

The discussion questions De Haro attaches are: **why** is only the egalitarian thesis a threat, and **under which of the two** would you classify Quine's empirical under-determination?

> [!intuition] Why only the stronger thesis bites
> Non-uniqueness says a rival exists. It does not say the rival is *equally good in every respect*, and it does not say that no further evidence or methodological consideration could separate them. Methodology survives non-uniqueness because its job is exactly to choose among the survivors. The egalitarian thesis denies that there is ever anything to choose on, which is what would make theory choice arbitrary and methodology idle.
>
> This is the same move as the Laudan quotation above: a logical possibility of a rival is not yet a reason to think the rules of theory choice are silent.

> [!intuition] Where Quine's empirical under-determination fits
> The slides leave the second question open. A defensible answer: Quine's empirical under-determination (section 5) is at most a **non-uniqueness** thesis. It claims that a theory has *at least one* empirically equivalent but theoretically inequivalent rival, and Quine (1975) himself calls it an open question whether any such case exists. It says nothing about *every* rival being equally well supported, so it does not reach the egalitarian thesis. It is also weaker than non-uniqueness in one respect: empirical equivalence (same observation sentences) is not yet equal evidential support, since, as Laudan and Leplin argue, evidential support can depend on more than a theory's observational consequences. Quine's holist claim in Two Dogmas ("any statement can be held true come what may") is the one that is sometimes read as egalitarian, and it is that claim Laudan's p. 293 objection targets. #needs-review (this answer is not given on the slides)

## 5. Empirical under-determination

A different and sharper thesis, from Quine's 1975 *On Empirically Equivalent Systems of the World*.

Imagine two theory formulations that are:

1. **Empirically equivalent**: they entail the same observation sentences
2. **Theoretically inequivalent**: they differ in their theoretical sentences

Then the theory is under-determined by the empirical data, because the data cannot decide between them. This is a direct problem for **scientific realism**: if you cannot tell which of two theories is right, which one are you a realist *about*?

Two varieties, and the distinction matters:

| Kind | Concerns | Turns on |
|---|---|---|
| **Semantic** | meaning and interpretation | independent of the evidence |
| **Epistemic** | the empirical evidence | what the evidence can settle |

### Does it ever actually happen?

The interesting question is whether real examples exist, and the honest answer in the literature is: not many, and suspiciously of one type.

- **Laudan and Leplin**: "It is noteworthy that contrived examples alleging empirical equivalence always involve the **relativity of motion**; it is the impossibility of distinguishing apparent from absolute motion to which they owe their plausibility." The pre-eminent historical cases are Ptolemy against Copernicus, which created the idea of empirical equivalence in the first place, and Einstein against Lorentz.
- **Quine (1975)** himself calls whether any cases exist an "open question".
- **Stanford (2006)**: critics "have been well within their rights to demand that serious, nonsceptical, and genuinely distinct empirical equivalents to a theory actually be produced" before withholding belief, rather than presuming such equivalents exist when none can be named.

So the strong thesis is widely asserted and thinly evidenced. That is a usable exam point.

### Where under-determination actually lurks: Quine on the common core

> [!definition] Added to the deck on 2026-09-15, under "On the idea of a common core"
> Quine's own account of when empirical equivalence does and does not generate a problem. Two quotations from *On Empirically Equivalent Systems of the World*:
>
> > If the implied observation conditionals (redundancies aside) are finite in number, we can simply take the conjunction of them, a single sentence, as our theory formulation. It contains its observation conditionals without remainder; they are all it is. It is implied by every empirically equivalent theory, and can conflict with none of them. (p. 323)
>
> > **Under-determination lurks where there are two irreconcilable formulations each of which implies exactly the desired set of observation conditionals plus extraneous theoretical matter, and where no formulation affords a tighter fit.** (p. 324)

This is worth having because it tells you what the thesis actually requires, and the requirement is demanding.

If you can collapse a theory into the bare conjunction of its observation conditionals, there is nothing left to be under-determined. That conjunction is implied by every empirically equivalent rival and conflicts with none of them, so it is a **common core** rather than a competitor. The problem only arises when two formulations are **irreconcilable**, each carries **extraneous theoretical matter** beyond the observation conditionals, and **neither fits more tightly** than the other.

> [!tip] Why this matters for the sceptical argument
> All three conditions have to hold at once. That is a substantial burden, and it explains the finding in the previous section: real examples are scarce because the conditions are hard to satisfy, not because nobody has looked. Anyone deploying under-determination against scientific realism owes you the three conditions, not just the slogan.
>
> Note also the page numbers: **pp. 324-326 are the Craig's-theorem passage De Haro told you on 2026-09-08 that you may skip.** The quotation above at p. 324 sits immediately before the skippable stretch begins. (Precisely, the announcement says to skip from the **last paragraph of p. 324** to the **first paragraph of p. 326**, so the p. 324 quotation is outside the skipped stretch only if it comes before that last paragraph.) #needs-review

## 6. Poincare's conventionalism

The best worked example in the lecture, and the one to reach for if asked to illustrate under-determination with something other than Duhem's abstract schema.

### Setup

Non-Euclidean geometries were shown to be mathematically consistent in the nineteenth century by Gauss and Riemann. That creates a question Kant's picture cannot easily answer.

- **Kant**: geometry and Newton's three laws are **synthetic a priori**. A priori, so independent of experience, yet synthetic, so they add something rather than being mere logical truths. Specific forces such as the inverse-square law and their parameters are empirical, that is synthetic a posteriori.
- **Poincare** (*Science and Hypothesis*, 1902): geometry is **neither** purely a priori **nor** a posteriori. It is **conventional**. We choose the geometric viewpoint that describes experience most simply, but we could take alternatives.

### First argument: perception does not hand us three dimensions

We do not perceive space directly as three-dimensional. The visual field is prima facie two-dimensional and only derivatively 3D, and "tactile space" could be multi-dimensional. We *discover* the structure of space by experiment, but that structure is not unique.

### Second argument: the thought experiment

Take a world inside a sphere of radius $R$. Two descriptions of it:

> [!example] Poincare's sphere
> **Description 1.** The world is a Euclidean disk with a temperature field $$T = R^2 - r^2$$ hottest at the centre, falling to zero at the boundary. Every body has the same coefficient of expansion, proportional to $T$. So any object is largest at the centre and shrinks to nothing as it approaches the edge.
>
> **Description 2.** Beings inside, whose own bodies shrink by the same law, experience the space as **infinite and curved**. A ruler shrinks exactly as fast as the thing it measures, so the boundary is never reached. If they develop a geometry, it will be **non-Euclidean**.

```
        ┌──────────────────────────────┐
        │        T = 0  (boundary)     │      Outside view:
        │    ╭──────────────────╮      │      a finite disk where
        │   ╱   objects shrink   ╲     │      everything shrinks
        │  │    ·  ·  ·  ·  ·  ·  │    │      towards the edge
        │  │  ·  ▪  ▪  ▪  ▪  ·   │    │
        │  │ · ▪  ███  ███  ▪ ·  │    │      Inside view:
        │  │  ·  ▪  ▪  ▪  ▪  ·   │    │      space is infinite,
        │   ╲   T = R²  (centre) ╱     │      geometry is
        │    ╰──────────────────╯      │      non-Euclidean
        └──────────────────────────────┘

        Same world. Two geometries. No experiment separates them.
```

The slide itself shows two circular pictures of the **Poincare disk model** of hyperbolic (non-Euclidean) geometry:

- **Left**: a disk tiled with black and white triangles. The label $T = R^2$ points at the centre and $T = 0$ at the rim. Near the centre the triangles are large; towards the rim they become smaller and smaller without end. In the hyperbolic geometry of the inhabitants every triangle is the **same size**; seen from outside with Euclidean eyes, they shrink towards the boundary, exactly as bodies shrink with falling temperature in Description 1.
- **Right**: M. C. Escher's *Circle Limit IV* (angels and devils), the same kind of tiling drawn with interlocking figures that repeat, ever smaller, towards the edge of the disk. Every figure is congruent to every other in the hyperbolic metric.

So the picture is literally one image with two readings: a finite Euclidean disk with a shrinking law, or an infinite hyperbolic plane of equal tiles.

### The conclusion

For us, with sensory organs habituated to Euclidean space, it seems easier to call this world a Euclidean disk and add a law of physics saying objects expand with temperature. But **there is no fact of the matter** about which description is right. Beings just like us who happened to live inside the sphere would describe it non-Euclidean.

No experiment decides, because **every experimental set-up can itself be interpreted in either picture**. The choice is a convention, in the same way that measuring distance in metres or yards is a convention.

This connects straight back to Duhem: laws cannot be tested directly, only indirectly, and the geometry is one of the auxiliary assumptions riding along in every test.

## 7. Where the lecture lands

- **Duhem**: under-determination is real, and resolved in practice by good sense rather than logic.
- **Quine**: scepticism about meanings undercuts the idea of non-empirical knowledge. The web of belief defeats both dogmas at once and rejects the verification principle.
- **Empirical under-determination**: theory is under-determined by data, and this is very common in scientific practice as **transient** under-determination, the temporary kind that more evidence later resolves.
- **Dualities** are potential cases of *semantic* under-determination, and on De Haro's own view are not a threat to a cautious scientific realism.

That last point is the lecturer's own research position, which is worth remembering given who marks the exam.

## Key Takeaways

> [!tip] Exam Focus
> **This lecture is examined directly.** Mock exam question 6:
>
> > The under-determination thesis was first formulated by Pierre Duhem, and it was later on strengthened by W.V.O. Quine. State this thesis in Duhem's own version, and explain it with an example.
>
> A model answer, at the half-page the rubric asks for:
>
> > Given that a set of premises $\{P_1,\dots,P_n\}$ deductively entails a statement $O$ describing a possible observation, and that experiment establishes $O'$ where $O'$ entails that $O$ is false, it follows that the conjunction of the premises is false. It does **not** follow that any particular premise is false. There is therefore no crucial experiment that can decide between two hypotheses irrefutably, because hypotheses are only ever tested in combination with auxiliary assumptions such as the theory of the experimental apparatus.
> >
> > Example: an astronomical theory conflicts with telescope observations. The conflict does not say whether the astronomical theory or the optical theory of the telescope is at fault.
>
> **It is also the escape hatch in question 3** (Einstein and Eddington). The expected answer says Popper must reject a falsified theory, but adds that by Duhem-Quine the scientist can always place blame elsewhere, and that Popper only permits this when there are **independent** reasons to reject the measurement, so no ad hoc rescue.
>
> Three things to be able to state cold:
> 1. Duhem's thesis in the formal version above, near-verbatim
> 2. Why no crucial experiment is possible: auxiliary assumptions
> 3. One worked example. Telescope is fastest; Poincare's sphere is the impressive one

> [!warning] The distinction people lose marks on
> "Logically possible" is not "rational". Quine shows you *can* save any statement. Laudan points out he never shows the rules of theory choice are too weak to say you *should not*. If a question invites you to conclude that under-determination makes science arbitrary, this is the reply.

## Exam questions

> [!exam]- Mock exam Q6: "The under-determination thesis was first formulated by Duhem and later strengthened by Quine. State this thesis in Duhem's own version, and explain it with an example." (half a page)
> **Key points:** premises entail $O$, experiment gives not-$O$; only the conjunction is false, no particular premise; no crucial experiment; hypotheses tested with auxiliary assumptions incl. apparatus; example (telescope or Poincare's sphere).
>
> Must hit:
> 1. **The formal version.** If premises $\{P_1, \dots, P_n\}$ deductively entail an observation statement $O$, and experiment establishes $O'$ where $O'$ entails not-$O$, then the **conjunction** of the premises is false. It does **not** follow that any particular premise is false.
> 2. **The consequence.** There is no crucial experiment that irrefutably decides between two hypotheses, because a hypothesis is only ever tested together with auxiliary assumptions (including the theory of the apparatus).
> 3. **An example.** An astronomical theory conflicts with telescope observations: the conflict does not say whether the astronomy or the optics of the telescope is at fault. (Poincare's sphere is the impressive alternative.)
>
> Losing marks: stating the slogan ("theory is under-determined by data") instead of the premises-and-conjunction version, or giving no example.

> [!exam]- Mock exam Q3: Einstein (1953) said that if the eclipse measurement had not shown the shift, "Then I am very sorry for the dear Lord, for the theory is correct." Would, and-or could, this be acceptable for Popper? (10 points)
> **Key points:** no, if the measurement is correct: falsified, dogmatic attitude; Duhem-Quine: blame can be put elsewhere (measurement, auxiliaries); acceptable only with independent reasons to reject the measurement; otherwise ad hoc, lowering scientific status.
>
> Follow the official key:
> 1. **No**, assuming the measurement was correct: the theory is falsified and has to be rejected. Einstein shows a **dogmatic attitude**.
> 2. **However**, perhaps the measurement was not correct. Given the **Duhem-Quine thesis**, scientists can always put the blame elsewhere (an auxiliary assumption, the measurement) and retain the theory.
> 3. But for Popper this is acceptable **only if there are independent reasons for rejecting the measurement result**. Otherwise it is an **ad hoc** hypothesis, which lowers the theory's scientific status.
>
> Losing marks: stopping at "no", or saying Duhem-Quine lets Einstein off without the independent-reasons condition.

> [!exam]- Does under-determination show that theory choice is arbitrary? Answer using Duhem, Quine and Laudan.
> **Key points:** Duhem: good sense decides, not formal is not arbitrary; Quine: any statement held true come what may; Laudan: logically possible is not rational, evaluative vacuum; only the egalitarian thesis would make choice arbitrary.
>
> - **Duhem:** no. Logic alone does not settle which premise to give up, but **good sense** does: some opinions are "perfectly unreasonable" even though no contradiction forces you to drop them. Choice is not *formal*, which is different from arbitrary.
> - **Quine:** pushes hardest. Any statement can be held true "come what may" if you make drastic enough adjustments elsewhere (pleading hallucination, even revising logic).
> - **Laudan:** logically possible is not rational. Quine argues in an "**evaluative vacuum**" and never examines the ampliative rules of theory choice to show they are too weak to decide. Only the **egalitarian** thesis (every theory is as well supported as every rival) would make choice arbitrary, and nothing in Quine establishes it.
> - Verdict: holism shows a theory *can* be saved, not that it is *reasonable* to save it.

> [!exam]- Explain under-determination using Poincare's sphere, and say what conclusion Poincare draws.
> **Key points:** Euclidean disk with $T = R^2 - r^2$, bodies shrink towards the edge; inhabitants find infinite non-Euclidean space; no experiment separates the two; geometry is conventional, no fact of the matter; geometry is an auxiliary assumption (Duhem).
>
> - A world inside a sphere of radius $R$. **Description 1:** a Euclidean disk with temperature $T = R^2 - r^2$ (hottest at the centre, zero at the edge), and every body expands in proportion to $T$, so everything shrinks towards the edge. **Description 2:** inhabitants, whose rulers shrink with them, never reach the edge and find space **infinite and non-Euclidean**.
> - No experiment separates the two, because every experimental set-up can itself be described in either picture.
> - Conclusion: geometry is **conventional**, neither a priori (against Kant) nor a posteriori. We pick the description that is simplest for us, like choosing metres over yards. There is no fact of the matter.
> - Link to Duhem: the geometry is one of the auxiliary assumptions that rides along in every test.

## Flashcards

> [!card]- For Popper, how does science come to accept its empirical basis?
> By **intersubjective agreement**. No objective facts force it, so accepting a falsifying report is ultimately a **human decision**.

> [!card]- Duhem's thesis: if premises $P_1, \dots, P_n$ entail $O$ and experiment establishes $O'$ incompatible with $O$, what follows?
> The **conjunction** of the premises is false. It does **not** follow that any particular premise is false.

> [!card]- Which thesis is summed up as "falsification is ambiguous"?
> **Duhem's under-determination thesis**: modus tollens refutes the conjunction, never a named premise.

> [!card]- What is a crucial experiment (*experimentum crucis*)?
> "An irrefutable procedure for transforming one of the two hypotheses before us into a **demonstrated truth**."

> [!card]- Why, for Duhem, is no crucial experiment possible?
> A hypothesis is only ever tested **together with auxiliary assumptions**, including the theory of the experimental apparatus.

> [!card]- Besides auxiliary assumptions, what else stops a crucial experiment from deciding between two hypotheses, according to Duhem?
> **Unconceived alternatives**: hypotheses nobody has thought of yet, which the experiment cannot rule out (developed later by Stanford).

> [!card]- What is the telescope example of Duhem's thesis?
> An astronomical theory conflicts with telescope observations, but nothing says whether the **astronomy or the optics** of the telescope is at fault.

> [!card]- According to Duhem, what decides which belief to give up when logic does not?
> **Good sense** (*bon sens*): some opinions no contradiction refutes are still "perfectly unreasonable".

> [!card]- What is the difference between Duhem's conclusion and saying theory choice is arbitrary?
> Duhem says choice is **not formal**: logic does not decide, but good sense still does.

> [!card]- What does De Haro question about Duhem's good sense?
> What good sense actually **is**, and how a scientist can be "an **impartial and faithful judge**".

> [!card]- What are Quine's two dogmas of empiricism?
> **Reductionism** and the **analytic-synthetic distinction**.

> [!card]- What does the dogma of reductionism claim, according to Quine?
> Every scientific statement is equivalent to one in a special empiricist language: the **verification theory of meaning**.

> [!card]- What is Quine's objection to reductionism?
> You **cannot test isolated statements**: no statement has its own private stock of empirical consequences.

> [!card]- What is Quine's objection to the analytic-synthetic distinction?
> The difference between conceptual scheme and fact is "**of degree only**".

> [!card]- For Quine, what are analytic statements?
> The **limiting case** of true statements: no empirical content, confirmed however the world turns out.

> [!card]- Why does Quine say the two dogmas are "at bottom identical"?
> Both assume statements can be assessed **one at a time** against experience. Deny that and both fall together.

> [!card]- Where does Quine's web of belief touch experience?
> **Only along the edges**. A conflict at the periphery forces readjustments in the interior.

> [!card]- What sits at the centre of Quine's web of belief?
> **Mathematics and logic**.

> [!card]- Why do logic and mathematics sit at the centre of Quine's web of belief?
> Revising them forces the **largest rearrangement**. They are revisable, just expensive, not a different kind of truth.

> [!card]- How does Quine extend Duhem's under-determination thesis?
> Duhem's bundle of auxiliary assumptions becomes **everything you believe**, so almost any statement can be kept true by paying elsewhere.

> [!card]- Which claim of Quine's says even a statement near the periphery can be saved by pleading hallucination or amending logical laws?
> Any statement can be **held true come what may**, given drastic enough adjustments elsewhere in the system.

> [!card]- What does Stein's criticism of Quine target?
> Quine **assumes** an empiricist must be sceptical about meanings, dismissing Frege-Carnap intensions instead of engaging them.

> [!card]- What is the difference between sense and reference (Frege, Carnap)?
> **Sense** (intension) is the linguistic meaning of the words. **Reference** (extension) is the object they pick out.

> [!card]- What does Laudan say discussions of under-determination have proceeded in?
> An "**evaluative vacuum**": assuming that if a move is logically possible, its rationality need not be asked about.

> [!card]- According to Laudan, what would Quine need to show for "held true come what may" and never does?
> That the **ampliative rules of theory choice** are always too ambiguous to decide between any two theories.

> [!card]- Why, for Laudan, is saving a theory by pleading hallucination not reasonable, even though it is possible?
> Rationality includes **epistemic warrant**, chiefly the empirical evidence. Logically possible is not rational.

> [!card]- What is Laudan's non-uniqueness thesis of under-determination?
> For any theory and evidence supporting it, there is **at least one** rival as well supported.

> [!card]- What is Laudan's egalitarian thesis?
> **Every** theory is as well supported by the evidence as any of its rivals.

> [!card]- Which of Laudan's two under-determination theses is a threat to epistemology and methodology?
> Only the **egalitarian** thesis.

> [!card]- Why does the non-uniqueness thesis of under-determination not threaten methodology?
> A rival existing does not make it equally good in every respect. Methodology's job is **choosing among the survivors**.

> [!card]- Why would the egalitarian thesis make methodology idle?
> It says there is **never anything to choose on**, so theory choice would be arbitrary.

> [!card]- What does Quine's empirical under-determination (1975) require of two theory formulations?
> **Empirically equivalent** (same observation sentences) but **theoretically inequivalent** (different theoretical sentences).

> [!card]- Why is empirical under-determination a problem for scientific realism?
> If the data cannot decide between rivals, which one are you a **realist about**?

> [!card]- What is the difference between semantic and epistemic under-determination?
> **Semantic** concerns meaning and interpretation, independent of the evidence. **Epistemic** concerns what the evidence can settle.

> [!card]- Dualities are potential cases of which kind of under-determination?
> **Semantic** under-determination.

> [!card]- On De Haro's own view, do dualities threaten scientific realism?
> No, not a **cautious scientific realism**.

> [!card]- Which kind of under-determination is very common in scientific practice?
> **Transient** under-determination: temporary, resolved later by more evidence.

> [!card]- According to Laudan and Leplin, what do contrived examples of empirical equivalence always involve?
> The **relativity of motion**: the impossibility of distinguishing apparent from absolute motion.

> [!card]- What are the pre-eminent historical cases of empirical equivalence, according to Laudan and Leplin?
> **Ptolemy vs Copernicus** and **Einstein vs Lorentz**.

> [!card]- What does Quine (1975) say about whether real cases of empirical under-determination exist?
> It is an "**open question**".

> [!card]- What does Stanford say critics may demand before anyone withholds belief on grounds of empirical equivalence?
> That serious, genuinely distinct empirical equivalents **actually be produced**, not presumed.

> [!card]- How well evidenced is the strong thesis of empirical under-determination?
> **Thinly**: widely asserted, with few real examples, and those mostly of one type.

> [!card]- Why does the conjunction of a theory's finitely many observation conditionals not create under-determination, for Quine?
> It is implied by every empirically equivalent rival and conflicts with none: a **common core**, not a competitor.

> [!card]- What three conditions must hold at once for under-determination to "lurk", according to Quine?
> - Two **irreconcilable** formulations
> - each with **extraneous theoretical matter** beyond the observation conditionals
> - **no formulation** fits more tightly

> [!card]- Why are real examples of empirical under-determination scarce, given Quine's common-core point?
> His **three conditions** are hard to satisfy at once.

> [!card]- What was Kant's view of the status of geometry and Newton's three laws?
> **Synthetic a priori**: independent of experience, yet adding content beyond logic.

> [!card]- What nineteenth-century development put pressure on Kant's view of geometry?
> **Non-Euclidean geometries** were shown to be mathematically consistent.

> [!card]- What is Poincare's view of the status of geometry?
> **Conventional**: neither a priori nor a posteriori. We choose the geometry that describes experience most simply.

> [!card]- What is Poincare's perceptual argument that space is not given as three-dimensional?
> The visual field is prima facie **two-dimensional**, and tactile space could be multi-dimensional.

> [!card]- In Poincare's sphere, what is the outside (Euclidean) description?
> A Euclidean disk with temperature $T = R^2 - r^2$. Bodies expand with $T$, so everything **shrinks towards the edge**.

> [!card]- In Poincare's sphere, what geometry do the inhabitants find?
> **Infinite, curved, non-Euclidean** space: their rulers shrink with what they measure, so they never reach the boundary.

> [!card]- Why can no experiment decide between the two descriptions of Poincare's sphere?
> **Every experimental set-up** can itself be interpreted in either picture.

> [!card]- Why do we find it easier to call Poincare's sphere world Euclidean?
> Our senses are **habituated to Euclidean space**, so a Euclidean disk plus a temperature law is simpler for us.

> [!card]- What analogy does Poincare use for the choice of geometry?
> Measuring distance in **metres or yards**: a convention, with no fact of the matter.

> [!card]- How does Poincare's conventionalism connect to Duhem's thesis?
> **Geometry** is one of the auxiliary assumptions riding along in every test.

> [!card]- When does Popper accept blaming a measurement instead of the theory it falsifies?
> Only with **independent reasons** to reject the measurement. Otherwise the rescue is **ad hoc**.

## Links

- **Course:** [[PhilSci - Overview|Course overview]]
- **Previous:** [[PhilSci-L01 - Introduction and Logical Empiricism]] · [[PhilSci-L02 - Kuhn on Scientific Practice]]
- **Source:** `3a Under-determination_online.pdf` and `Flipped classroom - Under-determination.mp4` on Canvas. The deck was **revised on 2026-09-15** and renamed from `3 ...` to `3a ...`; the revision appended two slides, both folded in above (sections 4 and 5). Nothing in the original was changed or removed.
- **Second half of the same session:** [[PhilSci-L03b - GenAI, Writing and Philosophical Learning]]
- **Further reading:** Curd & Cover, commentary to Chapter 3 · Quine (1975), *On Empirically Equivalent Systems of the World* · De Haro (2021), *The Empirical Under-Determination Argument against Scientific Realism for Dual Theories* · Le Bihan and Read (2018), *Duality and Ontology*
