---
type: lecture
course: 5354PHSC6Y
week: 3
lecture: 3b
date: 2026-09-15
format: flipped classroom, second half (slides on Canvas as `3b GenAI.pdf`)
status: complete
topics:
  - GenAI as a study aid versus a substitute for learning
  - Gartenberg et al. (2026) on AI in journal submissions and peer review
  - The institutional incentive diagnosis
  - Authorship, authenticity and responsibility
  - What writing with AI actually requires
  - The course's own AI policy
---

# PhilSci-L03b: GenAI, Writing and Philosophical Learning

> [!abstract] Overview
> The second half of the 15 September flipped classroom, and the only session in this course that is about how you are permitted to work rather than about a philosophical doctrine.
>
> De Haro's thesis is that the interesting question is not whether AI use is allowed. It is whether a given use **strengthens or replaces your own learning**, and he thinks that question has a determinate answer in most concrete cases. He supports it with one empirical study of what AI has done to a journal's submission pipeline, one public scandal about authorship, and one long anecdote about trying to write a philosophy paper with AI himself.
>
> The slides carry their own AI use statement, which is the argument being made by demonstration.

> [!warning] This is the deck that governs your paper
> Sections 7 and 8 are course policy, not commentary. The paper (30%) and the presentation (20%) are both governed by them, and the **Project Description** on Canvas repeats the rules in stricter language: generating or structuring your arguments or analysis is **not permitted**, and any use at all requires an AI use statement naming the tool, the purpose and the extent.
>
> The Jetten case slide (printed slide 10, "AI and public communication") also says: **"Pangram detection has become very good."** Read that as a statement of intent.

---

## 0. The slides declare their own provenance

The title slide carries this, verbatim:

> [!quote] De Haro, title slide
> AI was used to produce a first draft of these slides, based on an outline, detailed instructions, and texts that I supplied to AI Chat. Afterwards, I edited and adapted that draft to produce the final slides.

This is not decoration. It is the model of compliant use that sections 7 and 8 then formalise: a human supplies the outline, the instructions and the source texts; the machine drafts; the human edits and takes responsibility; the use is declared. Every restriction later in the deck is consistent with the way the deck itself was made.

The session opened with a Wooclap poll asking the room two questions, "Do you use GenAI to study?" and "How do you use it?", before any position was stated.

> [!tip] A note on the deck's ordering
> The page order in `3b GenAI.pdf` does not match the slide numbers printed on the pages (they run 1, 2, 14, 4, 5, 7, 8, 10, 11, 12, 3, 15, 16, 17; the title page, the conclusion and a figure-only page between slides 5 and 7, holding Figures 6 and 9, carry no number). The deck was reordered before export. This note follows the argument rather than the file, and flags nothing by it.

---

## 1. The framing: neither enthusiasm nor prohibition

De Haro refuses both of the available slogans. The slide is a two-column contrast.

| AI **can** support learning | AI should **not** replace learning |
|---|---|
| Ask for clarification of a difficult definition | Do not outsource the reading |
| Request alternative examples | Do not submit AI-generated answers as your own |
| Test whether you can explain an argument | Do not use generated text before you can judge its quality and accuracy |
| Use feedback to revise text **that you have written** | Always acknowledge AI use when applicable |

> [!definition] The organising question
> **"The central question is whether AI strengthens or replaces your own learning."**

Notice the structure of the permitted column. Every item presupposes that you have already done the work: you have read the text and got stuck on a definition, you have understood a concept well enough to judge whether a fresh example fits, you have an argument in your head to be tested on, you have **written something** for the feedback to be about. The AI is downstream of your effort in every case. The forbidden column is the same list with the effort removed.

---

## 2. Beyond writing: AI and mathematics

A short aside, included to show that the sceptical position is not a blanket one.

| What may become possible | What must remain visible |
|---|---|
| Systematic comparison of Cayley graphs and state transfer; De Haro cites Krystal Guo's work at the KdV Institute as striking | Access usually depends on a handful of commercial companies |
| AI may help discover relations across technical literatures and formulate hypotheses for verification | Advanced subscriptions cost around €100 per month |
| | **Generation is not verification, proof, or understanding** |

That last line is the one that carries across to the rest of the deck. The objection is not that machine output is worthless; it is that producing a candidate and establishing that it is correct are different acts, and only the first has got cheaper.

The access point is a distinct argument and easy to skip past. If the tools that confer an epistemic advantage are gated behind a subscription, then the distribution of who can do good work changes for reasons that have nothing to do with ability.

---

## 3. The research evidence: Gartenberg et al. (2026)

> [!info] What the study is
> Gartenberg et al. examine **submissions and peer reviews at the journal *Organisation Science***, over a **five-year period beginning January 2021**. They analyse both manuscripts and reviews.
>
> Source as given on the slides: "Gartenberg et al. (2026), summary document provided for this class." The deck works from a summary, not the full paper.

**Their measures of writing quality**, all six:

- Readability
- Jargon
- Nominalisation
- Passive voice
- Hedging
- Specificity

> [!quote] The central claim
> Current AI tools, combined with incentives, are pushing research towards **more, rather than better, writing.**

### 3.1 The headline results

| | |
|---|---|
| **+42%** | submission volume since late 2022 |
| **−1.28 SD** | Flesch Reading Ease in January 2026 relative to January 2021 |

Submissions up by nearly half; readability down by more than a standard deviation. The deck presents these together because the claim is about the pair, not either number alone.

**The qualifications are on the slide**, and De Haro leaves them there rather than burying them:

- The finding is **systems-level, not individual-level**
- It is **one journal**
- It **does not establish that every use of AI makes text worse**

> [!warning] Do not overstate this in an essay
> The qualifications are doing real work. The study licenses a claim about what happens to a publication system under a change in the cost of producing text. It licenses **no** claim about whether any particular author writes worse with AI. Conflating the two is the obvious error and it is pre-empted on the slide itself.

### 3.2 The figures

Four figures from the paper are reproduced on the slides. Their content, since the note must stand in for the deck:

- **Figure 1. Monthly Submission Volume at *Organization Science*, January 2013 to end of 2025.** A histogram of monthly submission counts, roughly 50-100 per month through the 2010s, drifting upward. Two dashed vertical markers: COVID-19 (March 2020) and ChatGPT (November 2022). The bars climb steeply after the ChatGPT line, reaching roughly 150-160 per month by 2025.
- **Figure 2. Monthly Submission Volume by AI Use Categories over Time, 2021-2026.** Five series: Total (black), and four AI-share bands (0-15%, 15-30%, 30-70%, 70%+) applied to first submissions' abstracts. The 0-15% band declines steadily after the ChatGPT marker (to about 50 per month by 2026) while the three higher-AI bands all rise from near zero (to roughly 25-35 each). Total coincides with the 0-15% line until the marker and drifts down from about 100 per month in 2021 (with a spike near 130 at the start of 2022) to about 75-80 in late 2022; after the marker it climbs to roughly 140 by 2026. The y-axis is number of submissions; the figure's note says each line is the share of first submissions' abstracts classified in that AI band.
- **Figure 6. Flesch Reading Ease of abstracts, 2013 to February 2026**, in SD units with a confidence band. Flat and slightly positive (around +0.1 to +0.25) for a decade, holding through the COVID marker, then turning sharply downward after the ChatGPT marker and falling to roughly −1.0 by 2026. The decline is the visual centrepiece of the deck.
- **Figure 9. AI Use in Reviews Over Time, 2021-2026.** Share of reviews by AI-use band. The 0-15% band sits near 1.0 until the ChatGPT marker, then declines to roughly 0.65 by 2026, with the 30-70% band rising to about 0.2 and the other two bands to about 0.1 each.

### 3.3 Problems in peer review

The effect is on both sides of the process.

**Manuscripts:**

- Above 30% AI use, **desk-rejection rate is 30% higher**
- High-AI submissions **rarely receive revise-and-resubmit decisions**

**AI-generated reviews:**

- **More than 30% use some AI**
- More difficult to read
- Narrow focus

> [!quote] The concern, stated on the slide
> Easy text generation **shifts costs to editors and reviewers**.

This is the sharpest point in the empirical section and it is an argument about externalities rather than about quality. Producing a submission became cheaper; evaluating one did not. The saving accrues to the author and the cost lands on people who did not make the choice.

### 3.4 Why "more rather than better": the institutional diagnosis

The authors' causal chain, reproduced from the slide as a four-box flow:

```
  Universities              Researchers              GenAI                    Journals
  ───────────               ───────────              ─────                    ────────
  Reward           ──►      Pressure to      ──►     Reduces the      ──►     Increase in
  publications in           increase                 cost of                  weak
  top journals              output                   producing text           submissions
```

> [!intuition] Why the diagnosis is institutional and not moral
> Nobody in this chain is behaving badly. Universities reward publication because they need a legible measure. Researchers respond to the measure they are judged on. GenAI lowers a cost. The degradation is a property of the **system**, which is exactly why the slide's qualification says "systems-level not individual-level". If you want to argue against the finding, the place to attack is the first box, not the third.

---

## 4. The case: AI and public communication

> [!example] The Jetten posts
> **Dutch Prime Minister Rob Jetten's social media.** What NRC reported:
>
> - **232 posts** by the Prime Minister and his party reported to have been AI-generated
> - The issue became especially sensitive where posts concerned **apologies to the Moluccan community**
> - Public debate focused on **authenticity and responsibility**
>
> Source: Denise Retera, "AI-gebruik van premier Jetten op social media ligt onder de loep", *NRC*, 11 September 2026.

The slide reproduces the top of the NRC article: the headline ("Prime Minister Jetten's AI use on social media under scrutiny"), a photo of Jetten standing in front of a wall with rough stone objects mounted on it, and the opening text. The standfirst says that Jetten is light-hearted about using AI for his "socials", but that according to experts politicians who do this risk damaging their image and authenticity. The article opens by recalling that in 2018 Jetten, then D66 parliamentary leader, repeated exactly the same sentences about abolishing the dividend tax three times in a row in front of television cameras, with an unchanged facial expression, which earned him the nickname "Robot Jetten".

The question for discussion, as posed on the slide:

> 'AI can make things better — but **who is speaking, and who is responsible for the message?**'

And the moral De Haro draws:

> **Using AI is not the whole issue. Context, transparency, and authorship matter.**

The apology is the detail that makes the case work. An apology is a **speech act whose validity depends on who performs it**; a generated apology is not a badly written apology but arguably not an apology at all. That is a point about the pragmatics of certain utterances rather than about text quality, and it generalises to any writing whose value depends on its being yours. A submitted essay is one of those.

The slide closes with the line quoted at the top of this note: **"Pangram detection has become very good."**

---

## 5. The anecdote: writing a philosophical review paper with AI

De Haro's own attempt, offered as evidence rather than as anecdote.

**The setup.** A chapter for an Elsevier reference work on Scientific Theories, co-written with **Hans Halvorson**. He gave the AI the chapter outline with a target word count per section, plus the LaTeX format taken from a previous chapter he had written on Dualities, and asked it to work section by section.

> [!quote] The prompt, as shown on the slide
> I have given you the outline of the Chapter for the Elsevier reference work where I am writing a Chapter on Scientific Theories together with Hans Halvorson. Next to each section is the number of words it needs to have.
>
> Here below I give you the latex format that I will use (taken from a previous chapter on Dualities I wrote). Start by writing section 2 on the received view, in latex. Put also the outline I am giving you into the latex, so that I will already have the structure with the outline of the content (everything up to section 6, conclusion: forget the rest, although you can use the references if you have access to them). We will go section by section, and I will show you how to write. I left some references so you can see how to write references.

**The starting point is stated explicitly: an outline for a review paper, i.e. not new research.**

### What the collaboration actually involved

| Human work | AI assistance |
|---|---|
| Construct detailed outline of the paper and argumentation | Produce text according to detailed instructions |
| Give detailed instructions about style and literature | Rewrite taking detailed feedback into account |
| Endless rewriting and feedback, additional writing | |

Two conclusions on the slide:

- The project is a **review paper**: it does not begin by asking AI to generate original philosophical ideas.
- **Expert guidance is needed throughout**: prompting, checking, selecting, writing, and rewriting.

> [!warning] The punchline
> **"500 pages of conversation, and the paper is still unfinished."**

The rhetorical work this does is worth naming. It is a concession and a warning at once: the person telling you this is a professional philosopher who supplied the argument, the structure, the literature and the style, on a paper that by design required no original ideas, and it still cost him five hundred pages and is not done. The implied question to a student is what the same process would cost someone who does not yet have the outline in their head.

---

## 6. Writing with AI requires expertise

The general conclusion, stated as a conjunction of two necessary conditions.

> [!definition] The two conditions
> You can responsibly use AI to help write **only if both conditions hold**:
>
> **(A) You are already an expert.** You can check claims, detect omissions, and revise to improve the argument.
>
> **(B) You can already write philosophy.** You recognise unnatural text and weak or incomplete arguments.
>
> "Before that stage, the priority is to **develop your own philosophical voice and judgement**."

This is the load-bearing slide of the deck and it has a definite logical shape. Both conditions are about **detection**: (A) is detecting that a claim is false or a step is missing, (B) is detecting that prose is bad or an argument incomplete. Neither is about generation. The claim is that AI-assisted writing is only safe for someone who could have caught the errors anyway, which entails that the tool is of least use to the person who most wants it.

> [!intuition] The bootstrapping problem
> If the conditions for safe use are expertise and the ability to write philosophy, and the way to acquire both is by writing philosophy without assistance, then the tool is unavailable precisely during the period in which you are acquiring the competence to use it. That is not a paradox, it is an ordering claim: unassisted practice first, assistance after. It is also why the course's permitted uses are all about testing and revising rather than drafting.

### Why this matters in Philosophy of Science specifically

> **GenAI is good at surface summaries, but poor at argumentative nuance.**

And the course is about exactly the thing the tool is worst at:

- Read and (re)construct arguments carefully
- Evaluate whether they succeed
- Produce and criticise written text by the same standards

The mismatch is the argument. A tool that is strong at summary and weak at nuance is a tool that fails at all three of the course's aims, and a summary that looks right is worse than one that looks wrong.

---

## 7. Course policy: appropriate uses

> [!definition] The permitted four
> AI may be used **as a study aid when it supports your own study**.
>
> | | |
> |---|---|
> | **Clarify** | Ask for an explanation of a term **after you have tried** to understand the text |
> | **Compare** | Request alternative examples, **then assess whether they really fit** the concept |
> | **Test** | Ask AI to **question you** on an argument you have read |
> | **Revise** | Request suggestions about a text **you have written**; **decide yourself** which suggestions to use |
>
> "In every case, use AI to deepen your understanding, rather than to replace reading and thinking."

Each of the four has a qualifier attached, and the qualifier is the operative part. *After* you have tried. *Then* assess. Question *you*. A text *you have written*, and *you* decide. Strip the qualifiers and all four become violations.

---

## 8. Course policy: limits and transparency

> [!warning] Not appropriate
> - Outsourcing assigned reading
> - **Generating a text for submission**
> - Using text that you cannot assess for accuracy, quality, and argumentation

> [!definition] Transparency in written work
> - **State what you did yourself**
> - **Explain whether and how AI was used, for which purpose, and which choices were your own**
>
> "Your submitted work must make your own reasoning and responsibility visible."

Note that the transparency requirement is not satisfied by a bare disclosure. It asks for three things: what you did, how the tool was used and for what, and **which choices were yours**. That last clause presupposes that the choices were in fact yours, so the statement is a claim about authorship and not a disclaimer.

The **Project Description** on Canvas states the same policy in stricter and more legally shaped language, and it is the version that governs the paper. Permitted there: improving clarity and grammar, refining phrasing, brainstorming and contextualising ideas, locating relevant papers or concepts as with a search engine. Not permitted: **generating or structuring your arguments or analysis**, submitting AI-generated text as your own, fabricating references or quotations. Any use requires an **AI use statement** at the end of the submission specifying the tool, the purpose and the extent. Misuse is handled under the UvA *Regulations Governing Fraud and Plagiarism* (2019).

---

## Conclusion, in De Haro's three lines

1. **Use AI to test your understanding**
2. **Develop your own philosophical voice before relying on AI to draft text**
3. **Be transparent about AI use and responsible for every claim you submit**

---

## Key Takeaways

> [!tip] Is this examined?
> **Almost certainly not.** The mock exam has eight questions and they map onto the doctrinal lectures; none of them is about research ethics or AI use. Read this note for the policy in sections 7 and 8, which governs 50% of your course grade through the paper and presentation, not for exam revision.
>
> The one place it could surface is a tutorial discussion, where the Jetten case and the "who is speaking" question are the obvious prompts.

> [!warning] The three things to actually retain
> 1. **The permitted uses all presuppose prior effort.** Clarify after trying, assess the examples, be questioned, revise your own text. Remove the qualifier and it is a violation.
> 2. **An AI use statement is mandatory if you use AI at all**, and it must name the tool, the purpose, the extent, and which choices were yours.
> 3. **Generating or structuring your arguments is the bright line.** Not phrasing, not grammar, not finding papers. Arguments and structure.

> [!intuition] The best line in the deck, for use elsewhere
> "Generation is not verification, proof, or understanding." It is stated about mathematics and it generalises to the whole session, and it is a compact statement of a real epistemological point: the cost of producing a candidate answer and the cost of establishing that it is correct have come apart, and only one of them has fallen.

## Links

- **Course:** [[PhilSci - Overview|Course overview]]
- **Same session:** [[PhilSci-L03 - Under-determination]] was the first half of the 15 September flipped classroom
- **Previous:** [[PhilSci-L01 - Introduction and Logical Empiricism]] · [[PhilSci-L01b - Popper and Lakatos]] · [[PhilSci-L02 - Kuhn on Scientific Practice]]
- **Tutorials:** [[PhilSci - Tutorials]]
- **Source:** `3b GenAI.pdf` on Canvas, uploaded 2026-09-15, 17 slides
- **Also governing:** `2026 Project Description Philosophy of Science.pdf` on Canvas, section "About the use of AI tools"
- **Further reading:** Gartenberg et al. (2026), studied via the summary document provided for the class · Denise Retera, "AI-gebruik van premier Jetten op social media ligt onder de loep", *NRC*, 11 September 2026
