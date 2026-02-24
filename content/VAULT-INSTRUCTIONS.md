# Vault Instructions — For Bob (AI Assistant)

> [!warning] Read this before touching any note in this vault.
> These instructions govern how notes are created, updated, and maintained. Follow them strictly.

---

## 1. Purpose

This Obsidian vault is Stanisław's **single source of truth** for university coursework. The goal: reading these notes alone should be **sufficient to pass every exam, solve every exercise, and understand every concept** — without needing to go back to lecture slides, textbooks, or recordings.

**Every note must be a full substitute for its source material.** If a lecture slide explains something, the note must capture that explanation completely. If a book chapter proves a theorem, the note must contain that proof. Nothing should be "see the book for details" — the details belong here.

---

## 2. What Goes Into Notes

### 2.1 Completeness Requirements

- **All theory** from lectures and book chapters — definitions, theorems, proofs, derivations
- **All algorithms** — full pseudocode, not "see Algorithm 5.1"
- **All formulas** — in LaTeX, named, and with intuition for what each term means
- **All examples** from lectures/book — worked through, not just referenced
- **All images and diagrams** — when source material contains a figure (backup diagrams, gridworlds, learning curves, etc.), **process the image and reproduce its content** in the note. Describe what the figure shows, redraw it as ASCII/Mermaid where possible, or transcribe its data. **Never write "see Figure X" and move on.**
- **All exercises and solutions** — full worked solutions with step-by-step reasoning, not just final answers
- **Intuitions and "why"** — not just "what." If a method has a known weakness, say why. If a formula looks a certain way, explain the intuition behind each term.

### 2.2 What NOT to Include

- Administrative info (deadlines, group policies, grading weights) — that lives in the course overview MOC only
- Redundant re-explanations of the same concept across multiple notes — use `[[wikilinks]]` to the Concept note instead
- Raw copy-paste of source text without formatting — everything must be structured and scannable

---

## 3. Note Types & Templates

### 3.1 Course Overview (MOC — Map of Content)

**Location:** `Courses/<COURSE>/<COURSE> - Overview.md`

The central hub for a course. Contains:
- Course metadata (credits, instructor, exam format, textbook)
- Week-by-week schedule with links to all lecture notes, book chapters, exercises
- Assessment breakdown
- Status tracker (which notes are done, which need review)

**Frontmatter:**
```yaml
---
type: moc
course: <COURSE_CODE>
tags: [moc]
---
```

### 3.2 Lecture Notes

**Location:** `Courses/<COURSE>/Lectures/<COURSE>-L<##> - <Title>.md`

Captures **everything** from the lecture slides plus context from the book. Should be self-contained — reading this note = attending the lecture + reading the relevant book sections.

**Structure:**
1. Frontmatter with metadata
2. Overview / motivation (1-2 paragraphs max)
3. Content sections mirroring lecture flow
4. Key takeaways / summary box
5. Links to related concepts, exercises, and book chapters

**Frontmatter:**
```yaml
---
type: lecture
course: <COURSE_CODE>
week: <N>
lecture: <N>
date: <YYYY-MM-DD>  # lecture date if known, otherwise omit
book_sections: ["Ch X.Y", "Ch X.Z"]
topics:
  - "[[Concept A]]"
  - "[[Concept B]]"
status: complete  # complete | draft | stub
---
```

### 3.3 Book Chapter Notes

**Location:** `Courses/<COURSE>/Book Notes/<COURSE>-Book Ch<N> - <Title>.md`

Covers material from the textbook. Focuses on content that **adds to or deepens** what's in the lecture notes. If the book and lecture cover the same thing identically, the lecture note handles it and the book note links there. But if the book has extra proofs, examples, or nuance — it goes here.

**Frontmatter:**
```yaml
---
type: book-chapter
course: <COURSE_CODE>
book: "<Book Title>"
chapter: <N>
sections: ["X.Y", "X.Z"]
topics:
  - "[[Concept A]]"
status: complete
---
```

### 3.4 Concept Notes (The Wiki Layer)

**Location:** `Concepts/<Concept Name>.md`

The **atomic units of knowledge**. Each concept gets one note. These are linked from everywhere — lectures, book notes, exercises, other concepts.

**Rules:**
- **One concept per note** (e.g., "Bellman Equation" not "Bellman Equations and Dynamic Programming")
- **Course-agnostic when possible** — a concept like "Gradient Descent" applies to RL, DL, ML1. Tag which courses use it.
- **Must include:** definition, intuition, math (if applicable), relation to other concepts, where it's used
- **Flat folder** — no nesting by course. Use tags/properties to filter.

**Frontmatter:**
```yaml
---
type: concept
aliases: [<alt name 1>, <alt name 2>]
course: [<COURSE_1>, <COURSE_2>]
tags: [<relevant-tags>]
status: complete
---
```

**Structure:**
1. **Definition** — precise, formal
2. **Intuition** — plain-language explanation, analogies
3. **Mathematical Formulation** — LaTeX, with term-by-term explanation
4. **Key Properties / Variants** — bullet points
5. **Connections** — links to related concepts
6. **Appears In** — which lectures, book chapters, exercises reference this

### 3.5 Exercise & Homework Notes

**Location:** `Courses/<COURSE>/Exercises/<COURSE>-HW<N> - <Title>.md`

**Every exercise and homework problem** must have:
- Full problem statement
- Complete worked solution with all steps shown
- Explanation of which concepts are being tested (with `[[links]]`)
- Common pitfalls or tricky parts highlighted

**Frontmatter:**
```yaml
---
type: exercise
course: <COURSE_CODE>
week: <N>
source: "Exercise Set <N> / Homework <N> / Coding Assignment <N>"
concepts:
  - "[[Concept A]]"
  - "[[Concept B]]"
status: complete
---
```

### 3.6 Coding Assignment Notes

**Location:** `Courses/<COURSE>/Coding Assignments/<COURSE>-CA<N> - <Title>.md`

Captures:
- What the assignment implements and why
- Key algorithms used (linked to concepts)
- Important implementation details / gotchas
- Core code snippets (not the entire notebook — just the important parts)
- Results and interpretation

**Frontmatter:**
```yaml
---
type: coding-assignment
course: <COURSE_CODE>
week: <N>
language: python
concepts:
  - "[[Concept A]]"
status: complete
---
```

### 3.7 Exam Prep Notes

**Location:** `Courses/<COURSE>/Exam Prep/`

- **Exam Analysis** — breakdown of past exams: what topics were tested, question types, weight
- **Cheat Sheet** — single note with every critical formula, named and linked to its concept note. Designed for last-minute cramming.

---

## 4. Formatting Standards

### 4.1 Math
All math in LaTeX. Inline: `$...$`. Display: `$$...$$`.

Always explain what variables mean the first time they appear:
```markdown
$$V(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

where:
- $V(s)$ — value of state $s$ under policy $\pi$
- $G_t$ — return (total discounted reward) from time step $t$
- $S_t$ — state at time $t$
```

### 4.2 Callouts
Use Obsidian callouts for visual scanning:

```markdown
> [!definition] Bellman Equation
> Recursive decomposition of the value function...

> [!formula] State-Value Function
> $$V(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$$

> [!intuition] Why This Works
> Think of it as asking: "What's the average reward I'll get..."

> [!example] Gridworld
> Consider a 4×4 grid where...

> [!warning] Common Mistake
> Don't confuse $V(s)$ with $Q(s,a)$...

> [!tip] Exam Hint
> This shows up in almost every exam...
```

### 4.3 Algorithms
Full pseudocode in code blocks with line-by-line comments where non-obvious:

````markdown
```pseudo
Algorithm: First-Visit MC Prediction (for estimating V ≈ v_π)
─────────────────────────────────────────────────────────────
Input: policy π to evaluate
Initialize:
  V(s) ∈ ℝ arbitrarily, for all s ∈ S
  Returns(s) ← empty list, for all s ∈ S

Loop forever (for each episode):
  Generate episode following π: S₀,A₀,R₁, ..., S_{T-1},A_{T-1},R_T
  G ← 0
  Loop t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    Unless S_t appears in S₀,...,S_{t-1}:     // first-visit check
      Append G to Returns(S_t)
      V(S_t) ← average(Returns(S_t))
```
````

### 4.4 Diagrams

- **Mermaid** for flowcharts, trees, and concept relationships
- **ASCII art** for simple grids, backup diagrams
- **Described figures** when reproducing complex plots — describe axes, trends, key data points

### 4.5 Links

- **Always wikilink** concepts: `[[Monte Carlo Methods]]` not "Monte Carlo methods"
- **Use aliases** where grammatically needed: `[[Monte Carlo Methods|MC methods]]`
- **Link exercises to concepts** they test
- **Link lectures to book chapters** they correspond to
- **Back-link from concepts** to where they appear

### 4.6 Tags

Use sparingly and consistently:
- `#exam-topic` — high-probability exam content
- `#key-formula` — must-memorize equations
- `#needs-review` — note needs another pass
- `#intuition` — contains a particularly good explanation

---

## 5. Process for Creating Notes

### 5.1 Source Material Priority

When creating notes for a topic, synthesize from **all available sources** in this order:
1. **Lecture slides** (PDF) — primary structure and flow
2. **Textbook chapters** — depth, proofs, additional examples
3. **Exercise sets with answers** — test understanding, reveal what's important
4. **Coding assignments** — practical implementation details

### 5.2 Handling Images from Source PDFs

**Do not skip images.** When a lecture slide or book page contains a figure:
1. Use the `image` tool to analyze the image and extract its full content
2. Reproduce the content in the note:
   - **Diagrams/flowcharts** → Mermaid or ASCII art
   - **Plots/graphs** → Describe axes, trends, key values; redraw as ASCII if simple
   - **Tables** → Markdown tables
   - **Backup diagrams** → ASCII art showing the tree structure
   - **Math on slides** → Transcribe to LaTeX
3. If the image is essential and cannot be adequately reproduced in text (e.g., complex plots with many data points), save a copy to `Assets/` and embed it: `![[image-name.png]]`

### 5.3 Workflow Per Topic

For each week/topic:
1. Extract and read the lecture PDF
2. Read corresponding book chapters
3. Create/update **Concept notes** for each key concept introduced
4. Create the **Lecture note** — referencing concepts via wikilinks
5. Create the **Book chapter note** — adding depth beyond the lecture
6. Create the **Exercise note** — full solutions with concept links
7. Create the **Coding assignment note** — if one exists for that week
8. Update the **Course Overview MOC** with links and status

---

## 6. Sub-Agent Usage

### 6.1 When to Use Sub-Agents

Sub-agents (via `sessions_spawn`) are useful for parallelizing note creation:
- **Good:** Spawning a sub-agent to process a specific lecture PDF into a lecture note while the main agent works on a different lecture or book chapter
- **Good:** Having a sub-agent extract and process all exercises from an exercise set PDF
- **Good:** Delegating a standalone concept note that doesn't depend on other unfinished notes

### 6.2 When NOT to Use Sub-Agents

- **Concept notes that require cross-referencing** other concept notes being written simultaneously — do these sequentially to ensure consistency
- **The Course Overview MOC** — this is the main agent's job since it links everything together
- **Quality review passes** — main agent should review sub-agent output for consistency
- **Small, quick notes** — overhead of spawning isn't worth it for a 5-minute note

### 6.3 Sub-Agent Instructions

When spawning a sub-agent for note creation, always include in the task:
1. The exact note type and template to follow (copy relevant template section from this file)
2. The source file path(s) to read
3. The output file path
4. Explicit instruction to **process all images** and not skip them
5. The list of concept names to wikilink to (so links are consistent across notes)
6. Reminder: **completeness over brevity** — the note must substitute the source material

### 6.4 Post-Sub-Agent Checklist

After a sub-agent delivers a note, the main agent must:
- [ ] Verify frontmatter is correct
- [ ] Check all wikilinks point to existing or planned concept notes
- [ ] Ensure no images/figures were skipped
- [ ] Confirm formulas render correctly
- [ ] Add the note to the Course Overview MOC

---

## 7. Vault Maintenance Rules

1. **Never delete content without asking** — mark with `#needs-review` instead
2. **Keep note filenames stable** — changing filenames breaks wikilinks everywhere
3. **Update the MOC** every time a new note is added
4. **Concept notes are shared** — if editing a concept note, check it doesn't break context for another course
5. **Status tracking** — every note must have a `status` field: `complete`, `draft`, or `stub`
6. **No orphan notes** — every note must be linked from at least the Course Overview MOC

---

## 8. Quality Bar

A note is `complete` when:
- [ ] Someone reading **only this note** (plus linked concept notes) could answer exam questions on the topic
- [ ] All formulas are in LaTeX and explained
- [ ] All algorithms have full pseudocode
- [ ] All images from source material are reproduced or described
- [ ] All exercises/solutions are worked through with reasoning
- [ ] Wikilinks connect to all relevant concepts
- [ ] The note has correct frontmatter and tags

A note is `draft` when it covers the main content but is missing some of the above.
A note is `stub` when it's a placeholder with minimal content.

---

## 9. File & Folder Conventions

```
university-vault/
├── Courses/
│   ├── <COURSE>/
│   │   ├── <COURSE> - Overview.md
│   │   ├── Lectures/
│   │   ├── Book Notes/
│   │   ├── Exercises/
│   │   ├── Coding Assignments/
│   │   └── Exam Prep/
│   └── Archive/                  ← past/completed courses
├── Concepts/                     ← flat, course-agnostic
├── Templates/                    ← note templates
├── Assets/                       ← images, diagrams
└── VAULT-INSTRUCTIONS.md         ← this file
```

Naming conventions:
- Lectures: `RL-L01 - Intro, MDPs & Bandits.md`
- Book chapters: `RL-Book Ch5 - Monte Carlo Methods.md`
- Exercises: `RL-ES03 - Exercise Set Week 3.md`
- Homeworks: `RL-HW01 - Homework 1.md`
- Coding assignments: `RL-CA01 - Dynamic Programming.md`
- Concepts: `Monte Carlo Methods.md` (plain name, no course prefix)
