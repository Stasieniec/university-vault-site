---
type: lecture
course: 5204MNLP6Y
topics:
  - "Multilingual NLP"
  - "Mini Project"
status: complete
---

# MNLP Mini Project

> [!note] Source
> Based on Christof Monz's Mini Project slides (PDF in Assets/). 9-slide Beamer presentation covering scope, timeline, deliverables, and quality criteria.

---

## Overview

The MNLP mini project is a **six-week group project** (Weeks 2-7 of Period 1) where teams of ~5 students work on **one concrete multilingual NLP problem**. It's the main practical component of the course alongside the exam.

Each team picks a problem (from the Canvas list or self-designed), implements neural approaches, compares architectures, does error analysis, and presents findings in a 4-page report + 10-15 minute presentation.

---

## Schedule & Deadlines

| Date | What |
|------|------|
| **Fri 2026-09-04, 17:00** | Submit team + problem choice via Google Sheet (link on Canvas) |
| Week 2-6 | Project work (see weekly breakdown below) |
| **Fri 2026-10-09, 12:00** | Submit report, slides, and code |
| Week 7 | Team presentations (10-15 min + 5 min Q&A) |

### Weekly Breakdown

> [!info] Week 1 (Now)
> - Choose your team (5 students)
> - Pick a problem from the Canvas list, or design your own
> - If self-designed: email Monz a 1-page description

> [!info] Week 2 — Literature & Planning
> - Identify and read relevant previous literature
> - Inspect data, pre-processing, split into train/val/test
> - Additional data crawling if needed
> - Decide on an initial (baseline) model

> [!info] Week 3 — First Model
> - Implement the first model
> - Evaluate
> - Debug
> - Document: what works, what doesn't?

> [!info] Week 4 — Refinement
> - Refine first model or try an alternative model
> - Bells and whistles: dropout, layer norms, more layers, residual connections, highway connections, etc.
> - Evaluate and compare
> - Document: what works, what doesn't?

> [!info] Week 5 — Error Analysis
> - Refine second model / try another alternative model
> - Error analysis, open problems
> - Conclusions, recommendations

> [!info] Week 6 — Wrap Up
> - Address any loose ends
> - Write the 4-page report with main findings
> - Prepare presentation slides

> [!info] Week 7 — Presentations
> - Presentation slots assigned at beginning of Week 6
> - 10-15 minutes presentation + 5 minutes for questions

---

## Project Requirements

### Team Formation
- Teams of **5 students**
- Fill out the Google Sheet by **Sep 4, 17:00** with:
  - All team members (full names, student IDs, emails)
  - The problem you choose to work on
  - Meeting time slot preferences (no guarantees)
- If designing your own problem: email Monz a 1-page description

### Problem Selection
- Canvas course page lists example problems with links to relevant datasets
- You can also choose your own problem (subject to approval via 1-page description)

### Deliverables (due Oct 9, 12:00 noon)

| Deliverable | Details |
|-------------|---------|
| **Report** | ~4 pages, PDF format; focus on most relevant findings |
| **Slides** | PDF or PowerPoint |
| **Code** | GitHub repo link (preferred) or gzipped tarball; must include `readme` explaining how to run |
| **Data** | If gzipped data < 10 MB: include it; otherwise provide a link |

### Evaluation Criteria

See the "What Makes a Good Project" section below. The evaluation focuses on:
- Quality of literature review and motivation
- Soundness of preprocessing and architecture choices
- Thoroughness of evaluation (multiple settings/architectures + error analysis)
- Clarity of report and presentation

---

## Practical Advice

> [!warning] Keep Data Small
> Most training will run on your laptop. If data is too big:
> - Down-sample
> - Select based on criteria: vocabulary, classes, length, etc.

---

## What Makes a Good Mini Project

Monz's slides list six quality dimensions:

1. **Literature Review** — good, succinct description of the most relevant research papers
2. **Data Description** — good description of preprocessing steps and experimental settings
3. **Architecture Motivation** — good motivation for why you chose the neural architectures you did
4. **Thorough Evaluation** — scores under **different settings/architectures**, plus **error analysis**
5. **Focused Report** — the report should center on the most relevant findings: **what works and what doesn't, and why**
6. **Strong Presentation** — focus on the most important aspects and findings, understandable by a wider audience, and **include examples**

---

## Example: How to "Pimp Up" a Project (Language Identification)

Monz provides a detailed walkthrough using **language identification** as a case study, showing how to elevate a seemingly simple problem into a thorough project:

### Literature & Input
- What has been tried? (neural and non-neural approaches)
- Process input: words, characters, or character $n$-grams?

### Models
1. **Baseline** (Week 2): word embeddings — how to combine embeddings; additional hidden layers?
2. **Better model** (Week 3): CNN sequence classifier — how to pool; strides, etc.
3. **Even better model** (Week 3/4): Transformer sequence classifier

### Evaluation Dimensions
- Standard accuracies (check literature for benchmarks)
- Short input problem (literature mentions this)
- Languages close to each other (e.g., closely related language families)
- Code-switching cases
- How accuracy changes after $n$ characters (useful for instant translation applications)
- Best overall metrics: accuracy, precision, recall, F1, etc.

### Data & Features
- Crawl/use more data
- Script information: can UTF-8 region information be used in a neural network?

### Presentation of Findings
- Score tables
- Visualizations
- Clusters
- Error analysis: what really makes a difference?
- Anything that contradicts previous literature? Is it a bug or a genuine finding?

---

## Key Takeaways

> [!tip] The Core Loop
> Literature $\rightarrow$ Baseline $\rightarrow$ Refine $\rightarrow$ Evaluate $\rightarrow$ Analyze Errors $\rightarrow$ Conclude

- **6 weeks** of work, culminating in a **4-page report**, **slides**, and **code** due **Oct 9 at noon**
- **Teams of 5** — form your team and pick a problem by **Sep 4, 17:00**
- The bar isn't just "does it work" — it's **"why does (or doesn't) it work?"** with thorough error analysis
- Keep data manageable for laptop training
- Presentations in Week 7 (10-15 min + 5 min Q&A)