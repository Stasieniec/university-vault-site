---
type: moc
course: 5204MNLP6Y
tags: [moc]
status: active
canvas_id: 59934
canvas_url: https://canvas.uva.nl/courses/59934
---

# Multilingual Natural Language Processing — Overview

> **Course:** Multilingual Natural Language Processing (5204MNLP6Y) — 2026/27 Sem. 1, Period 1
> **Programme:** MSc Artificial Intelligence, UvA
> **Credits:** 6 EC
> **Instructor:** Christof Monz (Informatics Institute)
> **Time zone:** Europe/Amsterdam

## Prerequisites

- Basic math: linear algebra, calculus, probability
- Basic machine learning: train/dev/test split, linear regression, classification
- Basic implementation experience

## Course Content

Provides an overview of NLP problems where multilinguality plays a role:
- **Multilingual scenarios:** NER for multiple languages, language-independent parsing, cross-lingual classification
- **Crosslingual scenarios:** Machine translation, crosslingual QA, crosslingual unlearning, crosslingual reasoning

**This course is NOT:** a comprehensive intro to ML or NLP — it focuses specifically on multilingual/crosslingual aspects.

## Assessment

| Component | Weight | Deadline | Notes |
|-----------|--------|----------|-------|
| Mini Project | TBD | Report & code: **2026-10-09 12:00**; Presentations: Week 7 | Teams of ~5; pick by **2026-09-04 17:00** |
| Exam | TBD | TBD | |

## Notes

- **Monz** always requires pass on both assignments and the exam. Realistically, you'll need at least a 5.5 on each.
- The weekly schedule is structure-light; most of the work is project-driven with lectures as supplements.

---

## Weekly Schedule

### Week 1: Overview and Multilinguality
Canvas module "Week 1" holds two decks.

| | Topic | Lecturer | Notes |
|---|-------|----------|-------|
| L1 | Introduction to NLP, applications, course admin | Monz | [[MNLP-L01 - Overview]] |
| L2 | Multilinguality and writing systems | Monz | [[MNLP-L02 - Multilinguality and Writing Systems]] |
| **Key date** | Team formation and problem selection by **Sep 4** | | |

### Week 2: Morphology and Segmentation
Canvas module "Week 2" holds two decks plus the Zoom recording of 9 September, which was moved online because of the public transport strike.

| | Topic | Lecturer | Notes |
|---|-------|----------|-------|
| L3 | Morphology and word formation | Monz | [[MNLP-L03 - Morphology and Word Formation]] |
| L4 | Subword segmentation | Monz | [[MNLP-L04 - Subword Segmentation]] |

> [!tip] Read L4 before choosing a mini-project
> Subword segmentation is where the tokenizer decisions live, and every mini-project makes them whether or not it thinks about them. L4 also flags an error in the lecturer's Algorithm 2, which prunes the highest-value tokens as printed.

### Week 3: First Model
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| **Project** | Implement first model, evaluate, debug | | | |

### Week 4: Refinement
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| **Project** | Refine or try alternative model; dropout, layernorm, residual connections | | | |

### Week 5: Error Analysis
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| **Project** | Second model refinement, error analysis, conclusions | | | |

### Week 6: Finalize
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| **Project** | Loose ends, 4-page report, presentation slides | | | |

### Week 7: Presentations
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| **Project** | Team presentations (10-15 min + 5 min Q&A) | | | |

---

## Key Deadlines

| Date | What |
|------|------|
| **2026-09-04 17:00** | Submit team + problem choice (Google Sheet) |
| **2026-10-09 12:00** | Submit report (4 pp PDF), slides, code (GitHub link) |
| Week 7 | Presentations |

## Mini Project: What Makes a Good One

- Good succinct description of most relevant research papers
- Good description of data preprocessing/settings
- Good motivation of neural architecture choices
- Thorough evaluation under different settings/architectures; error analysis
- Report focusing on most relevant findings: what works and what doesn't, why
- [[MNLP - Mini Project]] — full project description, schedule, deliverables, and evaluation criteria

## Resources

- Mini Project slides (PDF in Assets/)
- Canvas modules: Overview, Mini Project

---

- Mini Project slides (PDF in Assets/)
- Canvas modules: Overview, Mini Project