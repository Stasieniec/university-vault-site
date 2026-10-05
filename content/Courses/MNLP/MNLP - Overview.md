---
type: moc
course: 5204MNLP6Y
tags: [moc]
status: active
canvas_id: 59934
canvas_url: https://canvas.uva.nl/courses/59934
exam_date: 2026-10-22
---

# Multilingual Natural Language Processing — Overview

> **Course:** Multilingual Natural Language Processing (5204MNLP6Y) — 2026/27 Sem. 1, Period 1
> **Programme:** MSc Artificial Intelligence, UvA
> **Credits:** 6 EC
> **Instructor:** Christof Monz (Informatics Institute)
> **Time zone:** Europe/Amsterdam
> **Lectures:** Mondays 13:00–14:45 (SP L1.02) and Wednesdays 09:00–10:45 (SP C0.110, room varies), twice weekly through 2026-10-14
> **Seminar / lab:** Wednesdays 11:00–12:45, SP B0.208, **Group 4**
> **Exam:** Thursday 2026-10-22, 09:00–11:00, **Roeterseiland (REC), not Science Park**

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
| Mini Project | TBD | Report, code and slides: **Fri 2026-10-09 12:00**; presentation in week 7 | Teams of ~5; picked by **2026-09-04 17:00** |
| Exam | TBD | **Thu 2026-10-22, 09:00–11:00** | Rooms: REC B3.05-B3.06-B3.07 / REC B3.08-B3.09 |

> [!warning] The exam is at Roeterseiland, not Science Park
> Thursday 2026-10-22, 09:00–11:00, in REC B3.05-B3.06-B3.07 / REC B3.08-B3.09. That is the **Roeterseiland** campus. Every other MNLP session this period is at Science Park, so do not run the usual commute on autopilot.

> [!note] The mini-project deadline exists only in the project description
> MNLP publishes **zero** assignments through the Canvas API, so Fri 2026-10-09 12:00 will never appear in a Canvas to-do list, calendar feed or reminder. It lives in [[MNLP - Mini Project]] and here, and nowhere else.

## Notes

- **Monz** always requires pass on both assignments and the exam. Realistically, you'll need at least a 5.5 on each.
- Lectures run **twice a week through 14 October**, in parallel with the project. The project is not a substitute for attending: the exam is examined off the lectures.

---

## Weekly Schedule

> [!info] Standing times and rooms, confirmed from MyTimetable, synced 2026-09-16
> - **Lecture A:** Mondays **13:00–14:45**, **SP L1.02**
> - **Lecture B:** Wednesdays **09:00–10:45**, **SP C0.110**, except **SP H0.08 on 23 Sep** and **SP L1.01 on 14 Oct**
> - **Seminar / lab:** Wednesdays **11:00–12:45**, **SP B0.208**, **Group 4**
> - Teaching runs through **Wed 2026-10-14**. Nothing is timetabled after that except the exam.
> - **Exam:** Thursday **2026-10-22, 09:00–11:00**, REC B3.05-B3.06-B3.07 / REC B3.08-B3.09 (Roeterseiland)

### Week 1 (31 Aug–4 Sep): Overview and Multilinguality
Canvas module "Week 1" holds two decks.

| | Topic | Lecturer | Notes |
|---|-------|----------|-------|
| L1 | Introduction to NLP, applications, course admin | Monz | [[MNLP-L01 - Overview]] |
| L2 | Multilinguality and writing systems | Monz | [[MNLP-L02 - Multilinguality and Writing Systems]] |
| **Key date** | Team formation and problem selection by **Sep 4** | | |

### Week 2 (7–11 Sep): Morphology and Segmentation
Canvas module "Week 2" holds two decks plus the Zoom recording of 9 September, which was moved online because of the public transport strike.

| | Topic | Lecturer | Notes |
|---|-------|----------|-------|
| L3 | Morphology and word formation | Monz | [[MNLP-L03 - Morphology and Word Formation]] |
| L4 | Subword segmentation | Monz | [[MNLP-L04 - Subword Segmentation]] |

> [!tip] Read L4 before choosing a mini-project
> Subword segmentation is where the tokenizer decisions live, and every mini-project makes them whether or not it thinks about them. L4 also flags an error in the lecturer's Algorithm 2, which prunes the highest-value tokens as printed.

> [!info] From week 3, one deck runs across two lectures
> Canvas Modules (read 2026-10-05) shows Monz teaching one deck per week and finishing it at the start of the next: week 4 opens with "continuation of static embeddings", week 5 with "continuation of contextual embeddings", week 6 with "continuation of crosslingual NLP". So notes are numbered by deck, and the exact slide where each lecture stopped is not recorded.

### Week 3 (14–18 Sep): First Model
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| Mon | **Mon 14 Sep, 13:00–14:45, SP L1.02** | Monz | [[MNLP-L05 - Static Embeddings]] |
| Wed | **Wed 16 Sep, 09:00–10:45, SP C0.110** | Monz | [[MNLP-L05 - Static Embeddings]] |
| Lab | Wed 16 Sep, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Project** | Implement first model, evaluate, debug | | |

### Week 4 (21–25 Sep): Refinement
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| Mon | Mon 21 Sep, 13:00–14:45, SP L1.02 | Monz | End of [[MNLP-L05 - Static Embeddings]], then [[MNLP-L06 - Contextual Embeddings]] |
| Wed | Wed 23 Sep, 09:00–10:45, **SP H0.08** | Monz | [[MNLP-L06 - Contextual Embeddings]] |
| Lab | Wed 23 Sep, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Project** | Refine or try alternative model; dropout, layernorm, residual connections | | |

### Week 5 (28 Sep–2 Oct): Error Analysis
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| Mon | Mon 28 Sep, 13:00–14:45, SP L1.02 | Monz | End of [[MNLP-L06 - Contextual Embeddings]], then [[MNLP-L07 - Crosslingual NLP]] |
| Wed | Wed 30 Sep, 09:00–10:45, SP C0.110 | Monz | [[MNLP-L07 - Crosslingual NLP]] |
| Lab | Wed 30 Sep, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Project** | Second model refinement, error analysis, conclusions | | |

### Week 6 (5–9 Oct): Finalize
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| Mon | Mon 5 Oct, 13:00–14:45, SP L1.02 | Monz | Continuation of [[MNLP-L07 - Crosslingual NLP]] per the module. Note written from the deck as of 5 Oct; re-check after Wednesday in case slides are added |
| Wed | Wed 7 Oct, 09:00–10:45, SP C0.110 | Monz | Not yet given |
| Lab | Wed 7 Oct, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Due** | **Fri 9 Oct 12:00**: 4-page report, code, slides | | |

### Week 7 (12–16 Oct): Presentations, last teaching week
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| L13 | Mon 12 Oct, 13:00–14:45, SP L1.02 | Monz | Last Monday slot |
| L14 | Wed 14 Oct, 09:00–10:45, **SP L1.01** | Monz | **Last session of the course.** Room differs from the usual C0.110 |
| Lab | Wed 14 Oct, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Project** | Team presentations (10–15 min + 5 min Q&A) | | See the inference below |

> [!question] Presentation date: narrowed, not confirmed
> Canvas says "week 7" and gives no date. MyTimetable has no MNLP session after **Wed 14 Oct** except the exam, so the team presentation has to land in one of the final teaching slots, **Mon 12 Oct** or **Wed 14 Oct**. This is an inference from the timetable, **not a confirmed date**. Ask Monz or check an announcement before booking anything around it.

### Week 8 (19–23 Oct): Exam
| | Slot | Notes |
|---|------|-------|
| **Exam** | **Thu 22 Oct, 09:00–11:00** | REC B3.05-B3.06-B3.07 / REC B3.08-B3.09, **Roeterseiland campus** |

---

## Key Deadlines

| Date | What |
|------|------|
| **2026-09-04 17:00** | Submit team + problem choice (Google Sheet) |
| **2026-10-09 12:00** | Submit report (4 pp PDF), slides, code (GitHub link) |
| **12 or 14 Oct** | Team presentation (inferred from the timetable, not confirmed) |
| **Wed 2026-10-14** | Last teaching session |
| **Thu 2026-10-22 09:00–11:00** | **Exam**, Roeterseiland (REC), not Science Park |

## Mini Project: What Makes a Good One

- Good succinct description of most relevant research papers
- Good description of data preprocessing/settings
- Good motivation of neural architecture choices
- Thorough evaluation under different settings/architectures; error analysis
- Report focusing on most relevant findings: what works and what doesn't, why
- [[MNLP - Mini Project]] — full project description, schedule, deliverables, and evaluation criteria

## Source material

The Files tab returns HTTP 403 for students, so decks come from **Modules**, through the `canvas_modules` connector tool (added 2026-10-05), which returns a download URL per file. Module contents as of 2026-10-05:

| Module | Item | Note |
|---|---|---|
| Mini Project | `mini-project.pdf`, Project Suggestions and Project Requirements pages, team and cluster-credit sheets | [[MNLP - Mini Project]] |
| Week 1 | `overview.pdf`, `multilingual-nlp.pdf` | [[MNLP-L01 - Overview]], [[MNLP-L02 - Multilinguality and Writing Systems]] |
| Week 2 | `morphology.pdf`, `subword.pdf`, Zoom recording of 9 Sep (`video1951888080.mp4`, 180 MB) | [[MNLP-L03 - Morphology and Word Formation]], [[MNLP-L04 - Subword Segmentation]] |
| Week 3 | `embeddings_static.pdf` (71 slides, 404 pages with animation builds) | [[MNLP-L05 - Static Embeddings]] |
| Week 4 | `embeddings_context.pdf` (50 slides, 305 pages) | [[MNLP-L06 - Contextual Embeddings]] |
| Week 5 | `cross-lingual-nlp.pdf` (45 slides, 188 pages) | [[MNLP-L07 - Crosslingual NLP]] |
| Week 6 | Only a "continuation of crosslingual NLP" header so far | |

The decks are 8 to 23 MB each and are not stored in `Assets/`, matching weeks 1 and 2.

## Resources

- Mini Project slides (PDF in Assets/)
- Canvas modules: Mini Project, Week 1 to Week 6 (see Source material)