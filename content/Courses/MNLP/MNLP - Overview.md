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

### Week 3 (14–18 Sep): First Model
Lectures continue alongside the project work.

| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| L5 | **Mon 14 Sep, 13:00–14:45, SP L1.02** | Monz | Given. **No note yet**, topic not recorded, see **Source material not yet processed** below |
| L6 | **Wed 16 Sep, 09:00–10:45, SP C0.110** | Monz | Given. **No note yet**, topic not recorded, see **Source material not yet processed** below |
| Lab | Wed 16 Sep, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Project** | Implement first model, evaluate, debug | | |

### Week 4 (21–25 Sep): Refinement
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| L7 | Mon 21 Sep, 13:00–14:45, SP L1.02 | Monz | Not yet given |
| L8 | Wed 23 Sep, 09:00–10:45, **SP H0.08** | Monz | Not yet given. Room differs from the usual C0.110 |
| Lab | Wed 23 Sep, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Project** | Refine or try alternative model; dropout, layernorm, residual connections | | |

### Week 5 (28 Sep–2 Oct): Error Analysis
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| L9 | Mon 28 Sep, 13:00–14:45, SP L1.02 | Monz | Not yet given |
| L10 | Wed 30 Sep, 09:00–10:45, SP C0.110 | Monz | Not yet given |
| Lab | Wed 30 Sep, 11:00–12:45, SP B0.208 (Group 4) | | |
| **Project** | Second model refinement, error analysis, conclusions | | |

### Week 6 (5–9 Oct): Finalize
| | Slot | Lecturer | Notes |
|---|------|----------|-------|
| L11 | Mon 5 Oct, 13:00–14:45, SP L1.02 | Monz | Not yet given |
| L12 | Wed 7 Oct, 09:00–10:45, SP C0.110 | Monz | Not yet given |
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

## Source material not yet processed

> [!warning] Fetch these from Canvas on the laptop, in one pass
> The MNLP Canvas **Files tab returns HTTP 403**, and Canvas is unreachable from cloud sessions (egress-blocked), so these decks could not be downloaded or read. Their topics are deliberately left blank rather than guessed.

| Item | Slot | Status |
|---|---|---|
| **L05** | Mon 2026-09-14, 13:00–14:45, SP L1.02 | Lecture given. No slides in the vault, no note, topic unknown |
| **L06** | Wed 2026-09-16, 09:00–10:45, SP C0.110 | Lecture given. No slides in the vault, no note, topic unknown |

Both need the deck pulled from Canvas Modules (the Files tab is blocked) before any note is written.

## Resources

- Mini Project slides (PDF in Assets/)
- Canvas modules: Overview, Mini Project