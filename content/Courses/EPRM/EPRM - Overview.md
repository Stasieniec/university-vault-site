---
type: moc
course: 5204EPRM6Y
tags: [moc]
status: active
canvas_id: 59626
canvas_url: https://canvas.uva.nl/courses/59626
---

# Engineering Production-Ready ML/AI Systems - Overview

> **Course:** Engineering Production-Ready ML/AI Systems (5204EPRM6Y), 2026/27 Semester 1, Period 2
> **Programme:** MSc Artificial Intelligence, UvA
> **Credits:** 6 EC
> **Lecturer and coordinator:** Kemal T. Yesilbek (k.t.yesilbek@uva.nl)
> **Course director:** Prof. Evangelos Kanoulas
> **Lectures:** Wednesdays 15:00-17:45, SP H0.08
> **Labs:** Thursdays 09:00-10:45, SP L1.01
> **Exam:** none. Graded on weekly assignments, a course project and a quiz on the guest lectures.

## What the course is about

Building live systems backed by ML or AI. It needs data science and ML knowledge but teaches ML/AI engineering: the skills that make research code reliable and large experiments manageable, and the skills an ML engineer, data engineer or data scientist needs in industry. The course code lives in the lecturer's repository `kemalty/uva-mlprod-code`, branch `v4`.

**No (Gen)AI use is allowed** on the course work. The reasoning given: build the skills first, AI can assist later. TAs will ask you about your work. (This is the opposite of the Agentic AI course, where AI use is allowed.)

## Assessment

Weights are not published. Each Canvas item is worth 10 points. Assignments unlock on Wednesday 12:00 of their week.

| Component | Due (Amsterdam time) | Notes |
|---|---|---|
| Assignment 0: setting up your environment | no deadline, 0 points | WSL (on Windows), Python, venv, run `main.py` |
| Assignment 1: data-driven roles, ML lifecycle, coding and unit testing | Tue 3 Nov 23:59 | Hints from the week 1 lecture: Uber's structure, "a new feature request" |
| Assignment 2: distributed data processing | Tue 10 Nov 23:59 | Unlocks Wed 4 Nov 12:00 |
| Assignment 4: deployment | Tue 24 Nov 23:59 | Unlocks Wed 18 Nov 12:00 |
| Assignment 3: containerization | Fri 27 Nov 23:59 | Unlocks Wed 11 Nov 12:00. Due after Assignment 4, as Canvas has it |
| Assignment 5: telemetry | Tue 1 Dec 23:59 | Unlocks Wed 25 Nov 12:00 |
| Quiz: guest lectures | Thu 10 Dec 23:59 | Graded quiz on the two guest lectures |
| Course project | Tue 15 Dec 06:00 | Presented in week 8 |

**Course project.** A partially free project: build an end-to-end production-ready ML/AI system for your own idea, using the concepts taught in the course. Start it after week 4. Presentations are in exam week (week 8).

**Guest lectures** (attendance required, dates not given yet):
- Basak Tugce Eskili, Senior ML Engineer at Booking.com: "From AI Demo to Production"
- Mohamed Osama (Oss) el Nagdy, Engineering Manager at Uber: "AI Security"

**Setup.** macOS, Linux (Ubuntu) or WSL. Tools in rough order of importance: Python, Git, the command line, Python venv, Make; later (Py)Spark, Docker, Kubernetes or Airflow, Prometheus. Setup instructions are in the course repository under `w1-setup`. Google Cloud vouchers of $50 per person are available for 70 students.

---

## Weekly Schedule

| Week | Topic | Assignment |
|---|---|---|
| 1 (28 Oct) | Data-driven roles, ML lifecycle, clean code, unit testing | A1, due Tue 3 Nov |
| 2 | Distributed data processing (Spark, Parquet) | A2, due Tue 10 Nov |
| 3 | Containerization and software engineering processes (containers, MLflow, testing, GitFlow) | A3, due Fri 27 Nov |
| 4 | Deployment patterns, APIs | A4, due Tue 24 Nov |
| 5 | Telemetry and CI/CD | A5, due Tue 1 Dec |
| 8 | Course project presentations | Project, due Tue 15 Dec 06:00 |

### Week 1: Roles, lifecycle, clean code, unit testing

| | Topic | Notes |
|---|---|---|
| Lecture 1 (Wed 28 Oct) | Data-driven roles and teams, the MLOps lifecycle and Dev/Acc/Prod, clean code, unit testing | [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]] |
| Readings | Seiden, *Outcomes Over Output*; MIT Sloan Management Review, "Achieving Return on AI Projects"; Kästner, "Process and Teams" in *Machine Learning in Production*; Martin, *Clean Code* | Kästner's part is three chapters: [[EPRM-Book Ch20 - Data Science and Software Engineering Process Models]], [[EPRM-Book Ch21 - Interdisciplinary Teams]], [[EPRM-Book Ch22 - Technical Debt]]. The others are covered inside the lecture note |

---

## Status

| Note | Status |
|---|---|
| [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]] | complete |
| [[EPRM-Book Ch20 - Data Science and Software Engineering Process Models]] | complete |
| [[EPRM-Book Ch21 - Interdisciplinary Teams]] | complete |
| [[EPRM-Book Ch22 - Technical Debt]] | complete |

---

## Concept Index

[[MLOps]] · [[Clean Code]] · [[Unit Testing]]
