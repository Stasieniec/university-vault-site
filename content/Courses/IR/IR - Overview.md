---
type: moc
course: IR
tags: [moc]
---

# Information Retrieval 1 — Overview

> **Course:** Information Retrieval 1 (52041INR6Y) — 2025/26 Sem. 2, Period 4
> **Programme:** MSc Artificial Intelligence, UvA
> **Instructors:** Dr. Evangelos Kanoulas, Dr. Maria Heuss, Dr. Maarten de Rijke, Dr. Bhaskar Mitra, Philipp Hager
> **Exam:** Written digital exam (ANS), date TBD Week 7+
> **Cheat sheet:** A4, one-side only, handwritten

## Textbooks

- **SEIRiP:** *Search Engines: Information Retrieval in Practice* — Croft, Metzler, Strohman ([online](https://ciir.cs.umass.edu/downloads/SEIRiP.pdf))
- **IIR:** *Introduction to Information Retrieval* — Manning, Raghavan, Schütze ([online](https://nlp.stanford.edu/IR-book/))
- **PTR:** *Pretrained Transformers for Text Ranking: BERT and Beyond* — Lin, Nogueira, Yates ([online](https://arxiv.org/abs/2010.06467))
- Additional papers referenced in lecture slides

## Assessment

| Component | Weight | Notes |
|-----------|--------|-------|
| Assignment 0 (Warmup) | PASS/FAIL | Individual, workflow familiarization |
| Assignment 1 (Unsupervised Retrieval) | PASS/FAIL | Pairs, ≥80% automated tests |
| Assignment 2 (Neural Retrieval) | PASS/FAIL | Pairs, due March 5 |
| Assignment 3 | PASS/FAIL | Pairs |
| Paper Presentation | 30% | Pairs, rubric provided |
| Final Exam | 70% | Must score ≥ 5.5 |

Must pass all assignments + ≥5.5 on presentation + ≥5.5 on exam.

---

## Weekly Schedule

### Week 1 — Foundations of IR
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| L1.1 | Administration & Course Intro | Kanoulas | | [[IR-L01 - Introduction]] |
| L1.2 | Introduction to IR | Kanoulas | Lin et al. 1, 2-2.7, 3-3.1 | [[IR-L02 - IR Fundamentals]] |
| **Reading** | | | SEIRiP 2.3, 4.1-4.3, 5.3, 5.6-5.7, 6.2, 7, 8 | |
| **Book** | | | | [[IR-PTR Ch1 - Introduction]], [[IR-PTR Ch2 - Setting the Stage]], [[IR-SEIRiP Ch4 - Processing Text]], [[IR-SEIRiP Ch5 - Ranking with Indexes]], [[IR-SEIRiP Ch7 - Retrieval Models]] |
| **Assignment** | A0: Warmup | | | [[IR-A00 - Warmup]] |

### Week 2 — Retrieval Models & Evaluation
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| L2 | Term-Based Ranking (BM25, QL, etc.) | Kanoulas | | [[IR-L03 - Retrieval Models]] |
| L3 | IR Evaluation | Kanoulas | | [[IR-L04 - Evaluation]] |
| **Assignment** | A1: Unsupervised Retrieval | | Due Feb 17 | [[IR-A01 - Unsupervised Retrieval]] |

### Week 3 — Neural IR
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| L4 | Neural IR: Intro & Reranking | Heuss | Lin et al. 2+3 | [[IR-L05 - Neural IR Intro & Reranking]] |
| L5 | Dense Retrieval | Heuss | Lin et al. 4+5 | [[IR-L06 - Dense Retrieval]] |
| **Reading** | | | Dense Text Retrieval Survey | |
| **Book** | | | | [[IR-PTR Ch3 - Multi-Stage Architectures for Reranking]], [[IR-PTR Ch4 - Refining Query and Document Representations]], [[IR-PTR Ch5 - Dense Retrieval and Learned Sparse Retrieval]] |

### Week 4 — Advanced Neural IR *(current)*
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| L6 | Learned Sparse Retrieval | Heuss | LSR Tutorial, Unified Framework paper | [[IR-L07 - Learned Sparse Retrieval]] |
| L7 | Generative Retrieval (DSI, etc.) | de Rijke | | [[IR-L08 - Generative Retrieval]] |
| L8 | RAG | Heuss | | [[IR-L09 - RAG]] |
| **Assignment** | A2: Neural Retrieval | | Due March 5 | |

### Week 5 — Learning to Rank
| | Topic | Lecturer | Readings | Notes |
|---|-------|----------|----------|-------|
| L9 | Offline LTR | Hager | LTR for IR 1.2-1.3, 2-2.2.1, 2.4.2, 3, 4.2 | [[IR-L10 - Learning to Rank]] |
| L10 | LTR from Interactions | Hager | Unbiased LTR paper | [[IR-L11 - Unbiased Learning to Rank]] |

### Week 6 — Responsible IR
| | Topic | Lecturer | Notes |
|---|-------|----------|-------|
| L11 | Fairness & Biases in IR | Heuss | |
| L12 | Explainable IR / IR & Society | Heuss / Mitra | |

### Week 7 — Conversational & Wrap-Up
| | Topic | Lecturer | Notes |
|---|-------|----------|-------|
| L13 | Conversational Search & Search R1 | Kanoulas | |
| L14 | Wrap-Up, Q&A, Sample Exam | Kanoulas | |

---

## Concept Index

**Foundations:** [[Information Retrieval]] · [[Inverted Index]] · [[Tokenization]] · [[Stemming]] · [[Stop Words]] · [[Bag of Words]]

**Retrieval Models:** [[TF-IDF]] · [[BM25]] · [[Query Likelihood Model]] · [[Language Model for IR]] · [[Vector Space Model]]

**Evaluation:** [[Precision]] · [[Recall]] · [[F-Measure]] · [[MAP]] · [[NDCG]] · [[MRR]] · [[Precision at K]]

**Neural IR:** [[Neural Reranking]] · [[Cross-Encoder]] · [[Bi-Encoder]] · [[Dense Retrieval]] · [[Learned Sparse Retrieval]] · [[BERT for IR]] · [[ColBERT]] · [[SPLADE]]

**Generative & RAG:** [[Generative Retrieval]] · [[Differentiable Search Index]] · [[Retrieval-Augmented Generation]]

**Learning to Rank:** [[Learning to Rank]] · [[Pointwise LTR]] · [[Pairwise LTR]] · [[Listwise LTR]] · [[Click Models]] · [[Position Bias]] · [[Inverse Propensity Weighting]] · [[Counterfactual Learning to Rank]] · [[Examination Hypothesis]] · [[Doubly Robust Estimation]] · [[Trust Bias]] · [[Cascading Position Bias]] · [[Item Selection Bias]] · [[Outlier Bias]]
