---
type: lecture
course: 5204MNLP6Y
week: 1
lecture: 1
date: 2026-09-01
status: complete
topics:
  - Introduction to NLP
  - NLP applications
  - Course administrativa
  - Prerequisites
  - Multilingual vs crosslingual
  - Impact of deep learning on NLP
  - Large language models and translation
---

# MNLP-L01 — Overview

## Motivation

This lecture introduces the landscape of natural language processing (NLP): what it is, why it matters, and where multilinguality fits. Christof Monz frames NLP as the intersection of computer science, artificial intelligence, and linguistics, working to algorithmically model word formation, sentence structure, meaning, and discourse.

## Content

### 1. Information Types

Information can be conveyed/stored in several forms:

- **Structured:** tables, databases
- **Unstructured:** language, images/videos
- **Continuous signals:** spoken language/audio, images/video
- **Discrete signals:** written language

Human language is the medium of choice to convey complex information. A limited repertoire of words allows infinite expressivity — but after decades of NLP research, formal modelling has proven surprisingly hard.

### 2. What is NLP?

NLP sits at the intersection of:

- Computer science
- Artificial intelligence
- Linguistics

Its goal is to algorithmically and formally model aspects of human language:

1. Word formation (morphology)
2. Sentence structure (syntax/grammar)
3. Sentence meaning (semantics)
4. Document/discourse structure

### 3. Core NLP Tasks

| Task | Description |
|------|-------------|
| Text categorization | Assign documents to categories |
| Document summarization | Extract or generate condensed versions |
| Machine translation | Translate between languages |
| Question answering | Return actual answers, not ranked documents |
| Named entity recognition | Identify persons, organizations, dates, locations |
| Sentiment analysis | Estimate attitude in reviews (positive/negative, fine-grained) |

**Example — NER:** *"President [Biden]PER has received the French prime minister [Macron]PER"*

**Example — QA vs IR:** Information retrieval returns ranked documents; QA returns actual answers (e.g. "1962" for "When was the Cuba Crisis?").

### 4. Industrial Relevance

Companies investing in NLP technology:

- **Search/Cloud:** Google, Microsoft, Baidu, IBM, Huawei
- **E-commerce:** Amazon, Bol, Alibaba, Booking
- **Information:** Bloomberg, Reuters, Elsevier

Applications include web ad matching, user review analysis, speech recognition/synthesis, web page translation, dialog agents.

### 5. Pre-Deep-Learning NLP

Traditional approach: different application $\to$ different methodology.

- **ML methods:** SVMs, decision trees, generative Bayesian models, discriminative max-ent models
- **Features:** POS tags, morphology, parse trees, named entities, taxonomies, argument roles

ML was used to weigh the importance of individual features for prediction.

### 6. Impact of Deep Learning

Over the last few years, neural networks achieved state-of-the-art performance across all NLP tasks.

**Advantages:**

- Strong performance
- Very little or no feature engineering required
- A limited repertoire of neural network types applies to most/all NLP tasks

**Disadvantages:**

- Requires large amounts of training data
- Difficult to trace errors
- Can fail spectacularly at times

The success extends beyond NLP to CV, handwriting recognition, speech, robotics, and IR. Many insights transfer between areas due to uniform NN types and limited application-specific features.

### 7. The Role of LLMs

LLMs originated within NLP and now dominate the research landscape.

**Uses:**

- End-to-end NLP: QA, summarization, MT
- Substituting/supplementing humans: data annotation, evaluation (LLM-as-judge), dialog
- BUT still fragile on non-English text

**MT example** (Arabic $\to$ English, by Monz):

| Model | Output |
|-------|--------|
| GPT-5.6 Terra | "The embassy in South Sudan called for an investigation into the attack in which Ethiopian peacekeeping soldiers were killed." |
| Gemini 3.6 Flash | "The embassy in South Sudan has requested an investigation into the attack in which Ethiopian peacekeepers were killed." |
| Claude Sonnet 4.6 | "The embassy located in South Sudan requested that the attack in which Ethiopian peacekeeping soldiers were killed be investigated." |
| Mistral Small 3.2 | "The Ethiopian embassy in Khartoum, where the Ethiopian peacekeepers were detained, is closed." |

Mistral Small catastrophically hallucinated the embassy's location.

### 8. Multilingual vs Crosslingual

**Multilingual scenarios:**

- NER for multiple languages — train without being language-specific?
- Language-independent parsing — identify universal features?
- Cross-language classification — train without language-specific data?

**Crosslingual scenarios:**

- Machine translation — make information understandable across languages
- Crosslingual QA — extract information from resources in other languages
- Crosslingual unlearning — manipulate information across languages
- Crosslingual reasoning — combine information across languages

**LLMs and multilinguality:**

- LLMs are predominantly trained on Internet resources $\to$ English-centric
- Multilingual models (Llama 3.1, DeepSeek) perform better on high-resource languages
- Dedicated multilingual models (Aya-Expanse) tend to lag behind

## Key Takeaways

1. NLP is the algorithmic modelling of language structure and meaning
2. Deep learning removed most feature engineering but introduced new challenges (data hunger, opacity)
3. LLMs dominate NLP but are English-centric; multilinguality remains an open problem
4. Multilingual $\neq$ crosslingual: one is about building systems for multiple languages, the other about transferring knowledge across languages

## Related Concepts

- [[Information Retrieval]]
- Natural Language Processing
- Multilingual Models
- Machine Translation