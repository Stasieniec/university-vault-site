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

(Factual aside on the example sentence: Macron is the French president and has never been prime minister. The tagging point is unaffected. #needs-review: check whether the slide itself says "prime minister".)

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

## Exam questions

> [!exam]- Distinguish multilingual from crosslingual NLP. Give examples of each and explain why the English-centric nature of LLMs keeps both open problems.
> **Key points:** multilingual means systems for multiple languages; crosslingual means transferring information across languages; multilingual examples: NER, parsing, classification; crosslingual examples: MT, QA, unlearning, reasoning; LLMs English-centric and fragile on non-English text.
>
> - **Multilingual:** building systems that work for **multiple languages**. Examples: NER for multiple languages (can it be trained without being language-specific?), language-independent parsing (are there universal features?), cross-language classification (can it be trained without language-specific data?).
> - **Crosslingual:** **transferring information or knowledge across languages**. Examples: machine translation, crosslingual QA, crosslingual unlearning, crosslingual reasoning.
> - **LLMs:** trained predominantly on Internet resources, so they are **English-centric**. Multilingual models such as Llama 3.1 and DeepSeek do better on **high-resource languages**; dedicated multilingual models such as Aya-Expanse tend to **lag behind**. LLMs are still fragile on non-English text.
>
> Losing marks: treating the two terms as synonyms. One is about covering many languages, the other about moving information between them.

> [!exam]- Contrast pre-deep-learning NLP with deep-learning NLP. What did neural networks gain, what did they cost, and why do insights now transfer between fields?
> **Key points:** before: per-application methods weighing engineered features; gains: state of the art, little or no feature engineering, few network types; costs: data hunger, hard-to-trace errors, spectacular failures; transfer via uniform network types and few application-specific features.
>
> - **Before:** a different methodology per application. Classical ML (SVMs, decision trees, generative Bayesian models, discriminative max-ent models) **weighed hand-engineered features**: POS tags, morphology, parse trees, named entities, taxonomies, argument roles.
> - **Advantages of deep learning:** state-of-the-art performance across NLP tasks; very little or no feature engineering; a limited repertoire of network types covers most or all tasks.
> - **Disadvantages:** needs large amounts of training data; errors are hard to trace (opacity); can fail spectacularly.
> - **Transfer:** the same success appears in computer vision, handwriting recognition, speech, robotics and IR. Because network types are uniform and application-specific features are few, insights carry over between these areas.

> [!exam]- What roles do large language models now play in NLP, and what are their limits? Use the Arabic to English translation comparison as evidence.
> **Key points:** LLMs originated in and now dominate NLP; end-to-end QA, summarization, MT; substitute humans in annotation, LLM-as-judge evaluation, dialog; fragile on non-English text; Mistral Small 3.2 hallucinated the embassy's location.
>
> - LLMs originated within NLP and now dominate research.
> - **Uses:** end-to-end NLP (QA, summarization, MT); substituting or supplementing humans in data annotation, evaluation (**LLM-as-judge**) and dialog.
> - **Limit:** still fragile on non-English text, because training data is English-centric.
> - **Evidence (Monz's Arabic to English example):** GPT-5.6 Terra, Gemini 3.6 Flash and Claude Sonnet 4.6 all render the sentence as the embassy in South Sudan calling for an investigation into an attack that killed Ethiopian peacekeepers, differing only in wording. **Mistral Small 3.2** produced "The Ethiopian embassy in Khartoum, where the Ethiopian peacekeepers were detained, is closed": it **catastrophically hallucinated** the embassy's location, and its sentence no longer matches the other three.

## Flashcards

> [!card]- Which three fields does NLP sit at the intersection of?
> **Computer science**, **artificial intelligence**, **linguistics**.

> [!card]- Which four aspects of human language does NLP aim to model algorithmically and formally?
> Word formation (**morphology**), sentence structure (**syntax**), sentence meaning (**semantics**), document or **discourse** structure.

> [!card]- Is written language a continuous or a discrete signal?
> **Discrete**. Spoken language is a continuous signal.

> [!card]- How does question answering differ from information retrieval?
> IR returns **ranked documents**; QA returns the **actual answer**.

> [!card]- Which types of entity does named entity recognition identify?
> **Persons**, **organizations**, **dates**, **locations**.

> [!card]- Which NLP task estimates the attitude expressed in reviews, either positive/negative or fine-grained?
> **Sentiment analysis**.

> [!card]- What was machine learning used for in pre-deep-learning NLP?
> To **weigh the importance** of individual features (POS tags, parse trees, named entities) for prediction.

> [!card]- How does deep-learning NLP differ from pre-deep-learning NLP in feature engineering?
> Pre-deep-learning NLP relied on engineered features; deep learning needs **very little or none**.

> [!card]- What are the disadvantages of deep learning for NLP?
> Needs **large amounts of training data**, errors are **hard to trace**, can **fail spectacularly**.

> [!card]- Why do deep-learning insights transfer between NLP, vision, speech and other fields?
> Neural network types are **uniform** and there are few **application-specific features**.

> [!card]- What is the difference between multilingual and crosslingual NLP?
> Multilingual builds systems for **multiple languages**; crosslingual **transfers information across** languages.

> [!card]- Which scenarios count as multilingual NLP?
> NER for multiple languages, language-independent parsing, cross-language classification.

> [!card]- Which scenarios count as crosslingual NLP?
> Machine translation, crosslingual QA, crosslingual unlearning, crosslingual reasoning.

> [!card]- Why are LLMs English-centric?
> They are trained predominantly on **Internet resources**.

> [!card]- On which languages do general LLMs such as Llama 3.1 and DeepSeek perform better?
> **High-resource** languages.

> [!card]- How do dedicated multilingual models such as Aya-Expanse compare with general LLMs like Llama 3.1?
> They tend to **lag behind**.

> [!card]- In which tasks can LLMs substitute or supplement humans?
> Data **annotation**, evaluation (**LLM-as-judge**), **dialog**.

> [!card]- What weakness of LLMs does Monz's Arabic-to-English translation comparison show?
> **Fragility on non-English text**: Mistral Small 3.2 **hallucinated** the embassy's location.

> [!card]- True or false: Deep-learning NLP still requires hand-engineered features such as POS tags and parse trees.
> **False.** Deep learning needs very little or no feature engineering; those features belong to pre-deep-learning NLP.

> [!card]- True or false: In pre-deep-learning NLP, different applications used different methodologies.
> **True.** Each application had its own methodology.

> [!card]- True or false: A limited repertoire of neural network types applies to most NLP tasks.
> **True.** That is one of the listed advantages of deep learning.

> [!card]- True or false: Dedicated multilingual models such as Aya-Expanse outperform general LLMs such as Llama 3.1 and DeepSeek.
> **False.** Dedicated multilingual models tend to lag behind.

> [!card]- True or false: Crosslingual QA extracts information from resources in other languages.
> **True.** It is a crosslingual scenario, like machine translation and crosslingual reasoning.

> [!card]- True or false: Because they are trained on Internet resources, LLMs are equally robust across languages.
> **False.** Internet training data makes them English-centric and fragile on non-English text.

> [!card]- Multiple selection: which are true of multilingual and crosslingual NLP? (A) Machine translation is a crosslingual scenario (B) Language-independent parsing is a crosslingual scenario (C) Crosslingual reasoning combines information across languages (D) Multilingual NLP means transferring knowledge across languages
> **A and C.** B is false: parsing is a multilingual scenario. D is false: that defines crosslingual; multilingual means building systems for multiple languages.

> [!card]- Multiple selection: which are true of deep learning in NLP? (A) It needs very little or no feature engineering (B) It works well with small amounts of training data (C) Its errors are easy to trace (D) Insights transfer to vision and speech because network types are uniform
> **A and D.** B is false: it needs large amounts of training data. C is false: errors are hard to trace.

