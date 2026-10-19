---
type: exam-prep
course: 5204MNLP6Y
lecture: 1
status: complete
---

# MNLP-L01 Exam Prep - Overview

> [!abstract] What the exam can ask here
> A block on this lecture would be small: sorting scenarios into multilingual or crosslingual, what an NLP task takes in and gives back, and what deep learning and LLMs do and do not do. The traps are "classification across languages" (multilingual, despite "across"), "only" on sentiment analysis and summarization, and swapping the advantages and disadvantages of neural approaches.
> [[MNLP-L01 - Overview|Lecture note]]; the language shares of LLM training data are on the [[MNLP - Cheat Sheet|cheat sheet]].

## Multilingual against crosslingual

- Multilingual scenarios ask how to build a system that can be trained **without being language-specific**.
- NER for **multiple languages** is a multilingual scenario.
- **Language-independent parsing** is multilingual: how to identify language-independent or **universal** features.
- Sentence or document classification **across languages** is listed as **multilingual**.
- Crosslinguality asks how **knowledge transfers across** languages.
- **Machine translation** is crosslingual: how to make information **understandable** across languages.
- Crosslingual **question answering**: how to **extract** information from resources in other languages.
- Crosslingual **unlearning**: how to **manipulate** information across languages.
- Crosslingual **reasoning**: how to **combine** information across languages.

| scenario | type | its question |
|---|---|---|
| NER, many languages | multilingual | not language-specific |
| parsing | multilingual | universal features |
| classification | multilingual | not language-specific |
| MT | crosslingual | make understandable |
| QA | crosslingual | extract |
| unlearning | crosslingual | manipulate |
| reasoning | crosslingual | combine |

> [!warning] Traps
> - *"Sentence classification across languages is a crosslingual scenario."* False: the slides list it as multilingual.
> - *"Language-independent parsing is a crosslingual scenario."* False: multilingual.
> - *"Crosslingual unlearning asks how to combine information across languages."* False: unlearning manipulates; reasoning combines.
> - *"Crosslingual QA extracts answers from resources in the language of the question."* False: from resources in other languages.

## What each task takes in and gives back

- In document retrieval, users enter **keywords or key-phrases**; even a well-formed question is **chopped into words** (after stop word removal).
- An IR system returns a **ranked list of documents**.
- In QA the user enters a **well-formed question** and gets **the actual answer** (or a list of answers).
- NER: names are **important indicators of content**; entities are often persons, organisations, dates, locations.
- NER seems trivial for English but **still makes silly mistakes**, and is **much harder for other languages**.
- Summarization keeps the most important content for **quick browsing**.
- Summarization applies to **single as well as multiple** documents.
- Summarization systems **traditionally** used **frequency-based** (IR) term weights.
- Sentiment analysis estimates the **attitude** expressed in a text.
- Sentiment categories can be simple (positive/negative) **or fine-grained** (X likes Y about Z).
- Sentiment analysis **requires named entity recognition**.
- Sentiment analysis operates on the **sentence and** the **document** level.
- MT **typically translates individual sentences**.
- MT requires **understanding** the source sentence **and** the ability to **generate** target sentences.
- About **half** of web pages are in languages other than English.
- Neural MT often achieves impressive quality and **can also fail spectacularly**.

> [!warning] Traps
> - *"A QA system returns a ranked list of documents."* False: IR does that; QA returns the answer.
> - *"Sentiment analysis operates only at the document level."* False: sentence and document level.
> - *"Sentiment analysis can be done independently of named entity recognition."* False: it requires NER.
> - *"Sentiment analysis is restricted to positive and negative."* False: it can be fine-grained.
> - *"Summarization can only be applied to single documents."* False: single and multiple.
> - *"NER for English no longer makes mistakes."* False: it still makes silly mistakes.
> - *"NER is equally hard across languages."* False: much harder outside English.
> - *"MT typically translates whole documents at once."* False: individual sentences.

## Before and after deep learning

- Before deep learning, **each NLP application required a different approach**.
- Methods: support vector machines, decision trees, **generative** Bayesian approaches, **discriminative** maximum entropy approaches.
- Features were hand-built: word classes (POS, morphology), **syntactic** (parse trees), **semantic** (named entities, taxonomies, argument roles).
- Machine learning was used to **weigh the importance** of individual features.
- Neural approaches now hold the state of the art for **all** NLP applications.
- Advantages: strong performance; **very little or no feature engineering**; a **limited repertoire** of network types applies to most (all?) tasks.
- Disadvantages: need **large amounts of training data**; errors are **difficult to trace**; can get it **spectacularly wrong**.
- The success extends beyond NLP: computer vision, handwriting recognition, speech recognition, robotics, information retrieval.
- Insights transfer between areas because of **uniform network types** and **few application-specific features**.

| | before deep learning | neural |
|---|---|---|
| approach | one per application | few network types |
| features | hand-built, ML weighs them | little or none engineered |

> [!warning] Traps
> - *"Before deep learning, machine learning was used to discover the features."* False: features were hand-built; ML weighed them.
> - *"Pre-neural NLP used only discriminative approaches."* False: generative Bayesian ones too.
> - *"Parse trees were used as semantic features."* False: syntactic features.
> - *"Neural approaches need little training data."* False: they need large amounts.
> - *"Errors of neural approaches are easy to trace."* False: difficult to trace.
> - *"Insights transfer between AI areas because each area has its own specialised network types."* False: the network types are uniform.

## Large language models

- LLMs **originated within NLP** and now dominate the research landscape.
- LLMs carry out traditional NLP tasks **end to end**: QA, document summarization, MT.
- LLMs increasingly **substitute or supplement** human effort: **data annotation**, **evaluation of outputs** (LLM as a judge), mimicking human **dialog**.
- LLMs are trained **predominantly on Internet resources**, where English dominates, so most state-of-the-art models are **English-centric**.
- Multilingual models (Llama 3.1, DeepSeek) perform **better for high-resource** languages.
- **Dedicated** multilingual models (Aya-Expanse) **tend to lag behind**. The slide does not say behind what.
- In the slide's translation example, three LLMs give near-identical translations and Mistral Small 3.2 gives a fluent sentence that says something different.

> [!warning] Traps
> - *"LLMs originated outside NLP and were later adopted by it."* False: they originated within NLP.
> - *"LLMs are used for end-user tasks but not to replace human annotation or evaluation."* False: annotation, LLM as a judge, dialog.
> - *"Dedicated multilingual models such as Aya-Expanse outperform Llama 3.1 and DeepSeek."* False: they tend to lag behind.
> - *"Multilingual LLMs perform equally well across languages."* False: better for high-resource languages.

## What NLP models, and what kind of signal text is

- NLP sits at the intersection of **computer science**, **artificial intelligence** and **linguistics**.
- Its goal: algorithmically and formally model word formation, sentence structure (grammar), sentence meaning, document or discourse structure.
- Structured information: tables, databases. Unstructured: **language**, images and videos.
- Continuous signals: **spoken language**, audio, images and videos. Discrete signals: **written language**.
- A **limited** repertoire of words expresses an **almost infinite** amount of information.
- Every child learns a language, yet after decades of research it is **not easy to model formally**.
- NLP techniques address **concrete end-user tasks**: text categorization, document summarization, MT, QA.

> [!warning] Traps
> - *"Written language is a continuous signal."* False: discrete; spoken language is continuous.
> - *"Language is structured information."* False: unstructured, like images and videos.

## Practice questions

**1.** Which of the following are crosslingual scenarios? (multiple answers possible)
- [A] Machine translation, which asks how to make information understandable across languages
- [B] Sentence classification across languages
- [C] Crosslingual unlearning, which asks how to manipulate information across languages
- [D] Language-independent parsing

> [!example]- Answer
> **A and C.** B: listed as multilingual (train one classifier that is not language-specific). D: multilingual (universal features).

**2.** Which of the following statements about neural network based NLP are true? (multiple answers possible)
- [A] Neural approaches achieve state-of-the-art performance for all NLP applications
- [B] Because no feature engineering is needed, neural approaches require little training data
- [C] Insights transfer between NLP and computer vision because the network types are uniform
- [D] Errors of neural approaches are easy to trace

> [!example]- Answer
> **A and C.** B: they require large amounts of training data. D: errors are difficult to trace.

**3.** Which of the following statements about sentiment analysis are true? (multiple answers possible)
- [A] Sentiment analysis requires named entity recognition
- [B] Sentiment analysis operates only on the document level
- [C] Sentiment analysis can be fine-grained, such as X likes Y about Z
- [D] Sentiment analysis is restricted to the categories positive and negative

> [!example]- Answer
> **A and C.** B: it operates on the sentence and document level. D: it can be fine-grained.

**4.** Which of the following statements are true? (multiple answers possible)
- [A] In document retrieval, even a well-formed question is chopped into words
- [B] A question answering system returns a ranked list of documents
- [C] Machine translation typically translates individual sentences
- [D] Summarization can only be applied to single documents

> [!example]- Answer
> **A and C.** B: that is document retrieval; QA returns the actual answer. D: single as well as multiple documents.

**5.** Which of the following statements about large language models are true? (multiple answers possible)
- [A] LLMs originated within NLP
- [B] LLMs are used to evaluate the outputs of systems
- [C] Most state-of-the-art LLMs are English-centric because they are predominantly trained on Internet resources
- [D] Dedicated multilingual models such as Aya-Expanse outperform Llama 3.1 and DeepSeek

> [!example]- Answer
> **A, B and C.** D: dedicated multilingual models tend to lag behind.

**6.** Which of the following statements about NLP before deep learning are true? (multiple answers possible)
- [A] Each NLP application required a different approach
- [B] Machine learning was used to weigh the importance of individual features
- [C] Parse trees were used as semantic features
- [D] Only discriminative approaches were used

> [!example]- Answer
> **A and B.** C: parse trees are syntactic features. D: generative Bayesian approaches were used as well.
