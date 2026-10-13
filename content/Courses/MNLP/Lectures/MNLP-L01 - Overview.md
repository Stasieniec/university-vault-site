---
type: lecture
course: 5204MNLP6Y
week: 1
lecture: 1
date: 2026-09-01
status: complete
topics:
  - What NLP is and what it models
  - NLP tasks (QA, NER, summarization, sentiment analysis, machine translation)
  - NLP before and after deep learning
  - The role of large language models
  - English-centric training data
  - Multilingual against crosslingual NLP
---

# MNLP-L01: Overview

> [!abstract] In short
> What is NLP, how did deep learning and large language models change it, and where do other languages come in?
>
> NLP builds algorithms that model human language. It used to need a separate method and hand-built features for every task; neural networks replaced that with a few general architectures that learn their own features, at the price of needing lots of data and failing in ways that are hard to trace. Large language models (LLMs) now do whole tasks end to end, but they learn mostly from English web text, so they work best on English and other languages with plenty of web text.
>
> The course splits the problem in two. **Multilingual** NLP builds one system that works for many languages. **Crosslingual** NLP moves information or knowledge from one language to another.
>
> No sample exam question covers this lecture. Expect statements on that split, and on what deep learning and LLMs do and do not do.

## What NLP models, and why it is hard

Information can be sorted in two ways:

- **Structured** (tables, databases) against **unstructured** (language, images and video).
- **Continuous signals** (spoken language and audio, images and video) against **discrete signals** (written language).

So written text is unstructured and discrete; speech and video are unstructured and continuous.

Language is people's medium of choice for complex information: a limited set of words can express an almost unlimited amount of it. Every child learns one, so it looks easy, yet decades of NLP research show it is hard to model formally.

> [!example] Who did not press charges against Clarkson?
> In an example used by Blunsom (2017), a news item says "the BBC producer allegedly struck by Jeremy Clarkson will not press charges", and only a later sentence names "producer Oisin Tymon". To answer, a reader has to work out that the two phrases are the same person. People do this without noticing; a program has to be built to do it.

> [!definition] Natural language processing
> A field at the intersection of **computer science**, **artificial intelligence** and **linguistics**. Its goal is to algorithmically and formally model aspects of human language such as **word formation** (morphology), **sentence structure** (grammar), **sentence meaning**, and **document or discourse structure**.

## The tasks NLP serves

NLP techniques serve concrete end-user tasks, such as **text categorization** (sorting documents into classes), summarization, machine translation and question answering. That is why much of the IT industry (Google, Microsoft, Amazon, Bloomberg and others) invests in them. In industry that means matching web ads to users, analysing reviews, speech recognition and synthesis, translating web pages and dialog agents.

**Question answering differs from retrieval in what goes in and what comes out.** In document retrieval ([[Information Retrieval]]) the user types keywords, and even a well-formed question is chopped into words: "When was the Cuba Crisis?" becomes {cuba, crisis} after stop-word removal. The system returns a **ranked list of documents**. In question answering the user types a **well-formed question** and gets back **the actual answer**, or a short list of them: "1962" for "When was the Cuba Crisis?", or "Warsaw" for "What is the capital of Poland?".

**Named entity recognition (NER)** finds names, since names are strong signals of what a text is about. The usual entity types are **persons, organisations, dates and locations**: "President [Biden] has received the French prime minister [Macron]", with both names tagged as persons. A news site can use it to link each mention of a person to that person's topic page. NER **seems trivial for English, though it still makes silly mistakes, and is much harder for other languages**.

**Summarization** keeps the most important content so a reader can browse quickly. It works on a **single document or several**. Systems traditionally used **frequency-based term weights** borrowed from information retrieval to decide what is important.

**Sentiment analysis** estimates the attitude a text expresses, such as a product review. The categories can be simple (positive or negative) or **fine-grained** ("X likes Y about Z": a person likes some aspect of a product or entity). It **requires NER**, which finds who X is and what Z is, and works at both the **sentence and the document level**. A five-star review can be uninformative, or hide a real complaint in one sentence, which is why the sentence level matters.

**Machine translation (MT)** matters because roughly half of all web pages are in languages other than English. It translates from any language into any other, **typically one sentence at a time**, and needs both "understanding" of the source sentence and the ability to **generate** sentences in the target language. Neural MT often reaches impressive quality, and it can also **fail spectacularly**: a Dutch sentence meaning "Peter R. de Vries may not be present at Holleeder's questioning" came out as "Peter R. de Vries may not be Holleeder during questioning".

## From feature engineering to neural networks

**Before deep learning, every application needed its own approach.** Each combined a machine learning method (support vector machines, decision trees, generative Bayesian models, discriminative maximum-entropy models) with hand-built features:

- **word classes**: part-of-speech (POS) tags, morphology
- **syntactic features**: parse trees
- **semantic features**: named entities, taxonomies, argument roles (who did what to whom)

Machine learning's job was to **weigh how important each feature is** for making correct predictions. The features themselves came from people.

**Neural networks now hold the state of the art for all NLP applications.** Their advantages: strong performance, **very little or no feature engineering** (the network learns its own representations), and a **limited repertoire of network types** that applies to most, perhaps all, NLP tasks. Their disadvantages: they **require large amounts of training data**, their errors are **difficult to trace**, and they can **get it spectacularly wrong**.

The same success shows up across AI: computer vision, handwriting recognition, speech recognition, robotics, information retrieval. **Insights transfer between these fields** because the network types are uniform and few features are specific to one application.

## What large language models changed

[[Large Language Models (LLM)|LLMs]] **originated within NLP and now dominate the research landscape.** They are used in two ways:

- **carrying out traditional NLP tasks end to end**: question answering, document summarization, machine translation
- **substituting or supplementing human effort**: annotating data with labels, **evaluating outputs** (LLM as a judge), mimicking human dialog

> [!example] Four LLMs translate one Amharic sentence
> The sentence (Amharic, an Ethiopian language: the course does not name it, but the script and grammar are Amharic) says that the embassy in South Sudan asked for an investigation into the attack in which Ethiopian peacekeepers were killed. GPT-5.6 Terra, Gemini 3.6 Flash and Claude Sonnet 4.6 all translate it correctly, in slightly different words: "The embassy in South Sudan called for an investigation into the attack in which Ethiopian peacekeeping soldiers were killed."
>
> Mistral Small 3.2 gives: "The Ethiopian embassy in Khartoum, where the Ethiopian peacekeepers were detained, is closed." A different place, detained instead of killed, an embassy closing instead of asking for an investigation: a fluent translation of an event the source does not describe.

**Why LLMs are weaker outside English.** LLMs are trained mostly on Internet resources, and English makes up about half of that material, with Chinese, German, Spanish and Japanese far behind. So most state-of-the-art models are **English-centric**. General models that are also multilingual, such as Llama 3.1 and DeepSeek, **do better on high-resource languages** (those with plenty of data). Models built specifically to be multilingual, such as Aya-Expanse, **tend to lag behind**. The course does not say behind what; the natural reading, used in this note, is behind general models such as Llama 3.1 and DeepSeek.

## Multilingual against crosslingual

The rest of the course is organised around this split.

**Multilingual** means one system that handles many languages. The questions are about building it so that it is not tied to any one language:

- **NER for multiple languages**: how to build a system that can be trained without being language-specific?
- **Language-independent parsing**: how to find language-independent, or universal, features of sentence structure?
- **Sentence or document classification across languages**: again, how to build a system that can be trained without being language-specific?

**Crosslingual** means information or knowledge has to **cross from one language into another**. The guiding question: how does knowledge transfer across languages, for example between English, which has the most data, and Chinese?

- **Machine translation**: how to make information understandable across languages?
- **Crosslingual question answering**: how to extract information from resources written in other languages?
- **Crosslingual unlearning**: how to manipulate information across languages?
- **Crosslingual reasoning**: how to combine information that comes from different languages?

> [!definition] The split
> **Multilingual:** one system works for multiple languages without being language-specific; NER, parsing and classification for many languages.
> **Crosslingual:** information or knowledge crosses between languages; MT, crosslingual QA, crosslingual unlearning, crosslingual reasoning, and knowledge transfer from one language to another.

> [!warning] Classification across languages is multilingual
> "Sentence/document classification **across** languages" is listed as a **multilingual** scenario. Its question is how to train one classifier without making it language-specific. The word "across" in the name does not make it crosslingual. The crosslingual scenarios are the ones whose question is about information or knowledge moving from one language to another.

Later lectures fill in both halves: [[MNLP-L06 - Contextual Embeddings]] tests whether a multilingual model (mBERT) fine-tuned in one language works in another (zero-shot crosslingual transfer), and [[MNLP-L07 - Crosslingual NLP]] adds explicit crosslingual training signals and covers machine translation.

## What to be able to answer

- Where text, speech and tables fall on structured/unstructured and continuous/discrete.
- QA against IR: what goes in, what comes out.
- NER is harder outside English; sentiment analysis needs NER and works per sentence and per document; MT works sentence by sentence.
- Pre-deep-learning NLP (one approach per application, ML weighing hand-built features) against deep learning (no feature engineering, few architectures, data-hungry, opaque).
- The two uses of LLMs, and why they are English-centric.
- Which scenario is multilingual and which crosslingual (classification across languages is multilingual).

Model answers are in the exam questions below.

## Exam questions

> [!exam]- Distinguish multilingual from crosslingual NLP, give examples of each, and say what the English-centric training of LLMs means for both.
> **Key points:** multilingual means one system for many languages without being language-specific; crosslingual means information or knowledge crossing between languages; multilingual examples: NER, parsing, classification across languages; crosslingual examples: MT, crosslingual QA, crosslingual unlearning, crosslingual reasoning; LLMs are trained mostly on English web text and fragile elsewhere.
>
> - **Multilingual:** one system that works for **multiple languages**, without being language-specific. Examples: NER for multiple languages (can it be trained without being language-specific?), language-independent parsing (are there universal features?), sentence or document classification across languages (again, trained without being language-specific).
> - **Crosslingual:** information or **knowledge crosses from one language to another**. Examples: machine translation (make information understandable across languages), crosslingual QA (extract information from resources in other languages), crosslingual unlearning (manipulate information across languages), crosslingual reasoning (combine information across languages).
> - **LLMs:** trained predominantly on Internet resources, about half of which are English, so most state-of-the-art models are **English-centric**. General multilingual models such as Llama 3.1 and DeepSeek do better on **high-resource languages**; dedicated multilingual models such as Aya-Expanse tend to **lag behind** (behind what is not stated; read as behind the general models). Translation can still go badly wrong: asked to translate a sentence in Amharic (identified from its script and grammar; the course does not name the language), one of four models produced a fluent translation of a different event.
>
> Losing marks: treating the two terms as synonyms, or calling classification across languages crosslingual because of the word "across".

> [!exam]- Contrast pre-deep-learning NLP with deep-learning NLP. What did neural networks gain, what did they cost, and why do insights now transfer between fields?
> **Key points:** before: one approach per application, ML weighing hand-built features; gains: state of the art, little or no feature engineering, few network types; costs: data hunger, hard-to-trace errors, spectacular failures; transfer through uniform network types and few application-specific features.
>
> - **Before:** each application required a different approach. Classical ML (SVMs, decision trees, generative Bayesian models, discriminative maximum-entropy models) **weighed the importance of hand-built features**: word classes (POS tags, morphology), syntactic features (parse trees), semantic features (named entities, taxonomies, argument roles).
> - **Advantages of deep learning:** state-of-the-art performance for all NLP applications; very little or no feature engineering; a limited repertoire of network types covers most or all tasks.
> - **Disadvantages:** needs large amounts of training data; errors are hard to trace; can get it spectacularly wrong.
> - **Transfer:** the same success appears in computer vision, handwriting recognition, speech recognition, robotics and IR. Because network types are uniform and application-specific features are few, insights carry over between these areas.

> [!exam]- What roles do large language models now play in NLP, and what are their limits? Use the four-model translation comparison as evidence.
> **Key points:** LLMs originated in and now dominate NLP; end-to-end QA, summarization, MT; substitute or supplement humans in annotation, LLM-as-judge evaluation, dialog; English-centric training data; Mistral Small 3.2 produced a fluent translation of a different event.
>
> - LLMs originated within NLP and now dominate research.
> - **Uses:** traditional NLP tasks end to end (QA, summarization, MT); substituting or supplementing humans in data annotation, evaluation (**LLM as a judge**) and dialog.
> - **Limit:** trained mostly on English Internet text, so weaker on other languages, especially low-resource ones.
> - **Evidence:** asked to translate a sentence in Amharic (identified from its script and grammar; the course does not name the language), GPT-5.6 Terra, Gemini 3.6 Flash and Claude Sonnet 4.6 all render it as the embassy in South Sudan calling for an investigation into an attack that killed Ethiopian peacekeepers, differing only in wording. **Mistral Small 3.2** produced "The Ethiopian embassy in Khartoum, where the Ethiopian peacekeepers were detained, is closed": fluent, and about a different event.

## Flashcards

> [!card]- Which three fields does NLP sit at the intersection of?
> **Computer science**, **artificial intelligence**, **linguistics**.

> [!card]- Which four aspects of human language does NLP aim to model algorithmically and formally?
> Word formation, sentence structure (grammar), sentence meaning, document or discourse structure.

> [!card]- What distinguishes spoken from written language as a signal?
> Spoken language is **continuous**; written language is **discrete**.

> [!card]- What distinguishes structured from unstructured information?
> Structured: tables, databases. Unstructured: **language**, images, video.

> [!card]- How does document retrieval treat a well-formed question?
> It chops it into **keywords** (after stop-word removal).

> [!card]- What does a document retrieval system return?
> A **ranked list of documents**.

> [!card]- How does question answering differ from document retrieval?
> QA takes a **well-formed question** and returns the **actual answer**, not ranked documents.

> [!card]- Which types of entity does named entity recognition usually identify?
> **Persons**, **organisations**, **dates**, **locations**.

> [!card]- Is NER harder for English or for other languages?
> **Other languages**. For English it seems trivial, though it still makes silly mistakes.

> [!card]- What did summarization systems traditionally use to decide what is important?
> **Frequency-based term weights** from information retrieval.

> [!card]- Why does sentiment analysis require named entity recognition?
> Fine-grained sentiment ("X likes Y about Z") has to identify who X is and what Z is.

> [!card]- At which levels does sentiment analysis operate?
> **Sentence** and **document** level.

> [!card]- What unit does machine translation typically translate?
> **Individual sentences**.

> [!card]- Which two abilities does machine translation require?
> **Understanding** the source sentence and **generating** sentences in the target language.

> [!card]- In pre-deep-learning NLP, what was machine learning used for?
> To **weigh the importance** of hand-built features (POS tags, parse trees, named entities) for prediction.

> [!card]- How does deep-learning NLP differ from pre-deep-learning NLP in feature engineering?
> Pre-deep-learning NLP relied on hand-built features; deep learning needs **very little or none**.

> [!card]- What are the disadvantages of deep learning for NLP?
> Needs **large amounts of training data**, errors are **hard to trace**, can **fail spectacularly**.

> [!card]- Why do deep-learning insights transfer between NLP, vision, speech and other fields?
> Network types are **uniform** and there are few **application-specific features**.

> [!card]- In which roles can LLMs substitute or supplement humans?
> Data **annotation**, evaluation (**LLM as a judge**), **dialog**.

> [!card]- Why are most state-of-the-art LLMs English-centric?
> They are trained predominantly on **Internet resources**, where English dominates.

> [!card]- How does the performance of general multilingual LLMs such as Llama 3.1 and DeepSeek vary across languages?
> They do better on **high-resource** languages.

> [!card]- How do dedicated multilingual models such as Aya-Expanse compare with large general LLMs?
> They tend to **lag behind** (read as behind the general models; not spelled out).

> [!card]- What went wrong when Mistral Small 3.2 translated an Amharic sentence into English (language identified from the script)?
> It produced a **fluent translation of a different event** (embassy in Khartoum, closed).

> [!card]- What is the difference between multilingual and crosslingual NLP?
> Multilingual: one system works for **many languages**. Crosslingual: information or knowledge **crosses between** languages.

> [!card]- What is the guiding question of crosslinguality?
> **How does knowledge transfer across languages?**

> [!card]- Is sentence or document classification across languages a multilingual or crosslingual scenario?
> **Multilingual**: one classifier trained without being language-specific.

> [!card]- Which crosslingual scenario asks how to manipulate information across languages?
> **Crosslingual unlearning**.

> [!card]- Which crosslingual scenario asks how to combine information across languages?
> **Crosslingual reasoning**.

> [!card]- Which crosslingual scenario extracts information from resources in other languages?
> **Crosslingual question answering**.

> [!card]- Which end-user tasks does NLP address?
> Text categorization, summarization, machine translation, question answering.

> [!card]- Which traditional NLP tasks do LLMs carry out end to end?
> Question answering, document summarization, machine translation.

> [!card]- Why does a five-star review still need sentence-level sentiment analysis?
> It can be uninformative, or hide a real complaint in one sentence.

> [!card]- What made pre-deep-learning NLP costly to extend to a new task?
> Each task needed its own approach and hand-built features.

> [!card]- What distinguishes how pre-deep-learning NLP and neural networks get their features?
> Before: people hand-built features and ML weighed them. Neural networks learn their own.

> [!card]- How is a task like parsing made multilingual?
> By finding language-independent, or universal, features.

> [!card]- What distinguishes crosslingual QA from ordinary QA?
> The answer must be extracted from resources in another language.

> [!card]- Why did Mistral Small 3.2's translation in the four-LLM comparison show that fluency is no evidence of correctness?
> It was fluent English describing a different event from the source.

> [!card]- True or false: Deep-learning NLP still requires hand-engineered features such as POS tags and parse trees.
> **False.** Deep learning needs very little or no feature engineering; those features belong to pre-deep-learning NLP.

> [!card]- True or false: In pre-deep-learning NLP, each application required a different approach.
> **True.** Each application combined its own ML method and features.

> [!card]- True or false: Neural networks require large amounts of training data.
> **True.** Data hunger is one of their disadvantages, with hard-to-trace errors and spectacular failures.

> [!card]- True or false: Sentiment analysis operates only at the document level.
> **False.** It operates at both the sentence and the document level.

> [!card]- True or false: Language-independent parsing is a crosslingual scenario.
> **False.** It is multilingual: it looks for universal features of sentence structure.

> [!card]- True or false: Sentence or document classification across languages is a multilingual scenario.
> **True.** Its question is how to train one classifier without being language-specific; "across" does not make it crosslingual.

> [!card]- True or false: Crosslingual unlearning asks how to combine information across languages.
> **False.** Unlearning asks how to manipulate information across languages; reasoning combines it.

> [!card]- True or false: Dedicated multilingual models such as Aya-Expanse outperform general LLMs such as Llama 3.1.
> **False.** Dedicated multilingual models tend to lag behind (read as behind the general models).

> [!card]- True or false: LLMs originated within NLP.
> **True.** They originated within NLP and now dominate its research landscape.

> [!card]- True or false: Summarization works on a single document or on several.
> **True.** It keeps the most important content of one document or several, so a reader can browse quickly.

> [!card]- Multiple selection: which are crosslingual scenarios? (A) Machine translation (B) Language-independent parsing (C) Crosslingual reasoning (D) Sentence classification across languages
> **A and C.** B and D are multilingual: their question is how to build one system that is not language-specific.

> [!card]- Multiple selection: which are true of deep learning in NLP? (A) It needs very little or no feature engineering (B) It works well with small amounts of training data (C) Its errors are easy to trace (D) Insights transfer to vision and speech because network types are uniform
> **A and D.** B is false: it needs large amounts of data. C is false: errors are hard to trace.

> [!card]- Multiple selection: which are true? (A) QA returns a ranked list of documents (B) Sentiment analysis requires named entity recognition (C) Machine translation typically translates individual sentences (D) LLMs are used to evaluate outputs as judges
> **B, C and D.** A is false: IR returns ranked documents; QA returns the actual answer.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Next:** [[MNLP-L02 - Multilinguality and Writing Systems]]
- **Later on transfer:** [[MNLP-L06 - Contextual Embeddings]], [[MNLP-L07 - Crosslingual NLP]]
- **Related concepts:** [[Information Retrieval]], [[Large Language Models (LLM)]]
- **Source:** `overview.pdf`, Christof Monz, Informatics Institute, University of Amsterdam
