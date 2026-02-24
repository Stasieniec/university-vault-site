---
type: concept
aliases: [RAG, Retrieval-Augmented Generation]
course: [IR]
tags: [neural-ir]
status: complete
---

# Retrieval-Augmented Generation

> [!definition] Retrieval-Augmented Generation (RAG)
> **RAG** is a framework that combines a retrieval model (which finds relevant documents) with a generative model (like an LLM). Instead of relying solely on its internal knowledge, the generator "reads" the retrieved documents to provide more accurate, grounded, and up-to-date answers.

> [!formula] The RAG Pipeline
> 1. **Retrieve**: Given a query $q$, find the top-$k$ relevant documents $D = \{d_1, ..., d_k\}$ from a knowledge base.
> 2. **Augment**: Create an expanded prompt: `Context: [D] Question: [q]`.
> 3. **Generate**: The LLM produces an answer $a$ conditioned on the context: $P(a | q, D)$.

> [!intuition] Open Book Exam
> Traditional LLMs (like GPT-4) are like a student taking an exam from memory. They might get facts wrong (hallucinate). RAG turns the exam into an "open book" test. The system searches the library, finds the right page, and then writes an answer based on what it's looking at.

## Why Use RAG?

- **Factuality**: Reduces "hallucinations" by forcing the model to cite sources.
- **Freshness**: You don't need to retrain the model to update its knowledge; you just update the documents in the retriever's index.
- **Transparency**: Users can see the source documents used to generate the answer.
- **Privacy**: Allows LLMs to answer questions about private data without that data ever being part of the model's training set.

## Connections

- **Retriever part**: Can use [[BM25]], [[BERT for IR]] (Dense Retrieval), or [[uniCOIL]].
- **Generator part**: Large Language Models (LLMs).
- **Problem solved**: Hallucination, outdated information in weights.

## Appears In

- [[IR-L09 - LLMs for IR (Upcoming)]]
