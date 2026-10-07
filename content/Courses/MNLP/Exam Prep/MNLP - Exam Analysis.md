---
type: exam-prep
course: 5204MNLP6Y
status: complete
---

# MNLP - Exam Analysis

> [!abstract] What the exam looks like
> Two hours on paper, multiple choice and multiple selection only, questions grouped in blocks by topic with every question independent of the others, and one two-sided cheat sheet that can be printed. Practical details (date, rooms) are in [[MNLP - Overview]].
>
> The only sample material is `exam-examples.pdf` in the Canvas module "Exam" (uploaded 2026-10-06). Monz's note on it: *"The file below contains multiple choice examples from previous years. This is not representative of the content of this year's questions as these questions are from a different course, but only the type of questions you can expect."* So the questions below show the **form**. The content comes from the lectures.

## How a question is built

The example file has three question blocks:

| block | topic | sub-questions | marks |
|---|---|---|---|
| Question 1 | sequence classification | 1.1, 1.2 | 12 (6 + 6) |
| Question 2 | training models and architectures | 2.1 | 6 |
| Question 3 | low-resource machine translation | 3.1 | 6 |

- Each block opens with a line naming its topic and the block's total marks.
- Every sub-question is worth **6 marks** and is marked **(multiple answers possible)**.
- Options are lettered **[A]**, **[B]**, **[C]**, sometimes **[D]**: three or four statements per question.
- The answer is written as letters in a box ("Your answer(s):").
- The statements are short claims about **how a method works or what it does and does not do**. Several are true-sounding generalisations with one word wrong ("continuously improves", "does not suffer from", "independent parameter updates").

Not stated in the file: whether a multiple-selection question gives partial credit, and whether a wrong letter costs marks.

## The example questions

Quoted from the file, typos included. There is **no official answer key**. Question 3.1 is the only one on MNLP material; its worked answer is below. The other three come from a different course and test content outside these notes, so they are left unanswered here.

> [!example] Question 1: sequence classification [12 marks]
> **1.1** How do multi-layer convolutional neural networks (CNNs) model long-distance phenomena in natural language? (multiple answers possible) [marks: 6]
> - [A] Increasing the kernel size of each convolution
> - [B] Applying max-over-time pooling after the last convolutional layer
> - [C] Applying max-over-time pooling after each convolutional layer
> - [D] Increasing the filter size of each convolution
>
> **1.2** One way to learn how to represent longer inputs is to use Doc2Vec, which learns paragraph (or document) vectors. Which of the following statements are true? (multiple answers possible) [marks: 6]
> - [A] Paragraph vectors (i.e., embeddings) directly model the paragraph class
> - [B] Paragraph vectors for new, unseen documents have to computed first.
> - [C] Doc2Vec requires a separate classifier mapping paragraph vectors to classes.

> [!example] Question 2: training models and architectures [6 marks]
> **2.1** One way to utilize mutiple GPUs is to use data parallelism. Which of the following statements are true? (multiple answers possible) [marks: 6]
> - [A] Data parallelism can lead to more unstable gradients due to the smaller batch size per GPU.
> - [B] The speed improvement of data parallelism depends on the number of model parameters.
> - [C] Data parallelism allows for independent parameter updates on each individual GPU.

> [!example] Question 3: low-resource machine translation [6 marks]
> **3.1** Fine-tuning is used to adapt a neural machine translation system trained on a general domain to a specific domain with limited parallel training data in that domain (the so-called in-domain). Note, 'general domain' and 'in-domain' can also refer to different language pairs. Which of the following statements about fine-tuning are true? (multiple answers possible) [marks: 6]
> - [A] Fine-tuning uses parameters from the general domain for initialization
> - [B] Fine-tuning continuously improves with the number of in-domain updates
> - [C] Fine-tuning does not suffer from catastrophic forgetting

### Question 3.1, worked

The note on "different language pairs" makes this the parent-child transfer learning of [[MNLP-L07 - Crosslingual NLP]] (section 9): the general domain is the parent, the in-domain data is the child.

- **[A] true.** Transfer learning is "basically the same as fine-tuning": train on the large pair, then **continue training** the same model on the small one. The child starts from the parent's parameters.
- **[B] false.** The tell is "continuously". L07 section 9.4 shows that letting more parameters adapt to the small child data stops helping and then hurts: unfreezing the target embeddings lowers Uzbek→English from 15.0 to 14.7 and 13.7, because parameters learned from 300M tokens overfit on 1.8M. The same holds for the number of updates (outside the MNLP material, but the standard result): with little in-domain data, quality peaks after a number of updates and then degrades as the model overfits.
- **[C] false.** Outside the MNLP material, but standard: continuing to train only on in-domain data overwrites what the model knew about the general domain, so general-domain quality drops. That loss is what "catastrophic forgetting" names.

Answer: **A**.

## What this means for revising

- **Statements turn on one word.** In the example file, "continuously", "does not suffer" and "independent" decide the answer. Exact conditions ("only the target embeddings are frozen", "into English, not out of it") matter more than being able to explain a method at length.
- **Multiple-selection questions punish half-knowledge.** The lecture notes' flashcards include multiple-selection cards in this format (starting with L07) for practice.
- **The cheat sheet is two sides and can be typed**, so tables of numbers (which method wins on which benchmark, by how much) can go on it instead of being memorised. The decision rules (what to freeze, which sampling temperature, many-to-one against one-to-many) still have to be understood, because the statements test them in unfamiliar wording.

## Topics by lecture

| lecture | topic |
|---|---|
| [[MNLP-L01 - Overview]] | NLP applications, multilingual against crosslingual, deep learning and LLMs in translation |
| [[MNLP-L02 - Multilinguality and Writing Systems]] | resource tiers, typology, writing systems, Unicode normalisation and encodings |
| [[MNLP-L03 - Morphology and Word Formation]] | vocabularies, word segmentation (maximum matching, BMES with HMMs), inflection, derivation, compounding, morphological typology |
| [[MNLP-L04 - Subword Segmentation]] | Morfessor, BPE, WordPiece, SentencePiece Unigram, byte-level BPE, fertility |
| [[MNLP-L05 - Static Embeddings]] | Word2Vec, negative sampling, GloVe, embedding evaluation, fastText, crosslingual embeddings |
| [[MNLP-L06 - Contextual Embeddings]] | BERT, mBERT, language sampling, zero-shot crosslingual transfer, vocabulary overlap |
| [[MNLP-L07 - Crosslingual NLP]] | parallel data, LASER, XLM, InfoXLM, NMT, transfer learning, multilingual NMT, BART, mBART |
