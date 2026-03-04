---
type: lecture
course: IR
week: 4
lecture: 9
book_sections: []
topics:
  - "[[Retrieval-Augmented Generation]]"
  - "[[Dense Retrieval]]"
  - "[[Cross-Encoder]]"
  - "[[Transformers]]"
  - "[[Language Model for IR]]"
  - "[[Multi-Stage Ranking]]"
status: complete
---

# IR-L09: Retrieval-Augmented Generation (RAG)

**Lecturer:** Maria Heuss  
**Date:** 25.2.2026

---

## Overview & Motivation

[[Retrieval-Augmented Generation]] (RAG) addresses two critical limitations of modern [[Language Model for IR|language models]]: **hallucinations** (generating false or unsupported information) and **limited knowledge cutoff** (inability to answer questions about recent events or private data). RAG combines a pre-trained retriever with a generative [[Transformers|transformer]]-based language model to ground generation in retrieved evidence, enabling accurate, verifiable, and up-to-date answers.

Traditional language models rely entirely on parameters learned during pretraining. RAG decouples this into two stages: **(1) retrieve relevant documents** from an external datastore at test time, and **(2) generate answers conditioned on those retrieved passages**. This simple yet powerful paradigm has become the foundation for modern [[Language Model for IR|information retrieval]] systems like ChatGPT and Gemini.

---

## 1. Why Retrieval-Based Language Models?

### Problem 1: Hallucinations

[[Language Model for IR|Large language models]] generate text that is syntactically fluent but often factually false—inventing citations, dates, names, and statistics that never existed. This renders them unreliable for knowledge-intensive tasks where correctness is paramount.

### Problem 2: Information Access & Staleness

- Knowledge cutoff limits: A model trained on data up to April 2024 cannot answer questions about events in June 2024.
- Private data: LLMs cannot reason over user-specific documents, emails, or proprietary databases unless explicitly provided.
- Scalability: Retraining or fine-tuning on new data is expensive.

**Solution:** Use an external datastore (corpus, knowledge base, vector index) as a "source of truth" that the model can query at test time. The LLM can then generate answers grounded in retrieved evidence.

---

## 2. What is RAG?

> [!definition]
> **Retrieval-Augmented Generation (RAG)** is a [[Language Model for IR|language model]] that uses an external datastore at test time to retrieve relevant documents before generating an answer.

### Formal Definition

Given a query $x$ and corpus $D$, RAG:

1. **Retrieves** the top-$k$ documents $Z = \{z_1, z_2, \ldots, z_k\}$ most relevant to $x$
2. **Generates** output $y$ conditioned on both the query and retrieved passages: $P(y | x, Z)$

Critically, retrieval happens at **inference time**, not training time. The external datastore can be updated without retraining the model.

### Key Advantages

- ✅ **Factuality:** Answers grounded in retrieved evidence reduce hallucination
- ✅ **Freshness:** Corpus updates are immediate; no retraining required
- ✅ **Interpretability:** Retrieved passages provide explanations/citations
- ✅ **Scalability:** Handle large corpora without parameter explosion
- ✅ **Privacy:** Incorporate private data at inference without leaking it into model weights

### Historical Timeline

- **2017–2019:** Early retrieval-augmented approaches in QA (e.g., DrQA)
- **2020:** [[DPR]] (Dense Passage Retrieval) + [[RAG]] paper (Lewis et al.) establish modern paradigm
- **2021:** [[FiD|Fuse-in-Decoder]] shows effective use of many passages
- **2023:** [[Atlas]] (end-to-end retriever-generator), In-context RAG, [[Self-RAG]] (adaptive retrieval)
- **2024–2025:** Agentic RAG, context compression, attribution & faithfulness challenges

---

## 3. Basic RAG Architectures

### 3.1 Naive RAG (String Concatenation)

The simplest approach: retrieve top-$k$ documents, concatenate them into the prompt, and let the LLM generate.

```
Query: "Who won the Nobel Prize in Physics 2023?"

Retrieved passages:
[1] "The 2023 Nobel Prize in Physics was awarded..."
[2] "Previous winners include..."
[...up to k passages...]

Prompt = "Context: " + passages + "\nQuestion: " + query
Output = LLM.generate(prompt)
```

**Limitations:**
- Poor scaling: Naive concatenation rapidly exhausts context windows
- No learnable integration: Retriever and generator are disconnected
- Inefficient use of passages when context is limited

---

### 3.2 RAG (Lewis et al., 2020)

Treats retrieved documents as **latent variables** and marginalizes over them during generation.

#### Mathematical Formulation

Given query $x$ and documents $D$, the probability of output $y$ is:

$$P(y | x) = \sum_{z \in D} P(z | x) \cdot P(y | x, z)$$

Where:
- $P(z | x)$ = **Retriever score** — likelihood of document $z$ given query (dense or sparse retrieval)
- $P(y | x, z)$ = **Generator score** — seq2seq model likelihood of output given query and document

**Components:**
- **Query Encoder:** Dense vector representation of query (learned bi-encoder)
- **Document Index:** FAISS index for efficient Maximum Inner Product Search (MIPS) over 21M Wikipedia paragraphs
- **Seq2Seq Generator:** Pre-trained T5 model that learns to generate conditioned on retrieved passages

#### Two Marginalization Strategies

> [!formula]
> **RAG-Sequence**
> $$P_{\text{seq}}(y | x) = \sum_{z} P(z | x) \cdot P_{\text{seq}}(y | x, z)$$
> Each retrieved document independently generates the full output sequence. The final probability is a weighted average over complete sequences. **One document grounds the entire answer.**

> [!formula]
> **RAG-Token**
> $$P_{\text{token}}(y | x) = \prod_{i=1}^{|y|} \sum_{z} P(z | x) \cdot P(y_i | x, z, y_{<i})$$
> At each token position $i$, the model computes a weighted mixture over all retrieved documents. **Different parts of the answer can draw from different documents**, enabling multi-fact synthesis.

#### Training: Why End-to-End Works Despite Non-Differentiability

The top-$k$ selection step is discrete and non-differentiable. How does gradient-based training work?

> [!intuition]
> **Trick 1: Fix top-$k$ per training step.** Once the top-$k$ is fixed, everything downstream is differentiable (matrix multiplications in the generator). The query encoder gradually learns to move toward high-scoring documents through gradient updates.
>
> **Trick 2: Frozen document encoder.** Re-encoding all 21M documents per training step is infeasible. Only the query encoder $E_q$ is updated; the document encoder $E_d$ remains frozen. This means some documents in the original corpus are "unreachable" — their embeddings never change, so the query can never reach them no matter what. (Later methods like [[Atlas]] address this with periodic reindexing.)

**Key Insight:** End-to-end training improves the **retriever** by showing it which documents help the generator. The generator loss backpropagates to the query encoder, creating a feedback loop that optimizes retrieval quality jointly with generation.

#### Evaluation Results

- **Natural Questions (NQ):** 44.5 EM (vs. 40.4 FiD)
- Shows comparable or better performance to dense retrieval baselines while maintaining end-to-end learnability

---

### 3.3 FiD: Fuse-in-Decoder (Izacard & Grave, 2021)

**Problem:** RAG and naive concatenation struggle to effectively use many passages. Naive concatenation hits context limits; RAG's marginal likelihood is slow at inference.

**Solution:** [[FiD|Fuse-in-Decoder]] — separate passage encoding from passage fusion.

#### Architecture

```
For each passage z_i:
  passage_encoding_i = Encoder(question, z_i)  # Independent encoding with question
  
All encodings fused in decoder via cross-attention:
  output = Decoder(encoder_outputs=[passage_encoding_1, ..., passage_encoding_k])
```

#### Key Contributions

> [!tip]
> **1. Scalability:** Each passage is encoded independently with the question, then all representations are passed to a shared decoder. This enables **linear scaling** — effectively use up to **100+ passages** where naive concatenation fails.

> [!tip]
> **2. Cross-Document Synthesis:** The decoder cross-attention mechanism can combine evidence across passages. Answers not present verbatim in any single passage can be synthesized from multiple sources.

> [!tip]
> **3. Architectural Simplicity:** Uses standard seq2seq (T5) with modified input formatting. No new parameters or complex mechanisms.

#### Performance Gains

- Natural Questions: Monotonic improvement with passage count (up to 100 passages)
- Natural Questions with gold passages: 68.2 EM (vs. previous 44.5 RAG)
- Open-domain QA significantly outperforms RAG and naive approaches

#### Limitations

- **Disconnected retriever & generator:** Retriever is BM25 or sparse retrieval; no end-to-end gradient flow
- **Linear inference cost:** Inference time grows with passage count (though still tractable)
- **No adaptive retrieval:** Always retrieves a fixed number of passages regardless of query difficulty

---

## 4. Advanced RAG Approaches

### 4.1 Atlas: Adapting the Retriever to the LLM (Izacard et al., 2023)

[[Atlas]] addresses RAG's core limitation: the frozen document encoder. While RAG's query encoder learns to retrieve better documents, the document embeddings are fixed, making some documents permanently unreachable.

#### Key Innovation: Periodic Reindexing

```
For each training epoch:
  1. Freeze generator weights
  2. Re-encode entire corpus with updated document encoder
     → Update FAISS index
  3. Run retrieval on training queries
  4. Retrain generator on retrieved passages
```

**Cost:** Reencoding millions of documents is expensive but done offline/periodically, not per-step.

#### Design Decisions

- **Retriever:** Learned dense [[Bi-Encoder]] (shared embeddings with generator encoder)
- **Generator:** Seq2seq (T5-base to XL) or decoder-only (LLaMA)
- **Training:** Alternates between updating index and training generator
- **Few-shot:** Atlas excels in few-shot settings; fewer training examples make learnable retrieval especially valuable

#### Performance Highlights

- Few-shot QA (Natural Questions, TriviaQA): Strong gains over frozen-retriever [[FiD]]
- Competitive with fine-tuned dense retrievers while maintaining end-to-end learnable pipeline

---

### 4.2 In-Context RAG: Decoupled Retrieval & Generation (Ram et al., 2023)

**Motivation:** Large context windows (64k–128k+ tokens) have become standard in modern LLMs (GPT-4, Claude 3). Why rebuild tight coupling (FiD) when you can simply prepend passages?

#### Approach

```
Context = "Passage 1:\n" + p1 + "\nPassage 2:\n" + p2 + ... + "\nQuestion: " + query
Output = LLM.generate(context)  # In-context learning / in-context retrieval
```

**Advantages:**
1. **Zero model modification:** Works with any LLM as a black box
2. **Model-agnostic:** No retraining; applicable to proprietary models
3. **Simple & cheap:** Retrieve + format + call LLM
4. **Sufficient performance:** With modern 100k+ context windows, competitive with FiD

#### Trade-offs

- **Lost in the middle effect** (discussed in Challenges)
- **No learnable retriever-generator interaction**
- **High inference cost:** Each retrieved passage consumes tokens

**Verdict:** In-context RAG has become the de-facto standard in industry (e.g., ChatGPT, Gemini) because simplicity and compatibility often outweigh tight coupling benefits.

---

### 4.3 Self-RAG: Adaptive Retrieval through Self-Reflection (Asai et al., 2023)

[[Self-RAG]] introduces **learned decision-making:** the model itself decides when and what to retrieve, plus critiques its own outputs.

#### Architecture

The model generates **reflection tokens** to control behavior:

- `[Retrieve]` — Signal to retrieve passages
- `[IsRel]` — Relevance judgment of retrieved passages
- `[IsSup]` — Factual support check (does retrieved passage support the claim?)
- `[Utility]` — Overall utility of response

#### Training Pipeline

```
Step 1: Train Critic C (Llama2-7B, distilled from GPT-4 labels)
  - For each segment, C predicts whether retrieval is needed
  - C judges relevance and factual support of passages

Step 2: Augment Training Corpus using C + Retriever R
  - For each training segment:
    a. C decides: retrieval needed? (IsRel)
    b. If yes: retrieve top-k passages
    c. C labels: passage relevance (IsRel) and support (IsSup)
    d. Interleave reflection tokens & passages into training text

Step 3: Train Generator M (Llama2-7B/13B)
  - Standard next-token prediction loss on augmented corpus
  - Vocabulary expanded with reflection tokens
  - Retrieved passages are masked from loss (no memorization)
```

#### Inference: Beam Search with Reflection Scoring

During generation, the model can output reflection tokens. At each step:
- If model outputs `[Retrieve]`: fetch relevant passages
- Use reflection tokens to score generation paths
- Beam search over multiple retrieval decisions

#### Key Insights

- **Adaptive:** Simple queries skip retrieval; complex queries trigger it
- **Faithful:** Reflection tokens create explicit traces of "did I retrieve?" and "did I check support?"
- **Lightweight:** Runs on 7B–13B parameters, not 70B+

#### Performance

- Matches or exceeds larger models (13B model ≈ 70B models on some benchmarks)
- Fewer retrieval calls = cheaper inference
- Better calibration of when to retrieve

---

## 5. Modern RAG: Current Best Practices (2024–2025)

### Architecture Consensus

Modern production RAG systems are **decoupled and modular**:

```
User Query
    ↓
[Retrieval Module] → Dense retrieval (embeddings) + Sparse retrieval (BM25/SPLADE)
    ↓
[Reranking Module] → Cross-encoder reranking (optional but recommended)
    ↓
[Passage Formatting] → Context + query → prompt
    ↓
[Generation Module] → LLM (GPT-4, Claude, Llama, etc.)
    ↓
[Attribution Module] → Optional: cite which passages support claims
    ↓
User Response
```

### Retrieval Design Trends

#### Hybrid Search: Dense + Sparse

> [!tip]
> **Dense Retrieval** (e.g., [[DPR]], all-MiniLM): Captures semantic meaning. Fails on exact matches (proper nouns, codes, acronyms).
>
> **Sparse Retrieval** (e.g., [[BM25]], [[SPLADE]]): Lexical matching via term overlap. Handles exact entity names and rare terms.
>
> **Hybrid:** Combine both, rerank with [[Cross-Encoder|cross-encoder]], consistently outperforms either alone.

#### Reranking with Cross-Encoders

[[Multi-Stage Ranking|Multi-stage retrieval]] has become standard:
1. **Stage 1:** Fast retriever (BM25 or dense) returns top-100
2. **Stage 2:** Expensive [[Cross-Encoder|cross-encoder]] reranks top-100 → top-5

This inverts the traditional cost: reranking (per-query) often provides best quality gain per compute unit.

#### Agentic RAG

The LLM is given control to **decide when and what to retrieve**:

```
LLM: "I need to find information about X."
→ Retrieve(query="X")
← [Retrieved passages]
LLM: "These passages answer part of it. I need Y also."
→ Retrieve(query="Y")
← [Retrieved passages]
LLM: "Now I can synthesize the answer."
→ Return answer with citations
```

Enables multi-hop reasoning and self-directed search.

#### Context Curation

With 100k+ context windows, the bottleneck is no longer *fitting* passages but *filtering* noise:

- **Context compression:** Remove or summarize irrelevant parts (RECOMP, LongLLMLingua)
- **Position-aware ranking:** Place most important passages at beginning/end (mitigates "lost in the middle")
- **Retrieval with attribution:** Ensure generation cites sources, enabling verification

---

## 6. Challenges in RAG

### 6.1 Multi-Hop Questions

**Problem:** Standard RAG retrieves passages by similarity to the input query. This works for single-hop questions (answer in one passage) but fails for multi-hop reasoning.

#### Example

Query: "What movie did the actor who played X star in after 2020?"

- Simple retrieval: "Find passages about actor X"
- Multi-hop required: Find actor X → Find films post-2020 → Synthesize

#### Solutions

1. **Graph-Structured Retrieval** (HopRAG, Liu 2025)
   - Build passage graphs at indexing: vertices = text chunks, edges = LLM-generated pseudo-queries
   - At retrieval: Start from top-k, follow graph edges using LLM reasoning

2. **Iterative/Agentic Retrieval**
   - Interleave retrieval and reasoning steps
   - LLM decides when to retrieve again (e.g., Search-R1, Jin 2025)

3. **Adaptive Retrieval**
   - Classify query complexity
   - Route simple queries → single-hop; complex → multi-hop pipeline (Adaptive-RAG, Jeong 2024; [[Self-RAG]])

---

### 6.2 Lost in the Middle

**Phenomenon:** [[Language Model for IR|Language models]] attend poorly to information in the middle of long contexts.

#### Finding (Liu et al., 2024)

Multi-document QA experiment: place a gold (relevant) passage at varying positions among k retrieved documents.

**Result:** U-shaped attention curve
- Models attend well to **beginning** and **end** of context
- Information in the **middle is substantially ignored**
- Persists across models (GPT-3.5, Claude, MPT-30B)
- **Scales with context length** — larger windows → worse middle effect

#### Mitigations & Status (2024–2025)

| Mitigation | Mechanism | Status |
|---|---|---|
| **Reranking** | Cross-encoder pushes relevant docs to top | Effective but not complete |
| **Context compression** | Remove/summarize irrelevant sections; makes relevant text appear closer | Limited gains |
| **FiD** | Encode passages independently; bypass attention decay | Architectural change required |
| **Position-aware ordering** | Place best evidence at beginning + end; reduce k | Heuristic, not principled |

**Current Status:** U-shape persists. Compressing position indices per attention head helps slightly (Zhang 2024), but gains remain modest. **"Context rot"** effect: semantically similar distractors hurt much more than unrelated filler (Chroma 2025).

> [!warning]
> The "lost in the middle" problem is NOT solved. Production RAG systems must either:
> - Use reranking to keep relevant passages in top positions
> - Reduce retrieved passage count (fewer passages = less middle effect)
> - Accept quality degradation for cheaper inference

---

### 6.3 Semi-Relevant Documents as Distractors

**Problem:** When the retriever returns passages that are partially related but incorrect, they actively harm generation quality.

#### Cuconasu et al., 2024: "The Power of Noise"

Experiment: Add semi-relevant "distractor" passages to QA contexts.

**Finding:** Semi-relevant distractors degrade answer quality **more than completely unrelated documents**.

**Reason:** The model sees superficially similar content and gets confused about which passages actually support the answer. Completely unrelated passages are easier to ignore.

**Implication:** Retrieval quality matters more than retrieval quantity. One precise passage > 20 mediocre passages.

---

### 6.4 Hallucination Under Insufficient Context

> [!warning]
> **Even with RAG, LLMs can hallucinate** when:
> - Retrieved passages don't answer the question
> - Passages conflict with each other
> - The model decides to "synthesize" beyond the retrieved context

This is **semantic hallucination** (factually false output despite retrieved evidence) vs. **parametric hallucination** (false facts from parameters). RAG reduces parametric hallucination but cannot eliminate semantic hallucination.

---

### 6.5 Evaluation of RAG

RAG evaluation is fundamentally harder than standard retrieval or generation evaluation:

| Challenge | Why | Implication |
|---|---|---|
| **No stable ground truth** | Free-form answers lack a single correct reference | Multiple valid answers confuse metrics |
| **Attribution difficulty** | Decomposing answers into claims and verifying each against sources is itself a hard problem | Hard to isolate retrieval vs. generation failures |
| **Component interactions** | Strong generator masks retrieval failures; strong retriever compensates for weak generator | Can't evaluate in isolation |
| **LLM-as-judge circularity** | Using an LLM to evaluate LLM output inherits biases and creates correlated failures | Risk of overoptimizing for LLM preferences, not human quality |
| **Evaluation cost** | LLM-based claim verification is expensive | Cheaper proxies (lexical overlap, etc.) may not correlate with quality |

#### RAGAS: Automated Evaluation of RAG (Es et al., 2024)

Framework for component-wise evaluation without large human-annotated datasets:

> [!definition]
> **RAGAS** (Retrieval-Augmented Generation Assessment) uses LLMs as judges to evaluate RAG pipeline components.

**Retrieval Quality:**
- **Context Precision:** Are relevant chunks ranked higher than irrelevant ones?
- **Context Recall:** Was all necessary information retrieved?

**Generation Quality:**
- **Answer Faithfulness:** Is the answer grounded in the retrieved context? (No hallucination)
- **Answer Relevancy:** Does the answer address the question?

**Workflow:**
1. Ask LLM to decompose answer into claims
2. For each claim, ask if it's supported by retrieved passages
3. Aggregate into component scores

**Advantage:** Doesn't require ground-truth reference answers; only requires the retrieved passages and generated answer.

---

### 6.6 Faithful Generation and Attribution

Recent work (2025) reveals a critical gap: **correct answers ≠ faithful citations**.

#### Wallat et al., 2025: "Correctness is not Faithfulness in RAG Attribution"

**Setup:** Multi-choice QA with retrieved documents. Vary document content while keeping the answer text the same.

**Scenarios:**

| Scenario | Retrieved Docs | Answer | Correct? | Faithful? |
|---|---|---|---|---|
| A | Gold passage present | "Capital is Berlin" | ✅ | ✅ |
| B | Unrelated passage | "Capital is Berlin" | ✅ | ❌ (hallucination) |
| C | Adversarial passage (says "capital is Munich") | "Capital is Berlin" | ✅ | ❌ (ignores retrieval) |

**Key Finding:** Models cite passages post-hoc to rationalize answers they've already generated, rather than deriving answers from citations. **Post-rationalization bias.**

#### Citation Behavior: Scaling Effects

Experiment: Scale component magnitudes to steer citation behavior.

```python
# More aggressive retriever → retrieves more passages
# More aggressive critic → demands citations
# Result: Can increase citation rate from 30% → 90%
#         But do citations become more faithful?
```

**Result:** Higher citation rates do NOT guarantee higher faithfulness. Models simply cite more passages without necessarily improving grounding.

#### Implications

1. **Citation rate is not a proxy for quality.** An LLM that cites everything may produce less faithful outputs than one that cites selectively.
2. **User trust can be misplaced.** Users may overestimate accuracy of highly-cited answers (Sadeghi et al., 2024: especially for non-technical users).
3. **Need explicit fidelity constraints:** Training objectives must enforce that generation actually *uses* retrieval, not just cites it.

---

## 7. Conclusion: RAG Trade-offs & Cost Analysis

RAG systems involve multiple stages, each with different computational costs and quality–cost trade-offs.

### Cost Breakdown

| Stage | Cost per Query | When Performed | Key Consideration |
|---|---|---|---|
| **Indexing** (embed corpus) | — | Offline | Amortized cost; paid again on corpus update |
| **Retrieval** (dense ANN + BM25) | $ | Per query | Fast; rarely bottleneck |
| **Reranking** (cross-encoder) | $$ | Per query | Often best quality gain per compute unit |
| **Generation** (LLM) | $$$ | Per query | Dominates cost; scales with input tokens |
| **Agentic loops** | $$$$ | Per query | Multiplies generation (3–5× LLM calls) |

### Key Insight: Better Retrieval Pays for Itself

**Principle:** Retrieving 5 precise passages is vastly better than retrieving 20 mediocre ones.

**Why:** 
- Fewer input tokens → faster generation → reduced LLM cost
- Cleaner context → better answer quality
- Reduced "lost in the middle" effect
- Less distraction from semi-relevant passages

**Recommendation:** Invest compute budget in retrieval quality (hybrid search, reranking) rather than passage quantity. The savings on generation cost offset reranking expense while improving answer quality.

### Modern RAG Best Practices

1. **Hybrid retrieval:** Dense + sparse, reranked
2. **In-context over tight coupling:** Use large context windows; no reason to build FiD unless optimizing for older models
3. **Agentic loops for complex queries:** Let the LLM control retrieval for multi-hop reasoning
4. **Attribution as a training objective:** Enforce citations to prevent post-rationalization
5. **Component evaluation:** Use RAGAS or similar to diagnose whether failures are retrieval or generation

---

## Key Takeaways

> [!summary]
> **RAG Definition:** Language model + external datastore at test time. Trades inference cost for factuality, freshness, and interpretability.
>
> **Basic Architectures:**
> - **RAG (2020):** Marginalizes over documents; learnable retriever; slow at inference
> - **FiD (2021):** Independent passage encoding + decoder fusion; scales to 100+ passages
> - **In-context RAG (2023):** Retrieval → formatting → LLM; standard in production
> - **Atlas (2023):** Periodic reindexing; end-to-end learning; expensive but strong
> - **Self-RAG (2023):** Adaptive retrieval with reflection tokens; efficient inference
>
> **Modern Consensus (2024–2025):**
> - Decoupled retrieval + generation
> - Hybrid (dense + sparse) + reranking pipeline
> - Agentic loops for complex reasoning
> - In-context over tight coupling
>
> **Critical Challenges:**
> - Multi-hop questions need iterative/agentic approaches
> - Lost in the middle: models ignore middle of contexts (reranking + fewer passages help partially)
> - Semi-relevant distractors hurt more than irrelevant ones (quality > quantity)
> - Correct answers ≠ faithful citations; models post-rationalize
> - Evaluation is hard; no single metric; RAGAS provides component-wise assessment
>
> **Cost Principle:** Invest in retrieval quality (reranking, hybrid search) to reduce generation cost and improve quality. Fewer, better passages beat many mediocre ones.

---

## References

[1] Patrick Lewis et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks." *Advances in Neural Information Processing Systems*, 2020.

[2] Gautier Izacard and Edouard Grave. "Leveraging passage retrieval with generative models for open domain question answering." *EACL*, 2021.

[3] Gautier Izacard et al. "Atlas: Few-shot learning with retrieval augmented language models." *Journal of Machine Learning Research*, 2023.

[4] Ori Ram et al. "In-context retrieval-augmented language models." *TACL*, 2023.

[5] Akari Asai et al. "Self-rag: Learning to retrieve, generate, and critique through self-reflection." *ICLR*, 2023.

[6] Nelson F Liu et al. "Lost in the middle: How language models use long contexts." *TACL*, 2024.

[7] Florin Cuconasu et al. "The power of noise: Redefining retrieval for RAG systems." *SIGIR*, 2024.

[8] Shahul Es et al. "RAGAS: Automated evaluation of retrieval augmented generation." *EACL*, 2024.

[9] Yixuan Tang and Yi Yang. "Multihop-RAG: Benchmarking retrieval-augmented generation for multi-hop queries." *arXiv*, 2024.

[10] Jonas Wallat et al. "Correctness is not Faithfulness in RAG Attribution." *ICTIR*, 2025.

[11] Ian van Dort and Maria Heuss. "How do LLMs cite? A mechanistic interpretation of attribution in RAG." *ECIR*, 2026.

[12] Hao Liu et al. "HopRAG: Multi-hop reasoning for logic-aware retrieval-augmented generation." *ACL*, 2025.

[13] Bowen Jin et al. "Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning." *arXiv*, 2025.

[14] Soyeong Jeong et al. "Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity." *NAACL*, 2024.

[15] Mersedeh Sadeghi et al. "Explaining the unexplainable: The impact of misleading explanations on trust in unreliable predictions." *UMAP*, 2024.

