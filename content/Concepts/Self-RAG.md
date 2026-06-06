---
type: concept
aliases: [Self-Reflective RAG, Self-Reflective Retrieval-Augmented Generation]
course: [IR]
tags: [neural-ir, exam-topic]
status: complete
---

# Self-RAG

## Definition

> [!definition] Self-RAG (Self-Reflective Retrieval-Augmented Generation)
> **Self-RAG** (Asai et al., 2023) is a framework that trains a single language model to **adaptively retrieve on demand** and to **critique its own generation** through special **reflection tokens**. Unlike standard [[Retrieval-Augmented Generation|RAG]], which always retrieves a fixed number of passages, Self-RAG lets the model decide *whether* to retrieve, judge *whether* retrieved passages are relevant, verify *whether* its claims are supported, and rate the overall *utility* of its output — all by emitting tokens from an expanded vocabulary.

## Intuition

> [!intuition] Retrieval as a Learned Decision, Not a Fixed Step
> Standard RAG bolts a retriever onto a generator and always feeds back top-$k$ passages — wasteful for simple queries ("what is 2+2?") and risky when irrelevant passages distract the model. Self-RAG instead folds retrieval *control* and *self-critique* into the model's own token stream.
>
> The model learns to emit a `[Retrieve]` token only when external evidence would help, then learns to grade what comes back (`[IsRel]`), check whether its sentence is actually grounded in that evidence (`[IsSup]`), and score how useful the answer is (`[IsUse]`). At inference these critique tokens become **soft scores** that re-rank generation candidates, so the model can prefer outputs that are both relevant and faithful — without a separate reranker or critic model running alongside it.

## Mathematical Formulation

Self-RAG augments the language model vocabulary $V$ with reflection tokens and factors the output distribution over text **and** reflection decisions. For an input $x$ and (optionally) preceding generation $y_{<t}$, the model $M$ defines:

$$
p_M(y_t, r_t \mid x, d, y_{<t}), \qquad r_t \in \{\texttt{Retrieve}, \texttt{IsRel}, \texttt{IsSup}, \texttt{IsUse}\}
$$

where:
- $x$ — input query / task prompt
- $d$ — a retrieved passage (absent when no retrieval was triggered)
- $y_t$ — the generated text segment at step $t$ (typically one sentence)
- $r_t$ — a reflection token drawn from the four critique types
- `Retrieve` $\in \{$ yes, no, continue $\}$ — whether to call the retriever for segment $t$
- `IsRel` $\in \{$ relevant, irrelevant $\}$ — is passage $d$ relevant to $x$?
- `IsSup` $\in \{$ fully, partially, no support $\}$ — is segment $y_t$ supported by $d$?
- `IsUse` $\in \{1,\dots,5\}$ — overall utility of the response to $x$

**Training (standard LM loss over the augmented corpus).** A critic model $C$ first labels an offline corpus with reflection tokens; the generator $M$ is then trained with ordinary next-token prediction over the interleaved (text + reflection token + passage) sequences:

$$
\mathcal{L}(M) = -\sum_{(x,y,d,r)} \log p_M\big(y, r \mid x, d\big)
$$

where retrieved passage tokens $d$ are **masked from the loss** (the model conditions on them but is not trained to reproduce them).

**Inference (segment-level beam search with critique reranking).** When a `[Retrieve]=yes` token is emitted, $K$ passages are fetched and each spawns a candidate continuation. Each candidate segment $y_t$ is scored by combining its LM probability with a weighted sum of its critique-token probabilities:

$$
\text{score}(y_t) = p_M(y_t \mid x, d, y_{<t}) \;+\; \sum_{G \in \{\text{IsRel},\,\text{IsSup},\,\text{IsUse}\}} w_G \, s_G
$$

where:
- $s_G$ — the model's normalized probability of the **desirable** value of critique token $G$ (e.g. `IsSup`=fully)
- $w_G$ — a tunable weight controlling how strongly faithfulness vs. utility steer decoding (set at test time, no retraining)

The top-scoring segment is committed, then decoding continues to the next segment.

## Key Properties / Variants

- **Adaptive (on-demand) retrieval:** the `[Retrieve]` token gates retrieval per segment, so simple queries skip retrieval while knowledge-intensive ones trigger it — fewer retriever calls, cheaper inference.
- **Self-critique via reflection tokens:** relevance (`[IsRel]`), support/faithfulness (`[IsSup]`), and utility (`[IsUse]`) are learned and emitted by the *same* model, giving an explicit, inspectable trace of "did I retrieve? is it relevant? is my claim grounded?"
- **Controllable at test time:** the critique weights $w_G$ let you trade faithfulness against fluency/utility per task without retraining — soft constraints rather than hard decoding rules.
- **Lightweight:** runs at 7B–13B parameters; a Self-RAG 13B model matches or beats much larger (70B+) instruction-tuned baselines on several QA and long-form benchmarks while issuing fewer retrieval calls.
- **Two trained models:** a **critic** $C$ (distilled from a strong teacher like GPT-4) only labels the offline corpus; the deployed **generator** $M$ internalizes critique, so no separate critic runs at inference.
- **Contrast with neighbours:** uses ordinary token-level LM training instead of marginalizing over passages like Lewis-style [[RAG]] or fusing them in the decoder like [[FiD]]; it addresses *when/what to retrieve and self-verification*, complementary to learnable-retriever methods like [[Atlas]].

```pseudo
Algorithm: Self-RAG Inference (segment-level)
──────────────────────────────────────────────
Input: query x, generator M, retriever R, beam parameters, weights {w_G}
Initialize generation y ← ""

Loop for each output segment t:
  Predict Retrieve token from M(x, y_<t)
  if Retrieve == "no":
    generate next segment y_t directly from M     # use parametric knowledge
  else:
    D ← R(x, y_<t)                                 # retrieve K passages
    Candidates ← {}
    for each passage d in D:
      y_t^d ← M(x, d, y_<t)                        # candidate continuation
      compute critique probs: IsRel, IsSup, IsUse
      score(y_t^d) ← log p_M(y_t^d) + Σ_G w_G · s_G(y_t^d)
      add (y_t^d, d) to Candidates
    y_t ← argmax_{y_t^d} score(y_t^d)              # critique-weighted rerank
  append y_t (and its reflection tokens) to y
  until end-of-sequence
return y
```

## Connections

- Type of: [[Retrieval-Augmented Generation]] (adaptive / self-reflective variant)
- Contrasted with: [[RAG]] (fixed marginalization), [[FiD]] (decoder fusion), [[Atlas]] (learnable retriever via reindexing), [[Agentic RAG]] (LLM-driven multi-step retrieval)
- Uses: [[Dense Retrieval]] / [[Bi-Encoder]] retriever for the `[Retrieve]` step, [[Cross-Encoder]]-style relevance judgement absorbed into `[IsRel]`
- Tackles: hallucination and faithfulness, adaptive retrieval for multi-hop queries (cf. Adaptive-RAG)
- Built on: [[Transformers]] / [[Language Model for IR|LLM]] backbones (Llama2-7B/13B)
- Related concern: faithful generation and attribution (correctness ≠ faithfulness)

## Appears In

- [[IR-L09 - RAG]]
