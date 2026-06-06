---
type: lecture
course: RecSys
lecture: 3b
date: 2026-06-04
topics:
  - "[[Generative Recommendation]]"
  - "[[Large Recommendation Models (LRM)]]"
  - "[[LLM-based Generative Recommendation]]"
  - "[[Item Tokenization]]"
  - "[[Semantic IDs]]"
  - "[[RQ-VAE]]"
  - "[[Scaling Laws]]"
  - "[[HSTU]]"
  - "[[Next-Item Prediction]]"
  - "[[Collaborative Filtering]]"
  - "[[LoRA]]"
  - "[[Direct Preference Optimization (DPO)]]"
status: complete
---

# RS-L03b - From LLMs to LRMs

> [!overview] Lecture in one breath
> This is the second half of Lecture 3, by **Yuyue Zhao**. It traces the field's move from **discriminative** recommendation (learn a scoring function $f(\text{user}, \text{item})$ and rank a fixed candidate pool) to **[[Generative Recommendation]]** (directly *produce* the target item, token by token). Two distinct routes to "generative" are contrasted throughout:
> - **Part 1 — [[LLM-based Generative Recommendation]]:** treat recommendation as a *language task*. Borrow scaling from language models by squeezing behavior data into something an LLM can read. The central problems are **alignment** (injecting collaborative signal the LLM never saw) and **[[Item Tokenization]]** (turning items into discrete tokens an LLM can generate).
> - **Part 2 — [[Large Recommendation Models (LRM)]]:** the opposite move. Design *native* architectures for behavior data and let a recommendation-specific **[[Scaling Laws|scaling law]]** emerge there.
>
> Primary source cited throughout: **Hou et al., "A Survey on Generative Recommendation: Data, Model, and Tasks," arXiv:2510.27157, 2025** (§4.1 = LLM-based GR, §4.2 = LRMs). This course has **no textbook** — the slides are the only source.

**Lecturer:** Yuyue Zhao · **Course:** RecSys (UvA MSc AI), Lecture 3 (Part 2) · **Date:** 2026-06-04

---

## 0. The storyline and learning objectives

The whole lecture follows a 4-stage roadmap:

```
[1] Paradigm review        [2] How LLMs are        [3] How RecSys scales      [4] Comparison
    Discriminative   --->      used for RecSys -->     natively              -->  & Conclusion
    -> Generative              (prompt, align,         (LRM, scaling,             (when each
                                tokenize)               long sequences)            route wins)
```

By the end you should be able to answer **four questions** (these map directly onto exam-worthy content):

- **Q1. Why should recommendation "generate" instead of "score"?** — the shift from discriminative scoring to direct target generation.
- **Q2. How does an LLM get aligned to a recommendation task?** — three alignment paradigms: text prompting, collaborative-signal injection, item tokenization.
- **Q3. How do items become tokens an LLM can generate?** — [[Item Tokenization]]: from [[Atomic Item IDs|atomic IDs]] to learned codebook **[[Semantic IDs|semantic IDs]]**.
- **Q4. What is the essential difference between LRM and LLM-based GR?** — *borrowed* language scaling vs. *native* recommendation scaling.

---

## 1. Discriminative vs. Generative recommendation

The core paradigm contrast (Hou et al. 2025, §4, Fig. 1).

| | **Discriminative** | **Generative** |
|---|---|---|
| **Mechanism** | Learn a scoring function $f(\text{user},\text{item})$; score every candidate | Directly **generate** the target item / content, token by token |
| **Strengths** | Mature, optimized for ranking | World knowledge, semantic understanding, scaling law, creative / cross-domain transfer |
| **Limitations** | Fixed candidate pool; weak cold-start; hard to explain; stacking gives diminishing returns | Must *ground* output to real items; heavier compute |

**Discriminative data flow (bottom-up):**

```
        Matching Score
              ^
       Matching Function          <- trainable (the "discriminative recommender")
        ^            ^
  Representation  Representation   <- Representation Learning
        ^            ^
      User          Item
```

**Generative data flow (bottom-up):**

```
        Recommend Items
              ^
   Generative Model (e.g., LLM, Diffusion)   <- can be trainable OR frozen
              ^
   User Historical Interaction Data
```

> [!intuition] The one-line contrast
> Discriminative = "**score** every candidate against a **fixed pool**." Generative = "**directly produce** the target." Generation lets us escape the fixed candidate pool and lean on pre-trained world knowledge, but forces a new problem: making sure what we generate actually *exists* in the catalog.

---

# PART 1 — LLM-Based Generative Recommendation

> Framing: *Recommendation as a language task — translate users, items, and histories into something a pretrained language model can read, and generate.*

## 2. Three research lines for LLM-based GR

(Hou et al. 2025, §4.1 taxonomy.)

1. **LLM directly as RecSys** — *no extra fine-tuning.* Drive recommendation through prompt design and [[In-Context Learning|in-context learning]]. Best for cold-start and cross-domain. (See §3.)
2. **Align an LLM to the recommendation task** — adapt the model so it carries collaborative signal and item structure. (See §4.)
3. **Training objectives & inference** — *what loss · how to decode.* Choose among SFT / SSL / RL / DPO; ensure generated items actually exist. (See §6.)

---

## 3. Line 1 — Use a pretrained LLM directly

*Prompt design + in-context learning — strong on cold-start, cross-domain.* (Hou et al. 2025, §4.1.1; e.g. **Chat-REC**, Gao et al. 2023.)

Two sub-approaches:

- **[[LLM-as-Enhancer]]:** rewrite user/item profiles and history into natural-language features; feed the enriched features into a CF / sequential / re-ranking model. The LLM augments behavior data with external knowledge.
- **[[LLM-as-Recommender]]:** use a templated prompt; the LLM directly outputs item titles or IDs. Zero-shot transfer across scenarios; extends to multimodal LLMs (e.g. poster images).

**Pros:** zero training cost; strong semantics and world knowledge; good cold-start / cross-domain generalization.
**Cons:** sensitive to prompt phrasing; no collaborative signal; may **hallucinate non-existent items** → this motivates "generation grounding" (item tokenization, §5).

### 3.1 Diagrams

**LLM-as-Enhancer** (ref: *Towards Open-World Recommendation with Knowledge Augmentation from LLMs*):

```
   +-------------------+        +-------------+
   | User Behavior Data| <----> | RecSys Model|
   +-------------------+        +-------------+
            ^
            | augments with external knowledge
   +------------------------------------------+
   | Reasoning Knowledge | Factual Knowledge   |
   | (LLMs, Knowledge Graphs, Multi-domain)    |
   +------------------------------------------+
```

**LLM-as-Recommender** (**Chat-REC**, Gao et al. 2023) — a routing flowchart:

```
 Recommendation System R ┐
 User-Item History       │
 User Query Q_i          ├──> Prompt Constructor C ──> ChatGPT ──> "Use RecSys?"
 User profile            │                                              │
 History of Dialogue H  ─┘                                     No ──> Output A_i
                                                               Yes -> RecSys Candidate
                                                                       Set Construction
                                                                          │
                                                       ┌──────────────────┘
                                                       v
                                            History of Recommendation R
                                            + Intermediate Answer A  (looped back)
```

> [!example] Chat-REC dialogue
> User asks for action movies → ChatGPT recommends *Fargo / Heat / Die Hard* and **explains *why*** Fargo was recommended, drawing on the user's movie history and personal information. This is the "explainable + conversational" sweet spot of using a frozen LLM directly.

---

## 4. Line 2 — Align an LLM to recommendation (the most important slide of §4.1)

(Hou et al. 2025, §4.1.2, Fig. 6 — the alignment taxonomy.)

### 4.1 Why alignment is necessary

LLM pre-training does **NOT** carry the things recommendation actually needs:

- click / interaction signal,
- a Top-K ranking objective,
- long-tail coverage incentives,
- exposure-bias awareness.

→ So a vanilla LLM often **loses to a specialized rec model**. **Solution: fine-tune on recommendation data.** The taxonomy classifies alignment methods by *how the user/item profile is structured into the input*:

```
                 Three alignment paradigms
   (1) Text Prompting     (2) Inject Collab.    (3) Item Tokenization
   pure natural language      Signal                items as learned
                          language + CF info        discrete tokens
```

---

### 4.2 Paradigm ① — Text Prompting

*Profile = task description + temporal interaction history, both in pure language.* Express the entire recommendation problem in natural language; the LLM is fine-tuned (often parameter-efficiently, via [[LoRA]]) on these text instances.

> [!example] Instruction-tuning format (worked example)
> | Section | Content |
> |---|---|
> | **Task Instruction** | "Given the user's historical interactions, please determine whether the user will enjoy the target new movie by answering 'Yes' or 'No'." |
> | **Task Input** | User's liked items: *GodFather*. User's disliked items: *Star Wars*. Target new movie: *Iron Man* |
> | **Instruction Output** | No. |

**Representative methods:**
- **TALLRec** (Bao et al., RecSys'23) — explicit preference statements + lightweight [[LoRA]] fine-tuning.
- **LlamaRec** (Yue et al., 2023) — two-stage: a sequential model first narrows candidates, then the LLM ranks them.
- **Reason4Rec** — extract preferences from user reviews / reasoning chains.

**Limitation:** no collaborative signal; under-performs when inter-item dependencies dominate.

---

### 4.3 Paradigm ② — Inject Collaborative Signal

*Let the LLM see semantic AND relational knowledge at the same time.* (e.g. **iLoRA, LLaRA** (SIGIR'24), **CoLLM, CORONA, HyperLLM, CoRAL** — Hou et al. 2025, §4.1.2.)

Three sub-strategies:

| Sub-strategy | Methods | What it does |
|---|---|---|
| **Representation augmentation** | iLoRA, LLaRA, CoLLM | Project a learned **CF embedding** into the LLM's token-embedding space; concatenate it with the language tokens |
| **Summary-based** | CORONA, HyperLLM | Distill collaborative knowledge into a **short text summary** (e.g. LLM reasoning over a GNN) before feeding it in |
| **Sentence-ization** | CoRAL | Translate CF signals into readable sentences such as *"users who liked A also liked B, C, D"* — directly readable by the LLM |

**Item Representation Example (diagram):**

```
Prompt:  "This user has watched [item_1], [item_2], ..., [item_n].
          Predict the next movie this user will watch."
                       │
   each [item] token is slotted into the language token stream
                       │
              v
   Frozen LLM (snowflake)  +  LoRA (flame)   <- only LoRA params train
                       │
                       v
                  Output: [item_{n+1}]
```

> [!warning] The deeper issue (why paradigm ③ exists)
> Dense CF embeddings are **not directly readable** by an LLM. We must *project*, *summarize*, or *verbalize* them — which exposes a **fundamental gap between collaborative semantics and language semantics**. Closing this gap is exactly what paradigm ③ (item tokenization) attempts.

---

### 4.4 Paradigm ③ — Item Tokenization

*How do we give every item an identifier that is compact, semantic, AND generable by an LLM?* (e.g. **P5** (Geng et al. RecSys'22), **BIGRec/M6**, **TIGER** (Rajput et al. NeurIPS'23), **LC-Rec, LETTER, TokenRec, CCFRec, SIIT** — Hou et al. 2025, §4.1.2.)

Five levels (L1–L5) of item identifiers:

| Level | Name | Methods | What it does | Trade-off |
|---|---|---|---|---|
| **L1** | ID-based | **P5, CLLM4Rec** | one special token per item | vocabulary blows up, no semantics |
| **L2** | Text-based | **BIGRec, M6** | use item title / description | very long, no collaborative info |
| **L3** | **Codebook-based** | **TIGER, LC-Rec** | discrete **[[Semantic IDs]]** from **[[RQ-VAE]]** | **compact + semantic** |
| **L4** | Codebook + CF | **LETTER, TokenRec, CCFRec** | inject CF signal into the quantizer | language + collaboration in one ID |
| **L5** | Adaptive | **SIIT** | LLM refines the identifier during training | tokens evolve with the model |

The highlighted row is **L3 (codebook-based [[Semantic IDs]])** — the modern default. It uses [[RQ-VAE|residual-quantized VAE]] to turn an item's content embedding into a short tuple of discrete codes.

**Two readings of "Semantic ID":**
- **(a)** Use the LLM-era representation trick *into* rec, as a seq2seq recommendation task trained **from scratch**. Pipeline: `Item content → Embedding → Quantization → SID`.
- **(b)** An LLM is **tuned over** the semantic IDs — e.g. **LC-Rec**, Open **OneRec**.

**Open OneRec framework (diagram):** three phases — **Pre-Training, Post-Training, Evaluation** — all driven by a **Qwen LLM decoder**. Each short video is passed through a **Tokenizer** emitting **itemic tokens** of the form `<item_a_1><item_b_1><item_c_1>` (a hierarchical multi-codeword [[Semantic IDs|semantic ID]]; three codebook levels a/b/c) interleaved with **text tokens**.

```
... "After searching for baking tools and purchasing
     <item_a_1><item_b_1><item_c_1>  and click  <item_a_2><item_b_2><item_c_2>,
     she will next click ..."   <- decoder generates next itemic tokens autoregressively
                                    in Evaluation, generated codes are MAPPED back to real videos
```

This demonstrates items represented as **multi-level discrete codes** the decoder generates [[Autoregressive Generation|autoregressively]].

---

### 4.5 Three alignment paradigms — cross-comparison

(Adapted from Hou et al. 2025, §4.1.2, Tables 2/3/4.)

| Paradigm | Profile Format | Collaborative Signals? | Typical Backbone | Representative Models |
|---|---|---|---|---|
| **① Text Prompting** | Natural Language | **No** | LLaMA, ChatGLM | TALLRec, LlamaRec |
| **② Injecting CF** | Text + Dense Embeddings | **Yes (injected)** | LLaMA, Vicuna | CoLLM, CORONA, CoRAL |
| **③ Item Tokenization** | [[Semantic IDs]] / Tokens | **Yes (via codebook)** | T5, LLaMA | P5, TIGER, BIGRec |

---

## 5. Line 3 — Training objectives & inference

### 5.1 Training objectives

*The task is [[Next-Item Prediction|next-item prediction]] — but you can shape the loss four ways.* (Hou et al. 2025, §4.1.3, Table 5.)

| Objective | Methods | Idea | Trade-off |
|---|---|---|---|
| **[[Supervised Fine-Tuning (SFT)\|SFT]]** | P5, LGIR | learn only positive samples | simple, but no explicit negatives → ranking margin hard to learn |
| **SSL (self-supervised)** | FELLAS, EasyRec | [[Contrastive Learning|contrastive learning]] | reduces template dependence, improves zero-shot transfer |
| **[[Reinforcement Learning\|RL]]** | LEA, RPP | reward-driven; can encode explicit negatives + non-differentiable metrics | needs feedback, unstable to train |
| **[[Direct Preference Optimization (DPO)\|DPO]] / preference** | LettinGo, RosePO, SPRec | optimize on (preferred vs rejected) pairs directly | no reward model needed; training is stable |

### 5.2 Inference strategies

*Once trained, how does the system actually serve recommendations?* (Hou et al. 2025, §4.1.3.)

| Strategy | What it does | Representative work | Trade-off |
|---|---|---|---|
| **Direct inference** | send the prompt, decode an item | simplest pipeline | prompt-sensitive; no ranking signal; long histories blow up the context |
| **Rerank-style** | generate / score over a candidate set from a traditional retriever | RecRanker (two-stage, position-bias correction), LLM4Rerank, GFN4Rec | quality bounded by the upstream candidate generator |
| **Acceleration** | make LLM-based serving feasible at scale | FELLAS (LLM only for embeddings); GenRec (distill prompts); AtSpeed (**speculative decoding**, 2–2.5× speed-up at same Top-K) | engineering complexity grows |

---

## 6. Part 1 summary

> [!summary] LLM-based GR in one sentence
> **LLM-based generative recommendation = translating the recommendation problem into a language task.** Two challenges to remember:
> 1. **Alignment** — how do we inject collaborative signal the LLM never saw during pre-training?
> 2. **Item Tokenization** — how do items become discrete tokens the LLM can generate?

---

# PART 2 — Large Recommendation Models (LRM)

> *Building recommendation's own scaling law.* Where **LLM-based GR** borrows scaling from **language** (and must squeeze behavior data into text), **LRM** does the opposite: it designs **native architectures for recommendation data** and lets the scaling law emerge there.

## 7. Industrial bottlenecks that motivated LRMs

Header: *"Over the past decade, the rapid advancements in deep learning... have largely been driven by the growth in computational scale."* (Hou et al. 2025, §4.2.)

### 7.1 The two scaling curves

**LLM parameter scale (log-scale y-axis, params in B):**

```
params(B) (log)
  10^3 |                                  * GPT-4  (~1760B)
       |                       * GPT-3.5 (175B)
  10^0 |        * GPT-2 (1.5B)
       | * GPT (0.12B)
       +----------------------------------------> time
```

> Compared to GPT-2, **GPT-3.5 has ~100× more parameters**.

**RecSys efficiency vs. compute (a generalized [[Scaling Laws|scaling law]]):**

```
Accuracy
   ^                                    .--?  (Use long-sequence modeling:
   |                              ___,--'        DIN'18, DIEN'19, SIM'20, TWIN'23)
   |                    Deep Learning models
   |               ,--'  (Deep Crossing'16, DeepFM'16, Wide&Deep'16, DCN'17)
   |          Linear model (LogReg 2007, FM 2010)
   |      Rule-based (Grundy, Mass Marketing)
   +------------------------------------------------> FLOPS per item
        10^5            10^7              10^10
```

> The efficiency improvements of RSs in the past have **also followed a generalized scaling law** (a sigmoid in accuracy vs. compute-per-item).

### 7.2 The cascaded pipeline and its three bottlenecks

```
Infra & Data            "Heavy Comms & Storage"
(User/Item Data,  ----------------------------->  CascadedRec funnel:
 Log, Server)
                  Retrieval        Pre-rank        Rank
                  ~10^8 cands  -->  ~10^4      -->  ~10^3
                  (Rule-1,2,        (Model-1         (final
                   Model-i,j)        ...Model-j)      ranking)
                  <----------- funnel narrows ----------->
```

Three bottlenecks that motivated rebuilding the stack:

- **Fragmented Computing:** overall **Model FLOPs Utilization (MFU)** of industrial RSs is just **0.1%–1%**, whereas LLM inference reaches up to **~70%**.
- **Inconsistent Objectives:** balancing **hundreds** of optimization objectives degrades system consistency and efficiency.
- **Technological Gap:** an **architectural disconnect** with technologies already validated in the LLM domain.

---

## 8. Choices of industry — three tiers of scaling strategy

(Company in parentheses.)

1. **Pointwise scaling of the cascaded pipeline:**
   - *More data — lifelong sequences:* **TWIN V2** (Kuaishou), **LONGER** (ByteDance/TikTok).
   - *Better architecture — ranking model:* **Wukong** (Meta), **RankMixer / Zenith** (ByteDance).
2. **Joint scaling of a unified architecture:**
   - *Joint sequential & non-sequential features:* **OneTrans, MixFormer, HyFormer** (ByteDance).
   - *Unified scaling backbone:* **UniMixer** (Kuaishou), **HSTU / KunLun** (Meta), **MTGR** (Meituan), **UniScale** (Taobao/Alibaba).
3. **LLM-style end-to-end scaling:** `Data ⇒ Pre-Training ⇒ (Mid-Training) ⇒ Post-Training ⇒ Test Time`. Examples: **OneRec, OneRec-V2, OneRec-Think** (Kuaishou), **PLUM** (Google), **GPR** (Tencent).

### 8.1 The LRM landscape (taxonomy grid)

```
DATA scaling
  ├─ Sequence length:  LONGER, TWIN-V2, SIM ...   (TikTok, Kuaishou)
  └─ Feature dimension: Wukong ...                (Meta)

MODEL scaling
  ├─ Attention-oriented: HSTU, KunLun ...         (Meta)
  └─ FFN-oriented:       RankMixer, UniMixer ...  (TikTok, Kuaishou)

Unified Scaling (spans both): OneTrans, MixFormer, HyFormer, MTGR, UniScale ...
   (TikTok, Kuaishou, Meituan, Taobao)
```

> Footnote: a separate route — **end-to-end generative** (covered next course) — OneRec / -V2 / -Think · GPR · PLUM.

---

## 9. The recurring LRM synthesis — "scale up, stay inside the latency budget"

This table reappears each time a new model is introduced. The organizing axes are **data** (seq. length / feature dim.) × **model** (attention- vs FFN-oriented). The **recurring trick** is always the same:

> [!intuition] The universal LRM trick
> **Cheaper / approximate ops · cache & reuse · raise MFU → reclaim compute, then spend it on scale** — all while staying inside the serving latency budget. Effectively: drag RecSys's MFU from ~0.1–1% toward LLM-level ~70%, and pour the freed FLOPs into longer sequences, higher-order interactions, or more parameters.

| Paper | What it scales | How it stays inside the latency budget |
|---|---|---|
| **LONGER** | Data · sequence length | KV-cache (encode user once, reuse across candidates) + token-merge. *Result:* near-linear; throughput loss −40% → −6.8% |
| **Wukong** | Data · feature dimension (interaction order) | Stacked FM blocks + embedding-wise low-rank crosses. *Result:* high-order interactions stay affordable |
| **[[HSTU]]** | Model · attention-oriented (+ long behavior seq.) | Pointwise (non-softmax) attention + ragged fused-GEMM kernels. *Result:* 5–15× over FlashAttention-2 |
| **RankMixer** | Model · FFN-oriented (drops self-attention) | Parameter-free token-mixing + per-token FFN. *Result:* MFU 4.5% → 45%, latency flat |
| **OneTrans** | Both axes · one backbone | Causal attention + cross-request KV-cache + FlashAttention. *Result:* scales like an LLM, serves like one |

Reference IDs: HSTU arXiv:2402.17152 · Wukong 2403.02545 · RankMixer 2507.15551 · LONGER 2505.04421 · OneTrans 2510.26104.

---

## 10. Long-sequence scaling line — a brief history → LONGER

(Hou et al. 2025, §4.2; TWIN-V2: Si et al., KDD'24.)

| Year | Model | Idea |
|---|---|---|
| 2018 | **DIN** (Zhou et al., KDD) | Attention over user history to localize relevant items for a candidate |
| 2019 | **DIEN** (Zhou et al., AAAI) | Add an interest-evolution layer on top of DIN's attention |
| 2020 | **SIM** (Pi et al., CIKM) | Search-based retrieval over very long histories |
| 2024 | **TWIN / TWIN-V2** (Kuaishou, KDD'24) | Scale ultra-long sequence modeling for CTR prediction at industrial scale |

> [!intuition] Common thread → the gap LONGER fills
> All these methods retrieve a **short relevant slice** of a long history. **None genuinely models the full ultra-long sequence end-to-end.** LONGER picks up exactly that gap.

### 10.1 LONGER — long sequences, end to end (ByteDance, RecSys 2025)

*Model the ultra-long history end-to-end, efficiently, without a retrieval shortcut.* (Chai et al., RecSys 2025; arXiv:2505.04421.)

**Key mechanisms:**
- **Global tokens** (target item, user/CLS) with a **full receptive field** — an **attention sink** that stabilizes long-context attention.
- **Token Merge** + small inner Transformers + hybrid attention.
- **KV-cache serving:** encode the user sequence once, reuse it across all candidates.

**Scaling & impact:**
1. End-to-end length to **10,000 tokens** (industry-first claim).
2. Clean **power-law scaling**.
3. KV-cache cuts online throughput loss from **−40% → −6.8%**.
4. Online: e-commerce **+6.5% GMV/user** (live).

**Architecture (diagram):**

```
 Global Tokens                          User long sequence
 [User Profiles, Context &              (read latest -> earliest)
  Cross Features] + [Candidate    -->   item_t, item_{t-1}, ..., item_1
  Item Features]                              │
        │  full receptive field over the whole long sequence
        └────────────────> pooled outputs up
```

---

## 11. Feature-interaction scaling line → Wukong

### 11.1 Wukong — scaling feature interaction (Meta, ICML 2024)

*Establish a scaling law for feature interactions, not sequences.* (Zhang et al., *Towards a Scaling Law for Large-Scale Recommendation*, ICML 2024; arXiv:2403.02545.)

**Existing problem:**
- Recommenders scaled by growing **embedding tables** (sparse), not interaction power.
- DLRM can't capture **high-order** crosses.
- DCNv2 / AutoInt+ hit **diminishing returns & instability** when enlarged.

**Key idea:**
1. Stack **Factorization-Machine blocks** so interaction order grows exponentially with depth: layer $i$ covers all orders up to $2^i$.
2. **FMB (FM + MLP):** raises interaction order.
3. **LCB (linear compress):** preserves lower orders.

> [!formula] Exponential interaction growth
> $$\text{interaction order at layer } i = 2^{i}$$
> Stacking $L$ FM blocks therefore reaches interaction order $2^{L}$ — high-order crosses become affordable through depth rather than width.

**Architecture (bottom-up):**

```
        Output Predictions
              ^
             MLP
              ^
   Interaction Stack (stacked Wukong Layers)
              ^
       Dense Embeddings

   Each Wukong Layer:
        ┌──────────────┐   ┌────────────────────┐
        │ Factorization│   │ Linear Compress     │   (run in parallel)
        │ Machine Block│   │ Block (preserve     │
        │ (raise order)│   │  low orders)        │
        └──────┬───────┘   └─────────┬───────────┘
               └──── Concat ─────────┘
                        │
                  Add & Norm   (residual connection)
```

---

## 12. Attention-oriented scaling line → HSTU

### 12.1 HSTU — recommendation gets a scaling law (Meta, ICML 2024)

*Meta's branching point: reframe discriminative CTR prediction as **generative sequential modeling**.* (Zhai et al., *Actions Speak Louder than Words*, ICML 2024; arXiv:2402.17152.)

**Key reformulation:**
- Many pointwise samples per user → **one chronological behavior sequence** (items, actions, user/item features).
- Items & actions are interleaved; predict the next action / item **causally**.
- **Unifies retrieval and ranking** in one model.

**HSTU architecture:**
1. **[[Hierarchical Sequential Transduction Unit]]:** pointwise *aggregated* attention (**SiLU**, not softmax).
2. **Relative position + time bias;** fused, lean design.
3. **Ragged fused-GEMM kernels** → **5–15× faster than FlashAttention-2** at length 8192.

**DLRM vs GR/HSTU stacks (diagram):**

```
        DLRM (left)                          GR / HSTU (right)
  Top NNs: MMoE, PLE                   repeated HSTU blocks
        ^                                       ^
  Feature Interactions NN:             Preprocessing
   FMs, DCN, Transformers, DHEN               ^
        ^                              Sequentialized Unified Features
  Embedding Operators
        ^
  Feature extractions (Numerical / Categorical)
        ^
  Raw Features
```

> [!formula] HSTU block (read bottom-to-top inside each block)
> $$U, Q, K, V = \phi_1\!\big(f_1(X)\big)$$
> $$A(X) = \phi_2\!\left(Q K^{\top} + \mathrm{rab}^{p,t}\right)$$
> $$\mathrm{Norm}\!\big(A(X)\,V(X)\big) \odot U(X)$$
> $$Y(X) = f_2(\cdots)$$
> with Add & Norm residual connections between stacked blocks. Where:
> - $X$ — input token representation; $f_1, f_2$ — linear projections.
> - $U, Q, K, V$ — gating, query, key, value streams produced by a single projection.
> - $\phi_1, \phi_2$ — **pointwise nonlinearities (SiLU)**; crucially, $\phi_2$ replaces the usual softmax, giving non-softmax "pointwise aggregated attention."
> - $\mathrm{rab}^{p,t}$ — the **relative position + time attention bias** (encodes both sequence position $p$ and timestamp $t$).
> - $\odot$ — elementwise **gating** by $U(X)$.
>
> The three vertical bands of the figure label the stages: **Feature extractions → Feature interactions → Representation transformations.**

---

## 13. FFN-oriented scaling line → RankMixer

### 13.1 RankMixer — rethinking self-attention (ByteDance, 2025)

*Is self-attention even the right primitive for industrial ranking?* (RankMixer, *Scaling Up Ranking Models in Industrial Recommenders*, ByteDance 2025; arXiv:2507.15551.)

**Why not just use attention?**
- CPU-era handcrafted crossing is **memory-bound** on GPUs → MFU ≈ 4.5%.
- Self-attention assumes one **shared space**, but rec features are **heterogeneous** (user/item ID spaces), plus quadratic cost.

**Key idea — mix, not attend:**
1. Replace attention with a **parameter-free Multi-Head Token Mixing** (shuffle feature subspaces across tokens).
2. **Per-Token FFN:** each feature token gets its own MLP.
3. Optional **Sparse-MoE**.

**Architecture (diagram):**

```
 Top (SMoE variant of Per-Token FFN):
     tokens -> ReLU Routing (gateloss) -> Sparse-MoE of per-head experts
               PFFN_H^1, PFFN_H^2, ..., PFFN_H^E  -> combined -> output

 Bottom (Token Mixing), parameter-free reshuffle:
     [ T tokens x D dim ]  --Split-->  transpose ( T -> H*(D/H) )
                           --Merge-->  [ H tokens x (T*D/H) dim ]
     (no learned attention weights — just a reshape that mixes feature
      subspaces across tokens)
```

> *Result:* MFU lifted from **4.5% → 45%** with latency flat.

---

## 14. Unifying both axes → OneTrans

### 14.1 OneTrans — one backbone for both axes (ByteDance, WWW 2026)

*One Transformer backbone for both feature interaction **and** user-behavior sequences.* (Zhang et al., WWW 2026; arXiv:2510.26104.)

```
 Feature-interaction line (Wukong, RankMixer — scales feature crosses) ┐
                                                                       ├─> OneTrans
 Long-sequence line (LONGER, HSTU — scales user histories) ────────────┘   one backbone
                                                                            for both
```

**Key design choices:**
- **Unified tokenizer:** convert sequential AND non-sequential attributes into a **single token stream** (`Sequential Features` + `Non-Seq Features` → `Tokenizer` → OneTrans).
- **Mixed parameterization:** share parameters across similar sequential tokens; token-specific parameters for non-sequential tokens.
- **Causal attention + cross-request KV cache:** precompute and reuse intermediate states.

> Takeaway: *"One Transformer to rank them all" — recommendation now both **scales like an LLM and serves like one** (KV-cache, FlashAttention).*

---

# PART 3 — Comparison & Conclusion

## 15. LLM-based GR vs LRM — side by side (the crux of the lecture)

(Synthesis from Hou et al. 2025, §4.1 and §4.2.)

| Dimension | **LLM-based Rec** (Part 1) | **LRM** (Part 2) |
|---|---|---|
| **Data form** | text sequence: behavior expressed as language | native action sequence: items / actions as tokens |
| **Source of knowledge** | world knowledge from web-scale pre-training | massive behavior data from the platform |
| **Source of scaling** | borrowed from language-model scaling | native to recommendation data |
| **Typical scenarios** | cold-start · cross-domain · explainable | industrial-scale main-feed ranking |
| **Hardest challenges** | aligning to collaborative signal · grounding | training infra · long-context engineering |

---

## 16. When does generative beat discriminative?

(Hou et al. 2025, §4.) Three conditions:

1. **Sparse data and cross-domain** — discriminative models *starve* in low-signal regimes; generative models lean on world knowledge, prior text, or cross-task transfer.
2. **Inherently generative tasks** — dialog-based recommendation, explanation generation, content creation; discriminative scoring **cannot produce these outputs at all**.
3. **Sufficient training compute** — generative models keep gaining with more compute (the scaling law holds); discriminative models tend to **saturate** well before that point.

## 17. Five high-level advantages of generative recommendation

*Hook every concept from this lecture back to one of these five* — they are traced through **TIGER** and **OneRec** next week ([[RS-L04 - Generative Recommendation]]).

1. **World-knowledge integration** — free of cold-start; understands new items / domains via pre-trained semantics.
2. **Natural-language understanding** — users / items / interactions expressed and reasoned about in language.
3. **Reasoning ability** — multi-hop preference inference and explanation become possible.
4. **[[Scaling Laws|Scaling law]]** — more compute = better model; both the LLM-based and LRM lines now exhibit this.
5. **Creative generation** — synthesize content, explanations, conversations.

---

## Key Takeaways

> [!tip] Exam focus
> - **Discriminative vs. generative (Q1):** discriminative learns $f(\text{user},\text{item})$ and *scores a fixed candidate pool*; generative *produces the target token-by-token*, escaping the fixed pool and using world knowledge — at the cost of having to **ground** output to real catalog items.
> - **Two routes to generative recommendation (Q4):** **LLM-based GR borrows scaling from language** (squeeze behavior into text); **LRM builds a native recommendation scaling law** (design architectures for behavior data). Know the side-by-side table in §15 cold.
> - **Three alignment paradigms (Q2):** ① **text prompting** (pure NL, no CF — TALLRec, LlamaRec) → ② **inject collaborative signal** (project/summarize/sentence-ize CF embeddings — CoLLM, CoRAL) → ③ **item tokenization** (learned discrete [[Semantic IDs]] — P5, TIGER). The driving tension: **dense CF embeddings are not directly readable by an LLM.**
> - **Item-tokenization ladder (Q3):** L1 [[Atomic Item IDs|atomic ID]] → L2 text → **L3 [[RQ-VAE]] semantic ID** (the modern default: compact + semantic) → L4 semantic ID + CF → L5 adaptive.
> - **Training objectives:** SFT (positives only) · SSL (contrastive) · RL (rewards, can encode negatives & non-diff metrics, unstable) · [[Direct Preference Optimization (DPO)|DPO]] (preference pairs, stable, no reward model).
> - **The universal LRM trick:** cheaper/approximate ops + cache & reuse + raise **MFU** (from ~0.1–1% toward LLM-level ~70%) → reclaim compute, spend it on scale, *stay inside the latency budget*. Map each model to its axis: **LONGER** = sequence length, **Wukong** = feature-interaction order ($2^i$ per layer), **[[HSTU]]** = attention (non-softmax SiLU, 5–15× over FlashAttention-2), **RankMixer** = FFN/token-mixing (MFU 4.5%→45%), **OneTrans** = both axes.
> - **HSTU is the conceptual hinge:** it reframes discriminative CTR prediction as *generative sequential modeling* and unifies retrieval + ranking — the bridge between Part 1 and Part 2.
> - **When generative wins:** sparse/cross-domain data, inherently generative tasks, sufficient compute. Five enduring advantages: world knowledge, NL understanding, reasoning, scaling law, creative generation.

---

## Key References

> [!note] Surveys
> - Hou et al. "A Survey on Generative Recommendation: Data, Model, and Tasks." arXiv:2510.27157, 2025. *(primary source for this lecture)*
> - Lin et al. "How Can Recommender Systems Benefit from Large Language Models: A Survey." arXiv:2306.05817, 2024.

**Part 1 — LLM-based Generative Recommendation:**
- Gao et al. *Chat-REC.* 2023.
- Bao et al. *TALLRec.* RecSys 2023.
- Yue et al. *LlamaRec.* 2023.
- Liao et al. *LLaRA.* SIGIR 2024.
- Geng et al. *P5.* RecSys 2022.
- Rajput et al. "Recommender Systems with Generative Retrieval" (*TIGER*). NeurIPS 2023.
- Zheng et al. *LC-Rec.* ICDE 2024.

**Part 2 — Large Recommendation Models:**
- Zhai et al. "Actions Speak Louder than Words" (*HSTU*). ICML 2024 · arXiv:2402.17152.
- Zhang et al. *Wukong.* ICML 2024 · arXiv:2403.02545.
- *RankMixer* (ByteDance). 2025 · arXiv:2507.15551.
- Chai et al. *LONGER.* RecSys 2025 · arXiv:2505.04421.
- Zhang et al. *OneTrans.* WWW 2026 · arXiv:2510.26104.
- Deng et al. *OneRec.* 2025 · arXiv:2502.18965.

---

## Links

**Concepts:**
- [[Generative Recommendation]] · [[LLM-based Generative Recommendation]] · [[Large Recommendation Models (LRM)]]
- [[Item Tokenization]] · [[Semantic IDs]] · [[RQ-VAE]] · [[Atomic Item IDs]]
- [[LLM-as-Enhancer]] · [[LLM-as-Recommender]] · [[In-Context Learning]]
- [[Collaborative Filtering]] · [[LoRA]] · [[Contrastive Learning]]
- [[Supervised Fine-Tuning (SFT)]] · [[Direct Preference Optimization (DPO)]] · [[Reinforcement Learning]]
- [[Next-Item Prediction]] · [[Autoregressive Generation]] · [[Scaling Laws]]
- [[HSTU]] · [[Hierarchical Sequential Transduction Unit]]

**Related RecSys lectures:**
- [[RS-L01 - Course Overview & Introduction]]
- [[RS-L02 - Evaluation Beyond Accuracy]]
- [[RS-L03a - Sequential Recommendation Models]] (preceding half of this lecture)
- [[RS-L04 - Generative Recommendation]] (TIGER & OneRec traced through the five advantages)
