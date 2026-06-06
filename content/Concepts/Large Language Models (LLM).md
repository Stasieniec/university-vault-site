---
type: concept
aliases: [LLM, LLMs]
course: [RecSys]
tags: [llm, generative-rec, exam-topic]
status: complete
---

# Large Language Models (LLM)

## Definition

> [!definition] Large Language Model
> A **Large Language Model (LLM)** is a web-scale, **autoregressive** neural sequence model (a [[Transformer Model|Transformer]] decoder) pretrained on massive text corpora to predict the next token. The result is a single model carrying **world knowledge**, **natural-language understanding**, and **reasoning ability** that obeys a [[Scaling Law]]: accuracy improves predictably as parameters, data, and compute grow. In recommendation, the LLM is repurposed so that **recommendation becomes a language task** — users, items, and interaction histories are translated into a token sequence the model can read and continue.

## Intuition

> [!intuition] Why bring an LLM into RecSys?
> A classic discriminative recommender learns a scoring function $f(\text{user}, \text{item})$ and ranks a **fixed candidate pool**. It starves in low-signal regimes (cold-start, cross-domain) because it only knows what it saw in the interaction log. An LLM arrives **already knowing** that *Iron Man* is an action superhero film, that *The Godfather* is a crime drama, and that fans of one often like the other — semantics it absorbed from the web, never from clicks. So instead of *scoring* candidates we let the model **generate** the target token-by-token, leaning on pretrained world knowledge.
>
> The catch: pretraining gives the LLM semantics but **not** collaborative signal. It never saw your platform's click matrix, your Top-K ranking objective, your long-tail coverage incentives, or your exposure bias. A vanilla LLM therefore often *loses* to a specialized rec model — which is exactly why the field develops **alignment** and **item tokenization** on top of it.

## Mathematical Formulation

An LLM factorizes the probability of a token sequence $x = (x_1, \dots, x_T)$ autoregressively and is trained by minimizing the next-token cross-entropy (negative log-likelihood):

$$
p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}),
\qquad
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta\!\left(x_t \mid x_{<t}\right)
$$

where:
- $x_{<t} = (x_1, \dots, x_{t-1})$ — the prefix (context) of already-seen tokens
- $p_\theta(x_t \mid x_{<t})$ — softmax over the vocabulary, produced by a [[Transformer Model|Transformer]] decoder using causal [[Self-Attention]]
- $\theta$ — model parameters (the "scale" in the scaling law)
- $T$ — sequence length / context window

The single primitive inside each layer is masked **self-attention**, which mixes the prefix into each position:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V
$$

where:
- $Q, K, V$ — query, key, value projections of the token embeddings
- $d_k$ — key dimension (the $\sqrt{d_k}$ stabilizes gradients)
- $M$ — causal mask ($M_{ij} = -\infty$ for $j > i$), forbidding a token from attending to the future

**Cast as recommendation.** Let $S_u = (i_1, \dots, i_n)$ be user $u$'s history. The recommender's job — [[Next-Item Prediction]] — is the same autoregressive objective over an item/text vocabulary:

$$
p_\theta(i_{n+1} \mid S_u) = \prod_{k} p_\theta\!\left(c_k \mid c_{<k}, \, \text{prompt}(S_u)\right)
$$

where the next item $i_{n+1}$ is emitted as one or more discrete tokens $c_k$ (a title, an atomic ID, or a multi-codeword [[Semantic ID]]) decoded by [[Beam Search]] / [[Autoregressive Decoding]] from a prompt built out of $S_u$.

## Key Properties / Variants

- **Three alignment paradigms** (how the user/item profile enters the input) — the core taxonomy of [[LLM-based Recommendation|LLM-based GR]]:
  1. **Text Prompting** — pure natural language; the LLM is (often parameter-efficiently) fine-tuned on instruction instances. Carries *no* collaborative signal. Models: TALLRec, LlamaRec.
  2. **Inject Collaborative Signal** — project a learned [[Collaborative Filtering|CF]] embedding into the token space, summarize CF knowledge into text, or "sentence-ize" it ("users who liked A also liked B"). Models: CoLLM, LLaRA, CORONA, CoRAL.
  3. **Item Tokenization** — give each item a compact, semantic, *generable* identifier; see the L1–L5 ladder below. Models: P5, TIGER, LC-Rec.
- **Item tokenization ladder** (how items become tokens the LLM can generate): **L1** atomic ID (vocabulary blows up, no semantics) → **L2** text/title (long, no CF) → **L3** [[RQ-VAE]] [[Semantic ID]] (compact + semantic) → **L4** semantic ID + injected CF → **L5** adaptive IDs refined during training.
- **Two usage modes without alignment** (zero training cost): **LLM-as-Enhancer** (rewrite profiles/history into NL features fed to a CF/sequential model) and **LLM-as-Recommender** (templated prompt, LLM outputs titles/IDs directly).
- **Training objectives** for the rec task: [[Supervised Fine-Tuning (SFT)|SFT]] (positives only, weak ranking margin), self-supervised/contrastive, [[Reinforcement Learning|RL]] (encodes negatives + non-differentiable metrics, unstable), and [[Direct Preference Optimization (DPO)|DPO]] (preferred-vs-rejected pairs, stable, no reward model).
- **Inference strategies**: direct decode, rerank over a retriever's candidate set, or accelerated serving (speculative decoding, embedding-only use).
- **Strengths**: world knowledge, NL understanding, reasoning, scaling law, creative generation — strong cold-start and cross-domain transfer.
- **Weaknesses**: prompt-sensitive; **hallucination** of non-existent items (motivates constrained / grounded decoding); the fundamental gap between dense collaborative semantics and language semantics; serving cost at industrial scale.

> [!warning] Borrowed scaling, not native scaling
> LLM-based GR borrows its scaling law **from language**, which forces behavior data to be squeezed into text and inflates context length. This is the contrast with [[Large Recommendation Models (LRM)|LRMs]], which design architectures **native** to recommendation data (e.g., HSTU, RankMixer) so a recommendation-specific scaling law emerges directly — typically reaching far higher hardware utilization (MFU) than the ~0.1–1% of classic cascaded pipelines.

> [!warning] Hallucination must be grounded
> An LLM can fluently emit an item title or ID that **does not exist in the catalog**. Practical generative recommenders therefore constrain decoding (e.g., trie / [[Constrained Decoding]] over valid item tokens, or rerank over a retrieved candidate set) so every generated item is real.

## Connections

- Is a: [[Transformer Model]] decoder trained by [[Autoregressive Generation]] with [[Self-Attention]]
- Governed by: [[Scaling Law]]; adapted cheaply via [[LoRA]] / [[Parameter-Efficient Fine-Tuning]]
- Used for recommendation as: [[LLM-as-Recommender]], [[LLM-as-Enhancer]], [[LLM-based Recommendation]], [[Generative Recommendation]]
- Aligned via: [[Item Tokenization]], [[Semantic ID]], [[RQ-VAE]], injected [[Collaborative Filtering]] signal
- Trained/served with: [[Supervised Fine-Tuning (SFT)]], [[Direct Preference Optimization (DPO)]], [[Reinforcement Learning]], [[Beam Search]]
- Contrasted with: [[Large Recommendation Models (LRM)]] (native vs borrowed scaling), discriminative [[Recommender System]]s, [[Sequential Recommendation]] models
- Tackles: [[Cold Start]], [[Cross-Domain Recommendation]]

## Appears In

- [[RS-L03b - From LLMs to LRMs]]
