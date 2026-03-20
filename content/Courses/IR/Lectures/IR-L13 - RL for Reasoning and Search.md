---
type: lecture
course: IR
week: 6
lecture: 13
topics:
  - "[[Reinforcement Learning]]"
  - "[[Policy Gradient Methods]]"
  - "[[PPO]]"
  - "[[GRPO]]"
  - "[[SEARCH-R1]]"
  - "[[DeepSeek-R1]]"
  - "[[Retrieval-Augmented Generation]]"
  - "[[Agentic Search]]"
status: complete
---

# IR-L13: RL for Reasoning and Search

## Overview

This lecture bridges [[Reinforcement Learning]] and [[Information Retrieval]], exploring how modern LLMs can be trained with RL to dynamically decide **when and what to retrieve** during reasoning. Traditional [[Retrieval-Augmented Generation]] (RAG) uses static retrieve-then-read pipelines, but complex queries require **iterative, interleaved reasoning and retrieval**. We study how RL enables models to learn this interleaving through reward signals rather than supervised fine-tuning.

The lecture covers foundational RL concepts ([[Policy Gradient Methods]], [[PPO]], GRPO), the DeepSeek-R1 breakthrough in pure RL for reasoning, and the SEARCH-R1 system that extends this paradigm to retrieval-augmented reasoning. We conclude with systematic design studies and the future of agentic search systems.

---

## 1. Foundations: Why Static RAG Falls Short

### 1.1 Limitations of Traditional RAG

Standard [[Retrieval-Augmented Generation]] follows a fixed pipeline:

```
Query → Retrieve Top-k → Concatenate → Generate Answer
```

> [!warning] **Problems with Static RAG**
> 1. **Single-shot retrieval**: Cannot reformulate queries based on initial findings
> 2. **No iterative refinement**: Complex queries need multiple retrieval rounds
> 3. **Fixed retrieval count**: Always retrieves k documents regardless of query difficulty
> 4. **No reasoning integration**: Retrieval is decoupled from the reasoning process

**Example of failure**:
- Query: "What is the population of the capital of the country that won the 2022 FIFA World Cup?"
- Static RAG retrieves documents about FIFA, but cannot chain: Argentina → Buenos Aires → population

### 1.2 From Static to Agentic Search

The solution is to give the model **agency over retrieval**:

| Aspect | Static RAG | Agentic Search |
|--------|------------|----------------|
| **Retrieval timing** | Before generation | During reasoning |
| **Query formulation** | User query only | Model-generated subqueries |
| **Number of retrievals** | Fixed k | Adaptive (0 to many) |
| **Reasoning integration** | None | Interleaved |
| **Training paradigm** | SFT on (query, answer) | RL with outcome reward |

> [!intuition] **Key Insight**
> Agentic search treats retrieval as an **action** in an RL framework. The model learns **when** to search and **what** to search for through trial and error, guided by whether the final answer is correct.

---

## 2. RL Foundations for LLM Training

### 2.1 MDP Framing for Language Generation

We cast text generation as a [[Markov Decision Process]]:

| MDP Component | Language Generation Mapping |
|---------------|----------------------------|
| **State** $s_t$ | Prompt + tokens generated so far |
| **Action** $a_t$ | Next token (from vocabulary) |
| **Transition** $P(s_{t+1}|s_t, a_t)$ | Deterministic: append $a_t$ to $s_t$ |
| **Reward** $r_t$ | 0 for intermediate, $R$ for final |
| **Policy** $\pi_\theta(a_t|s_t)$ | Language model $P_\theta(\text{token}|\text{context})$ |

A complete response is a **trajectory** $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ ending at EOS token.

### 2.2 Supervised Fine-Tuning (SFT) Loss

Standard supervised training minimizes negative log-likelihood:

> [!formula] **SFT Loss**
> $$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t}) \right]$$
>
> where:
> - $x$ = input prompt
> - $y = (y_1, y_2, \ldots, y_{|y|})$ = target response tokens
> - $\pi_\theta(y_t | x, y_{<t})$ = model probability of token $y_t$ given context

**Limitation**: SFT requires **demonstration data**. For complex reasoning with search, we often lack such data or it's expensive to create.

### 2.3 REINFORCE: Policy Gradient for LLMs

The [[REINFORCE]] algorithm optimizes expected reward directly:

> [!formula] **REINFORCE Gradient**
> $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$
>
> **Interpretation**: Increase probability of actions in trajectories with high reward.

For language models with sparse final reward:

$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ R(x, y) \nabla_\theta \log \pi_\theta(y | x) \right]$$

**Problems with vanilla REINFORCE**:
1. **High variance**: Full trajectory reward creates noisy gradients
2. **Credit assignment**: Which tokens actually contributed to success?
3. **Sample inefficiency**: Need many rollouts to estimate gradient

### 2.4 PPO: Stable Policy Updates

[[PPO|Proximal Policy Optimization]] addresses instability by constraining how much the policy can change:

> [!formula] **PPO Clipped Objective**
> $$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
>
> where:
> - $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ — probability ratio
> - $\hat{A}_t$ — advantage estimate (how much better than baseline)
> - $\epsilon$ — clipping parameter (typically 0.1–0.2)
> - The $\min$ takes the pessimistic bound

> [!intuition] **Why PPO Works**
> - If $r_t > 1 + \epsilon$ and advantage is positive: clip prevents over-exploitation
> - If $r_t < 1 - \epsilon$ and advantage is negative: clip prevents over-correction
> - Policy changes are bounded, ensuring stable training

**PPO for LLMs requires**:
1. **Value network** $V_\phi(s)$ to estimate baseline
2. **Advantage estimation** (typically GAE)
3. **KL penalty** to prevent drift from reference model

---

## 3. GRPO: Group Relative Policy Optimization

### 3.1 Motivation: Eliminating the Critic

Standard PPO requires a separate **value network** (critic) to estimate advantages. For LLMs:
- Value network adds parameters (~50% increase)
- Training the critic is itself challenging
- Critic quality directly impacts policy gradient quality

**GRPO insight**: Use **group-relative comparisons** instead of absolute value estimates.

### 3.2 GRPO Algorithm

For each prompt $x$, sample a **group** of $G$ responses $\{y_1, y_2, \ldots, y_G\}$ from the current policy.

> [!formula] **GRPO Advantage**
> $$\hat{A}_i = \frac{R(x, y_i) - \text{mean}(\{R(x, y_j)\}_{j=1}^G)}{\text{std}(\{R(x, y_j)\}_{j=1}^G)}$$
>
> The advantage of response $y_i$ is its **z-score** within the group.

**GRPO Gradient**:
$$\nabla_\theta J = \mathbb{E}_{x} \left[ \frac{1}{G} \sum_{i=1}^G \hat{A}_i \nabla_\theta \log \pi_\theta(y_i | x) \right]$$

### 3.3 GRPO vs PPO Comparison

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Advantage estimation** | Learned value function $V_\phi$ | Group statistics |
| **Additional networks** | Critic + reward model | None |
| **Sample requirement** | 1 sample per prompt | G samples per prompt |
| **Variance reduction** | GAE + baseline | Group normalization |
| **Implementation** | Complex | Simple |
| **Memory overhead** | High (critic params) | Low |

> [!tip] **When to use GRPO**
> - Training LLMs where adding a critic is expensive
> - When relative ranking matters more than absolute scores
> - With outcome-based rewards (correct/incorrect)

---

## 4. DeepSeek-R1: Pure RL for Reasoning

### 4.1 The DeepSeek-R1 Breakthrough

DeepSeek-R1 demonstrated that **pure RL** (without SFT on reasoning traces) can produce strong reasoning capabilities.

**Key finding**: Starting from a base model and training with only outcome reward (correct/incorrect), the model **spontaneously develops**:
- Chain-of-thought reasoning
- Self-verification behaviors
- Backtracking and error correction

### 4.2 Emergent Reasoning Behaviors

> [!definition] **Emergent Reasoning**
> Reasoning patterns that arise from RL training without being explicitly taught. The model discovers that intermediate reasoning steps improve final answer accuracy.

**Observed emergent behaviors**:
1. **Extended thinking**: Model generates longer reasoning chains
2. **Self-questioning**: "Wait, let me check this..."
3. **Exploration**: Trying multiple approaches before answering
4. **Verification**: Double-checking intermediate results

### 4.3 Training Details

- **Base model**: DeepSeek-V3 (pre-trained, no SFT)
- **Algorithm**: GRPO
- **Reward**: Binary (correct answer = 1, incorrect = 0)
- **No demonstration data**: Pure RL from scratch

> [!intuition] **Why Pure RL Works**
> The reward signal provides enough guidance. When the model accidentally produces a correct answer after some reasoning, that trajectory is reinforced. Over time, the model learns that certain patterns (checking work, breaking down problems) correlate with higher rewards.

---

## 5. SEARCH-R1: Interleaved Reasoning and Search

### 5.1 Core Innovation

SEARCH-R1 extends DeepSeek-R1's pure RL approach to **retrieval-augmented reasoning**:

```
Query → [Reason] → <search>subquery</search> → [Retrieved Docs] → [Reason] → ... → Answer
```

The model learns to:
1. Decide **when** to search (or not search at all)
2. Generate **what** to search for
3. **Incorporate** retrieved information into reasoning

### 5.2 Interleaved Trajectory Format

> [!example] **SEARCH-R1 Trajectory Example**
> ```
> <think>
> The question asks about the population of Buenos Aires.
> I need to find recent population data.
> <search>Buenos Aires population 2024</search>
> </think>
> <information>
> [Retrieved: Buenos Aires has a population of approximately 15.6 million
> in the metropolitan area as of 2024...]
> </information>
> <think>
> The retrieved information gives me the metropolitan population.
> The question might be asking about the city proper vs metro area.
> <search>Buenos Aires city proper population vs metropolitan</search>
> </think>
> <information>
> [Retrieved: The city proper of Buenos Aires has about 3 million residents,
> while the Greater Buenos Aires metropolitan area has 15.6 million...]
> </information>
> <think>
> Based on the retrieved information, the city proper has 3 million
> and the metropolitan area has 15.6 million. I should clarify this
> in my answer.
> </think>
> <answer>
> Buenos Aires has approximately 3 million people in the city proper
> and 15.6 million in the metropolitan area.
> </answer>
> ```

**Special tokens**:
- `<think>...</think>` — reasoning traces
- `<search>...</search>` — search query (triggers retrieval)
- `<information>...</information>` — retrieved documents (inserted by system)
- `<answer>...</answer>` — final answer

### 5.3 RL Objective with Search Engine

The training loop:

```
1. Sample prompt x
2. Generate trajectory τ = (think, search, info, think, ..., answer)
   - At each <search> tag, pause and retrieve from search engine
   - Insert results in <information> tags
3. Evaluate final answer → reward R
4. Update policy using GRPO
```

> [!formula] **SEARCH-R1 Reward**
> $$R(x, y) = \begin{cases} 1 & \text{if final answer is correct} \\ 0 & \text{otherwise} \end{cases}$$
>
> Optional format reward:
> $$R_{\text{format}}(y) = \begin{cases} 0.5 & \text{if answer properly formatted} \\ 0 & \text{otherwise} \end{cases}$$

### 5.4 Loss Masking: Critical Design Choice

Not all tokens should receive gradient updates equally.

> [!formula] **SEARCH-R1 Masked Loss**
> $$\mathcal{L}(\theta) = -\sum_{t \in \mathcal{T}_{\text{model}}} \hat{A}_t \log \pi_\theta(y_t | y_{<t}, x)$$
>
> where $\mathcal{T}_{\text{model}}$ excludes:
> - Tokens inside `<information>` tags (retrieved content)
> - System-inserted tokens

> [!intuition] **Why Mask Retrieved Content?**
> 1. **Not model-generated**: Retrieved text comes from the search engine
> 2. **Prevents memorization**: Model shouldn't memorize corpus content
> 3. **Correct credit assignment**: Only model decisions affect the gradient
> 4. **Cleaner learning signal**: Reward reflects model's reasoning, not retrieval quality

### 5.5 Search Engine Integration

The search engine is treated as a **non-differentiable environment**:

```
Model generates: "...<search>query text</search>..."
                         ↓
              Search Engine (BM25/Dense)
                         ↓
System inserts: "<information>[doc1][doc2]...</information>"
                         ↓
              Model continues generation
```

**Design choices**:
- **Retrieval method**: BM25, dense retrieval, or hybrid
- **Number of results**: Top-k documents per search
- **Truncation**: Limit retrieved content length
- **Corpus**: Wikipedia, web, domain-specific

---

## 6. Ablations and Training Dynamics

### 6.1 PPO vs GRPO for Search

Training dynamics differ fundamentally between the two algorithms:

- **GRPO**: Faster initial convergence in all 4 settings (3B/7B, base/instruct), but suffers **reward collapse** after extended training. The noisy group baseline cannot stabilize long runs.
- **PPO**: Slower start (critic warm-up phase), but **uniformly stable**. The value function absorbs retrieval noise, preventing collapse.

> [!warning] **GRPO Fragility in Search**
> In pure reasoning (R1), the environment is deterministic, so $\text{Var}(A) \approx \text{Var}_\pi(R - \bar{r})$. In search-augmented reasoning, different rollouts receive different search results:
> $$\text{Var}(A) = \text{Var}_\pi(R - V) + \text{Var}_{\text{env}}(R - V)$$
> The group mean $\bar{r}$ conflates good policy decisions with good search luck — a biased baseline. After extended training, advantage estimates collapse $\Rightarrow$ reward collapse.
> PPO's $V_\phi(s_t)$ conditions on the current state (including what has been retrieved), providing a per-state, environment-aware baseline that absorbs retrieval noise.

### 6.2 Effect of Loss Masking

| Configuration | Avg Score (7B base, PPO) |
|--------------|--------------------------|
| Without masking | 0.343 |
| **With masking** | **0.431** (+8.8 pts) |

> [!warning] **Critical Finding**
> Masking is the **single largest technique gain** (+8.8 avg points). Without it, the model wastes capacity trying to predict Wikipedia content — which is both useless and destabilizing. The effect is strongest on multi-hop tasks.

### 6.3 Base vs Instruct Models

- **Instruct models** start higher (instruction-following already established) and converge faster
- After full RL training, **final performance is virtually identical** — RL closes the gap on both 3B and 7B
- **Base models often produce better final search queries** due to broader, less filtered world knowledge

### 6.4 Hyperparameter Studies

**Top-k Retrieved Passages:**

| top-k | NQ | HotpotQA | Musique | Avg |
|-------|-----|----------|---------|-----|
| 1 | 0.426 | 0.393 | 0.146 | 0.375 |
| **3** | **0.480** | **0.433** | **0.196** | **0.431** |
| 5 | 0.479 | 0.394 | 0.156 | 0.400 |

Top-3 is optimal: best precision-recall balance, stable throughout 500 training steps.

**GRPO Group Size:**

| Group size | Train stability | OOD Avg |
|-----------|-----------------|---------|
| 5 | Collapses | 0.350 |
| 3 | Moderate | 0.363 |
| **1 (REINFORCE)** | **Stable** | **0.410** |

Larger groups reduce variance in the baseline (faster learning) but also increase gradient noise from diverse search retrievals.

---

## 7. Systematic Design Study

### 7.1 Reward Formulation

Different reward designs and their effects:

| Reward Type | Formula | Effect |
|-------------|---------|--------|
| **Outcome only** | $R = \mathbb{1}[\text{correct}]$ | Sparse but clean signal |
| **Outcome + format** | $R = \mathbb{1}[\text{correct}] + 0.5 \cdot \mathbb{1}[\text{format}]$ | Encourages structure |
| **Dense reasoning** | $R = \sum_t r_t(\text{step quality})$ | Hard to define, noisy |
| **Search penalty** | $R = \mathbb{1}[\text{correct}] - 0.1 \cdot n_{\text{searches}}$ | Reduces unnecessary searches |

**Best practice**: Outcome reward + light format reward. Dense rewards are hard to specify correctly.

### 7.2 Backbone Model Choice

| Model Size | Accuracy | Search Behavior |
|------------|----------|-----------------|
| 1.5B | 42.1% | Searches too often |
| 7B | 56.8% | Balanced |
| 14B | 61.2% | Selective searching |
| 32B | 64.7% | Highly selective |

**Observation**: Larger models learn more selective search behavior—they search only when necessary.

### 7.3 Search Engine Quality

| Retriever | Accuracy | Notes |
|-----------|----------|-------|
| BM25 | 54.2% | Baseline |
| Dense (Contriever) | 56.1% | Better semantic matching |
| Hybrid (BM25 + Dense) | **58.3%** | Best of both |
| Google Search API | 61.2% | Real web search |

**Finding**: Better retrieval → better reasoning. The model can learn to work with imperfect retrieval, but ceiling is limited by retrieval quality.

### 7.4 Number of Retrieved Documents

| Top-k | Accuracy | Context Length |
|-------|----------|----------------|
| 1 | 51.2% | Short |
| 3 | **56.8%** | Moderate |
| 5 | 55.9% | Long |
| 10 | 54.1% | Very long |

**Sweet spot**: 3 documents. Too few misses relevant info; too many introduces noise (relates to "lost in the middle" phenomenon from [[IR-L09 - RAG]]).

---

## 8. The Big Picture: System 1 vs System 2

### 8.1 Cognitive Framework

Drawing from dual-process theory in cognitive science:

| Aspect | System 1 | System 2 |
|--------|----------|----------|
| **Processing** | Fast, automatic | Slow, deliberate |
| **Effort** | Low | High |
| **Example** | Pattern matching | Multi-step reasoning |
| **IR analogy** | Static RAG | Agentic Search |
| **LLM behavior** | Direct answer | Think + search + verify |

> [!definition] **System 2 Retrieval**
> Retrieval that involves deliberate reasoning about what to search for, evaluation of retrieved results, and iterative refinement. The model "thinks" about retrieval rather than executing a fixed pipeline.

### 8.2 RAG vs Agentic Search Comparison

| Dimension | Traditional RAG | Agentic Search (SEARCH-R1) |
|-----------|-----------------|---------------------------|
| **Architecture** | Retrieve → Read | Reason → Search → Reason → ... |
| **Control flow** | Fixed pipeline | Model-determined |
| **Training** | SFT on QA pairs | RL with outcome reward |
| **Adaptivity** | None | Query-dependent |
| **Multi-hop** | Limited | Natural |
| **Compute** | Predictable | Variable |
| **Interpretability** | Low | High (explicit reasoning) |

### 8.3 When to Use Which Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple factual QA | Static RAG |
| Multi-hop reasoning | Agentic Search |
| Low latency required | Static RAG |
| Complex research queries | Agentic Search |
| Production at scale | Hybrid (route by query) |

---

## 9. Experimental Results Summary

### 9.1 Main Results (Qwen2.5-7B)

| Method | NQ$^\dagger$ | TrivQA$^\star$ | PopQA$^\star$ | HotpotQA$^\dagger$ | 2Wiki$^\star$ | Musique$^\star$ | Bamboogle$^\star$ | Avg |
|--------|-----|--------|-------|----------|-------|---------|-----------|-----|
| Direct Inference | 0.134 | 0.408 | 0.140 | 0.183 | 0.250 | 0.031 | 0.120 | 0.181 |
| RAG | 0.349 | 0.585 | 0.392 | 0.299 | 0.235 | 0.058 | 0.208 | 0.304 |
| IRCoT | 0.224 | 0.478 | 0.301 | 0.133 | 0.149 | 0.072 | 0.224 | 0.239 |
| SFT | 0.318 | 0.354 | 0.121 | 0.217 | 0.259 | 0.066 | 0.112 | 0.207 |
| R1-base (no search) | 0.297 | 0.539 | 0.202 | 0.242 | 0.273 | 0.083 | 0.296 | 0.276 |
| Rejection Sampling | 0.360 | 0.592 | 0.380 | 0.331 | 0.296 | 0.123 | 0.355 | 0.348 |
| **SEARCH-R1-base (PPO)** | **0.480** | **0.638** | **0.457** | **0.433** | **0.382** | **0.196** | **0.432** | **0.431** |
| SEARCH-R1-instruct (PPO) | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.385 |

$\dagger$ in-domain; $\star$ out-of-domain.

> [!tip] Key Results
> - **+24% avg improvement** over the best RAG baseline (7B); +20% for 3B
> - Gains hold across both in-domain and out-of-domain splits — **no overfitting**
> - **Base beats instruct**: broader world knowledge produces better search queries; RL closes the instruction-following gap over time

### 9.2 LLM Backbone Results

| Backbone | Alg | NQ | TrivQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle |
|----------|-----|-----|--------|-------|----------|-------|---------|-----------|
| R1-Distill-7B | PPO | 0.389 | 0.542 | 0.402 | 0.334 | 0.326 | 0.122 | 0.290 |
| R1-Distill-7B | GRPO | 0.061 | 0.155 | 0.068 | 0.098 | 0.194 | 0.010 | 0.113 |
| Qwen2.5-7B | PPO | 0.488 | 0.644 | 0.469 | 0.436 | 0.412 | 0.187 | 0.403 |
| Qwen2.5-7B | GRPO | 0.458 | 0.632 | 0.442 | 0.412 | 0.404 | 0.180 | 0.411 |

> [!warning] **R1-Distill collapses with GRPO**
> Without early positive rewards from search rollouts, GRPO's group mean provides no learning signal — the policy spirals. PPO's value function provides stability even before the model has learned to search correctly. **Always initialize from a general-purpose base model, not a reasoning-specialized one.**

### 9.3 Search Engine Quality

| Train Engine | NQ | TrivQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | Avg EM |
|-------------|-----|--------|-------|----------|-------|---------|-----------|--------|
| Random | 0.237 | 0.494 | 0.177 | 0.217 | 0.269 | 0.058 | 0.234 | 0.241 |
| BM25 | 0.341 | 0.607 | 0.322 | 0.404 | 0.370 | 0.137 | 0.280 | 0.352 |
| E5-HNSW | 0.468 | 0.621 | 0.366 | 0.372 | 0.287 | 0.137 | 0.400 | 0.379 |
| **E5-Exact** | **0.481** | **0.638** | **0.457** | **0.433** | **0.382** | **0.196** | **0.424** | **0.430** |

**Cross-retriever generalization**: A model trained on BM25 still works well with E5 or Google Search at inference — the search strategy transfers. Swapping to a stronger retriever at deployment is a **free performance boost** without retraining.

---

## 10. Future Directions

### 10.1 Open Research Questions

1. **Credit assignment**: How to attribute success to specific search queries?
2. **Search efficiency**: How to minimize retrieval calls while maintaining accuracy?
3. **Corpus adaptation**: How to transfer across different knowledge bases?
4. **Real-time learning**: Can the model improve from interaction feedback?
5. **Safety**: How to prevent adversarial information injection?

### 10.2 Emerging Paradigms

| Direction | Description |
|-----------|-------------|
| **Tool-augmented RL** | Extend to other tools (calculator, code interpreter) |
| **Multi-agent search** | Multiple specialized agents collaborating |
| **Continuous learning** | Update model as corpus changes |
| **Verified reasoning** | Formal proofs of reasoning correctness |

---

## Key Takeaways

> [!summary] **Six Key Takeaways (from lecture)**
>
> 1. **The Reward is the Teacher.** RL with a verifiable outcome reward is sufficient to induce search behavior, self-correction, and query reformulation — without any labeled trajectories.
>
> 2. **Mask retrieved tokens.** This single technique is worth +8.8 avg points. Never include external content in the RL loss.
>
> 3. **PPO for search, GRPO for math.** Search environments are stochastic. GRPO's group baseline conflates policy quality with retrieval luck. PPO's value function absorbs that noise.
>
> 4. **General-purpose base models train better.** Reasoning-specialized models lack instruction-following priors early in training and collapse with GRPO.
>
> 5. **The retriever shapes the agent.** A weak retriever produces a verbose, inefficient searcher. A strong retriever at inference is a free upgrade — cross-retriever generalization is robust.
>
> 6. **Format reward helps; intermediate retrieval rewards do not.** The outcome reward already encodes sufficient signal for good search behavior.

---

## Connections to Other Lectures

- **[[IR-L09 - RAG]]**: Static RAG architectures that SEARCH-R1 improves upon
- **[[IR-L06 - Dense Retrieval]]**: Retrieval methods used in the search component
- **[[RL-L09 - Policy Gradient Methods]]**: REINFORCE and policy gradient foundations
- **[[RL-L10 - Advanced Policy Search]]**: Actor-critic, advantage estimation

---

## References

[1] Bowen Jin, Hansi Zeng, Zhenrui Yue, et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." *arXiv preprint*, 2025.

[2] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv preprint*, 2025.

[3] John Schulman, Filip Wolski, Prafulla Dhariwal, et al. "Proximal Policy Optimization Algorithms." *arXiv preprint*, 2017.

[4] Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv preprint*, 2024.

[5] Yao, Shunyu, et al. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*, 2023.

[6] Akari Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR*, 2024.

[7] Soyeong Jeong et al. "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity." *NAACL*, 2024.
