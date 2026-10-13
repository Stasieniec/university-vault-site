---
type: lecture
course: 5204MNLP6Y
week: 5
lecture: 7
date: 2026-09-28
status: complete
topics:
  - From multilingual to crosslingual models
  - Zero-shot crosslingual knowledge transfer
  - Parallel and comparable data
  - Parallel corpus acquisition (OPUS, document and sentence alignment)
  - LASER sentence embeddings
  - vecalign sentence alignment
  - XLM and translation language modeling (TLM)
  - CLM, MLM and TLM pretraining objectives
  - XNLI evaluation settings (translate-train, translate-test, zero-shot)
  - InfoXLM (MMLM, TLM, XLCo)
  - Within-language versus across-language transfer
  - Sequence-to-sequence models and conditional language modeling
  - Encoder-decoder neural machine translation
  - Transformer encoder-decoder
  - Data requirements of NMT
  - Low-resource NMT and universal translation
  - Parent-child transfer learning for NMT
  - Parameter freezing and language relatedness in transfer
  - BART denoising pretraining
  - BART noise functions
  - BART fine-tuning for classification, span prediction and MT
  - Multilingual NMT (Johnson et al.) and target language tags
  - Many-to-one, one-to-many and many-to-many translation
  - Language sampling and temperature
  - Transfer against interference
  - Zero-shot translation
  - mBART multilingual denoising pretraining
---

# MNLP-L07: Crosslingual NLP

> [!abstract] In short
> Multilingual models such as mBERT and XLM-R share one network and one vocabulary across many languages, but every training example stays inside one language, so nothing tells them that a German sentence and its English translation mean the same thing. This lecture asks whether crosslingual ability emerges anyway, and what to do when it does not.
>
> **First half:** parallel data (translations) is the missing signal. LASER, XLM's translation language modelling and InfoXLM's contrastive objective use it. A test that mixes two languages inside one input shows that purely multilingual models are much weaker crosslingually than standard benchmarks suggest, and that these objectives fix most of it.
>
> **Second half:** machine translation. Neural MT is conditional language modelling with an encoder and a decoder, and it needs a lot of parallel data. Three ways around the shortage: parent-child transfer (train on a big pair, continue on the small one, keep the target embeddings frozen), multilingual NMT (one model for all directions, with a trade-off between transfer and interference and a sampling temperature to balance it), and pretraining the encoder-decoder on monolingual text (BART, and its multilingual version mBART).
>
> On the exam: example question 3.1 (fine-tuning a general-domain NMT system on a small in-domain set) is parent-child transfer. The multilingual NMT choices (many-to-one against one-to-many, which temperature) suit one-word-wrong statements. Where Christof Monz's lecture and the papers behind it disagree (decoder freezing, MMLM), a box says which way to answer.

## Multilingual models never see two languages at once

The models of [[MNLP-L06 - Contextual Embeddings|L06]] are multilingual in four senses: they **train jointly on many languages**, use **one model for all languages**, use **one subword vocabulary for all languages**, and base their **predictions on context within the same language** (masked language modelling, MLM). The fourth is the problem. When mBERT predicts a masked German word, every token it can see is German.

So training on many languages makes a model multilingual, but **nothing explicitly links information across languages**. When predicting word $W$ in language $L$, there is **no benefit in having access to unrelated context in another language $L'$**: a random French sentence elsewhere in the batch says nothing about a masked German token. The languages share parameters; they never share context.

Yet what people want from these models is crosslingual.

> [!definition] Crosslingual knowledge transfer
> Given **fine-tuning data in language A for task X**, how well does the model generalise to **test data for task X in language B**?

This is the common scenario, because annotated fine-tuning data rarely exists outside high-resource languages. When no task data in B was seen at all, it is called **zero-shot crosslingual transfer**. Transfer of this kind is a crosslingual capability: the model has to map what it learned in A onto B.

So the question is: **can crosslingual capabilities emerge without any crosslingual training signal?** Earlier zero-shot results seem to say yes: mBERT fine-tuned on English works, to a degree, on other languages. Crosslingual tasks, where a single input mixes two languages, cast doubt on it. The answer comes in "The harder test" below. First, the signal that is missing.

## Parallel data: text known to correspond

The fix is **context alignment across languages**: give the model text in two languages that is known to correspond, so that context in $L'$ really is useful for predicting $W$ in $L$. Two kinds of data provide it.

**Parallel data** is sentence pairs in different languages that are **meaning equivalent**: translations of each other. Data-driven MT has used it for decades.

**Comparable data** is sentence or document pairs **on the same topic**, not fully parallel: the degree of parallelism is a **sliding scale**. Examples: news articles about the same event, Wikipedia articles about the same entity.

Over millions of aligned pairs, recurring co-occurrences (新加坡 on the Chinese side whenever *Singapore* is on the English side) let a model learn which words correspond, with no dictionary.

**Where parallel data comes from.** Sometimes it is a natural by-product: **multilingual news** (Xinhua, the Chinese state news agency), **international organisations** (the UN and EU translate every official document into all official languages), and **websites of countries with two or more languages** (Canada, Belgium, the US). **OPUS** is a website with a very large collection of parallel corpora for research, covering many languages: the first place to look.

Otherwise you **crawl your own corpus**, in two steps:

1. **find parallel documents** (document alignment);
2. **find parallel sentences** inside them (sentence alignment).

Or skip step 1 and **align sentences directly in vast raw web crawls such as Common Crawl**.

Sentence alignment is its own step because translators do not keep paragraph boundaries: in one English-Chinese news story, two English paragraphs map into a single Chinese paragraph, which the aligner must split at a sentence boundary.

## LASER: one vector per sentence, in any language

Finding parallel sentences means **measuring degrees of meaning equivalence**. The old-fashioned measures are **dictionary overlap** (how many words of one sentence have a dictionary translation in the other) and **relative length distribution** (translations have predictable length ratios). The more recent one is to **compute and compare sentence embeddings**, specifically **LASER** (Artetxe and Schwenk, 2019): embed every sentence of every language in one shared vector space, in which translations are near neighbours.

LASER is an encoder-decoder **trained to translate, on parallel data**. Only the encoder is kept.

- **Encoder.** The sentence is split into BPE subwords (byte-pair encoding, from [[MNLP-L04 - Subword Segmentation|L04]]), looked up in **one embedding table shared by all languages**, and passed through a **stack of BiLSTMs** (bidirectional LSTMs). The top layer's states are **max-pooled** over positions (an element-wise maximum) into one fixed-size **sentence embedding**, whatever the sentence length. **The encoder is never told which language its input is in.**
- **Decoder.** An LSTM generates the translation. The sentence embedding reaches it twice: through a linear map $W$ that sets its initial state, and concatenated to its input at every step, together with the embedding of the previous output token and a **language ID embedding** $L_{id}$ naming the language to produce.

> [!intuition] Why the vectors are language-independent
> The decoder knows the source only through the sentence embedding, and learns which language to produce from $L_{id}$. So the encoder gains nothing by encoding the input's language, and everything by encoding its meaning, which is all the decoder needs to translate into its target languages. After training, the decoder is discarded, and translations from all training languages land close together in the encoder's space.

**vecalign** is the sentence alignment approach built on this: it **uses LASER to match sentences** (the similarity of two sentences is the similarity of their LASER embeddings), with an **efficient search that scales to massive data sets**.

## XLM: predict a word from its translation

mBERT **never sees an explicit translation pair during pretraining**, so any crosslingual alignment it has is an **emergent side effect**: no objective trains it. Yet for many language pairs at least some parallel text exists (OPUS, crawled corpora). **XLM** (Lample and Conneau, 2019) **uses parallel data for an explicit crosslingual loss**.

XLM has three pretraining objectives:

- **CLM, causal language modelling:** given a prefix, predict the next word.
- **MLM, masked language modelling:** predict a masked word from monolingual context, as in BERT.
- **TLM, translation language modelling:** predict masked words in a context made of **two languages**: a sentence and its translation.

**TLM is the crosslingual objective**, and it is **always combined with one of the monolingual ones**: MLM + TLM, or CLM + TLM. BERT's **next sentence prediction (NSP) is dropped**. The parallel corpora come from OPUS.

In standard notation, with $\mathbf{x} = (x_1, \dots, x_n)$ a token sequence, $M$ the set of masked positions and $\mathbf{x}_{\setminus M}$ the sequence with those positions replaced by `[MASK]`:

> [!formula] CLM, MLM and TLM
> $$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{n} \log p_\theta(x_t \mid x_1, \dots, x_{t-1})$$
> $$\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log p_\theta(x_i \mid \mathbf{x}_{\setminus M})$$
> $$\mathcal{L}_{\text{TLM}} = -\sum_{i \in M_x} \log p_\theta\big(x_i \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y}\big) - \sum_{j \in M_y} \log p_\theta\big(y_j \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y}\big)$$
>
> where:
> - $p_\theta$ is the Transformer's softmax over the shared vocabulary, with parameters $\theta$
> - $\mathbf{x}$ is a sentence in one language and $\mathbf{y}$ its translation, with masked positions $M_x$ and $M_y$
>
> TLM is MLM applied to the concatenation $[\mathbf{x}; \mathbf{y}]$. The loss is the same; only the input changes.

> [!example] One TLM training example
> The pair *the curtains were blue* / *les rideaux étaient bleus*, concatenated, with words masked on both sides. Each input position is the sum of three embeddings: token, position and language.
> ```
> English half (language en)
> pos  token     predict
> 0    [/s]
> 1    the
> 2    [MASK]    curtains
> 3    [MASK]    were
> 4    blue
> 5    [/s]
> French half (language fr)
> pos  token     predict
> 0    [/s]
> 1    [MASK]    les
> 2    rideaux
> 3    étaient
> 4    [MASK]    bleus
> 5    [/s]
> ```
> To predict *curtains*, the English context *the ___ ___ blue* is almost useless, but the French side is unmasked there and says *rideaux*. The cheapest way to lower the loss is to attend across and learn that *rideaux* corresponds to *curtains*; likewise *les* and *bleus* can be read off *the* and *blue*. Context in $L'$ is now useful for predicting $W$ in $L$.

Two details matter. **Position embeddings restart at 0** in the second sentence, so roughly corresponding words get similar position signals. **Language embeddings** (en, fr) mark which half is which.

### What TLM buys

**XNLI** (natural language inference with a premise and a hypothesis, test sets in 15 languages; see [[MNLP-L06 - Contextual Embeddings|L06]]) can be approached in three ways, all starting from English-only training data:

- **Translate-train:** machine-translate the English **training set** into each language and fine-tune on the translation.
- **Translate-test:** machine-translate each **test set into English** and apply the model fine-tuned on English.
- **Zero-shot:** fine-tune on English only and test directly on every language.

The results for XLM (exact accuracies are on the cheat sheet):

- **Adding TLM to MLM helps every language**, by a few points on average. The gains are largest for languages far from English in script or structure (Vietnamese, Turkish, Arabic, Chinese) and smallest for English itself.
- **On average over the 15 languages, zero-shot XLM (MLM+TLM) slightly beats translate-test with the same XLM model**, although it never saw non-English NLI data. Not in every language: translate-test still wins for every European language except English (a tie), and for Turkish and Arabic; zero-shot wins for languages far from English or with little data (Vietnamese, Thai, Chinese, Hindi, Swahili, Urdu).
- **Translate-train is still best on average** (Urdu is the exception), but it needs an MT system and a translated training set for every language.
- Zero-shot XLM with MLM+TLM beats both mBERT and LASER in every language where they are reported.

**Related languages help most.** An XLM language model for Nepali gets lower perplexity (better) with data from a second language added. English helps; **Hindi, closely related and in the same script, helps far more**; both together are best.

**Better aligned word embeddings.** XLM's input embeddings put translation pairs closer together than two fastText baselines: MUSE (the mapping method of [[MNLP-L05 - Static Embeddings|L05]]) and "Concat" (one fastText model on all languages concatenated), though XLM was never trained to align words directly.

## InfoXLM: align whole sentences too

**InfoXLM** (Chi et al., 2021) **builds on XLM-R** (the purely multilingual masked language model of L06) **but also includes the parallel-data ideas of XLM**. It has three objectives:

1. **MMLM, multilingual masked language modelling:** masked language modelling on monolingual text in many languages. Monz describes it as **similar to MLM, but instead of directly optimising the loss of the ground truth, it uses negative sampling**.
2. **TLM**, as in XLM.
3. **XLCo, crosslingual contrastive learning:** take the **representations of the `[CLS]` tokens** of the two sentences of a **parallel sentence pair**, and **classify whether the pair is parallel**, with **negative sampling, using random sentences as negatives**.

> [!warning] How to answer a statement about MMLM
> The exam is set from the lecture, so a statement in Monz's wording ("MMLM uses negative sampling") is most likely marked true. Background from the InfoXLM paper: MMLM is the ordinary masked-LM loss of XLM-R. The paper only reads its softmax over the vocabulary as a contrastive loss, with the correct token as the positive and every other vocabulary entry as a negative; nothing extra is sampled.

XLCo is [[Contrastive Learning]] at sentence level. In the standard InfoNCE form, "is this pair parallel?" becomes "which of these candidates is the translation?", with one true translation among random sentences:

> [!formula] XLCo loss
> $$\mathcal{L}_{\text{XLCo}} = -\log \frac{\exp\big(f(\mathbf{x})^\top f(\mathbf{y})\big)}{\exp\big(f(\mathbf{x})^\top f(\mathbf{y})\big) + \sum_{\mathbf{y}' \in \mathcal{N}} \exp\big(f(\mathbf{x})^\top f(\mathbf{y}')\big)}$$
>
> where:
> - $(\mathbf{x}, \mathbf{y})$ is a parallel sentence pair, the positive
> - $f(\cdot)$ is the encoder's `[CLS]` representation of a sentence
> - $\mathcal{N}$ is a set of negatives: random sentences that are not translations of $\mathbf{x}$
>
> Minimising the loss pulls translations together and pushes non-translations apart.

The three losses are **equally weighted**:

$$\mathcal{L}_{\text{InfoXLM}} = \mathcal{L}_{\text{MMLM}} + \mathcal{L}_{\text{TLM}} + \mathcal{L}_{\text{XLCo}}$$

> [!intuition] Token level and sentence level
> TLM aligns languages at the **token level**: a masked word is predicted from the words of its translation. XLCo aligns them at the **sentence level**: a sentence's representation must be closer to its translation's than to anything else. LASER got sentence-level alignment from a translation decoder; InfoXLM gets it from a contrastive loss inside masked-LM pretraining.

**How InfoXLM differs from XLM:**

- **Starting point:** XLM is trained from scratch; InfoXLM is **initialised from XLM-R** and then pretrained further.
- **Monolingual data:** Wikipedia for XLM; for InfoXLM, a version of CC-100 web text rebuilt by the InfoXLM authors, covering 94 languages.
- **Vocabulary:** shared BPE for XLM; XLM-R's SentencePiece vocabulary (another subword vocabulary, from L04) for InfoXLM.
- **Language embeddings:** XLM has them; InfoXLM does not, because it keeps XLM-R's architecture in order to keep its weights. Its TLM has to tell the two halves apart from the tokens alone.

**Parallel data is the same shape in both.** Monz gives "15 XNLI languages" for XLM and "14 English-centric pairs" for InfoXLM, but in the XLM paper each of the 14 other XNLI languages is paired with English too. Both are English-centric (English on one side of every pair, no French-German data); InfoXLM just has more of it.

## The harder test: two languages in one input

This is the within/across test of Rajaee and Monz, met in [[MNLP-L06 - Contextual Embeddings#Within a language, across languages (Rajaee and Monz)|L06]] for mBERT and XLM-R. This lecture adds InfoXLM, and QA alongside XNLI.

Standard zero-shot benchmarks keep every part of an input in one language, so a model can do well by understanding each language on its own. The harder test fine-tunes on English only (zero-shot transfer) and then evaluates inputs whose **two parts are in different languages**:

- **XNLI**: the **premise** in one language and the **hypothesis** in another (15 languages, 225 combinations).
- **XSQuAD** (QA, an English reading-comprehension set translated into 10 languages): the **context** paragraph in one language and the **question** in another (11 languages, 121 combinations).

> [!definition] Within and across
> - **within**$(L)$: the score when both parts of the input are in $L$
> - **across**$(L)$: the average score over all mixed-language combinations in which $L$ is one of the two languages, in either position (formula in L06)

**What the three models show** (mBERT, XLM-R, InfoXLM; exact scores on the cheat sheet):

- **Every model drops from within to across.**
- **XLM-R is much better than mBERT within languages, on both tasks. Across languages it is better on XNLI, but worse than mBERT on QA on average.** Being better at each language separately did not make it better at relating two languages to each other.
- **InfoXLM is only slightly better than XLM-R within a language, but much better across languages**: a moderate gain on XNLI and a very large one on QA. Its within-across gap is the smallest of the three on both tasks; on QA it shrinks to about a quarter of XLM-R's.

The full grids of language pairs add detail:

- **English is the best partner.** In all three models, for every premise language, an English hypothesis gives the best of the mixed-language pairs, and sometimes even beats the same-language pair. English dominates the pretraining and fine-tuning data.
- **Swahili is the weakest partner.** With a Swahili hypothesis mBERT is barely above chance (XNLI has three classes, so chance is 33.3%). InfoXLM lifts that column most.
- **The grids are not symmetric**: swapping premise and hypothesis languages changes the score.
- **mBERT is weak on Thai even within Thai.** XLM-R fixes Thai within Thai, but a Thai question about a context in another language fails for both models.
- **XLM-R on QA works across languages mainly when the question is in English.** An English context with a foreign question is much weaker, and a question in one non-English language about a context in another (Hindi question, Arabic context) mostly collapses.
- **InfoXLM fills in the grid**: no cell collapses, and the pairs XLM-R failed on become reasonable.

> [!intuition] Why QA exposes this more than XNLI
> XNLI compares two sentences, and a coarse sentence-level gist in a shared space is often enough. Extractive QA must find the exact span that answers the question, which means matching the question's words to specific words of the context. With a Hindi question and an Arabic context, that is word-level matching between two non-English languages: never asked for by MLM on separate languages, and what TLM's "predict a word from its translation" trains.

> [!tip] The answer to the open question
> Standard zero-shot benchmarks make purely multilingual models look crosslingual. Tasks that relate two languages inside one input show that much of that ability **does not emerge** without a crosslingual training signal, and that adding one (parallel data, through TLM and XLCo) fixes most of it.

## Machine translation is conditional language modelling

The second half turns from models that understand several languages to models that translate between them. Translation has the same bottleneck as the first half: it needs parallel data, and most language pairs have little of it. First, how neural MT works. (MT moved from rule-based, symbolic systems to statistical, data-driven ones, and from the mid-2010s to neural ones.)

### Why translation is not sequence labelling

In sequence labelling (part-of-speech tagging, named entity recognition) there is exactly one output per input position. **Sequence-to-sequence** modelling drops two of its assumptions: it does **not assume an isomorphic relationship between input and output positions** (the $t$-th output need not be about the $t$-th input), and it does **not assume that input and output have the same length**.

> [!example] Hiermit hörte sie nicht auf
> German *Hiermit hörte sie nicht auf.* (word by word: with-this stopped she not [particle of the verb]) is English *She did not stop with this.*
> - **Different lengths:** 6 tokens against 7.
> - **Word order differences:** *sie* moves from third position to first.
> - **One-to-many:** *Hiermit* → *with this*; *nicht* → *did not*.
> - **Many-to-one:** *hörte ... auf* → *stop* (the separable verb *aufhören*, split around the clause).

Seq2seq can be cast as **conditional language modelling**:

> [!formula] Seq2seq as conditional language modelling
> $$p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X}) \qquad\text{so}\qquad p(\mathbf{Y} \mid \mathbf{X}) = \prod_{t=1}^{|\mathbf{Y}|} p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X})$$
>
> where:
> - $\mathbf{X}$ is the output of the **encoder**, a representation of the source sentence
> - $\mathbf{Y}_{<t}$ is a representation of the **decoder**'s output before time $t$ (the prefix $y_1, \dots, y_{t-1}$)
> - $y_t$ is the next output token
>
> It is an ordinary language model over the output in which every prediction is also conditioned on the input. Training minimises $-\log p(\mathbf{Y} \mid \mathbf{X})$ over parallel sentence pairs.

**How a sentence pair becomes data.** Both sentences are tokenised, and each token is replaced by its index in its language's vocabulary ($V_f$ for the foreign side, $V_e$ for English). The **source** sentence becomes a vector $\mathbf{z}$. The **target** sentence becomes **two** vectors: $\mathbf{x}$, what the decoder **reads**, is `<s>` followed by the sentence; $\mathbf{y}$, what it must **predict**, is the sentence followed by `</s>`. So $\mathbf{y}_i = \mathbf{x}_{i+1}$: the same sentence shifted by one position.

> [!warning] Which letter is which
> From here on, $\mathbf{z}$ is the **source** and both $\mathbf{x}$ and $\mathbf{y}$ are on the **target** side. (In the XLM section $\mathbf{x}$ was a source sentence; here it is not.)

```
z: Hiermit hörte sie nicht auf .
x: <s> She did not stop with this .
y: She did not stop with this . </s>
```

At step $i$ the decoder has read the gold words up to $\mathbf{x}_i$ and must predict $\mathbf{y}_i$. Feeding the gold previous words during training, instead of the model's own guesses, is called **teacher forcing** (the standard name for this). At test time there are no gold words: the decoder feeds its own predictions back in until it produces `</s>`.

### The RNN encoder-decoder and its bottleneck

**Sutskever et al. (2014)** made both the **encoder** and the **decoder** an LSTM. The encoder reads $\mathbf{z}$ token by token and **represents it as a whole**; the decoder **reads $\mathbf{x}$ token by token and learns to predict $\mathbf{y}$**. The important question is how the two connect, and Sutskever et al. simply **initialise the decoder with the last state of the encoder**:

$$\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$$

where $\mathbf{h}^{enc}_n$ is the encoder's hidden state after the last ($n$-th) source token and $\mathbf{h}^{dec}_0$ is the decoder's initial state. This creates an **encoder-decoder information bottleneck**: everything the decoder knows about the source must fit into one fixed-size vector, however long the sentence.

### The Transformer encoder-decoder removes the bottleneck

In the Transformer version (mechanics in [[Transformers]] and [[Self-Attention]]), source tokens pass through $n$ encoder layers, which output one vector per source position. Each decoder layer has three sublayers, each with a residual connection (its input added to its output):

1. **Target context layer:** masked self-attention over the target prefix. Position $t$ may look only at itself and earlier positions.
2. **Source-target context layer:** attention from each target position over **all** outputs of the top encoder layer (cross-attention).
3. **Feed-forward layer**, applied to each position separately.

The decoder stacks $n$ such layers and ends in a softmax over the target vocabulary. The source-target context layer is what removes the bottleneck: every decoder position, in every layer, can read every source position, instead of one summary vector.

## Neural MT needs a lot of data

NMT quality depends to a large degree on **the amount of data**, **the amount of variation within the data**, and **the relevance of the data to the actual task**. In **low-resource NMT** some of these conditions are not met. Typical cases are **domain adaptation** (plenty of data, but from the wrong domain) and **NMT for low-resource language pairs** (little data at all).

**Universal translation is far off.** Schwenk et al. (2019) found that **86% of all language directions are of poor quality**. The core problem: **limited parallel training data for the majority of directions**, and current MT models **do not generalise very well beyond the training data, and not at all beyond specific language directions**. Directions involving English are by far the best; many others had too little data to build a system at all.

**NMT is data hungry.** With under two million tokens of parallel data (Hausa, Turkish, Uzbek, Urdu into English; Zoph et al., 2016), plain NMT loses clearly to an older syntax-based statistical system.

**How much is enough?** Koehn and Knowles (2017) plotted BLEU (a translation quality score, higher is better) against corpus size, doubling the data at each step, for neural MT, phrase-based statistical MT, and phrase-based MT with a big language model:

- **Neural MT starts catastrophically low**: with under half a million words it scores close to zero, while phrase-based MT is already usable.
- **Neural MT improves much faster** as data grows.
- It **overtakes plain phrase-based MT at tens of millions of words**, and phrase-based MT with a big language model only at around a hundred million.

Below tens of millions of words statistical MT wins, and most of the world's language pairs live there. Three ways around this follow.

## Transfer learning: start the child from a parent

**Learning tens of millions of parameters from small data is challenging**, so **initialise the model with "reasonable" parameters**. Transfer learning is **basically the same as fine-tuning**:

1. **train an NMT model on a language pair with large amounts of data** (the **parent**);
2. **continue training this model on the low-resource language pair** (the **child**).

The same recipe adapts a general-domain model to a small in-domain set: the general-domain model is the parent, the in-domain data the child.

Monz lists four conditions under which **transfer learning for NMT works best**:

- the **target language of parent and child is identical** (for example, both translate into English);
- the **source languages of parent and child are related**;
- **decoder parameters are frozen** (frozen: kept fixed at the parent's values, never updated);
- **child and parent languages are made to share the same vocabulary**.

The experiments below refine the third condition and qualify the first two.

**The setup** (Zoph et al., 2016): the parent is French→English, trained on 300M English tokens; the child is Uzbek→English, with 1.8M. Every parameter of the trained parent is copied into the child, and then:

- **Not shared:** the **source vocabulary** and the **source embeddings**. The child has its own source vocabulary; each Uzbek word takes over the embedding row of some arbitrary French word, a starting value that means nothing for it.
- **Shared:** the **target language** and **all other parameters** (target embeddings, attention, ...).
- **Freeze target-specific parameters.** The parent learned its English embeddings from far more data than the child has. The child is then trained on its own data with strong regularisation (dropout 0.5).

**Results.** Transfer improves all four children (Hausa, Turkish, Uzbek, Urdu) by several BLEU, and the smallest corpus, Urdu, gains most. With further tricks the transfer model overtakes the syntax-based statistical system for Hausa and closes most of the gap for the other three.

### What to freeze

Start the Uzbek child fully frozen and let one more group of parameters train at a time:

- **Nothing trainable: zero BLEU.** The **source embeddings** must be retrained, and they give the biggest single jump.
- **Each further group helps, up to and including the attention**: the source RNN (the encoder's recurrent layers), then the target RNN (the decoder's recurrent layers), then the attention. The best model trains everything except the target embeddings.
- **Unfreezing the target embeddings hurts**: a little for the target input embeddings, more when the target output embeddings are trained too. They were learned from 300M English tokens; retraining them on 1.8M **overfits** (fits the small data at the expense of what generalises).

> [!warning] "Decoder frozen" or "target embeddings frozen": how to answer
> Monz's list of conditions says **decoder parameters are frozen**. The exam is set from the lecture, so a statement in that wording ("transfer learning for NMT works best if decoder parameters are frozen") is most likely marked true. Background from the experiment: only the **target embeddings** need to stay frozen, and training the decoder's recurrent layers and the attention still helps. If a statement asks what the freezing experiment shows, the answer is the target embeddings.

**Example question 3.1, fine-tuning on a small in-domain set** (general domain = parent, in-domain = child). (A) "uses parameters from the general domain for initialisation" is true: the child starts as a copy of the parent. (B) "continuously improves with the number of in-domain updates" is false. In the lecture, letting more parameters adapt to small data helps only up to a point, then hurts; outside the course material, the standard result is the same for training time: quality peaks, then drops as the model overfits. (C) "does not suffer from catastrophic forgetting" is false (outside the course material): training only on the new data overwrites what the model knew about the general domain, so quality there drops. That loss is **catastrophic forgetting**.

### Which parent

**Relatedness helps** (Zoph et al.). For a Spanish→English child, a French parent helps more than a German parent, and both help enormously compared with no parent. French is closer to Spanish than German is. Uzbek, unrelated to French, gains far less from the same French parent than Spanish does.

**Structure transfers, as well as words** (Zoph et al.). **French'** is French with its vocabulary randomly reshuffled: each word type is consistently swapped for another, so grammar and word order stay French but no word keeps its French meaning. A French parent still lifts the French' child a lot. What transfers includes word order, syntax and the shape of the attention.

**With a shared vocabulary, unrelated parents help too** (Kocmi and Bojar, 2018). With one vocabulary shared by parent and child, an English→Estonian child gains about as much from an English→Czech or English→Russian parent (unrelated to Estonian) as from an English→Finnish one (related). Note that the target languages differ here (Czech and Estonian), and transfer still works. With a shared vocabulary, the parent's size matters more than its relatedness. One more finding, explained in the Kocmi and Bojar paper: **the size direction matters.** Swap parent and child so that the parent is the smaller corpus, and transfer helps little or **hurts**. Transfer pays only when it flows from the larger corpus to the smaller.

**What transfer actually transfers** (Aji et al., 2020). Children are Burmese, Indonesian and Turkish into English; parents are German→English and English→German. Each group of child parameters (the embeddings, and the **inner**, non-embedding layers) is either transferred from the parent or randomly initialised, and then trained either way.

- **Transferring everything is best.**
- **The inner layers carry most of the benefit**: inner layers alone get about half of the full gain.
- **Embeddings alone are worse than training from scratch on average**: parent embeddings without the parent's inner layers to interpret them are a bad initialisation.
- But embeddings **on top of** inner layers add a further clear gain: the two work together.
- The English→German parent, whose **source** is English, transferred about as well into X→English children as the German→English parent, whose **target** is English. That sits awkwardly with the "identical target language" condition.

## Multilingual NMT: transfer in parallel

Parent-child transfer moves knowledge **in stages**: first the parent, then the child. **Multilingual joint training** moves it **in parallel**. **Johnson et al. (2017)** use **one single system for all translation directions**, with a **simple architecture and training setup**.

> [!definition] Language combinations in multilingual NMT
> - **Many-to-one:** many different source languages translated into **one single target** language. Used in **training**.
> - **One-to-many:** one single source language translated into **many different target** languages. Used in **training**.
> - **Many-to-many:** many different source languages translated into many different target languages. **Testing only.**

> [!definition] Zero-shot translation
> Translating in a **language direction not encountered during training**. A model trained on German→English and English→French that is asked for German→French is translating zero-shot.

Many-to-many is testing only because the training data is English-centric (next section): the model never trains on a pair without English, so asking for one is exactly zero-shot.

**Two indicators that knowledge transfer happens:**

1. **performance on zero-shot directions**, since anything above zero there can only come from transfer;
2. whether the multilingual model **outperforms a baseline trained only on the direction-relevant data** (a bilingual system for that one pair).

### How it is trained

> [!definition] Multilingual NMT training (Johnson et al., 2017)
> - **Single-language-centric training**, typically **English**, because that is where the resources are: only $xx \rightarrow en$ and $en \rightarrow xx$ directions are used.
> - **Every training example is annotated with a target language tag**, prepended to the source sentence.
> - **One system for all languages: all parameters are shared** for all language combinations.
> - **One WordPiece vocabulary for all languages** (32k, 64k, ...; WordPiece is a subword vocabulary, from L04).
> - **Source, target and output embeddings are shared.**

The tag names the **target**: `<2es> How are you?` gives *¿Cómo estás?*, `<2ja> How are you?` gives *お元気ですか？*, and `<2en> ¿Cómo estás?` gives *How are you?*. The same English sentence goes to Spanish or Japanese depending only on the tag. Nothing names the source language; the model recognises it from the input. The tag is part of the source as fed to the encoder.

The architecture is a standard encoder-decoder; only the data changes, by one token per sentence. Every multilingual ability, zero-shot included, comes from the shared parameters.

**Language directions are mixed during training, even within a batch**: one batch can hold English→Spanish, German→English, English→Japanese and Korean→English examples, each with English on one side. **Language balancing is important**, and is achieved through **(weighted) sampling** of languages (see "Sampling temperature" below).

### Many-to-one helps; one-to-many is mixed

**Many-to-one** (into English; Johnson et al., on WMT data and on production data 10 to 100 times larger): the multilingual model **beats the bilingual baseline in every direction**, by small margins. Every direction trains the same English decoder, so sharing costs nothing on the target side.

**One-to-many** (English into several target languages): **mixed**. Some directions gain and others lose, by up to about two BLEU. Now the **decoder** has to produce several languages, and they **compete** for the same parameters. **Balancing moves the loss around rather than removing it**: whichever direction gets less of the training signal pays.

### Transfer against interference

> [!definition] Transfer and interference
> **Transfer:** a direction gets better because it shares a model with other languages. **Interference:** a direction gets worse because the other languages compete with it for the same model capacity.

**Arivazhagan et al. (2019)** trained on **103 languages**, grouped into high-, medium- and low-resource languages, and compared bilingual baselines with three multilingual models: **Any→En** (many-to-one), **En→Any** (one-to-many) and **All→All** (one model for both $xx \rightarrow en$ and $en \rightarrow xx$: many-to-one and one-to-many together, still English-centric).

- **Low-resource languages gain from transfer**: a lot when translating into English, only a little out of it.
- **High-resource languages lose to interference.** **No multilingual model beats the bilingual baselines on the high-resource group.**
- **Into English transfers far more than out of English**, the same asymmetry as in Johnson et al.
- **All→All is the weakest multilingual option in every group**: one model for both directions spreads its capacity thinnest.

### Sampling temperature

How the languages are sampled during training decides who wins:

- **Proportional** sampling (by data size): **high-resource languages dominate**, and **low-resource languages perform poorly**.
- **Uniform** sampling (every language equally often): **high-resource languages suffer**, and **low-resource languages can see strong performance**.
- **The best compromise lies in between: over-sample low-resource languages and under-sample high-resource ones.** It is the same trade-off as language sampling in multilingual pretraining ([[MNLP-L06 - Contextual Embeddings#Sampling languages with a smoothing exponent|L06, sampling languages with a smoothing exponent]]).

One parameter sets where between these two the sampling sits: the **sampling temperature** $T$. The settings and results here are from Arivazhagan et al. (2019).

> [!formula] Temperature-based sampling
> $$p_l \propto \left(\frac{n_l}{\sum_{l'} n_{l'}}\right)^{1/T}$$
>
> where:
> - $p_l$ is the probability that the next training example comes from language (pair) $l$
> - $n_l$ is the amount of training data for $l$
> - $T$ is the temperature
>
> **$T = 1$ is proportional** sampling. **$T = 100$ is (effectively) uniform**: the exponent $1/T$ is close to 0, so all $p_l$ are close to equal. **$T = 5$ is in between.** Higher temperature means flatter sampling and more weight on low-resource languages.
>
> (Standard formula; only the settings are from the course.) It is the $p_l \propto n_l^{\alpha}$ of L06 with $\alpha = 1/T$, so $T = 5$ is $\alpha = 0.2$.

Results, for the same three groups:

- **$T = 1$ (proportional) is best for high-resource languages** among the multilingual models **and wrecks the low-resource ones**, which fall well below their bilingual scores.
- **$T = 100$ (uniform) is best for low-resource languages and worst for high-resource ones.**
- **$T = 5$ is the compromise**: nearly as good as uniform for low-resource languages, clearly better than uniform for high-resource ones, and best for the medium group. (The All→All model above was trained with $T = 5$.)

### Zero-shot translation

**Zero-shot translation, with no data for the given pair, has to rely entirely on transfer.** A model trained on 10 languages is compared with one trained on 102 (probably the 103-language model above without English), on six directions never seen in training:

- **Decent improvement in most cases**: five of the six directions gain from more languages (Yiddish→German more than doubles), but French→Chinese gets worse.
- **It probably also depends on which languages are included.** Belarusian→Russian (closely related) is high even with 10 languages; Hindi→Finnish (unrelated) stays very low.
- **Zero-shot translation is still lagging behind** supervised directions.

## BART: pretrain the encoder-decoder on monolingual text

Every MT method so far needed parallel data, which is what low-resource pairs lack. **Can we also reduce our dependence on parallel data by using monolingual data?**

Pretrained models so far came in two shapes:

- **BERT uses only an encoder.** It is bidirectional (every position sees the whole input) and predicts masked tokens from the encoder outputs.
- **GPT uses only a decoder.** It is autoregressive (position $t$ sees only earlier positions) and predicts the next token.

Translation and summarisation need both. **BART** combines them: a **bidirectional encoder** reads a **corrupted** document, and an **autoregressive decoder** reconstructs the **original** document, attending to the encoder through cross-attention. That makes BART a **denoising autoencoder**; its pretraining loss is the negative log-likelihood of the original document under the decoder. (BART was pretrained on English text only, per the BART paper.)

### The noise functions

Starting from the two-sentence document `A B C . D E .`:

| noise function | encoder input |
|---|---|
| token masking | `A _ C . _ E .` |
| token deletion | `A . C . E .` |
| text infilling | `A _ . D _ E .` |
| sentence permutation | `D E . A B C .` |
| document rotation | `C . D E . A B` |

In every case the decoder must produce the original `A B C . D E .`.

- **Token masking:** source tokens are replaced by `[MASK]`; the decoder predicts the masked tokens.
- **Token deletion:** source tokens are deleted; the decoder predicts the deleted tokens. Nothing marks the gap, so the model must also work out **where** tokens are missing.
- **Text infilling:** a **sequence** of source tokens is replaced by a **single** `[MASK]`; the decoder predicts the whole sequence. The model must also work out **how many** tokens the mask stands for. In the BART paper span lengths are drawn from a Poisson distribution with $\lambda = 3$, and a span of length 0 inserts a `[MASK]` where nothing was removed (the `D _ E` above).
- **Sentence permutation:** the sentences of the document are randomly shuffled; the decoder generates them in the original order.
- **Document rotation** (also called document permutation): a random token becomes the start, followed by the rest of the document, followed by the original beginning; the model has to identify where the document really starts.

**Which works best** (base-size models trained under the same conditions, compared on SQuAD question answering, MNLI inference and perplexity on four generation tasks):

- **Text infilling is the best single noise function**: best or close to best on almost every task.
- **Token deletion beats token masking** on every generation task, consistent with deletion being the harder task.
- **Document rotation and sentence permutation on their own are poor**: rearranging whole sentences gives too weak a signal at the token level. **Combined with text infilling**, sentence permutation gives the best perplexity on CNN/DailyMail summarisation.
- A plain **left-to-right language model** is best on one generation task (long-form question answering) but worst on SQuAD, because extractive QA needs bidirectional context.

At large scale, BART **matches RoBERTa** (BERT retrained longer, on more data) on SQuAD, so the decoder costs nothing on understanding tasks, and it is **the best on summarisation**, most clearly on the abstractive XSum (one-sentence summaries that cannot be copied from the article).

### Fine-tuning BART

- **Classification:** the same uncorrupted input goes into encoder and decoder, and the **label is predicted from the last hidden time step of the decoder**, which has attended to the whole input.
- **Span prediction** (SQuAD): **each token is labelled**, and the model learns to predict the **beginning and end** labels of the answer span.
- **Machine translation:** a **randomly initialised source-language encoder replaces the embedding layer of the BART encoder**. The new encoder maps the foreign sentence into vectors that pretrained BART can treat like (noisy) English, and BART "denoises" them into English. During fine-tuning **the pretrained BART encoder's parameters can remain fixed (frozen) or be updated**. **The source vocabulary can differ from BART's**, since only the new encoder sees foreign tokens.

```
foreign tokens
  ↓ new encoder (random init)
  ↓ pretrained BART encoder
  ↓ pretrained BART decoder
English
```

On Romanian→English, **Fixed BART** (pretrained BART frozen, only the new encoder trained) is slightly worse than a baseline MT system without BART; **Tuned BART** (BART parameters updated as well) beats the baseline by about one BLEU. Pretraining helps translation here only when the pretrained parameters may adapt. Since BART saw only English in pretraining, the new encoder has to do all the Romanian-specific work.

## mBART: BART in many languages

**BART is to mBART as BERT is to mBERT**: the same recipe, run on many languages with one shared model. **mBART** (Liu et al., 2020):

- is **one model trained on a concatenation of monolingual data in multiple languages**;
- uses the **BART objective, always applied within the same language**: one language in, the same language out, so there is **no crosslingual signal**.

So mBART sits with mBERT and XLM-R: unlike TLM or XLCo, nothing in its pretraining links one language to another.

**Pretraining.** The noise is **text infilling plus sentence permutation**. An instance can be several sentences long, up to a whole document, with `</s>` between sentences. A **language ID token** (`<En>`, `<Ja>`) **ends the encoder input and starts the decoder input**, so the decoder's first token tells it which language to generate.

```
enc in:  Where did __ from ? </s>
         Who __ I __ </s> <En>
dec in:  <En> Who am I ? </s>
         Where did I come from ? </s>
dec out: Who am I ? </s>
         Where did I come from ? </s> <En>
```

The two sentences are swapped (sentence permutation) and spans are masked (text infilling).

**Fine-tuning for translation.** The pretrained model is **fine-tuned directly on parallel data**, at sentence or document level. The source language's tag ends the encoder input and the **target** language's tag starts the decoder.

> [!tip] BART against mBART for translation
> BART was pretrained on English only, so translating with it needs a **new randomly initialised source encoder** in front of it. mBART has seen every language it translates during pretraining, so the **whole model is fine-tuned as is**, with no extra component. Neither has a crosslingual signal in pretraining.

**Results.** mBART pretrained on 25 languages ("mBART25") is compared with the same model trained from random initialisation, on language pairs with anywhere from 10K to 4.5M sentence pairs of parallel data:

- **mBART beats random initialisation for every pair, in both directions.**
- **The gains are largest in the low-to-medium range of a few hundred thousand pairs** (English-Vietnamese, English-Turkish).
- **Almost no data cannot be rescued**: with 10K pairs (English-Gujarati) BLEU stays near zero.
- **The gains mostly shrink as the parallel data grows** into the millions.

Pretraining matters most when parallel data is scarce but not absent: the same pattern as parent-child transfer.

## The methods side by side

| method | trained on | crosslingual signal | yields |
|---|---|---|---|
| mBERT, XLM-R | monolingual text | none | encoder |
| LASER | parallel text | translation decoder | sentence vectors |
| XLM | mono + parallel | TLM (tokens) | encoder |
| InfoXLM | mono + parallel | TLM + XLCo | encoder |
| parent-child | big pair, then small | translation pairs | NMT model |
| multilingual NMT | English-centric pairs | translation pairs | NMT model |
| BART | English monolingual | none | seq2seq model |
| mBART | multilingual mono | none | seq2seq model |

## What to be able to answer

- Why MLM on many languages leaves them unlinked; crosslingual and zero-shot transfer.
- Parallel against comparable data; document then sentence alignment, or mining Common Crawl.
- LASER: trained on parallel data, max pooling, language ID in the decoder only; vecalign.
- CLM, MLM, TLM: positions restart at 0, NSP dropped, TLM always paired; translate-train, translate-test, zero-shot, and what holds only on average.
- InfoXLM: initialised from XLM-R; MMLM (Monz: "negative sampling"), TLM, XLCo; token against sentence level.
- Within against across: XLM-R against mBERT on QA; InfoXLM across.
- Seq2seq assumptions, conditional LM, $\mathbf{z}$, $\mathbf{x}$, $\mathbf{y}$, the bottleneck, cross-attention.
- Parent-child: Monz's four conditions, what the experiment freezes, relatedness, French', shared vocabulary, size direction, inner layers; example question 3.1 (answer A).
- Multilingual NMT: target tag, sharing, English-centric data, zero-shot, many-to-one against one-to-many, interference, $T$ = 1, 5, 100.
- BART noise functions and fine-tuning, Fixed against Tuned BART; mBART's language tokens and where it helps.

Full model answers are in the exam questions below.

## Exam questions

> [!exam]- Why does multilingual pretraining (mBERT, XLM-R) not by itself make a model crosslingual, and what evidence shows that explicit crosslingual objectives fix it?
> **Key points:** MLM uses only same-language context, so alignment is an emergent side effect; standard zero-shot keeps the input in one language; within against across on mixed-language XNLI and QA; XLM-R beats mBERT within but loses across on QA; InfoXLM's TLM and XLCo close most of the gap.
>
> 1. **The mechanism.** Multilingual models train jointly on many languages with one model and one subword vocabulary, but MLM predicts a word only from **same-language context**. When predicting word $W$ in language $L$, unrelated context in $L'$ is of no use, so nothing in the objective links languages. Any alignment is an **emergent side effect**.
> 2. **Why it looked fine.** Standard zero-shot benchmarks (fine-tune on English, test on another language) keep all parts of the input in one language, and mBERT does reasonably there.
> 3. **The harder test.** Put premise and hypothesis (XNLI) or context and question (XSQuAD) in **different languages**, and compare **within** (both parts in one language) with **across** (mixed, with the language in either position).
> 4. **The result.** Every model drops from within to across. XLM-R is much better than mBERT within languages, but **worse than mBERT across languages on QA** on average: being good at each language separately does not make a model good at relating two of them. On QA it works across languages mainly when the question is in English.
> 5. **The fix.** InfoXLM adds TLM and XLCo on parallel data. Within a language it is only slightly better than XLM-R; across languages it is much better, most of all on QA, where the within-across gap shrinks to about a quarter.
>
> Losing marks: treating "multilingual" and "crosslingual" as the same thing, or citing only standard zero-shot scores as proof of crosslingual ability.

> [!exam]- You have English-only training data for NLI and need a system for 14 other languages. Compare translate-train, translate-test and zero-shot transfer, using the XLM results on XNLI.
> **Key points:** translate-train translates the training set into each language (best on average, but costly); translate-test translates the test set into English; zero-shot fine-tunes on English only; zero-shot XLM (MLM+TLM) slightly beats translate-test on average, though not in every language; TLM gives the gain over MLM alone.
>
> - **Translate-train:** machine-translate the English training set into each language and fine-tune on the translation. The best of the three for XLM on average (Urdu is the exception), but it needs an MT system and a translated training set for every language.
> - **Translate-test:** machine-translate each test example into English and apply the same XLM model fine-tuned on English.
> - **Zero-shot crosslingual transfer:** fine-tune on English only and test directly on each language.
> - On average over the 15 languages, zero-shot XLM with MLM+TLM **slightly beats translate-test**, although it never saw non-English NLI data. Per language it does not always win: translate-test is better for every European language except English (a tie), and for Turkish and Arabic; zero-shot wins for Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu. TLM is what gets it there: XLM with MLM alone is a few points lower on average, and the TLM gain is largest for languages far from English (Vietnamese, Turkish, Arabic, Chinese).
> - Zero-shot XLM (MLM+TLM) also beats mBERT and LASER in every language where they are reported.
>
> Losing marks: mixing up which side gets translated (training set into the target language, or test set into English).

> [!exam]- Explain translation language modelling (TLM): its input, its loss, why it forces crosslingual alignment, and how it differs from MLM and CLM.
> **Key points:** TLM is MLM over a sentence concatenated with its translation; same loss, different input; a masked word is predictable from its translation, which forces alignment; positions restart at 0 and language embeddings mark the halves; CLM predicts the next word, MLM uses monolingual context.
>
> - **XLM** (Lample and Conneau, 2019) has three objectives: **CLM** (predict the next word from a prefix), **MLM** (predict masked words from monolingual context, as in BERT), **TLM** (predict masked words in a **sentence concatenated with its translation**). TLM is always paired with MLM or CLM; NSP is dropped; the parallel data comes from OPUS.
> - **Loss:** TLM is MLM applied to $[\mathbf{x}; \mathbf{y}]$:
> $$\mathcal{L}_{\text{TLM}} = -\sum_{i \in M_x} \log p_\theta(x_i \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y}) - \sum_{j \in M_y} \log p_\theta(y_j \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y})$$
> The loss is the same as MLM's; only the input changes.
> - **Why it aligns:** in *"the [MASK] [MASK] blue" / "[MASK] rideaux étaient [MASK]"*, the English context barely constrains *curtains*, but the unmasked French *rideaux* gives it away. The cheapest way to lower the loss is to attend across languages and learn that *rideaux* corresponds to *curtains*. Context in $L'$ becomes useful for predicting $W$ in $L$.
> - **Two input details:** position embeddings **restart at 0** in the second sentence (so position cannot separate the halves, and corresponding words get similar position signals); **language embeddings** (en, fr) mark which half is which.
> - **Evidence:** on zero-shot XNLI, adding TLM to MLM improves every language.

> [!exam]- How would you build a parallel corpus from the web, and how do LASER embeddings make sentence alignment possible?
> **Key points:** OPUS and natural sources (UN, EU, multilingual news and countries); document alignment then sentence alignment, or direct mining from Common Crawl; LASER: trained on parallel data, BiLSTM encoder max-pooled into one sentence vector; the language ID goes only to the decoder, which makes the vector language-independent; vecalign aligns by LASER similarity at scale.
>
> - **Sources first:** natural by-products (multilingual news such as Xinhua, the UN and EU, websites of multilingual countries such as Canada and Belgium) and **OPUS**, a large research collection of parallel corpora.
> - **Crawl pipeline:** (1) document alignment (find parallel documents), (2) sentence alignment inside them. Or mine sentences directly from raw crawls such as **Common Crawl**.
> - **Why sentence alignment is a separate step:** translations do not keep paragraph boundaries (in one English-Chinese news story, two English paragraphs map into one Chinese paragraph, which must be split at a sentence boundary).
> - **Measuring equivalence:** old methods use dictionary overlap and relative length; the modern method compares **LASER** sentence embeddings (Artetxe and Schwenk, 2019).
> - **LASER:** an encoder-decoder trained to translate on parallel data. BPE embeddings, stacked BiLSTM encoder, max pooling into one sentence vector; the encoder is never told the input language. An LSTM decoder knows the source only through that vector, and is told the output language by a language ID embedding. So the vector must encode meaning and has no reason to encode the input language. Translations end up as near neighbours.
> - **vecalign** uses LASER similarities to match sentences, with an efficient search that scales to massive data sets.

> [!exam]- Explain neural machine translation as conditional language modelling, from the RNN encoder-decoder to the Transformer encoder-decoder.
> **Key points:** seq2seq drops $x_t \leftrightarrow y_t$ and $|X| = |Y|$; conditional LM $p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X})$; decoder input is the target shifted by one; RNN $\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$ is an information bottleneck; Transformer cross-attention over all encoder outputs removes it.
>
> - Seq2seq drops two assumptions of sequence labelling: that $x_t$ corresponds to $y_t$, and that $|X| = |Y|$. MT needs both dropped (*Hiermit hörte sie nicht auf* / *She did not stop with this*: different lengths, reordering, one-to-many, many-to-one).
> - It is **conditional language modelling**: $p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X})$, with $\mathbf{X}$ the encoder output and $\mathbf{Y}_{<t}$ the decoder's prefix. The sequence probability is the product of these terms over $t$.
> - **Data:** source ids $\mathbf{z}$; decoder input $\mathbf{x}$ = `<s>` + sentence; target $\mathbf{y}$ = sentence + `</s>` ($\mathbf{x}$ shifted by one). Training feeds the gold previous words.
> - **RNN encoder-decoder** (Sutskever et al., 2014, LSTMs): $\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$. Everything about the source must pass through one fixed-size vector: an **information bottleneck**.
> - **Transformer:** each decoder layer has a target context layer (masked self-attention over the target prefix), a **source-target context layer** (attention over all top-layer encoder outputs) and a feed-forward layer, each with a residual connection. Every decoder position in every layer can read every source position, which removes the bottleneck.

> [!exam]- Describe parent-child transfer learning for low-resource NMT. What should be frozen, and which factors decide whether transfer pays off?
> **Key points:** train a parent on a high-resource pair, continue on the child with the same target language; freeze the decoder (Monz's condition), which the experiment narrows to the target embeddings; related parents help, but structure transfers too (French'); with a shared vocabulary unrelated parents help; transfer must flow from the larger corpus to the smaller; inner layers carry most of the benefit.
>
> 1. **Procedure** (Zoph et al., 2016): train a parent on a high-resource pair (French→English); copy all parameters into the child (Uzbek→English, far less data); child source words take over rows of the parent's source embedding matrix; freeze the English embeddings; continue training with strong regularisation (dropout 0.5).
> 2. **Gains:** every child improves; the smallest corpus (Urdu) gains most.
> 3. **Freezing:** Monz lists "decoder parameters are frozen" as a condition, and a statement in that wording is most likely marked true. The freezing experiment narrows it: with everything frozen the child scores zero; training source embeddings, encoder, decoder RNN and attention each helps; unfreezing the **target embeddings** then lowers BLEU, because retraining embeddings learned from far more English data on the small child data overfits.
> 4. **Relatedness:** a Spanish child gains more from a French parent than from a German one. But French' (scrambled vocabulary) still gains a lot from a French parent, so structure transfers, as well as shared words.
> 5. **Shared vocabulary** (Kocmi and Bojar, 2018): English→Czech and English→Russian parents (unrelated) help an English→Estonian child about as much as an English→Finnish one (related).
> 6. **Direction:** transfer must flow from the larger corpus to the smaller; reversed, it helps little or hurts.
> 7. **What transfers** (Aji et al., 2020): inner layers carry most of the benefit; parent embeddings alone are worse than training from scratch; both together are best.

> [!exam]- Describe BART: its architecture, its pretraining noise functions, which noise works best, and how it is fine-tuned for classification, span prediction and translation.
> **Key points:** bidirectional encoder plus autoregressive decoder reconstructing corrupted text; five noise functions; text infilling best, rotation and permutation alone poor; classification from the last decoder state, start and end labels for spans; MT via a new randomly initialised source encoder, and only Tuned BART beats the baseline.
>
> - **Architecture:** bidirectional encoder (as BERT) plus autoregressive decoder (as GPT). The encoder reads a **corrupted** document, the decoder reconstructs the **original** through cross-attention: a denoising autoencoder. Suited to tasks needing both, such as translation and summarisation.
> - **Noise functions:** token masking, token deletion (the model must find where tokens are missing), text infilling (a span replaced by one `[MASK]`; the model must find how many tokens are missing), sentence permutation, document rotation.
> - **Best:** **text infilling**; deletion beats masking on generation tasks; rotation and sentence permutation alone are poor; infilling plus permutation gives the best CNN/DailyMail perplexity.
> - **Classification:** same input to encoder and decoder; label predicted from the last decoder hidden state. **Span prediction (SQuAD):** label each token, predict start and end of the answer.
> - **MT:** a randomly initialised source encoder replaces the embedding layer of BART's encoder; the pretrained BART encoder can be frozen or updated; the source vocabulary can differ from BART's. On Romanian→English, Fixed BART (frozen) is slightly below the baseline, Tuned BART (updated) beats it by about one BLEU.
> - At scale BART matches RoBERTa on SQuAD and is the best summariser, with its largest lead on the abstractive XSum.

> [!exam]- Why does crosslingual QA expose the weakness of purely multilingual models much more than XNLI does? Use the XLM-R and InfoXLM results.
> **Key points:** XNLI can rely on a sentence-level gist; QA needs word-level matching across languages; XLM-R QA works across languages mainly with an English question; InfoXLM fills the grid; InfoXLM's across gain is far larger on QA than on XNLI.
>
> - **XNLI** compares two sentences; a coarse sentence-level gist in a shared space is often enough.
> - **Extractive QA** must find the exact answer span, which means matching the question's words to specific context words. With a Hindi question and an Arabic context that is **word-level matching between two non-English languages**, never asked for by MLM on separate languages, and exactly what TLM trains.
> - **XLM-R QA:** strong when both parts share a language, and mainly works across languages with an English question; a question in one non-English language about a context in another mostly collapses. Its across score falls below mBERT's.
> - **InfoXLM QA:** no cell collapses; the pairs XLM-R failed on become reasonable.
> - InfoXLM's across gain over XLM-R is moderate on XNLI and very large on QA.

> [!exam]- Which factors predict whether crosslingual transfer will succeed? Support each with evidence.
> **Key points:** language relatedness; pairing with English; how well the model represents the language; an explicit crosslingual training signal; size and direction of the data in NMT transfer, and which parameters transfer.
>
> - **Language relatedness:** for Nepali language modelling, Hindi data helps far more than English data; a Spanish NMT child gains more from a French parent than from a German one; zero-shot Belarusian→Russian is good, Hindi→Finnish poor.
> - **Pairing with English:** English dominates pretraining and fine-tuning data, so an English hypothesis gives the best mixed-language pairs in every XNLI model, and MT directions involving English are by far the best.
> - **How well the model represents the language:** with a Swahili hypothesis mBERT is barely above chance; mBERT is weak on Thai even within Thai; XLM-R fixes Thai within Thai, but on QA a Thai question about a context in another language still fails.
> - **An explicit crosslingual training signal:** TLM and XLCo on parallel data close most of the within-across gap (InfoXLM).
> - **Amount and direction of data in NMT transfer:** transfer helps when it flows from a large parent to a small child; reversed it helps little or hurts. With a shared vocabulary and a large parent, relatedness matters less (Kocmi and Bojar).
> - **Which parameters transfer:** inner layers carry most of the benefit (Aji et al.).

> [!exam]- Describe multilingual NMT as in Johnson et al. (2017): training data, how the output language is chosen, what is shared, and how zero-shot translation arises.
> **Key points:** one system with all parameters shared, transfer in parallel; English-centric data ($xx \rightarrow en$, $en \rightarrow xx$) mixed within batches; target language tag prepended to the source; one WordPiece vocabulary and shared embeddings; non-English pairs are zero-shot.
>
> 1. **One single system for all translation directions**, trained jointly: knowledge is transferred **in parallel**, as opposed to parent-child transfer **in stages**.
> 2. **Data:** a collection of parallel corpora, **English-centric**: only $xx \rightarrow en$ and $en \rightarrow xx$ directions, because that is where the resources are. Directions are **mixed during training, even within a batch**.
> 3. **Target language tag:** every training example gets a tag such as `<2es>` prepended to the **source** as fed to the encoder. It names the target language; the source language is never named.
> 4. **Shared everything:** all parameters shared for all language combinations, one WordPiece vocabulary for all languages (32k, 64k), shared source, target and output embeddings. The architecture is a standard encoder-decoder.
> 5. **Zero-shot:** training covers many-to-one and one-to-many; asking for a non-English pair (many-to-many, testing only) such as German→French is a direction never seen in training and relies entirely on transfer.
> 6. **Balancing:** languages are sampled with weights, between proportional (high-resource languages dominate) and uniform (high-resource languages suffer), set by a temperature ($T = 1$ proportional, $T = 100$ uniform, $T = 5$ the compromise; the temperature settings are from Arivazhagan et al., 2019).
>
> Losing marks: saying the tag names the source language, or that the model trains on non-English pairs.

> [!exam]- Compare many-to-one and one-to-many multilingual NMT, and explain transfer, interference and the choice of sampling temperature.
> **Key points:** many-to-one beats bilingual baselines in every direction, one-to-many is mixed; the decoder's languages compete; low-resource languages gain (mostly into English), high-resource ones lose; All→All is weakest; $T = 1$ favours high-resource, $T = 100$ low-resource, $T = 5$ the compromise.
>
> - **Many-to-one** (into English): the multilingual model beats the bilingual baseline for every direction, by small margins, even on large production data. All directions share one target side.
> - **One-to-many** (out of English): some directions gain, others lose, because the decoder must produce several languages that compete for the same parameters. Oversampling only moves the loss to whichever direction gets less training signal.
> - **Transfer** (a direction improves by sharing a model) against **interference** (it gets worse because others compete for capacity). Across 103 languages (Arivazhagan et al., 2019): low-resource languages gain, far more into English than out of it; high-resource languages lose, and no multilingual model beats their bilingual baselines; All→All is the weakest multilingual model.
> - **Temperature** (Arivazhagan et al.) $T$ in $p_l \propto (n_l / \sum n)^{1/T}$: $T = 1$ (proportional) is best for high-resource languages but wrecks low-resource ones; $T = 100$ (uniform) is best for low-resource and worst for high-resource; $T = 5$ is nearly as good as uniform for low-resource languages and better for high-resource ones.

## Flashcards

> [!card]- Which of the four senses in which mBERT and XLM-R are "multilingual" leaves languages unlinked?
> Their predictions are based on context **within the same language** (MLM).

> [!card]- What is crosslingual knowledge transfer?
> Fine-tuning on task data in **language A**, then testing on the same task in **language B**.

> [!card]- What is crosslingual transfer called when no task data in the test language was seen at all?
> **Zero-shot** crosslingual transfer.

> [!card]- Why does training on many languages not by itself link them?
> Predicting word $W$ in language $L$ gains nothing from unrelated $L'$ context: languages share **parameters**, never **context**.

> [!card]- What is the difference between parallel and comparable data?
> Parallel: **meaning-equivalent** pairs (translations). Comparable: **same topic**, parallel only to a degree on a sliding scale.

> [!card]- What are the two steps of crawling your own parallel corpus?
> **Document alignment**, then **sentence alignment** inside the aligned documents.

> [!card]- Why is sentence alignment a separate step after document alignment?
> Parallel segments **do not follow paragraph boundaries**, so links other than paragraph-to-paragraph are needed.

> [!card]- How does the LASER encoder turn a sentence into one vector?
> BPE embeddings, a **stack of BiLSTMs**, then **max pooling** over time into a fixed-size sentence embedding.

> [!card]- Why are LASER sentence embeddings language-independent?
> The decoder sees the source only through the sentence vector and gets the output language from a **language ID embedding**, so the encoder need encode only **meaning**.

> [!card]- What data is LASER trained on?
> **Parallel** data: it is an encoder-decoder trained to translate.

> [!card]- Which sentence alignment method matches sentences by LASER embedding similarity, with a search that scales to massive data sets?
> **vecalign**.

> [!card]- What does XLM add that mBERT lacks?
> An **explicit crosslingual loss** on parallel data. mBERT never sees a translation pair, so its alignment is an emergent side effect.

> [!card]- What are XLM's three pretraining objectives?
> **CLM**, **MLM**, **TLM**.

> [!card]- Which XLM objective masks words in a sentence concatenated with its translation?
> **TLM**, translation language modelling.

> [!card]- How does TLM differ from MLM?
> The loss is the **same**; only the **input** differs: a translation pair instead of monolingual text.

> [!card]- Why does TLM force crosslingual alignment?
> A masked word can be predicted from its **unmasked translation** in the other language, so the model learns to attend across languages.

> [!card]- What is the difference between translate-train and translate-test on XNLI?
> Translate-train machine-translates the **English training set** into each language; translate-test translates each **test set into English**.

> [!card]- Does zero-shot XLM (MLM+TLM) beat translate-test on XNLI?
> **On average, slightly**, without ever seeing non-English NLI data; not in every language.

> [!card]- Which languages gain most on XNLI when TLM is added to MLM?
> Those **distant from English** in script or structure (e.g. Vietnamese, Turkish, Arabic, Chinese). English itself gains least.

> [!card]- For XLM language modelling of Nepali, does adding Hindi or English data help more?
> **Hindi**, the related language, by far. Adding both is best.

> [!card]- What are InfoXLM's three objectives?
> **MMLM**, **TLM**, **XLCo**.

> [!card]- What does InfoXLM's XLCo objective classify?
> Whether a sentence pair is **parallel**, from the two `[CLS]` representations, with **random sentences** as negatives.

> [!card]- What is the difference between the crosslingual signal of TLM and of XLCo?
> TLM aligns at the **token level**; XLCo aligns **whole-sentence** representations.

> [!card]- What is the difference between the within and across scores in mixed-language XNLI and QA?
> Within: both input parts in language $\ell$. Across: mean over **mixed-language** combinations with $\ell$ in **either** part.

> [!card]- What is paradoxical about XLM-R against mBERT in mixed-language evaluation?
> XLM-R is much better **within** languages but **worse across** languages on QA, on average.

> [!card]- How does InfoXLM compare with XLM-R on within and across scores?
> Only **slightly better within**, **much better across**: the within-across gap shrinks sharply.

> [!card]- Why does crosslingual QA expose purely multilingual models more than XNLI?
> Extractive QA needs **word-level matching** of question to context across languages; XNLI can rely on a coarse sentence gist.

> [!card]- What do mixed-language evaluations show about crosslingual ability without a crosslingual training signal?
> It largely **does not emerge**: standard zero-shot benchmarks make multilingual models look more crosslingual than they are.

> [!card]- What two assumptions of sequence labelling does sequence-to-sequence modelling drop?
> That $x_t$ **corresponds to** $y_t$, and that $|X| = |Y|$.

> [!card]- How is sequence-to-sequence modelling written as conditional language modelling?
> $p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X})$: the next output token given the **decoder prefix** and the **encoder output**.

> [!card]- Which four complex mappings make machine translation hard?
> **Different lengths**, **word order differences**, **one-to-many**, **many-to-one**.

> [!card]- How did Sutskever et al. (2014) connect the LSTM encoder and decoder?
> By **initialising the decoder LSTM with the encoder's last state**.

> [!card]- What problem does $\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$ cause in an RNN encoder-decoder?
> An **information bottleneck**: the whole source sentence must pass through one fixed-size vector.

> [!card]- How do the NMT decoder input $\mathbf{x}$ and the target $\mathbf{y}$ differ?
> $\mathbf{x}$ = `<s>` + sentence, $\mathbf{y}$ = sentence + `</s>`: the same sentence **shifted by one**.

> [!card]- Which Transformer decoder component replaces the RNN encoder-decoder bottleneck?
> The **source-target context layer** (cross-attention) over **all** top-layer encoder outputs, in every decoder layer.

> [!card]- With small parallel corpora, which wins: NMT or statistical MT?
> **Statistical** MT. NMT is data hungry: it starts far lower and only overtakes with much more data.

> [!card]- What is parent-child transfer learning for NMT?
> Train on a **high-resource pair** (parent), then **continue training** the same model on the low-resource pair (child).

> [!card]- Which conditions make parent-child NMT transfer work best?
> Same **target language**, **related** source languages, frozen **decoder** (in the experiments only the **target embeddings**), **shared vocabulary**.

> [!card]- In Zoph et al.'s freezing experiment, which parameters should stay frozen in the child?
> Only the **target-language embeddings**. Training the decoder RNN and attention still helps.

> [!card]- Why does unfreezing the target embeddings hurt a low-resource NMT child?
> The parent learned them from far more target-language data; retraining them on the small child data **overfits**.

> [!card]- Which child language gains most from parent-child transfer in Zoph et al. (2016)?
> The one with the **smallest corpus** (Urdu).

> [!card]- Does a related parent help NMT transfer more than an unrelated one in Zoph et al. (2016)?
> **Yes**: a Spanish child gains more from a French parent than from a German one.

> [!card]- What does NMT transfer to French' (French with randomly reshuffled vocabulary) show?
> Transfer is **not only about shared words**: structure such as word order and syntax transfers too.

> [!card]- With a shared parent-child vocabulary, do unrelated parents help NMT transfer (Kocmi and Bojar, 2018)?
> **Yes**, about as much as related ones. Having a large parent matters most.

> [!card]- According to Aji et al. (2020), which part of a parent NMT model carries most of the transfer benefit?
> The **inner** (non-embedding) layers.

> [!card]- According to Aji et al. (2020), how do parent embeddings transferred alone compare with training the child from scratch?
> **Worse** on average: without the parent's inner layers they are a bad initialisation.

> [!card]- What is the difference between parent-child transfer and multilingual NMT?
> Parent-child transfers knowledge **in stages**; multilingual NMT transfers it **in parallel** through joint training.

> [!card]- Which multilingual NMT language combination is used for testing only?
> **Many-to-many**. Many-to-one and one-to-many are used in training.

> [!card]- What is translating in a language direction never seen during training called?
> **Zero-shot translation**.

> [!card]- What are the two indicators that knowledge transfer happens in multilingual NMT?
> Performance on **zero-shot directions**, and beating a **bilingual baseline** trained only on that direction's data.

> [!card]- What does a tag like `<2es>` name in Johnson et al.'s multilingual NMT?
> The **target** language. Nothing names the source language.

> [!card]- Why does many-to-one multilingual NMT help more consistently than one-to-many?
> Many-to-one shares one **target** side; in one-to-many the decoder produces several languages that **compete** for parameters.

> [!card]- What is it called when a direction gets worse because other languages compete for the same model capacity?
> **Interference**.

> [!card]- Do low-resource languages gain more from multilingual NMT when translating into English or out of it?
> **Into English** (many-to-one).

> [!card]- Which sampling temperature gives proportional sampling?
> $T = 1$.

> [!card]- Which sampling temperature is the compromise between proportional and uniform in Arivazhagan et al. (2019)?
> $T = 5$.

> [!card]- What does training on more languages (10 against 102) do to zero-shot translation?
> Improves **most** directions, but not all.

> [!card]- How does zero-shot translation compare with supervised directions?
> It still **lags behind**.

> [!card]- Besides the number of training languages, what does zero-shot translation quality depend on?
> **Which languages** are included: related pairs (Belarusian→Russian) do far better.

> [!card]- What is BART's architecture?
> A **bidirectional encoder** (as BERT) plus an **autoregressive decoder** (as GPT).

> [!card]- What is BART's pretraining task?
> Reconstruct the **original** document from a **corrupted** one: a denoising autoencoder.

> [!card]- What are BART's five noise functions?
> Token masking, token deletion, text infilling, sentence permutation, document rotation.

> [!card]- What is the difference between BART's token masking and token deletion?
> Masking **marks** the gap with `[MASK]`; deletion leaves no mark, so the model must find **where** tokens are missing.

> [!card]- What extra thing must BART work out under text infilling?
> **How many** tokens the single `[MASK]` stands for.

> [!card]- Which single BART noise function works best?
> **Text infilling**.

> [!card]- How do sentence permutation and document rotation perform as BART's only noise function?
> **Poorly**: rearranging whole sentences gives too weak a token-level learning signal.

> [!card]- How is BART fine-tuned for classification?
> The same input goes to encoder and decoder; the label is predicted from the **last decoder hidden state**.

> [!card]- How is BART adapted for machine translation into English?
> A **randomly initialised source encoder** replaces the embedding layer of BART's encoder.

> [!card]- What is the difference between Fixed BART and Tuned BART for Romanian→English MT?
> Fixed (BART frozen) is **worse than the baseline**; Tuned (BART updated) **beats** it.

> [!card]- Which noise functions does mBART's pretraining use?
> **Text infilling** plus **sentence permutation**.

> [!card]- What is the difference between translating with BART and with mBART?
> BART (English-only) needs a **new source encoder**; mBART is **fine-tuned as is** on parallel data.

> [!card]- At what amount of parallel data does mBART pretraining help translation most?
> **Low to medium** (a few hundred thousand pairs). Almost no data cannot be rescued, and gains shrink with more data.

> [!card]- In Monz's description of InfoXLM's MMLM, what is used instead of directly optimising the ground-truth loss?
> **Negative sampling.**

> [!card]- What model is InfoXLM initialised from?
> **XLM-R**, then pretrained further with MMLM, TLM and XLCo.

> [!card]- Why does InfoXLM have no language embeddings, unlike XLM?
> It keeps **XLM-R's architecture** so it can keep XLM-R's weights.

> [!card]- How are InfoXLM's three losses combined?
> **Equally weighted**: a plain sum.

> [!card]- On mixed-language XNLI, is XLM-R better or worse than mBERT across languages?
> **Better.** Only on mixed-language QA does XLM-R fall below mBERT.

> [!card]- In mixed-language QA, when does XLM-R work across languages?
> **Mainly when the question is in English.**

> [!card]- In parent-child NMT transfer, what does the Uzbek child score with every parameter frozen?
> **Zero BLEU**: the source embeddings must be retrained first.

> [!card]- In parent-child NMT transfer, which parameter group gives the biggest single gain when unfrozen?
> The **source embeddings**.

> [!card]- In which direction of corpus size must NMT transfer flow to pay off?
> From the **larger** corpus (parent) to the **smaller** (child).

> [!card]- Which sampling temperature is effectively uniform?
> $T = 100$.

> [!card]- Which sampling temperature is best for high-resource languages and worst for low-resource ones?
> $T = 1$ (proportional).

> [!card]- Which multilingual model was weakest in every resource group in Arivazhagan et al. (2019)?
> **All→All**, one model for both directions.

> [!card]- Do multilingual NMT models beat bilingual baselines for high-resource languages in Arivazhagan et al. (2019)?
> **No.** High-resource languages lose to interference.

> [!card]- Where does mBART put the language ID token?
> At the **end of the encoder input** and the **start of the decoder input**.

> [!card]- True or false: TLM uses a different loss function from MLM.
> **False.** The loss is the same; only the input (a sentence plus its translation) differs.

> [!card]- True or false: In TLM, the position embeddings restart at 0 for the second sentence.
> **True.** So position cannot tell the halves apart; language embeddings mark them instead.

> [!card]- True or false: LASER's decoder is told the output language by a language ID embedding, not by the encoder.
> **True.** So the sentence vector has no reason to encode the input language.

> [!card]- True or false: InfoXLM is only slightly better than XLM-R within a language but much better across languages.
> **True.** The explicit crosslingual objectives mainly close the within-across gap.

> [!card]- True or false: In the Transformer encoder-decoder, every decoder layer can attend to all encoder output positions.
> **True.** The source-target context layer does this, removing the RNN's single-vector bottleneck.

> [!card]- True or false: With a shared vocabulary, an unrelated parent can help an English→Estonian NMT child about as much as a related Finnish parent.
> **True.** Kocmi and Bojar: Czech and Russian parents help as much as Finnish.

> [!card]- True or false: Transferring only a parent's embeddings, without its inner layers, beats training the NMT child from scratch.
> **False.** Embeddings alone are worse than scratch on average; the inner layers carry the benefit (Aji et al.).

> [!card]- True or false: mBART pretraining makes translation work even with almost no parallel data (around 10K pairs).
> **False.** With 10K pairs (English-Gujarati) BLEU stays near zero; pretraining cannot rescue almost no data.

> [!card]- True or false: Adding more languages to a multilingual NMT model always improves zero-shot translation.
> **False.** Most zero-shot directions improve, but not all (Fr→Zh got worse).

> [!card]- True or false: LASER's sentence embedding is obtained by max pooling over the BiLSTM states.
> **True.** Max pooling gives one fixed-size vector, whatever the sentence length.

> [!card]- True or false: Many-to-one multilingual NMT beat the bilingual baselines in every direction in Johnson et al. (2017).
> **True.** By small margins; one-to-many was mixed.

> [!card]- True or false: XLM-R is better than mBERT across languages on mixed-language QA.
> **False.** It is better within languages but, on average, **worse** across them on QA.

> [!card]- True or false: BART's decoder is bidirectional.
> **False.** The encoder is bidirectional; the decoder is **autoregressive**.

> [!card]- True or false: mBART's pretraining objective uses a crosslingual signal.
> **False.** Denoising is always within the **same** language.

> [!card]- True or false: Transfer learning for NMT works best if decoder parameters are frozen.
> **True.** It is in Monz's list of conditions; the freezing experiment narrows it to the target embeddings.

> [!card]- True or false: InfoXLM's MMLM uses negative sampling.
> **True.** That is Monz's description; the InfoXLM paper reads the vocabulary softmax as contrastive.

> [!card]- True or false: Zero-shot XLM (MLM+TLM) beats translate-test on XNLI for every language.
> **False.** Only on average; translate-test wins for the other European languages, Turkish and Arabic.

> [!card]- True or false: In multilingual NMT, many-to-many translation is used for testing only.
> **True.** Training is English-centric, so non-English pairs are zero-shot.

> [!card]- Multiple selection: which are true of multilingual NMT as in Johnson et al. (2017)? (A) Each language gets its own encoder. (B) A tag naming the target language is added to the source sentence. (C) Source, target and output embeddings are shared. (D) Training uses every language pair, including pairs without English.
> **B and C.** (A) false: all parameters are shared across languages. (D) false: training is English-centric ($xx \rightarrow en$, $en \rightarrow xx$); non-English pairs are zero-shot.

> [!card]- Multiple selection: which are true of language sampling in multilingual NMT? (A) Proportional sampling lets high-resource languages dominate. (B) Uniform sampling hurts low-resource languages. (C) A temperature of T = 100 approximates uniform sampling. (D) The best compromise over-samples low-resource languages.
> **A, C and D.** (B) false: uniform sampling hurts **high**-resource languages; low-resource ones can do well under it.

> [!card]- Multiple selection: which are true of Arivazhagan et al.'s (2019) 103-language NMT results? (A) Low-resource languages gain most when translating into English. (B) Multilingual models beat bilingual baselines on high-resource languages. (C) All→All is weaker than the dedicated Any→En and En→Any models. (D) One-to-many gains more than many-to-one for low-resource languages.
> **A and C.** (B) false: no multilingual model beats the bilingual baselines on the high-resource group (interference). (D) false: low-resource gains are far larger into English.

> [!card]- Multiple selection: which are true of mBART? (A) It is pretrained on parallel data. (B) Its pretraining objective has no crosslingual signal. (C) A language ID token starts the decoder input. (D) Translation fine-tuning requires a new randomly initialised source encoder.
> **B and C.** (A) false: it is pretrained on concatenated monolingual data. (D) false: that is how English-only BART is used; mBART is fine-tuned as is.

> [!card]- Multiple selection: which are true of parent-child transfer learning for low-resource NMT? (A) The child model is initialised with the parent's parameters. (B) The target-language embeddings stay frozen during child training. (C) The source embeddings stay frozen during child training. (D) Transfer from a smaller parent to a larger child works as well as the reverse.
> **A and B.** (C) false: the source embeddings must be retrained. (D) false: reversed transfer helps little or hurts.

> [!card]- Multiple selection: which are true of XLM and InfoXLM pretraining? (A) XLM keeps BERT's next sentence prediction. (B) XLM's TLM is always combined with MLM or CLM. (C) InfoXLM's XLCo classifies whether a sentence pair is parallel from `[CLS]` representations. (D) InfoXLM is trained from scratch.
> **B and C.** (A) false: XLM drops NSP. (D) false: InfoXLM is initialised from XLM-R.

> [!card]- Multiple selection: fine-tuning adapts a general-domain NMT system to a small in-domain set (possibly another language pair). Which are true? (A) It uses the general-domain parameters for initialisation. (B) It continuously improves with the number of in-domain updates. (C) It does not suffer from catastrophic forgetting.
> **A only.** (B) false: with little data, quality peaks and then drops. (C) false: training only on in-domain data overwrites general-domain knowledge.

> [!card]- Multiple selection: which are true of BART's noise functions? (A) Text infilling replaces a span with a single `[MASK]`. (B) Token deletion marks each deleted position with `[MASK]`. (C) Sentence permutation alone performs poorly. (D) Text infilling is the best single noise function.
> **A, C and D.** (B) false: deletion leaves no mark, so the model must find where tokens are missing.

## Links

- **Course:** [[MNLP - Overview|Course overview]] · [[MNLP - Exam Analysis|Exam analysis]] (example question 3.1 is parent-child transfer)
- **Previous:** [[MNLP-L06 - Contextual Embeddings]] (mBERT and XLM-R, the multilingual models whose crosslingual ability this lecture tests; language sampling with a smoothing exponent)
- **Earlier lectures used here:** [[MNLP-L05 - Static Embeddings]] (crosslingual static embeddings and MUSE) · [[MNLP-L04 - Subword Segmentation]] (BPE, WordPiece and SentencePiece vocabularies of LASER, XLM, XLM-R and multilingual NMT) · [[MNLP-L02 - Multilinguality and Writing Systems]] (why Thai, Arabic and CJK behave differently)
- **Related concepts:** [[Transformers]] · [[Self-Attention]] · [[Contrastive Learning]] · [[Word Embeddings]] · [[Tokenization]] · [[Large Language Models (LLM)]] · [[Supervised Fine-Tuning (SFT)]]
- **Project:** [[MNLP - Mini Project]]
- **Numbers:** exact results for every comparison in this note are in `docs/mnlp-cheat-sheet-numbers/L07.md`, for the cheat sheet.
- **Primary references:** Artetxe and Schwenk (2019), LASER · Lample and Conneau (2019), XLM · Chi et al. (2021), InfoXLM · Sutskever et al. (2014) · Zoph et al. (2016), transfer learning for low-resource NMT · Koehn and Knowles (2017), six challenges for NMT · Schwenk et al. (2019), WikiMatrix · Kocmi and Bojar (2018) · Aji et al. (2020), on what transfer actually transfers · Lewis et al. (2020), BART · Johnson et al. (2017), multilingual NMT with target language tags · Arivazhagan et al. (2019), 103-language NMT, transfer against interference and temperature sampling · Liu et al. (2020), mBART
- **Source:** `cross-lingual-nlp.pdf`, Christof Monz, MNLP: Cross-Lingual NLP, slides 1 to 56 (deck as re-uploaded 2026-10-07), taught Mon 28 Sep to Wed 7 Oct
