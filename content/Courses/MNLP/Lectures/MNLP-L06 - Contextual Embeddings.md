---
type: lecture
course: 5204MNLP6Y
week: 4
lecture: 6
date: 2026-09-21
status: complete
topics:
  - Static versus contextual word representations
  - Pre-training and fine-tuning workflow
  - Encoder-decoder, encoder-only and decoder-only architectures
  - BERT masked language modelling and next sentence prediction
  - The CLS token, BERT fine-tuning and SQuAD span prediction
  - WordPiece segmentation
  - Whole-word masking
  - Multilingual BERT (mBERT)
  - Language sampling with a smoothing exponent (temperature)
  - Zero-shot cross-lingual transfer
  - Vocabulary overlap and transfer
  - WALS typological features
  - Translation retrieval with mBERT layers
  - Fake-English, word-order and depth experiments
  - Cross-lingual inference within a task (within and across)
  - XLM-R
---

# MNLP-L06: Contextual Embeddings

> [!abstract] In short
> A static embedding gives *train* one vector whether it is a vehicle, a verb or a train of thought. A contextual model gives every occurrence its own vector. BERT made this standard: a Transformer encoder pre-trained to fill in masked words, then fine-tuned on a real task.
>
> mBERT is BERT trained on 104 Wikipedias with one shared vocabulary and nothing cross-lingual in its loss, yet a model fine-tuned on English works in German or Hindi (zero-shot cross-lingual transfer). Shared wordpieces help but are not the main cause; similar structure and model depth matter more, and no single factor explains it. When one input mixes two languages, mBERT and XLM-R both do much worse.
>
> Exam statements: the masking rule, what mBERT's training lacks, the sampling exponent, and what does and does not drive transfer. Exact figures are on the cheat sheet.

## One vector per word is not enough

**Static embeddings.** So far a word representation has been a [[Word Embeddings|word embedding]]: one row per word type in a table (see [[MNLP-L05 - Static Embeddings]]). You can train it directly, as Word2Vec does with [[Negative Sampling]] over a small window, or get it as a by-product: train a deep network on a real task (a text classifier, a translation system) and keep its input embedding layer. Either way the result is static: the same vector in every sentence.

> [!example] Three trains
> - *He took the **train** to Munich.* (a vehicle)
> - *She had to **train** for her next match.* (a verb: to practise)
> - *He interrupted her **train** of thought.* (a sequence)
>
> $$E(\text{train}_1) = E(\text{train}_2) = E(\text{train}_3)$$
>
> where $E(\cdot)$ is the embedding lookup. Three senses, two parts of speech, one vector.

In the by-product route the deep network did read the whole sentence, and its hidden layers computed exactly the context that tells the three *trains* apart. Keeping only the input layer throws that away.

**Contextual embeddings keep it.** In a deep network (stacked [[Recurrent Neural Network (RNN)|RNN]] layers, say, or a Transformer), look at the column of hidden states above one word. At the bottom is the static vector for *train*; each layer up mixes in more of the sentence. Any of these states can serve as "the embedding of *train* in this sentence", and they differ from sentence to sentence.

Which one is useful depends on the training. With an output at every position (tagging, next-word prediction), each top-layer state is pushed to stay about its own word. With all top states combined (pooled) into one vector for a sentence classifier, the top layer is pushed to be about the sentence, and the word's own contribution can wash out.

> [!definition] The contextual representation trade-off
> Strike the right balance between
> - **integrating sufficient context** to distinguish different occurrences of the *same* word, and
> - **not capturing too much context**, where the word's own contribution to the representation becomes unclear.

Three ways to manage it:

1. **Choose the right layer(s).** Lower layers are closer to the word, higher ones closer to the sentence. Pick or combine layers to suit the task.
2. **Choose the right training task(s).** The objective decides what the hidden states encode (per-position outputs against one pooled output, as above).
3. **Tighten the connections between layers**, for example with **residual connections**: $h^{(l+1)} = h^{(l)} + f(h^{(l)})$, where $h^{(l)}$ is the state at layer $l$ and $f$ the layer's transformation. The word's own information is carried upward unchanged next to whatever context the layer adds. (The formula is the standard definition, added here.)

## Pre-train on a big generic task, fine-tune on the real one

Contextual models are used in two phases: a big network first learns general language from a task with practically unlimited data, then a small task head is put on top and the combination is trained on the real task. In symbols: $M_{C,\theta}$ is the contextual model (C for contextual) with parameters $\theta$, $M_{R,\theta'}$ is the task head with its own parameters $\theta'$, and $M_{F,\theta\cup\theta'}$ is the two combined, whose parameters are the union of both sets.

> [!definition] Pre-training and fine-tuning
> **Pre-training.** Take a model $M_{C,\theta}$ that can model wide contexts ([[LSTM]], [[Transformers|Transformer]], CNN). Train it on a general task $T_P$ for which (very) large amounts of data exist, such as word prediction in language modelling, updating $\theta$.
>
> **Fine-tuning.** Choose the real task of interest $T_R$ (question answering, part-of-speech tagging) and a model $M_{R,\theta'}$ suited to it. Combine the two into $M_{F,\theta\cup\theta'}$ and train it on $T_R$, updating $\theta'$ **and maybe also** $\theta$.

"Maybe also $\theta$" is the important switch. If only $\theta'$ is trained, the pre-trained model is a frozen **feature extractor**. If $\theta$ is updated too, that is full **fine-tuning**, BERT's default. Partial versions, where some lower layers stay frozen, come back in the transfer experiments below.

**Which network plays $M_C$?** Most current contextual models are built from [[Transformers|Transformer]] layers ([[Self-Attention|self-attention]]), in three shapes:

- **Encoder-decoder**: the encoder builds a representation of the input, the decoder generates the output the task needs. This is the original Transformer (Vaswani et al. 2017), used for translation, summarisation and question answering.
- **Encoder-only**: predicts its own input after it has been scrambled by some noise function; in its purest form an **auto-encoder**. BERT is the prominent example.
- **Decoder-only**: a classical next-word-prediction language model. Encoder-decoder tasks can be framed in it by putting input and output into one sequence. **All current [[Large Language Models (LLM)|LLMs]] are decoder-only.**

## BERT: an encoder trained to fill in blanks

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al. 2019) is the **encoder half of the Transformer**. Every position attends to every other position, to its left and to its right: that is what "bidirectional" means, unlike a left-to-right RNN or a decoder.

It is **pre-trained on missing-word prediction**, which is **self-supervised**: the labels are the words themselves, so any amount of raw text can be used, relevant to a downstream task or not. It is then **fine-tuned** on the actual task, which is **supervised** and works even with little task data.

There are two pre-training tasks: **masked language modelling** (MLM: predict hidden tokens) and **next sentence prediction** (NSP: does segment B follow segment A?).

### The input: three embeddings added up

The input is `[CLS]`, the tokens of segment A, `[SEP]`, the tokens of segment B, `[SEP]`. Each position's input vector is the sum of three learned embeddings:

$$x_i = E_{\text{token}(i)} + E_{\text{segment}(i)} + E_{\text{position}(i)}$$

where $E_{\text{token}}$ is the embedding of the WordPiece token at position $i$, $E_{\text{segment}}$ is $E_A$ or $E_B$ depending on which segment the token belongs to, and $E_{\text{position}}$ is the position embedding.

> [!example] `[CLS] my dog is cute [SEP] he likes play ##ing [SEP]`
> Everything up to and including the first `[SEP]` gets $E_A$, the rest $E_B$. WordPiece splits *playing* into `play ##ing`; `##` marks a word-internal piece. If *dog* is selected for masking, its token embedding becomes $E_{[MASK]}$.

### Masked language modelling: 15%, then 80/10/10

> [!definition] BERT masking rule
> Tokens are selected for masking with probability **0.15**. A selected token is replaced
> - **80%** of the time by `[MASK]`,
> - **10%** of the time by a **random word**,
> - **10%** of the time by **the original word** (left unchanged).
>
> **The loss is computed only for the selected positions**, regardless of whether the input there shows `[MASK]`, a random word or the original word.

Why not always `[MASK]`? Because `[MASK]` never occurs at fine-tuning time. A model that only ever predicted at `[MASK]` positions could learn to build good representations only there. With random and unchanged tokens mixed in, it cannot tell which positions will be scored, so it has to represent every token well. (This rationale is Devlin et al.'s.)

So the loss covers only the selected 15%, and within those it covers the random and unchanged positions as well as the `[MASK]` ones. In BERT the masks are drawn once, during pre-processing, and reused in every epoch (**static masking**; XLM-R below changes this).

```pseudo
Algorithm: BERT MLM masking and loss
──────────────────────────────────────────
Input: tokens x_1..x_n, vocabulary V,
       p_mask = 0.15
S ← ∅;  x̃ ← copy of x
// S = positions that will be scored
for i = 1..n:
    if x_i is [CLS] or [SEP]: continue
    with probability p_mask:
        S ← S ∪ {i}
        r ← uniform(0,1)
        if r < 0.8:   x̃_i ← [MASK]
        elif r < 0.9: x̃_i ← random token
        else:         x̃_i ← x_i
        // unchanged, but still scored
T_1..T_n ← BERT(x̃)
// W = output projection onto V
loss ← − Σ_{i∈S} log softmax(W T_i)[x_i]
// predict the ORIGINAL token, only at S
```

### `[CLS]` and next sentence prediction

> [!definition] The class embedding `[CLS]`
> - **not tied to a specific word, and never masked**;
> - **not meant to learn a context-specific word representation**;
> - captures a **general representation of the entire input sequence**;
> - **its loss is computed with respect to a classification task.**

In pre-training that task is NSP. Each input holds two segments A and B. **50%** of the time B really follows A in the document (*IsNext*); **50%** of the time B is a random segment from the corpus (*NotNext*). A classifier on $C$, the final-layer output at `[CLS]`, predicts the label. In practice A and B are stretches of running text, much longer than single sentences.

BERT was pre-trained on BookCorpus plus English Wikipedia, in two sizes, base and large. BERT-base has 12 attention heads and BERT-large 16 (Devlin et al.), although the lecture's table gives 16 for both, so a statement taken from that table will likely be keyed with 16 (other dimensions on the cheat sheet).

### Fine-tuning, entailment and SQuAD spans

To fine-tune:

1. add a small **task-specific model**, randomly initialised;
2. for tasks that need word representations (tagging, span extraction), connect it to the **top-layer token outputs** $T_i$;
3. for classification tasks (sentiment, entailment), connect it to **the `[CLS]` output** $C$;
4. train on the task data, with or without updating BERT's own parameters.

Fine-tuning is short: small batches (16 or 32 sequences) and **2 to 4 epochs**, because the encoder already starts out good and long training on little data would overfit. (In current terms, [[Supervised Fine-Tuning (SFT)|supervised fine-tuning]] of an encoder.)

> [!definition] Natural language inference (NLI), or entailment
> Given a **premise** and a **hypothesis**, the premise can **entail** the hypothesis, **contradict** it, or be in **no relationship** (neutral) to it. A **3-way classification** task.

*A soccer game with multiple males playing* entails *Some men are playing a sport*. In BERT the premise is segment A, the hypothesis segment B, and the label is predicted from $C$. **SNLI** and **MultiNLI** are large English NLI datasets. For NER (named entity recognition, tagging names of people, places and organisations) the input is a single sentence, and the head sits on the $T_i$.

**SQuAD** (a reading-comprehension benchmark) gives a question and a paragraph; the answer is a **span** of the paragraph, given by its start and end position. Nothing is generated: *At what pressure is water heated?* is answered by pointing at *high pressure* in the paragraph.

> [!formula] Span prediction
> Fine-tuning learns **one start vector $S$ and one end vector $E$**, both in $\mathbb{R}^H$; these are the only new parameters. Each paragraph position is scored by an inner product, normalised with a softmax:
> $$P_{\text{start}}(i) = \frac{\exp(T_i \cdot S)}{\sum_k \exp(T_k \cdot S)}$$
> $$P_{\text{end}}(j) = \frac{\exp(T_j \cdot E)}{\sum_k \exp(T_k \cdot E)}$$
> The predicted span maximises $T_i \cdot S + T_j \cdot E$ subject to $i \le j$.
>
> where $T_i$ is BERT's final-layer output at token $i$, $H$ the hidden size, and $i \le j$ rules out spans that end before they start.

## Subwords meet masking

BERT works on subword pieces, which matters twice: for how words are split, and for which pieces get masked.

### WordPiece

BERT and mBERT split words with **WordPiece** (Schuster and Nakajima 2012). It is very similar to BPE and predates it as an NLP segmentation method. Like BPE, and unlike Unigram LM, it works **bottom-up**: start from characters and merge pairs. Every non-initial character starts with a `##` prefix (*hug* is `h ##u ##g`), and merged non-initial pieces keep it. See [[MNLP-L04 - Subword Segmentation]] for BPE and Unigram LM, and [[Tokenization]].

The difference from BPE is the merge criterion. BPE merges the most frequent pair. WordPiece merges the pair whose parts occur together most often relative to how often they occur at all.

> [!formula] WordPiece pair score
> $$\text{score}(a,b) = \frac{\text{count}(a,b)}{\text{count}(a)\cdot\text{count}(b)}$$
>
> where $\text{count}(a,b)$ is how often symbol $a$ is immediately followed by $b$, and $\text{count}(a)$, $\text{count}(b)$ are the symbols' own frequencies. A pair scores high when its parts occur together more often than their frequencies would suggest. Training merges the highest-scoring pair, adds it to the vocabulary, and repeats until the vocabulary has the target size.

> [!example] One training step
> Word counts: *hug* 10, *pug* 5, *pun* 12, *bun* 4, *hugs* 5, so the words start as `h ##u ##g`, `p ##u ##g`, `p ##u ##n`, `b ##u ##n`, `h ##u ##g ##s`. Symbol counts: `##u` 36, `##g` 20, `p` 17, `##n` 16, `h` 15, `##s` 5, `b` 4.
>
> The most frequent pair, `##u ##g` (20), scores $20/(36\cdot20) \approx 0.028$; every pair containing the very common `##u` scores the same $1/36$. `##g ##s` occurs only 5 times but scores $5/(20\cdot5) = 0.05$, because `##s` only ever follows `##g`. WordPiece merges `##g ##s`; BPE would merge `##u ##g`.

At inference WordPiece **does not replay the merges** the way BPE does. It runs greedy longest match against the final vocabulary, which can give a different segmentation. Its unknown handling is also the harshest: a word containing a character it cannot match becomes `[UNK]` as a whole, where BPE and Unigram LM fall back to characters or bytes.

```pseudo
Algorithm: WordPiece inference
(greedy longest match, per word)
──────────────────────────────────────────
Input: word w, vocabulary V
pieces ← [];  start ← 1
while start ≤ |w|:
    end ← |w|;  found ← none
    // try the LONGEST candidate first
    while end ≥ start:
        cand ← w[start..end]
        // non-initial pieces carry ##
        if start > 1: cand ← "##" + cand
        if cand ∈ V: found ← cand; break
        end ← end − 1
    // the WHOLE word becomes [UNK]
    if found = none: return [UNK]
    append found to pieces
    start ← end + 1
return pieces
```

| | BPE | WordPiece | Unigram LM |
|---|---|---|---|
| Direction | bottom-up merges | bottom-up merges | top-down pruning |
| Criterion | pair frequency | count ratio | corpus likelihood loss |
| Inference | replay merges | greedy longest match | Viterbi or sampling |
| Unknown word | character/byte fallback | whole word `[UNK]` | character fallback |
| Typical models | GPT, RoBERTa, XLM | BERT, mBERT | T5, XLM-R |

### Easy pieces, and whole-word masking

Some words are harder to predict than others: in *The developers [MASK] the property*, guessing the verb is hard, while guessing *the* would be easy. Subword pieces differ in the same way, in any subword scheme. This example uses BPE's notation, where `@@` marks a piece that continues into the next one (WordPiece would write `view ##ed`):

- `The deve@@ lop@@ ers [MASK] ed the proper@@ ty .` hides the stem of *viewed*: still hard, which verb?
- `The deve@@ lop@@ ers view@@ [MASK] the proper@@ ty .` hides only the suffix: given `view@@` and a past-tense context, `ed` is nearly certain.

**Pieces that mostly carry morphology (tense, number, case) are easier to predict than pieces that carry the root or stem.** So under per-token random masking, many masked positions are suffixes or word-internal pieces the model can complete from the visible rest of the word. It earns loss signal there without learning much.

> [!definition] Whole-word masking
> Instead of masking tokens independently, **always mask all subword tokens that belong to the same word**. The **masking rate stays unchanged** (still about 15% of tokens).
>
> `The deve@@ lop@@ ers view@@ ed the proper@@ ty .` becomes `The deve@@ lop@@ ers [MASK] [MASK] the proper@@ ty .`

The effect: with the same model, data and masking rate, BERT-Large improves on SQuAD and on MultiNLI, in both its cased and lowercased (uncased) versions. The only change is that the model can no longer cheat by completing a word from its own visible pieces.

## mBERT: BERT on 104 Wikipedias, and nothing more

> [!definition] mBERT (multilingual BERT)
> - Released **on GitHub in 2018, with no paper**.
> - **Exactly the same architecture and loss as BERT** (MLM + NSP): **no cross-lingual loss, nothing language-specific**. That means **no parallel data** (at least not intentionally), **no language-ID embedding**, **no language-specific parameters** such as adapters, and **no alignment loss** rewarding translation equivalents for being close.
> - **Trained on 104 languages**: the concatenated Wikipedias.
> - **NSP pairs are always within the same language**, and there is **no mixing of languages inside a sequence** (a batch can contain several languages).
> - **One shared subword vocabulary** for all languages (WordPiece; BPE or Unigram LM would work similarly well).

Nothing in the objective says that *Hund* and *dog* mean the same thing. The languages share only the **parameters** and the **subword vocabulary**, so every cross-lingual ability mBERT shows has to come from those two. Most of what follows asks which.

### Sampling languages with a smoothing exponent

Wikipedia sizes vary enormously, and the joined corpus is extremely skewed: **English alone is larger than dozens of smaller languages combined**. Sampling training data in proportion to size would let a few big languages dominate, so mBERT samples languages from a smoothed distribution:

> [!formula] Weighted language sampling
> $$p_l = \frac{n_l^{\alpha}}{\sum_{l'} n_{l'}^{\alpha}}$$
>
> where $p_l$ is the probability of drawing training data from language $l$, $n_l$ the size of language $l$'s data (in characters, tokens, etc.), $\alpha$ the exponent that sets how strongly raw sizes count, and the sum over all languages $l'$ makes the probabilities add up to 1.

- **$\alpha = 1$**: sampling in proportion to raw size; English and a few other large Wikipedias **dominate** training.
- **$\alpha = 0$**: every $n_l^0 = 1$, so sampling is **uniform**; tiny Wikipedias are oversampled so much that the model would mostly **memorise** them.
- **$\alpha = 0.7$** (mBERT's choice): **up-weights low-resource languages and down-weights high-resource ones**, a compromise.

Why it works: $x^\alpha$ with $\alpha < 1$ is concave, so it shrinks large values more than small ones. The size ratio between two languages, $n_1/n_2$, becomes $(n_1/n_2)^\alpha$. Every language keeps its rank; the gaps get smaller.

> [!example] English 1000, Swahili 10
> - $\alpha = 1$: $p_{sw} = 10/1010 = 0.0099$, $p_{en} = 0.990$.
> - $\alpha = 0.7$: $1000^{0.7} = 125.9$ and $10^{0.7} = 5.01$, so $p_{sw} = 5.01/130.9 = 0.038$ and $p_{en} = 0.962$. Swahili gets about 3.9 times its raw share; English still gets 96%. The 100:1 ratio has become $100^{0.7} \approx 25$:1.
> - $\alpha = 0$: $p_{sw} = p_{en} = 0.5$. Each unit of Swahili text is now drawn 100 times as often as each unit of English ($0.5/10$ against $0.5/1000$): the memorisation problem.

The same knob is often written as a **temperature** $T$, with $\alpha = 1/T$ ([[MNLP-L07 - Crosslingual NLP]] uses this form). $T = 1$ is proportional sampling, and **raising the temperature flattens the distribution toward uniform**. Lowering $\alpha$ and raising $T$ are the same move.

### One shared vocabulary, and who pays for it

mBERT has **one vocabulary of about 120k entries for all 104 languages**, against 30k for English BERT. The architecture is identical, so mBERT's larger parameter count comes **entirely from the bigger embedding matrix**.

- **Chinese characters, Japanese kanji and Korean hanja are split into single characters** before WordPiece is applied.
- **The sampling smoothing also applies to the subword frequencies** used to build the vocabulary, so small languages get more entries than their raw size would give them.
- Even so, **the shared vocabulary still favours high-resource languages and Latin script**, and **low-resource and morphologically rich languages are split into many more pieces per word** (higher **fertility**, see [[MNLP-L04 - Subword Segmentation#Fertility: what segmentation costs a language|fertility in L04]]).
- An **uncased** mBERT lowercases and **strips accents**, which damages languages that depend on diacritics. (Examples added here: Spanish *papá* "dad" collapses into *papa* "potato", and Vietnamese loses its tone marks.)

## Zero-shot cross-lingual transfer: does mBERT generalise across languages?

> [!definition] Zero-shot cross-lingual transfer
> Fine-tune on task data in language A only, then evaluate on the same task in language B, with **no task data in B at all**. In a table of fine-tuning language against evaluation language, the **diagonal** is ordinary in-language performance and every **off-diagonal** cell is zero-shot.

Pires et al. (2019) ran this on NER and POS tagging across 16 languages. **Performance off the diagonal is decent**: in NER, a model fine-tuned on English tags Dutch entities reasonably well without a single Dutch training example. Zero-shot stays below in-language performance. POS scores off the diagonal are higher than NER's, though the two use different measures (accuracy against F1). In POS, closely related languages transfer best (Spanish and Italian). Transfer is also **asymmetric**: in NER, English to Dutch is clearly better than Dutch to English. So mBERT generalises beyond its fine-tuning language. The question is how.

### Different scripts: transfer survives

If transfer came from shared wordpieces, languages written in different scripts should barely transfer, since they share almost no pieces. They do transfer.

**Hindi and Urdu** are essentially one spoken language written in two unrelated scripts (Devanagari and Perso-Arabic), which makes them the cleanest test of "same grammar, no shared script" (background added here). POS tagging transfers well between them in both directions. **English to Bulgarian** (Latin to Cyrillic) also transfers well. Pairs involving **Japanese** drop sharply, possibly because of **larger typological differences**, as Pires et al. suggest (Japanese puts the verb last and uses postpositions).

### Vocabulary overlap: mBERT is flat, English BERT is not

Pires et al. then measured overlap directly, using only the wordpieces inside labelled named entities:

> [!formula] Entity wordpiece overlap (Jaccard similarity)
> $$\text{overlap} = \frac{|E_{l,t} \cap E_{l',e}|}{|E_{l,t} \cup E_{l',e}|}$$
>
> where $E_{l,t}$ is the set of wordpieces in named entities of the fine-tuning data ($t$) in language $l$, and $E_{l',e}$ the same for the evaluation data ($e$) in language $l'$. It is 0 for disjoint sets and 1 for identical ones.

Plotting zero-shot NER F1 against this overlap, one point per language pair, gives two different pictures:

- **English BERT** (monolingual, applied to other languages) scores **near zero at low overlap and rises roughly linearly** with it. It can only transfer through strings that literally occur in both languages: names, loanwords.
- **mBERT** is **roughly flat**: decent F1 across the whole range, already at overlap near zero.

So whatever mBERT transfers is **not surface string matching**. It has a representation shared across languages below the level of the wordpieces.

### Typology: similar structure transfers better

**WALS**, the World Atlas of Language Structures, is a database of structural features of languages (word order, tone, number of cases and so on), compiled from reference grammars. Coverage is wide but sparse: no language has every feature filled in. In NLP each language becomes a vector of its feature values, so two languages' **structural similarity** can be compared along dimensions such as word order or number of cases.

Pires et al. grouped languages by word order and averaged POS transfer:

- **Subject-verb-object order**: from an SVO language, SVO targets are far better than SOV (verb-final) targets such as Japanese or Hindi. From an SOV language the two are almost equal.
- **Adjective-noun order**: transfer is again better within the same order, with a much smaller effect.
- **Number of shared WALS features**: zero-shot accuracy **rises as two languages share more features**, for both mBERT and English BERT, with mBERT far above English BERT throughout. Individual pairs vary a lot around the trend.

### Translation retrieval: language-neutral in the middle

Do mBERT's sentence representations line up across languages, and in which layer?

> [!definition] Translation retrieval with a mean difference vector (Pires et al.)
> - Take a sample of $M$ translation pairs (from WMT16, a machine-translation benchmark).
> - For each layer $l$, represent a sentence by the **average of its hidden states, excluding `[CLS]` and `[SEP]`**: $v^{(l)}_{\text{EN}}$, $v^{(l)}_{\text{DE}}$.
> - Compute **one average difference vector** for the language pair:
> $$\bar v^{(l)}_{\text{EN}\to\text{DE}} = \frac{1}{M}\sum_{i=1}^{M} \left( v^{(l)}_{\text{DE}_i} - v^{(l)}_{\text{EN}_i} \right)$$
> - "Translate" each English sentence by shifting it, $v^{(l)}_{\text{EN}_i} + \bar v^{(l)}_{\text{EN}\to\text{DE}}$, and find its **nearest German neighbour by $\ell_2$ distance** (Euclidean distance, not cosine similarity). It is a hit when that neighbour is the true translation.

For English-German, English-Russian and Urdu-Hindi the accuracy follows the same **inverted U**: low in layer 1, **peaking in the middle-upper layers (6 to 8 of 12)**, then falling in the last layers. English-Russian (Latin against Cyrillic) starts lowest and catches up by the peak.

The reading: lower layers are close to the surface tokens, which differ between languages. Middle layers are the most language-neutral: **a single constant shift maps one language onto the other**. The top layers are shaped by predicting a word *in this language*, and become language-specific again. So accuracy does **not** keep rising up to the top layer: "choose the right layer", measured.

## Zero-shot mBERT against task-specific systems (Wu and Dredze)

Wu and Dredze (2019) tested zero-shot transfer from **English fine-tuning** on five tasks: document classification (MLDoc), NLI (on XNLI), NER, POS tagging and dependency parsing, over 39 languages in total.

**MLDoc** sorts news documents into four classes. Zero-shot, mBERT comes **within a fraction of a point of the best system, which was trained with parallel text**, without ever seeing a translation. Going zero-shot still costs a lot against in-language training, most of all for Japanese.

**XNLI** is a crowd-sourced English NLI test and dev set **translated into 14 languages** (15 in total), so a premise can be paired with its hypothesis in any of the 15 languages. On XNLI:

- **Translate-train beats zero-shot**: machine-translating the English training set into each target language and fine-tuning on that (pseudo-supervision) gives mBERT a clearly better average than fine-tuning on English alone.
- Zero-shot mBERT **narrowly beats X-LSTM**, an older LSTM system that was pre-trained with cross-lingual signal (bitext or a bilingual dictionary). It **loses to the two stronger systems with cross-lingual signal**: Artetxe and Schwenk (parallel data) and XLM with MLM+TLM.
- It also loses to **XLM trained with MLM alone**, which, like mBERT, uses no parallel data. **XLM** (Lample and Conneau 2019, covered in [[MNLP-L07 - Crosslingual NLP]]) is a cross-lingual BERT-style model whose full version adds translation language modelling (TLM), an objective on parallel sentence pairs. Its MLM-only variant drops that. XLM-R, below, is a different and later model. One caveat on these comparisons: the X-LSTM and XLM results chose their model on the target-language dev set, which mBERT did not.
- mBERT's weakest languages are low-resource and distant ones: Swahili, Thai, Urdu, Hindi, Turkish.

**NER**: zero-shot mBERT **beats a dedicated cross-lingual NER system** (Xie et al.) by several points on average. Chinese collapses (a plausible reason, not given in the source: a different script and a very different typology).

**POS**: zero-shot mBERT **does not match** Kim et al.'s low-resource cross-lingual tagger, which also gets a few hundred target-language sentences (320). For POS, adding a little target-language data beats zero-shot mBERT on average.

### Freezing layers during fine-tuning

Wu and Dredze then fine-tuned on English with every layer **up to and including layer $k$ frozen**, for $k$ = 0, 3, 6 or 9, where layer 0 is the embedding layer: so $k = 0$ freezes only the embeddings. They compared these with fine-tuning all layers, and with "Feat": mBERT not fine-tuned at all, only the task head trained (the feature-extractor setting).

- **Feat is below the best fine-tuned setting on average on every task**, so fine-tuning is necessary. The loss is largest on MLDoc and smallest on POS.
- **Freezing the lowest layers helps or is neutral**, compared with fine-tuning everything. The best level is layer 0 or 3 for NER, 3 for POS, and 6 for MLDoc and XNLI. Fine-tuning only on English would otherwise pull the lower layers toward English; frozen, they keep the multilingual representation from pre-training.
- **Freezing up to layer 9 hurts** NER (most), POS and XNLI, but not MLDoc: for most tasks the upper layers need to adapt.

### Vocabulary sharing: correlation, task by task

Finally, for each target language Wu and Dredze measured the share of its test wordpieces that also occur in the English training data, counted once per wordpiece type (type level) or weighted by frequency (token level). Correlating that share with zero-shot scores across languages:

- **NER**: an **almost perfect correlation**. It is the most lexical task, and named entities are often copied verbatim across languages (though there are only five languages).
- **XNLI**: **no correlation** at type level, and only a weak, non-significant one at token level. Sentence-level inference does not ride on shared wordpieces.
- **MLDoc, POS, parsing**: in between. Low-overlap languages score lowest, with a wide spread.

Correlation is not cause: languages that share many wordpieces with English also tend to be structurally close to English. The next experiments separate the two.

## Isolating the causes (K et al.)

K et al. (2020; K is the first author's surname) trained their own **bilingual BERT models (B-BERT)** on English plus one other language, so they could switch individual factors off.

### Removing all subword overlap: fake English

**Fake English** is English with every Unicode character shifted by a large constant, into a range no other language uses. Words, grammar, word order and frequencies stay identical; only the characters change, so the vocabulary it shares with the other language is **exactly zero**.

Train B-BERT on fake English plus Spanish, Hindi or Russian, fine-tune on (fake) English XNLI, test on the other language: **removing all subword overlap costs little, at most about a point and a half**. The cleanest case is a B-BERT trained on English and fake English, two copies of one language with disjoint alphabets: fine-tuned on fake English, it does almost as well on real English. **Shared wordpieces are not what makes multilingual BERT multilingual.**

### Scrambling word order

If not vocabulary, perhaps structure. K et al. **randomly permuted a fraction of the words during pre-training**, with **no permutation during fine-tuning**. The more words permuted, the larger the drop. It is a **significant drop, but transfer is not eliminated**: even with every word permuted, accuracy stays well above chance. Of Spanish, Hindi and Russian, Hindi suffers most. (A possible reason, not from the source: Hindi is verb-final, the furthest of the three from English word order.)

### Depth

Of the architecture factors K et al. varied, **depth had the largest impact**. Keeping the parameter count roughly constant, they went from 1 to 24 layers. In-language accuracy (on fake English) saturates quickly, by about 6 layers. Zero-shot accuracy (on Russian) keeps rising with depth. So the gap between the two shrinks: **deeper networks learn more language-independent representations**, and depth buys cross-lingual transfer more than in-language performance.

### Two languages in one input

So far "cross-lingual" meant: fine-tune on English, test on monolingual Spanish or Russian inputs. What if the **task itself is multilingual**, with the premise in one language and the hypothesis in another?

K et al. tested B-BERT on XNLI with one side in fake English and the other in the target language. Both monolingual conditions work. **Mixed pairs fall below both**, for all three languages, and a fake-English hypothesis with a target-language premise does better than the reverse. Hindi has the lowest mixed scores, but its monolingual score is also the lowest, so this does not show that mixing hurts Hindi most: measured as a drop from each language's own score, all three lose a similar amount.

This is the surprising result. In a truly language-neutral space, an English premise and a Spanish hypothesis would be as easy to compare as two English sentences. They are not: the models are **good at doing the same task in another language, and poor at relating two languages inside one input**.

## Within a language, across languages (Rajaee and Monz)

Rajaee and Monz (2024) scaled this up: mBERT fine-tuned on English NLI (the zero-shot setting) and evaluated on XNLI for **every combination of premise language and hypothesis language** among the 15.

> [!definition] Within and across
> - **Within** for language $L$: premise and hypothesis both in $L$.
> - **Across** for language $L$: the average over all pairs where one side is in $L$ and the other in a different language:
> $$\text{across}(L) = \frac{1}{2(N-1)}\Big(\sum_{L' \ne L} A_{L L'} + \sum_{L' \ne L} A_{L' L}\Big)$$
> where $A_{PH}$ is the accuracy with premise language $P$ and hypothesis language $H$, and $N = 15$.
>
> (Formula reconstructed here; it reproduces the reported averages.)

**Across is below within for every language**, often by a wide margin, and English, the fine-tuning language, has the highest within score. Patterns in the full matrix:

- For any hypothesis language, a **premise in the same language is best**.
- **An English hypothesis helps**: a foreign premise with an English hypothesis does better on average than an English premise with a foreign hypothesis. For some premise languages (German, Urdu, Hindi, Swahili, Thai) an English hypothesis even beats a hypothesis in the premise's own language.
- **A Swahili or Thai hypothesis does badly** with any foreign premise. Swahili is barely above chance (33% for three balanced classes), and Thai is only a little better.
- **Related languages pair well**: Russian with Bulgarian, Spanish with French, Hindi with Urdu.

### The heuristics hypothesis

> [!definition] Hypothesis (Rajaee and Monz)
> **Transfer of heuristics can contribute to cross-lingual generalisation.**

The idea builds on Rajaee et al. (2022), who counted **word overlap between premise and hypothesis in SNLI**. When every hypothesis word appears in the premise, the label is almost always entailment; as overlap falls, entailment becomes the minority. A model can learn the shortcut "high overlap means entailment".

How this would explain the across drop is an interpretation, not an established result. Within any one language the shortcut still works, so it transfers from English to Spanish. When premise and hypothesis are in different languages, overlap is near zero by construction, so the shortcut says "not entailment" even for true entailments.

## XLM-R: scaling mBERT up

The most prominent extension of mBERT is **XLM-R** (XLM-RoBERTa, Conneau et al. 2020), a **RoBERTa-style extension of mBERT**. All four of its changes come under that heading:

- **Drop NSP, MLM only**: next sentence prediction **contributes little and sometimes hurts**.
- **Dynamic masking** instead of static.
- **More training data**: **CC-100**, filtered CommonCrawl web text in **100 languages**, about **two orders of magnitude more** monolingual data than mBERT's Wikipedia.
- **A larger shared vocabulary**: **250k** entries, double mBERT's, built with **Unigram LM (SentencePiece)**, giving **better fertility** (fewer pieces per word) at the cost of a bigger embedding matrix.

(Background, not from the lecture: RoBERTa is BERT retrained with a tuned recipe, which already dropped NSP, used dynamic masking, more data and a larger vocabulary. The CC-100 corpus and the Unigram LM vocabulary are specific to XLM-R.)

> [!definition] Static and dynamic masking
> - **Static masking** is applied **at the data level, as part of pre-processing**: each sequence is masked once, and the same masked copy is reused in every epoch.
> - **Dynamic masking** masks **multiple iterations over the same input differently**: a fresh mask each time the sequence is fed to the model.

With static masking, a token not selected during pre-processing is never a prediction target, however many epochs you run. Dynamic masking turns every pass into new training signal at no extra data cost.

| | mBERT | XLM-R |
|---|---|---|
| Objective | MLM + NSP | MLM only |
| Masking | static | dynamic |
| Data | Wikipedia, 104 languages | CC-100, 100 languages |
| Vocabulary | 120k WordPiece | 250k Unigram LM |
| Cross-lingual signal | none | none |

### Better within, not uniformly better across

Rajaee and Monz ran the within/across evaluation for both models, on XNLI and on cross-lingual question answering (XSQuAD: SQuAD-style QA where the question and the context paragraph can be in different languages).

- **Within a language, XLM-R beats mBERT on both tasks**, on average and in almost every language (English QA is a tie).
- **Across languages on XNLI, XLM-R also beats mBERT**, and its within-across gap is a little smaller.
- **Across languages on QA, XLM-R is worse than mBERT on average**, although it is clearly better within. Its within-across gap on QA is far larger than mBERT's. Thai shows the split most clearly: XLM-R lifts Thai within-language QA enormously, while its across score barely moves.

**mBERT degrades smoothly**: an English question works with every context, a Thai question with almost none. **XLM-R is "own language, or English question"**: strong when question and context share a language or the question is English, collapsed almost everywhere else.

The conclusion: more data and a bigger vocabulary made **each language better on its own**, and did **not** make the model better at **relating two languages inside one input**.

## What drives mBERT's cross-lingual transfer?

> [!summary] Conclusions
> - **Shared (sub)word vocabulary does play a role, but it is not the main cause of transfer** (cross-script transfer, the flat overlap curve, fake English).
> - **Structure (grammar, word order) matters.** Randomly permuting words **hurts performance significantly**, but **does not entirely eliminate transfer**.
> - **Model architecture matters.** **Deeper networks can learn more language-independent representations.**
> - **Bottom line: no single factor explains transfer (or the lack of it).**

What is missing to improve transfer is left as an open question. One candidate is what mBERT's training lacks: parallel data and an **alignment loss** that pulls translation equivalents together, typically a [[Contrastive Learning|contrastive]] objective (a pointer added here; [[MNLP-L07 - Crosslingual NLP]] takes it up). The mixed-language results show why it would be needed.

## What to be able to answer

- Static against contextual embeddings; the context trade-off and its three remedies.
- Feature extractor against full fine-tuning; the three Transformer families.
- BERT: input sum, 15% and 80/10/10, which positions are scored, `[CLS]` and NSP, where heads attach, SQuAD spans.
- WordPiece against BPE; why whole-word masking helps at the same rate.
- mBERT: what its training has and lacks; $p_l$ for given $\alpha$ or $T$; the vocabulary's biases.
- Zero-shot transfer: what scripts, entity overlap, typology, layers, freezing and the fake-English, permutation and depth experiments show.
- Within against across; the heuristics hypothesis; XLM-R's changes and where it beats mBERT.

Model answers are in the exam questions below.

## Exam questions

> [!exam]- Why can mBERT transfer a task from one language to another when nothing in its training is cross-lingual? Argue from the experimental evidence.
> **Key points:** only parameters and subword vocabulary are shared; vocabulary overlap is a minor cause (cross-script transfer, flat overlap curve, fake English costs little); structure and word order matter; depth matters; no single factor, and mixed-language inputs fail.
>
> 1. **What mBERT lacks.** Same architecture and loss as BERT (MLM + NSP) on 104 Wikipedias; no parallel data, no language-ID embedding, no language-specific parameters, no alignment loss, NSP pairs always within one language. Languages share only the **parameters** and the **subword vocabulary**.
> 2. **Vocabulary overlap is a minor cause.** Pires et al. (2019): mBERT transfers well across scripts (Hindi and Urdu, English to Bulgarian), and its zero-shot NER F1 is roughly flat in entity-wordpiece overlap, while English BERT's is near zero at low overlap and rises with it. K et al. (2020): fake English removes all subword overlap and costs little, at most about a point and a half of XNLI accuracy.
> 3. **Structure matters.** From an SVO language, transfer to SVO targets is far better than to SOV targets and rises with the number of shared WALS features. Permuting words during pre-training drops XNLI significantly, most for Hindi, yet transfer stays well above chance.
> 4. **Depth matters.** At roughly constant parameter count, in-language accuracy saturates early while zero-shot accuracy keeps rising with depth, so the gap shrinks: deeper networks learn more language-independent representations.
> 5. **Bottom line:** no single factor explains transfer, or the lack of it.
> 6. **The limit.** When premise and hypothesis are in different languages, accuracy falls below both monolingual settings, so the shared space is not truly language-neutral. Rajaee and Monz hypothesise that transfer of heuristics (such as premise-hypothesis word overlap) contributes to cross-lingual generalisation; on one reading, mixed-language inputs remove that shortcut (an interpretation, not an established result).
>
> Losing marks: calling shared wordpieces the main cause; claiming mBERT saw parallel data or an alignment objective.

> [!exam]- Explain why static word embeddings are inadequate and how contextual embeddings fix this. What design tension do contextual models face, and how can it be managed?
> **Key points:** one vector per word type ($E(\text{train}_1)=E(\text{train}_2)$); contextual models give each occurrence its own vector; tension: enough context to separate senses without losing the word's own contribution; managed by choice of layer, choice of task, residual connections.
>
> - **Static embeddings**, whether a by-product of a task (the embedding layer of a classifier or translation system) or trained directly (Word2Vec with negative sampling, limited window), give one vector per word type: $E(\text{train}_1)=E(\text{train}_2)=E(\text{train}_3)$ for the vehicle, the verb "to practise" and "train of thought".
> - In the by-product route the network did see the whole sentence, but only the input embedding layer was kept, so the context computed in deeper layers was thrown away.
> - **Contextual embeddings** keep it: a deep network reads the whole sentence and every occurrence gets its own vector (a hidden state in that word's column).
> - **Tension:** integrate enough context to distinguish occurrences of the same word, without capturing so much that the word's own contribution becomes unclear.
> - **Managed by:** (1) choosing the right layer(s): lower layers are closer to the word, higher ones to the sentence; (2) choosing the right training task(s): per-token outputs keep each top state about its word, a single pooled output pushes the top layer toward the sentence; (3) tightening connections between layers, e.g. residual connections, which carry the word's own information upward alongside the added context.

> [!exam]- Describe how BERT is pre-trained and fine-tuned: input representation, both pre-training objectives, the role of `[CLS]`, and how task heads attach.
> **Key points:** input = token + segment + position embeddings; MLM: 15% selected, 80/10/10, loss on selected positions only; NSP: 50% IsNext, 50% random, classified from `[CLS]`; head on `[CLS]` for classification, on token outputs for tagging and spans; short fine-tuning, BERT optionally updated.
>
> - **Model:** encoder half of the Transformer, bidirectional (every position attends left and right).
> - **Input:** `[CLS] A [SEP] B [SEP]`, with $x_i = E_{\text{token}(i)} + E_{\text{segment}(i)} + E_{\text{position}(i)}$. WordPiece tokens.
> - **MLM:** select 15% of tokens; replace 80% of those with `[MASK]`, 10% with a random token, leave 10% unchanged; the loss is on the selected positions only, predicting the original token.
> - **NSP:** 50% B really follows A (IsNext), 50% B is random (NotNext); classified from $C$, the final-layer output at `[CLS]`.
> - **`[CLS]`:** never masked, not tied to a word, represents the whole sequence, trained through a classification loss.
> - **Fine-tuning:** a small randomly initialised head on the token outputs $T_i$ for tagging or span extraction (for SQuAD, a start vector and an end vector), on $C$ for classification; train on the task with or without updating BERT, with small batches for 2 to 4 epochs.
> - Self-supervised pre-training can use huge amounts of task-irrelevant data; supervised fine-tuning then works with little task data.
>
> Losing marks: saying the loss covers all tokens, or only the positions showing `[MASK]`.

> [!exam]- Describe how mBERT is trained and what it deliberately lacks. Explain its language sampling scheme and the problems a single shared multilingual vocabulary causes.
> **Key points:** same architecture and MLM + NSP loss on 104 Wikipedias; no parallel data, language ID or alignment loss, NSP within one language; smoothed sampling $p_l \propto n_l^\alpha$ with $\alpha = 0.7$; one shared WordPiece vocabulary; favours high-resource languages and Latin script, higher fertility for low-resource languages, uncased strips accents.
>
> - Released on GitHub in 2018 with no paper. Exactly BERT's architecture and loss (MLM + NSP), on the concatenated Wikipedias of **104 languages**.
> - **Absent:** parallel data (at least intentionally), language-ID embedding, language-specific parameters such as adapters, any alignment loss. NSP pairs are always in one language; no language mixing within a sequence (a batch can mix languages).
> - **Sampling:** Wikipedia sizes are extremely skewed (English alone exceeds dozens of small languages combined), so languages are drawn with $p_l = n_l^\alpha / \sum_{l'} n_{l'}^\alpha$. $\alpha=1$: proportional, big languages dominate; $\alpha=0$: uniform, tiny languages are memorised; $\alpha=0.7$: up-weights low-resource and down-weights high-resource languages. Written as a temperature, $\alpha = 1/T$, and a higher $T$ is flatter.
> - **Vocabulary:** one shared WordPiece vocabulary, four times English BERT's, which accounts for all of mBERT's extra parameters. CJK characters are split into single characters before WordPiece. The sampling smoothing is also applied to the counts used to build the vocabulary.
> - **Problems:** even smoothed, the vocabulary favours high-resource languages and Latin script; low-resource and morphologically rich languages get much higher fertility (more pieces per word); the uncased variant lowercases and strips accents, which damages languages that depend on diacritics.

> [!exam]- Calculate: English has 1000 units of training text and Swahili 10. Using $p_l = n_l^\alpha/\sum_{l'} n_{l'}^\alpha$, give both sampling probabilities for $\alpha = 1$, $0.7$ and $0$, and interpret each.
> **Key points:** $\alpha=1$: 0.0099 and 0.990, English dominates; $\alpha=0.7$: 0.038 and 0.962, Swahili up-weighted; $\alpha=0$: 0.5 and 0.5, small language memorised; the ratio becomes $(n_1/n_2)^\alpha$.
>
> - $\alpha = 1$: $p_{sw} = 10/1010 = 0.0099$, $p_{en} = 0.990$. Proportional to size; English dominates.
> - $\alpha = 0.7$: $1000^{0.7} = 125.9$, $10^{0.7} = 5.01$, so $p_{sw} = 5.01/130.9 = 0.038$ and $p_{en} = 0.962$. Swahili gets about 3.9 times its raw share; English still gets 96%.
> - $\alpha = 0$: $p_{sw} = p_{en} = 0.5$. Each unit of Swahili text is drawn 100 times as often as each unit of English ($0.5/10$ against $0.5/1000$): the model would memorise the small language.
> - General rule: the ratio between two languages goes from $n_1/n_2$ to $(n_1/n_2)^\alpha$, because $x^\alpha$ with $\alpha<1$ is concave and compresses large values more. Here 100:1 becomes about 25:1 at $\alpha=0.7$. With a temperature, $\alpha = 1/T$: $\alpha = 0.7$ is $T \approx 1.4$.

> [!exam]- Compare XLM-R with mBERT: what changed in training, and how do the two compare within a language and across languages on XNLI and cross-lingual QA?
> **Key points:** MLM only, dynamic masking, CC-100, 250k SentencePiece Unigram vocabulary; better within a language on XNLI and QA; better across languages on XNLI; worse across languages on QA; better per language, not better at relating languages.
>
> - **XLM-R** (Conneau et al. 2020) is a RoBERTa-style extension of mBERT: **MLM only** (NSP dropped: it contributes little and sometimes hurts), **dynamic masking**, more data (**CC-100**: filtered CommonCrawl, 100 languages, about two orders of magnitude more than Wikipedia) and a larger **250k** Unigram LM vocabulary via SentencePiece (better fertility, bigger embedding matrix). Still no cross-lingual signal.
> - **Within** a language XLM-R is better on both XNLI and QA, on average and in almost every language.
> - **Across** languages on XNLI it is also better, with a slightly smaller within-across gap.
> - **Across** languages on QA it is **worse** than mBERT on average, with a far larger within-across gap. Thai within-language QA improves enormously while Thai across barely moves.
> - XLM-R's QA results are "own language, or English question": strong diagonal and English-question column, little in between. mBERT degrades more smoothly.
> - Conclusion: more data and a bigger vocabulary made each language better on its own without making the model better at relating two languages inside one input.

> [!exam]- What happens when the two inputs of one task are in different languages? Give the evidence for B-BERT, mBERT and XLM-R, and a possible explanation.
> **Key points:** B-BERT mixed pairs fall below both monolingual conditions; mBERT across below within for every language; XLM-R worse across on QA; the space is not language-neutral; heuristics hypothesis: the word-overlap shortcut fails across languages.
>
> - Standard zero-shot transfer is English fine-tuning, then monolingual testing in another language. Cross-lingual inference within a task means, for example, an English premise with a Spanish hypothesis.
> - **K et al. (2020), B-BERT on XNLI:** fake-English pairs and target-language pairs both work; mixed pairs fall below **both** monolingual conditions for all three languages (Hindi scores lowest, but its monolingual baseline is lowest too). A fake-English hypothesis beats a fake-English premise.
> - **Rajaee and Monz (2024), mBERT on XNLI, 15 languages:** across is below within for every language. An English hypothesis helps; a Swahili hypothesis is barely above chance and a Thai one little better.
> - **XLM-R:** better than mBERT within and across on XNLI (on average), but worse than mBERT across languages on QA.
> - **Interpretation:** the models can do the same task in another language but are poor at relating two languages inside one input, so the space is not language-neutral.
> - **Heuristics hypothesis:** transfer of heuristics can contribute to cross-lingual generalisation. In SNLI, full premise-hypothesis word overlap almost always means entailment; a model can learn "high overlap means entailment", which works within any single language but fails when premise and hypothesis are in different languages, where overlap is near zero. (How exactly this explains the drop is an interpretation, not an established result.)

> [!exam]- What do layer-wise analyses reveal about where mBERT holds language-neutral information? Use translation retrieval and parameter freezing, and link to the choice of layer.
> **Key points:** translation retrieval follows an inverted U peaking at layers 6 to 8; middle layers most language-neutral, top layers language-specific; Feat (no fine-tuning) below fine-tuning on every task; freezing lower layers helps or is neutral, freezing up to layer 9 hurts; ties to choosing the right layer.
>
> - **Translation retrieval (Pires et al.):** per layer, mean-pool hidden states (excluding `[CLS]`, `[SEP]`), shift each English vector by the one mean EN-to-DE difference vector, retrieve the nearest German sentence by $\ell_2$ distance. Accuracy follows an **inverted U**: low at layer 1, peak at layers 6 to 8, falling in the last layers.
> - Reading: low layers are close to language-specific surface tokens; middle layers are the most language-neutral (one constant offset maps one language onto the other); top layers are shaped by predicting words in a specific language.
> - **Freezing (Wu and Dredze 2019):** fine-tune on English with every layer up to and including layer $k$ frozen ($k = 0$ freezes only the embeddings), compared with fine-tuning all layers. Feature-based use without fine-tuning is below the best fine-tuned setting on average on every task. Freezing the lowest layers helps or is neutral (best at 0 or 3 for NER, 3 for POS, 6 for MLDoc and XNLI), since English-only fine-tuning would pull lower layers toward English. Freezing up to layer 9 hurts NER (most), POS and XNLI, though not MLDoc: upper layers must adapt to the task.
> - Both measure the "choose the right layer" strategy for balancing word and context information.

> [!exam]- Explain how subword segmentation interacts with masked language modelling and with multilingual models: WordPiece, whole-word masking, and the vocabulary choices of mBERT and XLM-R.
> **Key points:** WordPiece: count-ratio merges, `##`, greedy longest match, whole word to `[UNK]`; suffix pieces easy to predict, so whole-word masking at the same rate, which improves SQuAD and MultiNLI; mBERT's shared vocabulary biased to high-resource languages and Latin script; XLM-R 250k Unigram LM, better fertility; overlap contributes little to transfer.
>
> - **WordPiece** (BERT, mBERT): bottom-up merges by $\text{count}(a,b)/(\text{count}(a)\,\text{count}(b))$; `##` marks non-initial pieces; inference is greedy longest match, not replaying merges; a word with an unmatchable character becomes `[UNK]` as a whole.
> - **Masking problem:** pieces carrying morphology (tense, number, case) are easy to predict from their visible stem, so per-token masking wastes many masked positions on trivially recoverable suffixes.
> - **Whole-word masking:** mask all pieces of a word together at the same overall rate (about 15%); it improves BERT-Large on SQuAD and MultiNLI, cased and uncased.
> - **mBERT:** one shared WordPiece vocabulary built from smoothed counts, CJK split into characters, still biased toward high-resource languages and Latin script, higher fertility for low-resource and morphologically rich languages; the uncased variant strips accents.
> - **XLM-R:** a 250k Unigram LM vocabulary via SentencePiece, fewer pieces per word, at the cost of a larger embedding matrix.
> - Wordpiece overlap itself contributes little to transfer (fake English costs at most about a point and a half).

## Flashcards

> [!card]- What is the shared flaw of static word embeddings, however they are trained?
> One vector per **word type**: *train* (vehicle), *train* (to practise) and *train of thought* all get the **same** vector.

> [!card]- Why do embeddings taken from a trained task model's input layer (e.g. a translation system's) still lack context?
> The network saw the whole sentence, but only the **input embedding layer** is kept, so the context computed in deeper layers is **thrown away**.

> [!card]- What trade-off must a contextual word representation strike?
> Enough context to **separate occurrences** of the same word, but not so much that the word's **own contribution** becomes unclear.

> [!card]- Name the three strategies for balancing word and context information in contextual representations.
> Choosing the right **layer(s)**, choosing the right **task(s)**, tightening connections between layers (e.g. **residual** connections).

> [!card]- What is the difference between using a pre-trained encoder as a feature extractor and full fine-tuning?
> Feature extractor: encoder **frozen**, only the task head trained. Full fine-tuning: head **and** encoder updated (BERT's default).

> [!card]- Which Transformer family is BERT, and which family are current LLMs?
> BERT is **encoder-only**; all current LLMs are **decoder-only**.

> [!card]- What makes BERT "bidirectional"?
> Every position attends to positions on **both** its left and its right.

> [!card]- What three embeddings are summed to form BERT's input vector at each position?
> **Token**, **segment** (A or B) and **position** embeddings.

> [!card]- What fraction of tokens does BERT select for masked language modelling?
> **15%**.

> [!card]- In BERT pre-training, what happens to a token selected for masking?
> **80%** replaced by `[MASK]`, **10%** by a random word, the rest left **unchanged**.

> [!card]- At which positions is BERT's MLM loss computed?
> Only at the **selected** positions, whether they show `[MASK]`, a random word or the original word.

> [!card]- Why does BERT not replace every selected token with `[MASK]`?
> `[MASK]` never appears at fine-tuning time; random and unchanged tokens force good representations of **every** input token.

> [!card]- How are BERT's next sentence prediction pairs built?
> **50%** B truly follows A (IsNext), **50%** B is a random segment (NotNext).

> [!card]- Which BERT special token is never masked and has a final output representing the whole input sequence?
> **`[CLS]`**.

> [!card]- From which output does BERT predict next sentence prediction?
> **C**, the final-layer output at the **`[CLS]`** position.

> [!card]- When fine-tuning BERT, where does the task head attach for classification versus tagging?
> Classification: the **`[CLS]` output C**. Tagging and span extraction: the **token outputs $T_i$**.

> [!card]- What does BERT output for SQuAD question answering?
> A **start/end span** in the paragraph; nothing is generated.

> [!card]- Which new parameters does BERT learn when fine-tuned for SQuAD?
> Only a **start vector S** and an **end vector E**, scored against each token output by inner product.

> [!card]- Which segmentation method merges the pair maximising count(a,b) / (count(a) · count(b))?
> **WordPiece**.

> [!card]- What is the difference between BPE's and WordPiece's merge criterion?
> BPE merges the **most frequent** pair; WordPiece the pair with the highest **count(a,b) / (count(a) · count(b))**.

> [!card]- Why can WordPiece merge a rare pair before a frequent one?
> The score is high when the pair occurs together **more often than its parts' frequencies suggest**; pairs containing a very common symbol score low.

> [!card]- What does WordPiece do with a word containing a character missing from its vocabulary?
> The **whole word** becomes `[UNK]`, where BPE and Unigram LM fall back to characters.

> [!card]- Why does per-token random masking weaken MLM when words are split into subwords?
> Morphological pieces (suffixes) are **trivially recoverable** from the visible stem, so many masked positions teach little.

> [!card]- Which masking scheme masks all subword pieces of a word together, at an unchanged masking rate?
> **Whole-word masking**.

> [!card]- Which cross-lingual signals does mBERT's training lack?
> **Parallel data**, **language-ID embedding**, language-specific parameters, **alignment loss**.

> [!card]- Since mBERT's training objective is not cross-lingual, what do its languages share?
> Only the **parameters** and the **subword vocabulary**.

> [!card]- Can an mBERT next sentence prediction pair contain two languages?
> **No**: NSP pairs are always within the **same language**.

> [!card]- Why does mBERT sample languages with a smoothed distribution?
> Wikipedia sizes are **extremely skewed**: English alone is larger than dozens of small languages combined.

> [!card]- What goes wrong with language sampling at α = 0?
> Sampling becomes **uniform**: tiny languages are oversampled so much that the model **memorises** them.

> [!card]- What does language sampling with α = 0.7 do relative to raw data sizes?
> **Up-weights low-resource** and down-weights high-resource languages: a compromise between proportional and uniform.

> [!card]- Why does an exponent α < 1 in $p_l \propto n_l^\alpha$ shrink the gap between languages?
> $x^\alpha$ is **concave** and compresses large values more: the ratio $n_1/n_2$ becomes $(n_1/n_2)^\alpha$.

> [!card]- How does a language-sampling temperature T relate to α, and what does raising T do?
> $\alpha = 1/T$; raising T makes sampling **flatter**, closer to uniform.

> [!card]- Why does mBERT have more parameters than BERT-base?
> Entirely the **bigger vocabulary** (embedding matrix); the architecture is identical.

> [!card]- Even with smoothing, which languages does mBERT's shared vocabulary favour?
> **High-resource** languages and **Latin script**.

> [!card]- Which languages get high fertility under mBERT's shared vocabulary?
> **Low-resource** and **morphologically rich** languages: many more pieces per word.

> [!card]- Why is uncased mBERT bad for some languages?
> It lowercases and **strips accents**, damaging languages that depend on **diacritics**.

> [!card]- In natural language inference, what relations can a premise have to a hypothesis?
> **Entailment**, **contradiction** or **neutral**: a 3-way classification.

> [!card]- What is zero-shot cross-lingual transfer?
> Fine-tune on task data in language A only, evaluate on language B with **no task data in B**.

> [!card]- Does mBERT transfer between languages with different scripts, such as Hindi and Urdu?
> **Yes, well**, despite almost no shared wordpieces.

> [!card]- Why does mBERT's zero-shot POS transfer drop for pairs involving Japanese?
> Possibly **larger typological differences** (Japanese is verb-final, with postpositions).

> [!card]- In Pires et al.'s overlap measure, which wordpieces are compared?
> Those in **labelled named entities** of the fine-tuning and evaluation data (Jaccard similarity).

> [!card]- How does zero-shot NER F1 depend on entity-wordpiece overlap for mBERT versus English BERT?
> mBERT: roughly **flat**, decent even near zero overlap. English BERT: near zero at low overlap, **rising** with overlap.

> [!card]- What does mBERT's flat zero-shot F1 against wordpiece overlap imply?
> Its transfer is **not surface string matching**: representations are shared below the level of surface wordpieces.

> [!card]- From an SVO fine-tuning language, how does mBERT's zero-shot POS transfer to SVO targets compare with SOV targets?
> SVO targets are **far better** than SOV (verb-final) targets.

> [!card]- From an SOV fine-tuning language, does mBERT's zero-shot POS transfer favour SOV targets over SVO targets?
> **Barely**: the two are almost equal.

> [!card]- Which typological database records structural features of languages and is used in NLP to measure structural similarity?
> **WALS** (World Atlas of Language Structures).

> [!card]- How does zero-shot POS accuracy change as two languages share more WALS features?
> It **rises**, for both mBERT and English BERT (mBERT higher throughout).

> [!card]- In mBERT translation retrieval, how is an English sentence "translated" before the nearest-neighbour search?
> Shifted by **one average EN-to-DE difference vector**, then matched by **$\ell_2$ distance**.

> [!card]- In which mBERT layers is translation retrieval most accurate?
> The **middle-upper** layers (about 6 to 8 of 12).

> [!card]- Why does mBERT translation retrieval get worse in the top layers?
> Top layers are shaped by **predicting words in a specific language**, so they become language-specific again.

> [!card]- For mBERT on XNLI, which is better: machine-translated training data (translate-train) or zero-shot transfer?
> **Translate-train** beats zero-shot.

> [!card]- On zero-shot XNLI, how does mBERT compare with systems pre-trained with cross-lingual signal?
> It narrowly **beats** the older X-LSTM but **loses** to the stronger ones (Artetxe and Schwenk; XLM with MLM+TLM).

> [!card]- Which XLM variant beats zero-shot mBERT on XNLI although, like mBERT, it uses no parallel data?
> **XLM with MLM** alone (its model was chosen on target-language dev data).

> [!card]- What distinguishes XLM from XLM-R?
> XLM's full version adds **TLM on parallel sentences**; XLM-R is a later RoBERTa-style mBERT with **no** parallel data.

> [!card]- On zero-shot MLDoc, how does mBERT compare with the best system trained with parallel text?
> **Nearly matches** it, within a fraction of a point, without seeing translations.

> [!card]- On zero-shot NER, how does mBERT compare with Xie et al.'s dedicated cross-lingual system?
> mBERT is **better** on average.

> [!card]- Does zero-shot mBERT beat Kim et al.'s cross-lingual POS tagger that also uses a few hundred target-language sentences?
> **No**, not on average: adding a little target-language data wins.

> [!card]- In Wu and Dredze's experiment, how does Feat (mBERT not fine-tuned, only the task head trained) compare with fine-tuning?
> **Worse** on average on every task than the best fine-tuned setting: fine-tuning is necessary.

> [!card]- In Wu and Dredze's freezing experiment, what does "frozen up to layer 0" mean?
> Only the **embedding layer** is frozen; all Transformer layers are fine-tuned.

> [!card]- Why does freezing mBERT's lower layers during English fine-tuning help zero-shot transfer?
> English-only fine-tuning would pull them **toward English**; frozen, they keep their **multilingual** pre-trained representation.

> [!card]- What happens to mBERT zero-shot transfer if layers up to 9 are frozen during fine-tuning?
> It **hurts** NER (most), POS and XNLI, though not MLDoc: the upper layers must **adapt to the task**.

> [!card]- For which task does wordpiece overlap with English correlate most strongly with mBERT's zero-shot score?
> **NER**, almost perfectly: the most lexical task, with entities often copied verbatim.

> [!card]- For which task does wordpiece overlap with English show no correlation with mBERT's zero-shot score?
> **XNLI** (sentence-level inference).

> [!card]- Why does a correlation between wordpiece overlap and transfer not show that overlap causes transfer?
> High-overlap languages are also **typologically closer** to English, so the two are confounded.

> [!card]- What is fake English (K et al. 2020)?
> English with every Unicode codepoint **shifted by a large constant**, so it shares **no characters** with the other language.

> [!card]- Why does fake English isolate the effect of subword overlap?
> Words, grammar, **word order** and frequencies stay identical; only the **characters** change, so shared vocabulary is zero.

> [!card]- How much XNLI transfer does B-BERT lose when fake English removes all subword overlap?
> **Very little**: transfer survives almost intact.

> [!card]- How did K et al. test whether word order matters for cross-lingual transfer?
> **Randomly permuting words during pre-training**, with no permutation during fine-tuning.

> [!card]- Of Spanish, Hindi and Russian, which loses most transfer when B-BERT is pre-trained with permuted words?
> **Hindi**.

> [!card]- Which model-architecture factor has the largest impact on B-BERT's cross-lingual transfer?
> **Depth**.

> [!card]- As B-BERT gets deeper at roughly constant size, which improves more: in-language or zero-shot accuracy?
> **Zero-shot** keeps improving; in-language **saturates** early, so the gap shrinks.

> [!card]- What does the mixed-language XNLI drop (premise and hypothesis in different languages) show about mBERT-style models?
> The space is **not truly language-neutral**: good at a task in another language, poor at **relating two languages** in one input.

> [!card]- In K et al.'s mixed-language XNLI test, why does Hindi have the lowest mixed scores?
> Its **monolingual baseline is lowest** too, so the low mixed scores do not show that mixing hurts Hindi most.

> [!card]- What is the difference between "within" and "across" in Rajaee and Monz's XNLI evaluation?
> Within: premise and hypothesis in the **same** language. Across: in **different** languages.

> [!card]- For mBERT on XNLI, which works better: a foreign premise with an English hypothesis, or an English premise with a foreign hypothesis?
> A foreign premise with an **English hypothesis**.

> [!card]- What is the heuristics hypothesis (Rajaee and Monz)?
> **Transfer of heuristics** can contribute to cross-lingual generalisation.

> [!card]- Which shortcut does SNLI training data offer an NLI model?
> **Full** premise-hypothesis **word overlap** almost always means **entailment**; low overlap mostly non-entailment.

> [!card]- Why would an NLI word-overlap shortcut fail when premise and hypothesis are in different languages?
> Lexical overlap is **near zero by construction**, so the shortcut predicts non-entailment. (An interpretation, not an established result.)

> [!card]- Why does XLM-R drop next sentence prediction and train with MLM only?
> NSP **contributes little** and sometimes **hurts** performance.

> [!card]- What is the difference between static and dynamic masking?
> Static: masked **once in preprocessing**, reused every epoch. Dynamic: a **fresh mask** each time a sequence is fed.

> [!card]- Why is dynamic masking better than static masking?
> Under static masking unselected tokens are **never targets**; dynamic masking gives **new training signal** each pass at no data cost.

> [!card]- What data is XLM-R trained on?
> **CC-100**: filtered CommonCrawl in 100 languages, two orders of magnitude more than mBERT's Wikipedia.

> [!card]- What vocabulary does XLM-R use?
> **250k** entries, **Unigram LM** via SentencePiece: better fertility, bigger embedding matrix.

> [!card]- Is XLM-R better than mBERT within a language, on XNLI and on QA?
> **Yes**, on both, on average.

> [!card]- On which task is XLM-R worse than mBERT across languages, on average?
> **Cross-lingual QA**; on XNLI it is better across too.

> [!card]- Which segmentation method builds XLM-R's 250k vocabulary?
> **Unigram LM** via SentencePiece.

> [!card]- What structure does XLM-R's cross-lingual QA (context by question language) matrix show?
> "**Own language, or English question**": both work, little in between.

> [!card]- What is the bottom line on what drives mBERT's cross-lingual transfer?
> **No single factor**: vocabulary overlap plays a minor role, structure and depth matter.

> [!card]- Which training signal, absent from mBERT, could improve its cross-lingual transfer?
> One candidate: parallel data with an **alignment loss** pulling translation equivalents together. The question is left open.

> [!card]- True or false: BERT's MLM loss is computed only at positions that show the `[MASK]` token.
> **False.** It covers all selected positions, including those given a random or unchanged word.

> [!card]- True or false: Whole-word masking keeps the overall masking rate unchanged.
> **True.** Still about 15% of tokens; only the choice of which tokens changes.

> [!card]- True or false: WordPiece inference replays the learned merges in order, like BPE.
> **False.** It uses greedy longest match against the final vocabulary.

> [!card]- True or false: mBERT was trained with a small amount of parallel data to align its languages.
> **False.** mBERT uses no parallel data and no alignment loss.

> [!card]- True or false: Language sampling with α = 0.7 down-weights high-resource languages.
> **True.** It up-weights low-resource and down-weights high-resource languages.

> [!card]- True or false: Raising the sampling temperature T moves language sampling closer to proportional sampling.
> **False.** A higher temperature moves it closer to uniform.

> [!card]- True or false: English BERT's zero-shot NER F1 is roughly flat in entity-wordpiece overlap.
> **False.** That is mBERT; English BERT's F1 rises with overlap.

> [!card]- True or false: Removing all subword overlap between the two languages (fake English) destroys most of B-BERT's cross-lingual transfer.
> **False.** Transfer survives almost intact.

> [!card]- True or false: Permuting words during pre-training hurts cross-lingual transfer significantly but does not eliminate it.
> **True.** Accuracy drops, but stays well over chance.

> [!card]- True or false: In mBERT translation retrieval, accuracy peaks in the middle layers.
> **True.** It follows an inverted U: middle layers are the most language-neutral, top layers language-specific again.

> [!card]- True or false: Freezing mBERT's lower layers during English fine-tuning helps or is neutral for zero-shot transfer.
> **True.** Frozen lower layers keep their multilingual representation.

> [!card]- True or false: In Wu and Dredze, mBERT's zero-shot XNLI score shows no type-level correlation with wordpiece overlap with English.
> **True.** Sentence-level inference does not depend on shared wordpieces.

> [!card]- True or false: Deeper B-BERT models at roughly constant parameter count narrow the gap between in-language and zero-shot accuracy.
> **True.** Zero-shot keeps improving with depth while in-language saturates.

> [!card]- True or false: XLM-R outperforms mBERT across languages on every task.
> **False.** It is better across on XNLI but, on average, worse across on cross-lingual QA.

> [!card]- Multiple selection: which are true of mBERT's training? (A) It uses the same loss as BERT, MLM plus NSP (B) NSP pairs may mix two languages (C) It has a language-ID embedding (D) All languages share one subword vocabulary
> **A and D.** B: NSP pairs are always in one language. C: there is no language-ID embedding.

> [!card]- Multiple selection: which are true of language sampling with $p_l \propto n_l^\alpha$? (A) α = 1 samples in proportion to data size (B) α = 0 samples uniformly (C) α = 0.7 down-weights low-resource languages (D) mBERT also smooths the counts used to build its vocabulary
> **A, B and D.** C: α = 0.7 up-weights low-resource languages.

> [!card]- Multiple selection: which are true of what drives mBERT-style transfer? (A) Removing all subword overlap eliminates most transfer (B) Permuting words in pre-training reduces transfer without eliminating it (C) Deeper models narrow the in-language to zero-shot gap (D) A single factor explains transfer
> **B and C.** A: removing all overlap barely changes transfer. D: no single factor explains it.

> [!card]- Multiple selection: which are true of zero-shot mBERT in Wu and Dredze? (A) Translate-train beats zero-shot on XNLI (B) It beats Kim et al.'s POS tagger, which also uses 320 target-language sentences (C) It beats Xie et al.'s cross-lingual NER system (D) Its XNLI score correlates strongly with wordpiece overlap
> **A and C.** B: Kim et al.'s tagger is better on average. D: XNLI shows no correlation with overlap.

> [!card]- Multiple selection: which are true of XLM-R compared with mBERT? (A) It drops next sentence prediction (B) It uses static masking (C) It has a larger vocabulary (D) Its average within-language score beats mBERT's on both XNLI and QA
> **A, C and D.** B: XLM-R uses dynamic masking.

> [!card]- Multiple selection: which are true when premise and hypothesis (or question and context) are in different languages? (A) B-BERT accuracy falls under both monolingual conditions (B) mBERT's across score exceeds its within score (C) XLM-R's average across-language QA score beats mBERT's (D) For mBERT an English hypothesis is easier than an English premise
> **A and D.** B: across is lower than within. C: XLM-R's average across QA score is lower than mBERT's.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L05 - Static Embeddings]] (the static word embeddings that this lecture replaces)
- **Next:** [[MNLP-L07 - Crosslingual NLP]]
- **Subword background:** [[MNLP-L04 - Subword Segmentation]] (BPE, Unigram LM, SentencePiece, fertility)
- **Concepts:** [[Word Embeddings]] · [[Negative Sampling]] · [[Tokenization]] · [[Transformers]] · [[Self-Attention]] · [[Recurrent Neural Network (RNN)]] · [[LSTM]] · [[Gated Recurrent Unit (GRU)]] · [[Large Language Models (LLM)]] · [[Supervised Fine-Tuning (SFT)]] · [[Contrastive Learning]]
- **Exact figures:** the MNLP cheat-sheet numbers for this lecture (`docs/mnlp-cheat-sheet-numbers/L06.md`)
- **Source:** `embeddings_context.pdf`, Christof Monz, MNLP: Contextual Embeddings, slides 1 to 50
- **Primary references named in the lecture:** Devlin et al. (2019), *BERT* · Vaswani et al. (2017), *Attention Is All You Need* · Schuster and Nakajima (2012), WordPiece · Pires et al. (2019), *How multilingual is Multilingual BERT?* · Wu and Dredze (2019), *Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT* · K et al. (2020), *Cross-Lingual Ability of Multilingual BERT* · Bowman et al. (2015), SNLI · Rajaee et al. (2022) · Rajaee and Monz (2024) · Conneau et al. (2020), XLM-R
