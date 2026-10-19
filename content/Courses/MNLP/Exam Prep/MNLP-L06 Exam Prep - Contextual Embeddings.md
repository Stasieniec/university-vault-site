---
type: exam-prep
course: 5204MNLP6Y
lecture: 6
status: complete
---

# MNLP-L06 Exam Prep - Contextual Embeddings

> [!abstract] What the exam can ask here
> Expect a BERT block (the masking rule, which positions carry the loss, `[CLS]` and NSP, how fine-tuning attaches a head) and an mBERT block (what its training lacks, the sampling exponent, the shared vocabulary's biases, and which experiment shows what about cross-lingual transfer). The traps are swapped percentages ("15% replaced by `[MASK]`"), swapped models (mBERT against XLM-R, English BERT against mBERT), and conclusions stated one notch too strong ("eliminates", "main cause", "always", "top layer").
> Explanations are in the [[MNLP-L06 - Contextual Embeddings|lecture note]]; figures are on the [[MNLP - Cheat Sheet|cheat sheet]].

## Static against contextual representations

- A static embedding gives every occurrence of a word **the same** vector: $E(\text{train}_1) = E(\text{train}_2) = E(\text{train}_3)$.
- Embeddings learned as a **by-product** of a task model (text classification, MT) are **static**, even though that model can be arbitrarily deep and read the **whole sentence**.
- Word2Vec is a **simple** model with a **limited** context window; its embeddings are also static.
- In a deep network, the hidden states above a word at each layer are **contextual** representations of that occurrence.
- The challenge is a **balance**: enough context to tell occurrences of the same word apart, but **not so much** that the word's own contribution becomes unclear.
- Strategies: choose the right **layer(s)**, choose the right **task(s)**, tighten connections between layers (e.g. **residual** layers).

> [!warning] Traps
> - *"Word embeddings learned inside an MT system are contextual, because the MT model uses the whole sentence."* False: the embedding layer is still static; only the hidden states are contextual.
> - *"The more context a word representation integrates, the better it represents the word."* False: too much context makes the word's own contribution unclear.

## Pre-training, fine-tuning and the three Transformer shapes

- Pre-training uses a model that can model **wider contexts**: LSTM, Transformer **or CNN**.
- The pre-training task $T_P$ is a **general** task with (very) large amounts of data, such as word prediction in language modelling; it updates $\theta$.
- Fine-tuning combines $M_{C,\theta}$ with a task model $M_{R,\theta'}$ into $M_{F,\theta\cup\theta'}$ and trains it on the real task, updating $\theta'$ **and maybe also** $\theta$.
- **Most** current contextual models are based on the Transformer.

| | What it does | Examples |
|---|---|---|
| Encoder-decoder | encoder represents the input, decoder generates the output | original Transformer (Vaswani 2017); MT, summarisation, QA |
| Encoder-only | predicts its **own input**, scrambled by noise; purest form an auto-encoder | **BERT** |
| Decoder-only | classical **next-word** prediction | **all** current LLMs |

- Encoder-decoder tasks **can be framed** in a decoder-only model.

> [!warning] Traps
> - *"Fine-tuning always updates the pre-trained parameters $\theta$."* False: $\theta'$ always, $\theta$ only maybe.
> - *"BERT is a decoder-only model trained on next-word prediction."* False: encoder-only, predicts its own noised input.
> - *"Tasks like machine translation require an encoder-decoder architecture."* False: they can be framed in a decoder-only model.
> - *"Contextual pre-training requires a Transformer."* False: LSTMs and CNNs also qualify.

## BERT pre-training: masking, `[CLS]` and next sentence prediction

- BERT is the **encoder** part of the Transformer.
- Pre-training on missing-word prediction is **self-supervised**: large amounts of data can be used, and task relevance is **not required**.
- Fine-tuning on the actual task is **supervised**, and possible even for **small** amounts of task data.
- Two prediction tasks: predict **masked tokens** (MLM), and predict whether sentence B is the **next sentence** (NSP).
- Tokens are **selected** with probability $p_{mask} = 0.15$. A selected token becomes `[MASK]` **80%** of the time, a **random word** 10%, and stays **the original word** 10%.
- The loss is computed **only** for the selected positions, **regardless** of whether they show `[MASK]`, a random word or the original word.
- `[CLS]` is **not tied** to a specific word and is **not masked**.
- `[CLS]` is **not meant** to learn context-specific word representations; it captures a general representation of the **entire** input sequence.
- The loss for `[CLS]` is computed with respect to a **classification** task (in BERT: NSP).
- NSP: **50%** of the time B directly follows A in a document, **50%** of the time B is chosen **at random from the corpus**.
- In practice segments A and B are **much longer** than individual sentences.
- BERT-large is deeper and wider than BERT-base. The slide lists **16** attention heads for both sizes; the BERT paper gives 12 for base.
- Monolingual BERT was trained on **BookCorpus plus English Wikipedia**, with **WordPiece** tokenisation.

> [!warning] Traps
> - *"15% of the input tokens are replaced by `[MASK]`."* False: 15% are selected; only 80% of those get `[MASK]`.
> - *"The MLM loss is computed only at positions showing `[MASK]`."* False: at every selected position, including random and unchanged ones.
> - *"Tokens left unchanged are excluded from the loss."* False: if selected, they are predicted like the others.
> - *"`[CLS]` is masked with the same probability as any other token."* False: it is never masked.
> - *"In NSP, a negative example takes the sentence that precedes A."* False: B is a random sentence from the corpus.
> - *"BERT pre-training requires data relevant to the downstream task."* False: task relevance is not required.

## Fine-tuning BERT, NLI and SQuAD spans

- The task-specific model is **small-ish** and **randomly initialised**.
- Word-level tasks connect the task model to the **top encoder layers** at the token positions; classification tasks connect it to the **`[CLS]` top layer**.
- Fine-tuning runs **with or without** updates to BERT's own parameters.
- Fine-tuning uses a **smaller** batch size (16 or 32 sequences) and a **limited** number of epochs (2 to 4).
- NLI (entailment): the premise **entails**, **contradicts** or has **no relationship** to the hypothesis: a **3-way** classification.
- SNLI is large-scale and **English only**, commonly used for training/fine-tuning.
- XNLI is a crowd-sourced set of **test and dev** pairs, **translated** into **14** languages (15 with English); a premise can be combined with its hypothesis in any of the 15 languages.
- SQuAD output is a **start/end span** in the paragraph: nothing is generated.
- SQuAD fine-tuning learns **one** start vector $S$ and **one** end vector $E$; each position $i$ is scored by the inner products $T_i \cdot S$ and $T_i \cdot E$, normalised with a **softmax**.
- The predicted span maximises $T_i \cdot S + T_j \cdot E$ subject to $i \le j$.

> [!warning] Traps
> - *"The fine-tuning head is initialised from the pre-trained model."* False: randomly initialised.
> - *"Classification heads read the top-layer output of the first word."* False: they read the `[CLS]` output.
> - *"SQuAD fine-tuning learns a separate start and end vector for every position."* False: one $S$ and one $E$, shared across positions.
> - *"XNLI was collected independently in each of its 15 languages."* False: English pairs translated into 14 languages.
> - *"Fine-tuning runs for many epochs because task data is small."* False: 2 to 4 epochs.

## WordPiece against BPE and Unigram LM

- WordPiece (Schuster and Nakajima 2012) is very similar to BPE, and **predates** it.
- It works **bottom-up**, like BPE and unlike Unigram LM (**top-down** pruning).
- Training: assume word boundaries, split words into characters, prefix every **non-initial** character with `##`.
- Merge criterion: $\text{score}(a,b) = \dfrac{\text{count}(a,b)}{\text{count}(a)\cdot\text{count}(b)}$; merge the highest-scoring pair until the vocabulary reaches the **target size**.
- Inference: split into characters, then **left to right greedy longest match** against the vocabulary.
- If no matching string exists at a position, WordPiece returns `[UNK]` for the **whole word**.

| | BPE | WordPiece | Unigram LM |
|---|---|---|---|
| Direction | bottom-up merges | bottom-up merges | top-down pruning |
| Criterion | raw pair frequency | association score | corpus likelihood loss (EM) |
| Inference | replay merges in order | greedy longest match | Viterbi or sampling |
| Output | deterministic | deterministic | deterministic or sampled |
| Unknowns | character/byte fallback | whole word `[UNK]` | character fallback |
| Models | GPT, RoBERTa, original XLM | BERT, mBERT, DistilBERT | T5, XLM-R, ALBERT |

> [!warning] Traps
> - *"WordPiece merges the most frequent adjacent pair."* False: that is BPE; WordPiece divides by both parts' counts.
> - *"WordPiece inference replays the learned merges in order."* False: greedy longest match.
> - *"WordPiece builds its vocabulary top-down by pruning."* False: bottom-up; Unigram LM prunes.
> - *"WordPiece was developed after BPE as a refinement of it."* False: it predates BPE.
> - *"WordPiece maps only the unknown character to `[UNK]`."* False: the whole word.
> - *"WordPiece can sample different segmentations of the same word."* False: deterministic; only Unigram LM samples.

## Masking subword pieces, and whole-word masking

- Nouns, adjectives and verbs are **harder** to predict than determiners and prepositions.
- Subword segments that mostly carry **morphological** information (tense, number, case) are **easier** to predict than segments carrying the **root/stem**.
- Whole-word masking **always** masks all subword tokens of the same word, and the masking rate **remains unchanged**.
- Whole-word masking improves BERT-Large on SQuAD and MultiNLI, in **both** cased and uncased versions.

> [!warning] Traps
> - *"Whole-word masking raises the masking rate to cover entire words."* False: the rate is unchanged.
> - *"Suffix pieces such as `-ed` are harder to predict than stems."* False: easier.
> - *"Whole-word masking only helps the uncased model."* False: cased and uncased both improve.

## mBERT: what its training has and lacks

- mBERT was introduced on **GitHub** in 2018, with **no paper**.
- It uses the **exact same architecture** as BERT and the **same** training loss: MLM plus NSP.
- **No** cross-lingual loss and **nothing** language-specific.
- **No** parallel data (at least not intentionally), **no** language-ID embedding, **no** language-specific parameters (e.g. adapters), **no** alignment loss rewarding translation equivalents for being close.
- Trained on the concatenated **Wikipedias** of **104** languages.
- NSP sentence pairs are **always within the same language**; languages are **not mixed** within a sequence, but a **batch can** contain several languages.
- **One shared** subword vocabulary: mBERT uses WordPiece, but BPE or Unigram LM **work similarly well**.
- High-resource and low-resource languages **share the same** subword vocabulary.

> [!warning] Traps
> - *"mBERT adds a language-ID embedding to each token."* False: no language-ID embedding.
> - *"mBERT builds NSP pairs from sentences in two different languages."* False: always the same language.
> - *"Each mBERT training batch contains a single language."* False: a batch can mix languages.
> - *"mBERT drops NSP and trains with MLM only."* False: that is XLM-R.
> - *"mBERT's shared vocabulary only works with WordPiece."* False: BPE or Unigram LM work similarly well.

## Sampling languages with the exponent $\alpha$

- The Wikipedia corpus is **extremely skewed**: English **alone** is larger than dozens of smaller languages combined.
- Weighted sampling: $p_l = \dfrac{n_l^{\alpha}}{\sum_{l'} n_{l'}^{\alpha}}$, where $n_l$ is the language's resource size (in characters, tokens, etc.) and $\alpha$ indicates **importance**.
- $\alpha = 1$: sampling **in proportion** to raw size; English and a few other large Wikipedias **dominate**.
- $\alpha = 0$: **uniform** sampling, which oversamples tiny Wikipedias so much that the model would mostly **memorise** them.
- $\alpha = 0.7$ **up-weights** low-resource languages and **down-weights** high-resource ones.
- The same smoothing **also applies to subword segmentation frequencies** when building the shared vocabulary.

> [!warning] Traps
> - *"$\alpha = 0$ samples languages in proportion to their size."* False: $\alpha = 1$ does; $\alpha = 0$ is uniform.
> - *"$\alpha = 0.7$ down-weights low-resource languages."* False: it up-weights them.
> - *"Uniform sampling is best for low-resource languages, since they get the most data."* False: the model mostly memorises them.
> - *"Lowering $\alpha$ from 1 to 0.7 increases English's share."* False: it decreases it.
> - *"Language smoothing affects training data but not the vocabulary."* False: it also applies to subword frequencies.

## The shared vocabulary and who pays for it

- mBERT's vocabulary has about **120k** entries for all 104 languages, against 30k for English BERT.
- mBERT has **more** parameters than BERT-base, and the difference is due **only to the vocabulary**: the architecture is the same.
- **Chinese** characters, Japanese **kanji** and Korean **hanja** are split into **individual characters before** WordPiece is applied.
- **Even with** smoothing, the shared vocabulary still favours **high-resource** languages and **Latin script**.
- Low-resource and **morphologically rich** languages are split into **more** pieces per word (**higher** fertility).
- The **uncased** mBERT lowercases and **strips accents**, which damages languages that depend on diacritics.

> [!warning] Traps
> - *"mBERT has more parameters than BERT-base because it has more layers."* False: same architecture; the bigger embedding matrix accounts for all of it.
> - *"All Korean text is split into single characters before WordPiece."* False: only hanja (Chinese characters).
> - *"Smoothing gives every language equally good coverage in the vocabulary."* False: high-resource and Latin-script languages are still favoured.
> - *"Morphologically rich languages get lower fertility."* False: higher.

## Zero-shot transfer in Pires et al.: scripts, overlap, typology

- Zero-shot transfer: training language A for task X improves performance in language B on task X **without any** task data in B.
- Pires et al. (2019) test **16** languages on **sequence labelling**: NER and POS tagging.
- In the fine-tuning-by-evaluation table, the **off-diagonal** cells are zero-shot; performance there is **decent**, showing generalisation beyond the fine-tuning language.
- Zero-shot stays **below** in-language performance, and transfer is **not symmetric** (NER English to Dutch beats Dutch to English).
- mBERT **generalises well across scripts** (Hindi and Urdu both ways, English to Bulgarian).
- Pairs involving **Japanese** drop, which **may** be due to larger **typological** differences.
- Vocabulary overlap is measured as **Jaccard** similarity over wordpieces occurring in **named entities** only: $|E_{l,t} \cap E_{l',e}| / |E_{l,t} \cup E_{l',e}|$.
- **mBERT**'s zero-shot NER F1 is high **even at low** overlap and depends far less on it; **English BERT** starts near zero and **rises** with overlap.
- Word order: from an **SVO** language, SVO targets transfer far better than SOV targets; from an **SOV** language the two are about **equal**.
- Adjective-noun order: same-order transfer is better, with a **smaller** effect.
- Zero-shot POS accuracy **rises** with the number of shared WALS features, for **both** mBERT and English BERT, with mBERT **higher** throughout.

> [!warning] Traps
> - *"mBERT transfers poorly between languages written in different scripts."* False: it generalises well across scripts.
> - *"The Japanese drop is explained by Japanese's script."* False: the slide suggests typological differences.
> - *"Like English BERT, mBERT's zero-shot NER depends strongly on entity wordpiece overlap."* False: mBERT scores high even at low overlap.
> - *"Pires et al.'s overlap counts all wordpieces in the training and test sets."* False: only wordpieces in named entities.
> - *"Zero-shot transfer between two languages is symmetric."* False.

## WALS

- WALS (World Atlas of Language Structures) records **structural** properties of languages: phonological, grammatical, lexical.
- Its data is gathered from **descriptive materials** such as reference grammars.
- Each feature has a **small set** of possible values (Tone: no tones, simple, complex).
- Coverage is wide but **most cells are empty**: **no** language has data for every feature, and **no** feature has values for every language.
- **Well-studied** languages dominate; English has the most features filled in.
- In NLP, WALS serves as a measure of **structural similarity** between languages, which can be represented as **vectors**.

> [!warning] Traps
> - *"WALS has complete feature data for well-studied languages such as English."* False: no language has data for every feature.
> - *"WALS features are induced automatically from corpora."* False: gathered from reference grammars.

## Translation retrieval across mBERT layers

- Sample of 5k sentence pairs from **WMT16**.
- For each layer, a sentence is the **average of all hidden activations, excluding `[CLS]` and `[SEP]`**.
- **One average** layer-wise difference vector per language pair: $\bar v^{(l)}_{EN\to DE} = \frac{1}{M}\sum_i \big(v^{(l)}_{DE_i} - v^{(l)}_{EN_i}\big)$.
- Each sentence is "translated" by **adding** that vector, then matched to its **nearest neighbour** by **$\ell_2$ distance**, not cosine similarity.
- Accuracy rises from the low layers, **peaks in the middle layers** (around layers 7 to 8 of 12), and **falls** towards the top, for EN-DE, EN-RU and UR-HI alike.

> [!warning] Traps
> - *"Each sentence is represented by its `[CLS]` vector."* False: the mean of the token states, excluding `[CLS]` and `[SEP]`.
> - *"Nearest neighbours are found by cosine similarity."* False: $\ell_2$ distance.
> - *"Retrieval accuracy increases steadily up to the top layer."* False: it peaks in the middle layers.
> - *"A separate difference vector is computed for each sentence pair at test time."* False: one average vector per layer and language pair.

## mBERT against other systems (Wu and Dredze)

- Wu and Dredze (2019) test zero-shot transfer from **English** on five tasks: MLDoc document classification, NLI (XNLI), NER, POS tagging, dependency parsing.
- MLDoc is document classification with **four** classes; only the **first two sentences** of each document are used, because of **memory** constraints.
- MLDoc: zero-shot mBERT comes a **close second** to Artetxe and Schwenk, a model pre-trained with **bitext**; in-language training is far higher.
- XNLI: **translate-train** (machine-translated training data, pseudo-supervision) **beats** zero-shot mBERT.
- XNLI zero-shot: mBERT narrowly **beats** X-LSTM, but **loses** to Artetxe and Schwenk, to XLM with MLM+TLM, and **even to XLM with MLM only**, which uses no cross-lingual signal.
- NER: zero-shot mBERT **beats** the dedicated cross-lingual system of Xie et al.
- POS: zero-shot mBERT does **not** match Kim et al., who use **small amounts** of target-language training data.

> [!warning] Traps
> - *"Zero-shot mBERT outperforms translate-train on XNLI."* False: translate-train is better.
> - *"On XNLI, zero-shot mBERT beats every model pre-trained with cross-lingual signal."* False: it beats only X-LSTM.
> - *"Zero-shot mBERT beats a POS tagger trained with a small amount of target-language data."* False: Kim et al. win.
> - *"MLDoc classifies complete documents."* False: only the first two sentences.

## Freezing layers during English fine-tuning

- Setup: fine-tune on English with the $k$ lowest layers **frozen**, $k \in \{0, 3, 6, 9\}$, where layer 0 is the **embedding** layer.
- Using mBERT **without** fine-tuning (feature extraction, "Feat") scores **below** the best fine-tuned setting on average on **every** task.
- Freezing the **lowest** layers is neutral or **helps** zero-shot transfer compared with fine-tuning everything.
- Freezing up to layer **9** hurts NER, POS and XNLI on average.

> [!warning] Traps
> - *"Layer 0 refers to the first Transformer layer."* False: the embedding layer.
> - *"Using mBERT as a frozen feature extractor gives the best zero-shot transfer."* False: fine-tuning is needed.
> - *"The more layers are frozen, the better the zero-shot transfer."* False: freezing up to layer 9 hurts most tasks.

## Vocabulary sharing: correlation by task

- $V^\ell_{obs} = V^{en}_{train} \cap V^\ell_{test}$: the test wordpieces of language $\ell$ that also occur in the English training data.
- $p^\ell_{type}$ counts each wordpiece **type once**; $p^\ell_{token}$ weights each wordpiece by its **frequency** $c^\ell_w$ in the test set.
- NER: an **almost perfect** correlation between overlap and zero-shot score.
- XNLI: **no** correlation at type level, a weak non-significant one at token level.
- MLDoc, POS and parsing: **positive** correlations, between those two extremes.

> [!warning] Traps
> - *"Wordpiece overlap with English strongly predicts zero-shot XNLI accuracy."* False: no correlation at type level.
> - *"Token-level overlap counts each wordpiece type once."* False: it weights by test-set frequency.

## Isolating the causes (K et al.)

- K et al. (2020) pre-train **bilingual** BERT (B-BERT) on (fake) English plus one other language, and test on XNLI.
- **Fake English** shifts every English UTF-8 codepoint by a **large constant**, so it shares **no** characters with any other language; everything else about English stays the same.
- Removing **all** subword overlap costs **little** zero-shot accuracy: (sub)word overlap does **not** affect transfer much.
- Word order: words are randomly **permuted during pre-training** only, with **no** permutation during fine-tuning.
- More permutation gives a **significant** drop, but still **reasonable** transfer; Hindi drops most.
- Of the architecture variants, **depth** has the **largest** impact; the parameter count is kept roughly **constant**.
- With more layers, in-language accuracy (fake English) **saturates** early while zero-shot accuracy (Russian) **keeps rising**, so the gap **shrinks**.
- When the task input mixes two languages (premise in one, hypothesis in the other), performance drops **significantly**, below **both** monolingual conditions.
- A target-language premise with a fake-English hypothesis does **better** than the reverse.

> [!warning] Traps
> - *"Removing all subword overlap eliminates most cross-lingual transfer."* False: the cost is small.
> - *"Word permutation is applied during pre-training and fine-tuning."* False: pre-training only.
> - *"Permuting words during pre-training eliminates transfer."* False: significant drop, reasonable transfer remains.
> - *"The number of attention heads has the largest impact on transfer."* False: depth.
> - *"Deeper models gain more in-language than cross-lingually."* False: in-language saturates, zero-shot keeps rising.
> - *"B-BERT handles an English premise with a Spanish hypothesis as well as two Spanish sentences."* False: mixed pairs drop significantly.

## Within and across languages (Rajaee and Monz)

- Rajaee and Monz (2024) evaluate mBERT on XNLI for **every** premise-language and hypothesis-language combination of the 15.
- **Within**: premise and hypothesis in the same language. **Across**: in different languages.
- Across is **below** within for **every** language; English has the highest within score.
- For each hypothesis language, a premise in the **same** language works best.
- A foreign premise with an **English hypothesis** scores higher than an English premise with a foreign hypothesis.
- Hypothesis: transfer of **heuristics** **can contribute** to cross-lingual generalisation.
- Rajaee et al. (2022) count **word overlap** between premise and hypothesis in SNLI: full overlap is **overwhelmingly** entailment, low overlap **mostly** non-entailment.

> [!warning] Traps
> - *"For some languages mBERT's across score exceeds its within score."* False: across is lower for every language.
> - *"An English premise helps mBERT more than an English hypothesis."* False: the English hypothesis helps more.
> - *"Rajaee and Monz show that heuristics are the main cause of cross-lingual generalisation."* False: a hypothesis that they can contribute.
> - *"In SNLI, pairs with full word overlap are mostly contradictions."* False: mostly entailment.

## XLM-R against mBERT

- XLM-R (Conneau et al. 2020) is the **most prominent** extension of mBERT, in **RoBERTa** style.
- It **drops NSP**, because NSP contributes little and **sometimes hurts**: MLM only.
- **Static** masking is applied once at the data level, as part of **pre-processing**; **dynamic** masking masks multiple passes over the same input **differently**. XLM-R uses dynamic masking.
- Data: **CC-100**, filtered CommonCrawl in **100** languages, about **two orders of magnitude** more monolingual data than mBERT's Wikipedia.
- Vocabulary: **250k** entries, **double** mBERT's, built with **Unigram LM** (SentencePiece), with **better** fertility.

| | mBERT | XLM-R |
|---|---|---|
| Objective | MLM + NSP | MLM only |
| Masking | static | dynamic |
| Data | Wikipedia, 104 languages | CC-100, 100 languages |
| Vocabulary | 120k WordPiece | 250k Unigram LM |

- **Within** a language, XLM-R beats mBERT on both XNLI and QA (XSQuAD), on average and in nearly every language; English QA is about **equal**.
- **Across** languages on XNLI, XLM-R also beats mBERT.
- **Across** languages on QA, XLM-R is **worse** than mBERT on average, with a much larger within-across gap.
- XLM-R's cross-lingual QA is strongest when the question is in **English**.

> [!warning] Traps
> - *"XLM-R covers more languages than mBERT."* False: 100 against 104.
> - *"XLM-R keeps NSP but adds dynamic masking."* False: it drops NSP.
> - *"XLM-R uses a WordPiece vocabulary."* False: Unigram LM via SentencePiece.
> - *"XLM-R's larger vocabulary worsens fertility."* False: it improves it.
> - *"XLM-R beats mBERT whenever question and context are in different languages."* False: its average across-language QA score is lower.

## What drives mBERT's cross-lingual transfer

- Shared (sub)word vocabulary **does play a role**, but it is **not the main cause** of transfer.
- Structure (grammar, word order) matters: permuting words hurts **significantly** but does **not entirely** eliminate transfer.
- Architecture matters: **deeper** networks can learn **more language-independent** representations.
- Bottom line: **no single factor** explains transfer, or the lack of it.

> [!warning] Traps
> - *"Shared vocabulary plays no role in mBERT's transfer."* False: it plays a role, but is not the main cause.
> - *"Word order similarity fully explains mBERT's transfer."* False: no single factor does.

## Practice questions

**1.** Which of the following statements about BERT pre-training are true? (multiple answers possible)
- [A] Of the tokens selected for masking, some are left unchanged in the input and the model is still trained to predict them.
- [B] About 15% of all input tokens are replaced by the `[MASK]` symbol.
- [C] The `[CLS]` token is masked with the same probability as word tokens.
- [D] In next sentence prediction, a negative segment B is drawn at random from the corpus.

> [!example]- Answer
> **A and D.** B: 15% are selected, and only 80% of those become `[MASK]`. C: `[CLS]` is never masked.

**2.** Which of the following statements about subwords in BERT are true? (multiple answers possible)
- [A] WordPiece merges the adjacent pair with the highest $\text{count}(a,b)/(\text{count}(a)\cdot\text{count}(b))$.
- [B] At inference, WordPiece replays its learned merges in the order they were learned.
- [C] Whole-word masking masks a larger share of tokens than standard BERT masking.
- [D] A masked piece carrying tense or number is easier to predict than a masked piece carrying the stem.

> [!example]- Answer
> **A and D.** B: inference is greedy longest match; replaying merges is BPE. C: the masking rate is unchanged.

**3.** Which of the following statements about mBERT's training are true? (multiple answers possible)
- [A] Next sentence prediction pairs can combine a sentence in one language with a sentence in another.
- [B] A single training batch can contain several languages.
- [C] With $\alpha = 0.7$, low-resource languages are sampled more often than their raw share of the data.
- [D] mBERT has more parameters than BERT-base because it has more layers.

> [!example]- Answer
> **B and C.** A: NSP pairs are always within the same language. D: same architecture; the extra parameters come from the larger vocabulary.

**4.** Which of the following statements about what drives mBERT-style cross-lingual transfer are true? (multiple answers possible)
- [A] A bilingual BERT trained on fake English and Spanish still transfers well from fake English to Spanish, although the two share no wordpieces.
- [B] In K et al.'s word-order experiment, words are permuted during both pre-training and fine-tuning.
- [C] With the parameter count held roughly constant, adding layers raises zero-shot accuracy more than in-language accuracy.
- [D] English BERT's zero-shot NER F1 stays roughly constant as entity wordpiece overlap grows.

> [!example]- Answer
> **A and C.** B: permutation is applied during pre-training only. D: that describes mBERT; English BERT's F1 rises with overlap.

**5.** Which of the following statements about zero-shot transfer from English with mBERT are true? (multiple answers possible)
- [A] On XNLI, zero-shot mBERT is outperformed by XLM trained with MLM only, which uses no cross-lingual signal.
- [B] Zero-shot mBERT outperforms Kim et al.'s POS tagger, which uses a small amount of target-language training data.
- [C] In translation retrieval with mBERT, accuracy is highest in the top layer.
- [D] Keeping the lowest layers frozen during English fine-tuning can improve zero-shot transfer compared with fine-tuning all layers.

> [!example]- Answer
> **A and D.** B: Kim et al. beat zero-shot mBERT on POS. C: accuracy peaks in the middle layers and falls towards the top.

**6.** Which of the following statements about inputs that combine two languages are true? (multiple answers possible)
- [A] For mBERT on XNLI, accuracy across languages is below accuracy within the language for every language.
- [B] XLM-R scores higher than mBERT on average on cross-lingual QA, where question and context are in different languages.
- [C] For mBERT, a foreign premise with an English hypothesis scores higher than an English premise with a foreign hypothesis.
- [D] Rajaee and Monz show that premise-hypothesis word overlap heuristics are the main cause of cross-lingual generalisation.

> [!example]- Answer
> **A and C.** B: XLM-R is better within languages, but worse than mBERT across languages on QA. D: transfer of heuristics is a hypothesis that it can contribute.
