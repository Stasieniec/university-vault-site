---
type: exam-prep
course: 5204MNLP6Y
lecture: 7
status: complete
---

# MNLP-L07 Exam Prep - Crosslingual NLP

> [!abstract] What the exam can ask here
> Expect two kinds of block: crosslingual pretraining (which model sees parallel data, and what each objective does) and low-resource MT (parent-child transfer, multilingual NMT, sampling, BART and mBART). The sample exam's question 3.1 comes from this lecture. Statements turn on the data a model sees (monolingual or parallel), what is shared or frozen, which side gains (into or out of English, high- or low-resource) and quantifiers like "every", "always" and "only". Two places need care: the slide's "decoder parameters are frozen" against its own freezing experiment, and MMLM, where the slide and the InfoXLM paper differ. Answer with the slide's wording.
> [[MNLP-L07 - Crosslingual NLP|lecture note]] · [[MNLP - Cheat Sheet|cheat sheet]] for numbers.

## Multilingual models get no crosslingual signal

- mBERT and XLM-R are **multilingual**: joint training, **one** model and **one** subword vocabulary for all languages, predictions from context **within the same language** (MLM).
- **Nothing** explicitly links information across languages: predicting word $W$ in language $L$ gains **nothing** from unrelated context in language $L'$.
- mBERT **never** sees a translation pair in pretraining. Any crosslingual alignment it has is an **emergent side effect**: no objective is designed to produce it.
- Crosslingual knowledge transfer: fine-tune on task X in language **A**, test on task X in language **B**. It matters because annotated fine-tuning data is **scarce** outside high-resource languages.
- Zero-shot results **seem to support** crosslingual ability without a crosslingual signal. Crosslingual tasks (two languages in one input) **cast doubt** on it.

| model | parallel data in pretraining |
|---|---|
| mBERT, XLM-R | no |
| XLM | yes (TLM) |
| InfoXLM | yes (TLM, XLCo) |
| BART | no |
| mBART | no |

> [!warning] Traps
> - *"Training jointly on many languages gives the model an explicit crosslingual signal."* False: masked language modelling stays inside one language. The signal needs parallel or comparable data.
> - *"mBERT's crosslingual alignment comes from a dedicated pretraining objective."* False: it is an emergent side effect.
> - *"mBART is trained with a crosslingual objective."* False: the BART objective is applied within the same language.

## Parallel data, comparable data and LASER

- **Parallel** data: sentence pairs in different languages that are **meaning equivalent**. Data-driven MT has used it for decades.
- **Comparable** data: sentence or document pairs on the **same topic**, **not fully** parallel. The degree of parallelism is a **sliding scale** (news on the same event, Wikipedia).
- Parallel data is sometimes a **natural by-product**: multilingual news, the UN and EU, countries with two or more official languages. **OPUS** collects parallel corpora for research.
- Or crawl your own: find parallel **documents** (document alignment) and parallel **sentences** (sentence alignment), or align sentences **directly** from raw web crawls such as Common Crawl.
- Measuring meaning equivalence: old-fashioned **dictionary overlap** and **relative length** distribution; more recent **sentence embeddings** (LASER, Artetxe and Schwenk 2019).
- LASER encoder: BPE embeddings, a stack of **BiLSTMs**, **max pooling** into one sentence embedding. The LSTM decoder gets the sentence embedding plus a **language ID** $L_{id}$ for the output. The encoder gets **no** language ID.
- **vecalign**: sentence alignment that **uses LASER** to match sentences, with an efficient search that **scales to massive** data sets.

> [!warning] Traps
> - *"Comparable data consists of sentence pairs that translate each other."* False: that is parallel data. Comparable data shares a topic and is only partly parallel.
> - *"LASER's encoder receives a language ID for the input sentence."* False: the language ID goes into the decoder.
> - *"LASER builds the sentence embedding from a `[CLS]` token."* False: it max-pools the BiLSTM states. `[CLS]` vectors belong to InfoXLM's XLCo.
> - *"vecalign matches sentences by dictionary overlap."* False: by LASER embedding similarity.

## XLM and translation language modelling

- XLM (Conneau and Lample) uses parallel data for an **explicit** crosslingual loss.
- **CLM**: given a prefix, predict the next word. **MLM**: predict a masked word from its context, as BERT. **TLM**: predict masked words in a context made of **two languages** (a sentence and its translation).
- **TLM is the crosslingual objective**, used in two settings: **MLM + TLM** or **CLM + TLM**.
- **Next sentence prediction is dropped.**
- TLM input: the pair concatenated, with words masked in **both** halves. **Position embeddings restart at 0** in the second sentence; **language embeddings** (en, fr) mark each half.
- A masked English word can be read off its unmasked French translation, so the model learns to attend across languages.
- Parallel data comes from **OPUS**.
- XNLI settings: translate-train, translate-test and zero-shot. The slide only names them. Per the lecture note, translate-train machine-translates the English **training** set into each language; translate-test translates each **test** set into English.
- XNLI: MLM + TLM beats MLM alone in **every** language. Zero-shot MLM + TLM beats translate-test **on average**, but translate-test still wins most European languages. Translate-train is best **on average**. Zero-shot XLM (MLM + TLM) beats LASER in **every** language.
- Nepali language model: adding a second language lowers perplexity. **Hindi** (related) helps **more** than English; both together is best.
- Word alignment: XLM's embeddings beat the fastText baselines MUSE and Concat (Concat = joint fastText training over all languages).

> [!warning] Traps
> - *"XLM keeps BERT's next sentence prediction."* False: dropped.
> - *"TLM is used as XLM's only pretraining objective."* False: always paired with MLM or CLM.
> - *"In TLM, positions keep counting from the end of the first sentence."* False: they restart at 0.
> - *"In TLM, words are masked only in the source-language sentence."* False: masks appear in both halves.
> - *"Zero-shot XLM beats translate-test in every XNLI language."* False: only on average.
> - *"English data helps a Nepali language model more than Hindi data."* False: Hindi helps more.

## InfoXLM against XLM

- InfoXLM (Chi et al. 2021) **builds on XLM-R** and adds XLM's parallel-data ideas.
- **MMLM**: like MLM, but instead of directly optimising the loss of the ground truth it uses **negative sampling**. This is the slide's description; the InfoXLM paper describes MMLM as the ordinary masked-LM loss. Answer with the slide.
- **TLM** as in XLM.
- **XLCo**: takes the **`[CLS]`** representations of a **parallel sentence pair** and **classifies whether the pair is parallel**, with **negative sampling** using **random sentences** as negatives ([[Contrastive Learning]]).
- The three losses are **equally weighted**.

| | XLM | InfoXLM |
|---|---|---|
| start | from scratch | initialised from XLM-R |
| mono data | Wikipedia | CC-100 (94 languages) |
| parallel data | 15 XNLI languages | 14 English-centric pairs |
| vocabulary | shared BPE | XLM-R's SentencePiece |
| language emb. | yes | no |
| objectives | MLM (or CLM) + TLM | MMLM + TLM + XLCo |
| signal | token level only | token and sentence level |

> [!warning] Traps
> - *"InfoXLM is trained from scratch."* False: initialised from XLM-R, then pretrained further.
> - *"InfoXLM uses language embeddings, like XLM."* False: it has none, because it keeps XLM-R's architecture.
> - *"XLCo predicts masked words from the translation."* False: that is TLM. XLCo classifies whether a pair is parallel.
> - *"XLCo's negatives are translations of other sentences in the batch."* False: random sentences.
> - *"XLM aligns languages at the sentence level."* False: token level only. Sentence level comes with InfoXLM's XLCo.

## Within a language against across languages

- The test: fine-tune on English, then evaluate with the two parts of an input in **different** languages. XNLI: premise and hypothesis. XSQuAD (QA): context and question.
- **within**: both parts in the same language. **across**: the two parts in different languages.
- **Every** model (mBERT, XLM-R, InfoXLM) scores **lower across** than within, on both tasks.
- XLM-R beats mBERT **within** on both tasks (on average). **Across**, XLM-R is better on **XNLI** but **worse** than mBERT on **QA**.
- InfoXLM is **slightly** better than XLM-R within and **much** better across, with the **smallest** within-across gap on both tasks. Its biggest gain is on **QA**.
- The grids are **not symmetric**: swapping which language takes which part changes the score.
- XLM-R on cross-lingual QA works mainly when the **question** is in English.

> [!warning] Traps
> - *"XLM-R outperforms mBERT across languages on both XNLI and QA."* False: on QA it is worse across languages on average.
> - *"InfoXLM's gain over XLM-R comes mainly within a single language."* False: small within, large across.
> - *"Better within-language performance guarantees better across-language performance."* False: XLM-R against mBERT on QA.

## Machine translation as sequence-to-sequence modelling

- Seq2seq differs from sequence labelling: **no** isomorphic relation between $x_t$ and $y_t$, and $|X|$ and $|Y|$ **can differ**.
- Conditional language modelling: $p(y_t \mid Y_{<t}, X)$, with $X$ the **encoder** output and $Y_{<t}$ the **decoder** output before $t$ (the prefix).
- MT mappings: different lengths, word order differences, **one-to-many** (*Hiermit* → *with this*), **many-to-one** (*hörte ... auf* → *stop*).
- Data: source as one vector $z$; target as **two** vectors: $x$ = `<s>` + sentence (what the decoder reads), $y$ = sentence + `</s>` (what it predicts), so $y_i = x_{i+1}$.
- Sutskever et al. (2014): encoder **and** decoder are LSTMs. The decoder is initialised with the encoder's **last** state, $h^{dec}_0 = h^{enc}_n$: an **information bottleneck**.
- Transformer decoder layer: **target context** layer, **source-target context** layer (attends to the encoder output), **feed-forward** layer, each with a residual connection.

> [!warning] Traps
> - *"Seq2seq models assume input and output sequences have equal length."* False.
> - *"In Sutskever et al. (2014) the decoder attends to every encoder state."* False: it starts from the last encoder state only, hence the bottleneck.
> - *"The decoder input $x$ ends with `</s>`."* False: $x$ starts with `<s>`; $y$ ends with `</s>`.

## How much data NMT needs

- NMT quality depends on the **amount** of data, the **variation** in it, and its **relevance** to the task. Typical low-resource problems: **domain adaptation** and **low-resource language pairs**.
- Schwenk et al. (2019): **most** language directions are of poor quality. Core problem: **limited parallel data** for most directions. Models do not generalise **very well** beyond the training data, and **not at all** beyond specific language directions.
- Zoph et al. (2016): NMT trained on small child data alone is **below** syntax-based SMT for all four languages.
- Koehn and Knowles (2017): with little data NMT is **far below** phrase-based MT, improves **faster** as data grows, and overtakes it only at **large** sizes (later still against phrase-based MT with a big language model).

> [!warning] Traps
> - *"NMT outperforms phrase-based MT regardless of training data size."* False: it loses on small data.
> - *"Current MT models generalise well to language directions unseen in training."* False: not at all.

## Parent-child transfer learning

- Transfer learning is **basically the same as fine-tuning**: train on a high-resource pair (**parent**), then **continue training** the same model on the low-resource pair (**child**). Reason: learning tens of millions of parameters from small data is hard.
- The slides list when it works best: **identical target** language; **related source** languages; **decoder parameters frozen**; parent and child **share the same vocabulary**.
- Zoph et al. setup: **not shared**: source vocabulary and source embeddings. **Shared**: target language and all other parameters (target embeddings, attention). **Freeze target-specific** parameters.
- The flow: copy every parent parameter; the child's (Uzbek) words take over the parent's (French) **source-embedding rows**; freeze the **English (target) embeddings**; continue training on the child.
- Freezing experiment (Uzbek→English): everything frozen gives **zero** BLEU; unfreezing the **source embeddings** gives the **biggest** jump; unfreezing the source RNN, the **target RNN** and **attention** keeps helping; then unfreezing the **target embeddings lowers** BLEU.
- The list says "decoder parameters are frozen", but the experiment only needs the **target embeddings** frozen (training the target RNN helps). A statement in the list's wording is most likely true; one about the experiment is about the target embeddings.
- Transfer helps **all four** children (Hausa, Turkish, Uzbek, Urdu); Urdu, with the **least** data, gains most. Syntax-based SMT stays **ahead** of the transfer model in all four; the final system beats it only for Hausa.
- Relatedness (Zoph): a Spanish child gains more from a **French** parent than from a German one. **French'** (French with its vocabulary randomly reshuffled) still gains a lot from a French parent, so the gain does not come only from shared words.
- Kocmi and Bojar (2018), **shared** vocabulary: **unrelated** parents (English→Czech, English→Russian for an English→Estonian child) help as much as the related English→Finnish one, even with a different target language.
- Aji et al. (2020): transferring embeddings **and** inner layers is best; inner layers alone carry **most** of the gain; embeddings alone are **worse** than random initialisation on average. An English→German parent transfers about as well into X→English as a German→English parent.

> [!warning] Traps
> - *"Parent and child share the source vocabulary and source embeddings."* False: those are the parts not shared.
> - *"Letting the target embeddings train on the child data gives the best child."* False: it lowers BLEU.
> - *"The source embeddings should stay frozen during child training."* False: retraining them gives the biggest jump.
> - *"Transferring only the parent's embeddings beats training the child from scratch."* False: worse on average.
> - *"With a shared vocabulary, only a related parent helps."* False: unrelated parents help as much.
> - *"With transfer learning, NMT beats syntax-based SMT for every child language."* False: the transfer model stays below it for all four.
> - Sample exam 3.1 (see [[MNLP - Exam Analysis|exam analysis]]), where fine-tuning means parent-child transfer: *"Fine-tuning continuously improves with the number of in-domain updates."* False. The slides show the parameter version (letting more parameters adapt to the small child data helps, then hurts); that quality also peaks and then drops with more updates is standard knowledge beyond the slides.
> - Sample exam 3.1: *"Fine-tuning does not suffer from catastrophic forgetting."* False, by standard knowledge beyond the slides (the Canvas sample question asks it): training only on in-domain data overwrites what the model knew about the general domain.

## Multilingual NMT

- Parent-child transfers knowledge **in stages**; multilingual joint training transfers it **in parallel**.
- Johnson et al. (2017): **one single system** for all translation directions, with a **simple** architecture and training setup.
- **Many-to-one** and **one-to-many**: used in **training**. **Many-to-many**: **testing only**.
- **Zero-shot**: a direction **not encountered** during training.
- Signs of knowledge transfer: **zero-shot** performance, and the multilingual model **beating a baseline** trained only on that direction's data.
- Training is **single-language-centric** (typically English): only $xx \rightarrow en$ and $en \rightarrow xx$.
- **Every** training example carries a **target** language tag (`<2es>`), part of the source fed to the **encoder**.
- **All** parameters shared; **one WordPiece** vocabulary for all languages (32k, 64k); source, target and output embeddings **shared**.
- Directions are mixed during training, **even within a batch**. Language balancing through **(weighted) sampling**.
- **Many-to-one**: multilingual beats the bilingual baseline in **every** direction shown, by small margins. **One-to-many**: **mixed**, some directions lose.

> [!warning] Traps
> - *"The language tag tells the model the source language."* False: it names the target.
> - *"Each target language gets its own decoder."* False: all parameters are shared.
> - *"Many-to-many translation is a training configuration."* False: testing only.
> - *"Training data includes pairs without English."* False: English-centric.
> - *"One-to-many training beats the bilingual baseline in every direction."* False: that holds for many-to-one; one-to-many is mixed.

## Transfer against interference, and sampling

- Arivazhagan et al. (2019): 103 languages, grouped into high-, medium- and low-resource.
- **Low-resource**: multilingual beats bilingual (with balanced sampling), **far more into English** than out of English (transfer).
- **High-resource**: **every** multilingual model is **below** the bilingual baseline (interference).
- All→All (one model for both directions) is below the dedicated Any→En or En→Any model in every group.
- Sample **proportionally**: high-resource languages **dominate**, low-resource ones perform **poorly**. Sample **uniformly**: high-resource languages **suffer**, low-resource ones can do **well**. Best compromise: **over-sample** low-resource, **under-sample** high-resource.
- Temperature: **$T = 1$ proportional**, **$T = 100$ uniform**, $T = 5$ in between.

| sampling | high-resource | low-resource | medium |
|---|---|---|---|
| $T = 1$ | best | worst, below bilingual | worst |
| $T = 100$ | worst | best | middle |
| $T = 5$ | middle | close to $T = 100$ | best |

> [!warning] Traps
> - *"Uniform sampling hurts low-resource languages."* False: it hurts high-resource languages.
> - *"Raising the temperature moves sampling towards proportional."* False: towards uniform; $T = 1$ is proportional.
> - *"Multilingual training improves high-resource languages over bilingual baselines."* False: they lose to interference.
> - *"Low-resource languages gain about equally into and out of English."* False: far more into English.

## Zero-shot translation

- With no data for the pair, translation relies **entirely** on transfer.
- From 10 to 102 training languages: **decent improvement in most** cases, **not all** (French→Chinese gets worse).
- Probably also depends on the **actual languages** included; scores vary widely by pair (Belarusian→Russian high, Hindi→Finnish low).
- Zero-shot translation is **still lagging behind**.

> [!warning] Traps
> - *"Adding training languages always improves zero-shot translation."* False: most directions improve, but French→Chinese gets worse.
> - *"Zero-shot translation matches supervised directions."* False: it lags behind.

## BART

- Goal: reduce dependence on parallel data by using **monolingual** data.
- BERT: **encoder only** (bidirectional). GPT: **decoder only** (autoregressive). BART: **bidirectional encoder** plus **autoregressive decoder**, for tasks that need both.
- The source is corrupted by a noise function; the decoder generates the original.

| noise | source | decoder must |
|---|---|---|
| token masking | tokens → `[MASK]` | predict masked tokens |
| token deletion | tokens removed, no mark | predict deleted tokens |
| text infilling | a span → one `[MASK]` | predict the whole span |
| sentence permutation | sentences shuffled | restore the order |
| document rotation | starts at a random token | find the real start |

- Document rotation is called "document permutation" in the slide text.
- Comparison: **text infilling** is the best single noise function on almost every task; **token deletion** beats token masking on the generation tasks; **document rotation** and **sentence shuffling** alone are poor; infilling plus sentence shuffling gives the best CNN/DM perplexity; a left-to-right language model is **worst** on SQuAD.
- BART **matches RoBERTa** on SQuAD 1.1 F1 and is **best** on summarisation (CNN/DailyMail and XSum).
- Fine-tuning for classification: the label comes from the **last hidden time step**. Span prediction (SQuAD): each token is labelled; the model predicts **begin and end**.
- Fine-tuning for MT: a **randomly initialised source-language encoder** replaces the **embedding layer** of BART's encoder. BART's encoder parameters can be **frozen or updated**. The source vocabulary **can differ** from BART's.
- Romanian→English: **Fixed** BART is slightly **below** the baseline; **Tuned** BART is **above** it.

> [!warning] Traps
> - *"Text infilling replaces each removed token with its own `[MASK]`."* False: one `[MASK]` per span.
> - *"Token deletion marks the deleted positions."* False: nothing marks them.
> - *"BART's decoder is bidirectional."* False: the encoder is; the decoder is autoregressive.
> - *"For MT, the new encoder replaces BART's whole pretrained encoder."* False: it replaces the embedding layer and feeds the pretrained encoder.
> - *"Keeping BART frozen during MT fine-tuning beats the baseline."* False: only Tuned BART does.

## mBART

- BART is to mBART as BERT is to **mBERT**.
- Liu et al. (2020): **one** model trained on a **concatenation of monolingual** data in multiple languages. The BART objective is **always applied within the same language**: **no** crosslingual signal.
- A **language ID** token (`<En>`, `<Ja>`) **ends the encoder input** and **starts the decoder input**. The pretraining example masks spans and swaps sentence order (text infilling plus sentence permutation).
- Fine-tuning on MT at **sentence** or **document** level: the **target** language token starts the decoder. No extra encoder is added.
- mBART25 against random initialisation: **better** on **every** pair in **both** directions. With almost no data (English-Gujarati, 10K pairs) BLEU stays **near zero**. Gains are biggest in the low-to-medium range and mostly smaller at millions of pairs.

> [!warning] Traps
> - *"mBART is pretrained on parallel data."* False: concatenated monolingual data.
> - *"mBART needs a new randomly initialised encoder for translation."* False: that is BART's MT recipe; mBART is fine-tuned directly.
> - *"mBART pretraining makes translation work with almost no parallel data."* False: at 10K pairs it stays near zero.

| | BART | mBART |
|---|---|---|
| pretraining data | monolingual | multilingual monolingual |
| crosslingual signal | none | none |
| MT fine-tuning | new source encoder | whole model, as is |

## Practice questions

**1.** Which of the following models receive an explicit crosslingual training signal from parallel data during pretraining? (multiple answers possible)
- [A] mBERT
- [B] XLM trained with MLM + TLM
- [C] InfoXLM
- [D] mBART

> [!example]- Answer
> **B and C.** A: mBERT never sees a translation pair; its alignment is emergent. D: mBART applies the BART objective within the same language, on monolingual data.

**2.** Which of the following statements about XLM are true? (multiple answers possible)
- [A] In TLM, a masked English word can be predicted using the unmasked French translation in the same input.
- [B] Zero-shot XLM (MLM + TLM) outperforms the translate-test setting in every XNLI language.
- [C] Adding TLM to MLM improves zero-shot XNLI accuracy in every language.
- [D] For a Nepali language model, adding English data lowers perplexity more than adding Hindi data.

> [!example]- Answer
> **A and C.** B: zero-shot wins only on average; translate-test still wins most European languages. D: Hindi, the related language, helps more.

**3.** Models fine-tuned on English are tested on XNLI and XSQuAD with the two parts of each input in different languages. Which of the following statements are true? (multiple answers possible)
- [A] mBERT, XLM-R and InfoXLM all score lower across languages than within a language.
- [B] InfoXLM improves over XLM-R more across languages than within a language.
- [C] On question answering, XLM-R scores higher than mBERT across languages on average.
- [D] For XLM-R on question answering, mixed-language pairs work best when the question is in English.

> [!example]- Answer
> **A, B and D.** C: XLM-R is better within languages, but across languages on QA it falls below mBERT.

**4.** A French→English NMT model is used as the parent for an Uzbek→English child. Which of the following statements are true? (multiple answers possible)
- [A] The child does not share the parent's source vocabulary and source embeddings.
- [B] The best child model also lets the target embeddings train on the child data.
- [C] Once parent and child share a vocabulary, transfer requires a related parent language.
- [D] Initialising only the embeddings from the parent, with randomly initialised inner layers, beats random initialisation of everything on average.

> [!example]- Answer
> **A only.** B: unfreezing the target embeddings lowers BLEU. C: with a shared vocabulary, unrelated parents help as much (Kocmi and Bojar). D: embeddings alone are worse than training from scratch on average (Aji et al.).

**5.** Which of the following statements about multilingual NMT are true? (multiple answers possible)
- [A] In Johnson et al. (2017), the tag added to the source sentence tells the model which language to translate into.
- [B] Sampling with temperature $T = 1$ gives each language the same probability.
- [C] Low-resource languages gain more from multilingual training into English than out of English.
- [D] With $T = 100$, the multilingual model beats the bilingual baselines for high-resource languages.

> [!example]- Answer
> **A and C.** B: $T = 1$ is proportional sampling; $T = 100$ is uniform. D: under $T = 100$ high-resource languages do worst, and no multilingual model beats their bilingual baselines.

**6.** Which of the following statements about encoder-decoder models are true? (multiple answers possible)
- [A] Sequence-to-sequence modelling does not assume that input and output have the same length.
- [B] In Sutskever et al. (2014), the decoder LSTM is initialised with the last state of the encoder LSTM.
- [C] To use BART for MT, a randomly initialised source encoder replaces BART's entire pretrained encoder.
- [D] On Romanian→English, BART with frozen parameters outperforms the baseline without BART.

> [!example]- Answer
> **A and B.** C: the new encoder replaces only the embedding layer of BART's encoder. D: Fixed BART is slightly below the baseline; only Tuned BART beats it.
