---
type: exam-prep
course: 5204MNLP6Y
lecture: 4
status: complete
---

# MNLP-L04 Exam Prep - Subword Segmentation

> [!abstract] What the exam can ask here
> A block on this lecture most likely contrasts BPE with Unigram LM (bottom-up or top-down, greedy or global, one segmentation or several), tests what SentencePiece and byte-level BPE add, and asks which way the Bostrom and Durrett comparison and fertility go. The traps: which tokens Unigram LM prunes, OOV "very low" against "zero", which language shows the bigger Unigram LM advantage (English against gold segmentations, Japanese downstream), whether SentencePiece is an algorithm, and how BPE is applied at inference time.
> Lecture: [[MNLP-L04 - Subword Segmentation|lecture note]]. Numbers: [[MNLP - Cheat Sheet|cheat sheet]].

## Why subwords

- The vocabulary is **one of the main bottlenecks** of training an NMT system.
- How well a word is represented depends on its **frequency** and on the **number of different contexts** it occurs in.
- Many word occurrences come from word formation: **inflections** and **compounding**.
- A subword is a **character n-gram** or a **morphologically meaningful unit** (language dependent).
- Handling low-frequency words (including zero-frequency, OOV) is **one of the main motivations** for subword segmentation.

> [!warning] Traps
> - *"How well a word is represented depends only on its frequency."* False: also on the number of different contexts it occurs in.

## Rule-based analysers and Morfessor

- Morphological analysers were **historically** often **finite-state transducers** (FSTs): states, and transitions that digest a string and output a string (`a:x`).
- FST input is a **morpheme**; output is an **analysis** (1st person, noun, singular).
- FSTs are very powerful, **efficient**, and handle cases and exceptions with **great accuracy**.
- FSTs are **extremely laborious** to produce and **totally language dependent**.
- With ambiguities an FST gives **no preference**; **probabilistic** FSTs do exist.
- Data-driven analysers **learn** morphological regularities from data instead of hand-writing rules.
- The **first** data-driven approach: **Morfessor** (Creutz and Lagus, 2002), based on **Minimum Description Length**: what it costs the model to **generate the data**, plus what the **model itself** costs.
- Current approaches follow similar principles but **do not label** the data (no actual analysis), **do not account for** character substitutions or deletions, and are **strictly concatenative**.
- They need **no** data annotation or linguistic knowledge.
- They can be **fitted to a desired vocabulary size**.
- BPE (Sennrich et al., 2016), WordPiece (Schuster et al., 2012) and Unigram LM (Kudo, 2018) split words **based on frequencies**, with **no linguistic knowledge**.

> [!warning] Traps
> - *"FSTs handle exceptions poorly."* False: they handle cases and exceptions with great accuracy; their problems are effort and language dependence.
> - *"FSTs rank the analyses of an ambiguous word."* False: no preference, though probabilistic FSTs exist.
> - *"Current data-driven segmenters label each segment with its grammatical function."* False: they do not label the data.
> - *"Data-driven segmenters account for character substitutions and deletions."* False: they do not; they are strictly concatenative.
> - *"MDL minimises only the size of the model."* False: model cost plus the cost of generating the data.

## BPE: training and inference

- Training: (1) split words into characters, **keeping word boundaries** (`</w>`), and collect frequencies; (2) count **all neighbouring** symbol pairs; (3) merge the **most frequent** pair and repeat from (2).
- Training stops when a **maximum number of merge operations** is reached.
- Pairs are counted **within words**, weighted by each word's frequency.
- On *the tall man is taller than the tallest man*, the first merge is `t+h`, the second `t+a`.
- Inference: split the input into characters and apply the merges **in the same order they were learned**.
- BPE is **greedy**: once a sequence has been merged, **it stays that way**.
- BPE is **deterministic**: a word is **always** segmented the same way.
- Typically **30K to 128K** merge operations, the high end for **multilingual** models.
- Pieces need not be morphemes: *The preside@@ nt of France arr@@ ived in K@@ yrg@@ yz@@ stan .*

> [!warning] Traps
> - *"At inference time, BPE merges the most frequent pair of the input sentence."* False: it replays the learned merges in the learned order.
> - *"BPE training stops when no pair occurs more than once."* False: at a maximum number of merges.
> - *"BPE can return several segmentations of the same word."* False: it is deterministic.
> - *"BPE uses linguistic knowledge to place splits at morpheme boundaries."* False: frequencies only.

## BPE in translation

- Three advantages: **significantly reduces** the vocabulary size (less memory, better speed), **better translation quality**, **significantly reduces** the number of OOV items.
- These benefits apply to **both** high- and low-resource language pairs.
- In the BPE translation results, the subword and character-bigram systems use **far smaller** vocabularies than the word-level ones.
- **Joint** BPE (one merge set for both languages) beats **separate** BPE.
- Separate BPE as a single model scores **below** the best word-level single model.
- **Character bigrams** tie joint BPE for the best single model and give the **best ensemble**.
- The syntax-based system beats **every single** neural model; only **ensembles** of subword or character-bigram models beat it.

> [!warning] Traps
> - *"BPE's benefits apply only to low-resource language pairs."* False: both high- and low-resource.
> - *"Every BPE system beats every word-level system."* False: separate BPE is below the best word-level single model.
> - *"A single joint-BPE model beats the syntax-based system."* False: only ensembles do.
> - *"Joint BPE gives the best ensemble."* False: character bigrams do.

## Unigram LM against BPE

| | BPE | Unigram LM |
|---|---|---|
| builds the vocabulary | **bottom-up**, merging | **top-down**, shrinking |
| decisions | **greedy** | best **global** segmentation |
| segmentations of a word | **one**, deterministic | **several**: sampling, n-best |
| in SentencePiece | yes | yes |

- Unigram LM is by **Kudo (2018)**, part of Google's **SentencePiece**; the **other** part of SentencePiece is **BPE**.
- Unigram LM **considers all possible** segmentations of a word.
- It optimises for the best **global** segmentation.
- It supports **multiple** segmentations, allowing **sampling** and **n-best** lists.
- It is **top-down**: it starts from an **existing large** vocabulary and **shrinks** it to the desired size.

> [!warning] Traps
> - *"Unigram LM builds its vocabulary bottom-up by merging frequent pairs."* False: top-down, pruning a large vocabulary.
> - *"Like BPE, Unigram LM always segments a word the same way."* False: it supports sampling and n-best lists.
> - *"SentencePiece contains only the Unigram LM."* False: it contains BPE too.

## Unigram LM vocabulary pruning

- Start: $V$ = **all substrings occurring more than once** in the data $D$, **not crossing words**.
- While $|V| > k$: fit a unigram LM $\theta$ to $D$.
- For each token $t$: loss $L_t = p_\theta(D) - p_{\theta'}(D)$, with $\theta'$ the LM **without** $t$.
- Remove $\min(|V| - k, \lfloor \alpha |V| \rfloor)$ tokens with the **highest** $L_t$; $\alpha \in [0, 1]$ is a **hyperparameter**.
- So **at most** a fraction $\alpha$ goes per round, and the LM is **refit** each round.
- Finally fit the final unigram LM and return $V$ and $\theta$.
- The slide removes the **highest** $L_t$; Kudo's paper removes the **lowest**. Answer with the slide.

> [!warning] Traps
> - *"Unigram LM starts from the single characters of the corpus."* False: from all substrings occurring more than once.
> - *"The initial candidates include substrings that cross word boundaries."* False: not crossing words.
> - *"All surplus tokens are removed in a single step."* False: at most $\lfloor \alpha |V| \rfloor$ per round, with a refit in between.

## Finding the best segmentation: brute force and dynamic programming

- A string of $n$ characters has $2^{n-1}$ possible segmentations.
- Brute force: generate **all** $2^{n-1}$, score each, select the best.
- Scoring needs **scores for segments** (units not split further) and a way to **combine** the scores of sub-segmentations.
- Sequence cutting: with values 1, 5, 8, 9 for lengths 1 to 4, the best cut of length 4 is **2 + 2** (10), better than keeping it whole (9).
- [[Dynamic Programming]]: solve each sub-problem **only once** and **store** its solution.
- DP uses **additional memory** to save computation time: a **time-memory trade-off**.
- Two strategies, **top-down with memoization** and **bottom-up**, with the **same asymptotic** run time.
- The segmentation algorithm fills spans by **increasing length**: $m[i,j]$ starts at $p(c[i:j])$ (span kept whole), and a split at $k$ replaces it if $m[i,k] \cdot m[k+1,j]$ is **larger**; $s[i,j] = k$ records the split.
- The Viterbi segmentation returns **only the most probable** segmentation.
- For n-best: **Yen's and Eppstein's** algorithms, or **disallow certain sub-solutions**.

> [!warning] Traps
> - *"Bottom-up DP has a better asymptotic run time than top-down memoization."* False: the same.
> - *"DP saves both time and memory."* False: it spends memory to save time.
> - *"The Viterbi segmentation returns an n-best list."* False: only the best; n-best needs Yen's or Eppstein's algorithm.
> - *"A string of n characters has $n^2$ possible segmentations."* False: $2^{n-1}$.
> - *"In the DP segmentation algorithm, the probabilities of two sub-segments are added."* False: they are multiplied. (In the sequence-cutting example, segment values are added.)

## SentencePiece and lossless segmentation

- **Standard BPE implementations assume word boundary detection.**
- SentencePiece treats the input as a **raw stream of Unicode characters, including spaces**.
- The space character is **escaped as a visible symbol** (▁).
- Without it, `[Hello] [world] [.]` cannot be desegmented unambiguously (*Hello world .*, *Helloworld.*, *Hello world.*).
- The same question arises for languages **without clear word boundaries** (Japanese).
- With the space symbol, `Hello▁world.` becomes `[Hello] [▁wor] [ld] [.]` and desegments **losslessly**.

> [!warning] Traps
> - *"SentencePiece requires text that is already split into words."* False: that is standard BPE; SentencePiece takes a raw stream.
> - *"SentencePiece discards spaces and restores them with language-specific rules."* False: the space is kept as a visible symbol.

## Bostrom and Durrett: BPE against Unigram LM

- Unigram LM pieces sit closer to **morphemes**: `▁tri cycle s` against BPE `▁t ric y cles`; `▁corrupt ed` against `▁cor rupted`; `▁nano technology`; `▁1848` against `▁184 8`.
- In Japanese, BPE **attaches particles** (は, に) to neighbouring words; Unigram LM keeps them **separate**.
- Unigram LM produces **longer** segments on average, in **English and Japanese**.
- In English, Unigram LM uses its vocabulary space **more effectively**, with **more tokens of moderate frequency**.
- In English, BPE has more **word-initial single capitals** (▁H, ▁K, ...); Unigram LM more **suffixes and punctuation** (s, ing, ed, ly, commas, full stops).
- Unigram LM needs **fewer** tokens per word and per word type (English).
- Against gold segmentations (CELEX2 English, MeCab Japanese), Unigram LM has higher precision, recall **and** F1 in **both** languages; the gap is **much larger in English**.
- English recall is **low** for both tokenizers.
- Downstream, Unigram LM is ahead on **every** task: **slightly** in English (SQuAD, MNLI, CoNLL NER), by **a large margin** on Japanese TyDi QA.
- So the direction flips: the gold-segmentation gap is larger in **English**, the downstream gap larger in **Japanese**.
- Neither of the authors' models beats BERT-base everywhere: **both** are below it on MNLI and CoNLL NER.

> [!warning] Traps
> - *"BPE produces longer segments on average than Unigram LM."* False: Unigram LM does, in both languages.
> - *"Unigram LM's longer segments appear in English only."* False: in Japanese too.
> - *"BPE has higher precision against CELEX2 than Unigram LM."* False: Unigram LM wins precision, recall and F1.
> - *"Unigram LM's downstream gain is large in English and small in Japanese."* False: the reverse.
> - *"Against gold segmentations, Unigram LM's advantage is larger in Japanese than in English."* False: much larger in English.
> - *"The Unigram LM model outperforms BERT-base on every English task."* False: below it on MNLI and CoNLL NER.
> - *"Word-initial capitals are more frequent in Unigram LM output."* False: in BPE output; suffixes and punctuation are more frequent in Unigram LM output.
> - *"Unigram LM uses more tokens per word than BPE."* False: fewer.

## OOV and byte-level BPE

- BPE and Unigram have **very low** OOV rates, **not zero**.
- **Typos** can **sometimes** still cause OOVs; **unknown characters will** result in OOVs.
- Byte-level BPE **guarantees zero** OOV.
- It operates on **raw UTF-8 bytes** instead of Unicode characters.
- The base alphabet is **fixed at 256** symbols, so any input in **any script** can be represented.
- **Even unseen scripts** can be encoded byte by byte, then merged where patterns emerge.
- Non-Latin, non-ASCII scripts cost **more bytes per character**: extra merges are needed to get from bytes to characters, giving **shorter** subsegments.

> [!warning] Traps
> - *"BPE and Unigram LM guarantee a zero OOV rate."* False: very low; only byte-level BPE guarantees zero.
> - *"Byte-level BPE operates on Unicode characters."* False: on raw UTF-8 bytes.
> - *"Byte-level BPE cannot encode a script it never saw in training."* False: any script, byte by byte.
> - *"Under byte-level BPE, non-ASCII scripts get longer subsegments."* False: shorter, because they cost more bytes per character.

## Fertility

- Fertility = the **average number of segments** a word is split into.
- It is influenced by the **maximum number of merges** (BPE) or **target vocabulary size** (Unigram LM), the **average length** of the surface words, and **morphological richness**.
- **Isolating, Latin-script** languages typically have fertility **close to 1** for common vocabulary words.
- When a word is chopped into many small, less meaningful parts, the model has to **work harder** to reassemble its meaning.
- **No single** tokenization scheme dominates across **all** languages and scripts.
- Commercial LLM APIs price **by token count**: high-fertility languages **pay more** for the same content.
- Fixed context windows fill up **faster** for high-fertility languages.

> [!warning] Traps
> - *"Fertility depends only on the vocabulary size."* False: also word length and morphological richness.
> - *"Isolating, Latin-script languages typically have high fertility."* False: close to 1 for common words.
> - *"A well-chosen tokenization scheme dominates across all languages and scripts."* False: none does.
> - *"High-fertility languages fill fixed context windows more slowly."* False: faster.

## Practice questions

**1.** Which of the following statements about Byte Pair Encoding (BPE) are true? (multiple answers possible)
- [A] At inference time, BPE applies the learned merge operations in the order they were learned.
- [B] BPE training stops when a maximum number of merge operations has been reached.
- [C] BPE can return an n-best list of segmentations for a word.
- [D] The benefits of BPE for translation are restricted to low-resource language pairs.

> [!example]- Answer
> **A and B.** C: BPE is deterministic, one segmentation per word. D: the benefits apply to both high- and low-resource pairs.

**2.** Which of the following statements about the translation results comparing word-level, character-bigram and BPE systems are true? (multiple answers possible)
- [A] The character-bigram model ties joint BPE for the best single-model score.
- [B] Joint BPE gives the best ensemble score.
- [C] A single model with separate BPE scores below the best word-level single model.
- [D] A single joint-BPE model outperforms the syntax-based system.

> [!example]- Answer
> **A and C.** B: the character-bigram ensemble is best. D: no single neural model beats the syntax-based system; only ensembles of subword or character-bigram models do.

**3.** Which of the following statements about the SentencePiece Unigram LM are true? (multiple answers possible)
- [A] It starts from a large vocabulary and shrinks it to the desired size.
- [B] Its initial vocabulary contains all substrings occurring more than once, including ones that cross word boundaries.
- [C] In the algorithm as presented, each round removes the tokens with the highest loss $L_t$.
- [D] Like BPE, it is greedy: once a sequence is merged, it stays merged.

> [!example]- Answer
> **A and C.** B: the substrings do not cross words. C follows the slide (Kudo's paper removes the lowest). D: Unigram LM optimises the best global segmentation; greediness is BPE's limitation.

**4.** Which of the following statements about finding the best segmentation are true? (multiple answers possible)
- [A] A string of $n$ characters has $2^{n-1}$ possible segmentations.
- [B] Bottom-up dynamic programming has a better asymptotic run time than top-down dynamic programming with memoization.
- [C] Dynamic programming spends extra memory to save computation time.
- [D] The Viterbi segmentation directly returns the n best segmentations.

> [!example]- Answer
> **A and C.** B: both strategies have the same asymptotic run time. D: Viterbi returns only the most probable segmentation; Yen's or Eppstein's algorithms give n-best.

**5.** Which of the following statements about SentencePiece are true? (multiple answers possible)
- [A] Standard BPE implementations assume that word boundaries have already been detected.
- [B] SentencePiece treats its input as a raw stream of Unicode characters, including spaces.
- [C] SentencePiece removes spaces and restores them with language-specific rules.
- [D] SentencePiece implements only the Unigram LM.

> [!example]- Answer
> **A and B.** C: the space is escaped as a visible symbol, so desegmentation is lossless. D: BPE is the other part of SentencePiece.

**6.** Bostrom and Durrett (2020) compare BPE and Unigram LM. Which of the following statements are true? (multiple answers possible)
- [A] Unigram LM produces longer segments on average in both English and Japanese.
- [B] Against gold-standard segmentations, BPE has the higher precision in English.
- [C] The downstream advantage of Unigram LM is larger on Japanese than on English tasks.
- [D] In English, BPE produces more suffix tokens such as *ing* and *ed*.

> [!example]- Answer
> **A and C.** B: Unigram LM wins precision, recall and F1 in both languages. D: suffixes and punctuation are more frequent in Unigram LM; BPE over-produces word-initial capitals.

**7.** Which of the following statements about out-of-vocabulary items and fertility are true? (multiple answers possible)
- [A] BPE and Unigram LM guarantee a zero OOV rate.
- [B] Byte-level BPE has a base alphabet fixed at 256 symbols.
- [C] Under byte-level BPE, non-ASCII scripts end up with shorter subsegments.
- [D] Fixed context windows fill up more slowly for high-fertility languages.

> [!example]- Answer
> **B and C.** A: their OOV rates are very low, but typos and unknown characters can still cause OOVs. D: they fill up faster.
