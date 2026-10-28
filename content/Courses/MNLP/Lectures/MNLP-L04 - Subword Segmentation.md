---
type: lecture
course: 5204MNLP6Y
week: 2
lecture: 4
status: complete
topics:
  - The vocabulary bottleneck
  - Out-of-vocabulary items
  - Rule-based morphological analyzers (FSTs)
  - Morfessor and Minimum Description Length
  - Byte Pair Encoding
  - WordPiece
  - SentencePiece Unigram LM
  - Viterbi segmentation by dynamic programming
  - Lossless segmentation
  - Byte-level BPE
  - Fertility and cross-lingual segmentation quality
---

# MNLP-L04: Subword Segmentation

> [!abstract] In short
> How can a model with a fixed vocabulary handle words it has never seen? Whole words turn every unseen word into `<UNK>`. Single characters carry almost no meaning and make sequences long. The answer is a vocabulary of word pieces learned from frequency counts, at a size you choose.
>
> **BPE** builds it bottom-up by merging the most frequent adjacent pair. **Unigram LM** prunes a huge candidate list top-down and scores every segmentation of a word, so it can return the best one or alternatives. Its pieces are more morpheme-like, especially in English, but its downstream gain is small in English and large in Japanese. **SentencePiece** is a toolkit that runs either one on raw text. **Byte-level BPE** starts from the 256 bytes, so nothing is out of vocabulary.
>
> The sample exam has no question on this lecture. Expect statements on which method does what: merging up or pruning down, greedy or not, deterministic or not, and which languages pay for each choice.

## The problem: a fixed vocabulary meets an open-ended language

The vocabulary is one of the main bottlenecks in training a neural machine translation (NMT) system. How well a word is represented depends on its **frequency** in the training data and the **number of different contexts** it occurs in, so rare words get poor embeddings and unseen words none.

Many word occurrences come from **word formation**: **inflection** (`tall`, `taller`, `tallest`) and **compounding** (German `Donaudampfschifffahrtsgesellschaft`, "Danube steamship company"). A word-level vocabulary treats `tall` and `tallest` as two unrelated entries.

The two obvious vocabularies both fail:

- **Whole words.** Systems carried 300,000 to 500,000 entries and still met unknown words, each becoming `<UNK>` (the unknown-word symbol): with "Kyrgyzstan" unseen, "The president of France arrived in Kyrgyzstan" becomes an arrival somewhere unspecified. The matrices are huge, and related forms share nothing.
- **Single characters.** Almost nothing is unknown, but each symbol carries almost no meaning and sequences become long and slow.

> [!definition] Subword
> Instead of whole words, use **sub-words**, where a subword is either
> - a **character n-gram** (a sequence of $n$ characters), or
> - a **morphologically meaningful unit** (language dependent).

A subword vocabulary sits between the extremes: few unknown words and moderate sequence length, at a vocabulary size you set.

## Doing it by hand: finite-state transducers

Historically, morphological analysers were hand-built **finite-state transducers (FSTs)**: states, with transitions that "digest" input and output a string (`a:x` reads `a`, writes `x`). The input is a morpheme (the smallest unit with meaning, see [[MNLP-L03 - Morphology and Word Formation|L03]]) and the output an analysis such as *1st person, noun, singular*.

- **Strength:** a very powerful, efficient framework that handles cases and exceptions with great accuracy.
- **Weaknesses:** **extremely laborious** to produce, **totally language dependent**, and given an ambiguous word they express **no preference** among its analyses. Probabilistic FSTs, which do rank the analyses, exist.

For multilingual NLP the language dependence is fatal: one expert-built analyser per language does not scale.

## Learning segments from data: Morfessor and description length

The alternative is to learn morphological regularities from data. The first approach was **Morfessor** (Creutz and Lagus, 2002), based on the **Minimum Description Length (MDL)** principle, which adds up two costs:

- how much does it cost the model to **generate the data**?
- how much does the **model itself** cost?

Whole words make the data cheap but the model enormous; characters make the model tiny but the data expensive. Minimising the sum favours pieces reused across many words, roughly the morphemes.

Current subword methods share Morfessor's aim, reusable pieces learned from unannotated data, with three simplifications:

- they **do not label** the data in any way: a split, with no actual analysis;
- they **do not account for character substitutions or deletions**, so `city`/`cities` or `run`/`ran` get no special treatment;
- they are **strictly concatenative**: a word is exactly its pieces joined together.

In return they need **no data annotation or linguistic knowledge**, and they **can be fitted to a desired vocabulary size** (which fixes the embedding and output layers). The three common methods, all learned from frequency counts, are **BPE** (Sennrich et al., 2016), **WordPiece** (Schuster and Nakajima, 2012) and **Unigram LM** (Kudo, 2018).

## BPE: merge the most frequent pair, again and again

Byte Pair Encoding (BPE) starts from characters and repeatedly fuses the most frequent pair of adjacent symbols into one new symbol, so frequent words end up whole and rare words split into frequent pieces.

> [!definition] BPE training
> 1. **Split words into characters** (keep word boundaries) and collect frequencies.
> 2. **Consider all neighbouring occurrences** (adjacent symbol pairs) and collect frequencies.
> 3. **Merge the most frequent sequence** and repeat from (2).
>
> Stop when a maximum number of merge operations has been reached.

The number of merges is the only hyperparameter, and it sets the final vocabulary size: the characters plus one new symbol per merge. **BPE is typically used with 30K to 128K merge operations**; multilingual models sit at the top of that range, because a vocabulary shared by many scripts needs more room.

Training runs over **word types with their frequencies**, so it is cheap; a pair's count sums the frequencies of the words it occurs in. Each word is counted on its own, so **no pair ever spans two words**. Each word ends in an end-of-word marker `</w>`, so word-final `est</w>` (*tallest*) differs from word-internal `est` (*establish*), which is what lets pieces be turned back into words.

> [!definition] BPE at inference time
> Split the input into characters and **apply the merge operations in the same order they were learned**.

A trained BPE model is this **ordered list of merges**, so the same word always gets the same pieces, whatever sentence it is in. A segmented sentence as Sennrich et al.'s system prints it:

```
The preside@@ nt of France
arr@@ ived in K@@ yrg@@ yz@@ stan .
```

`@@` marks a piece that continues into the next one. That is only how the output is displayed, `</w>` turned round (a piece without `@@` ends a word); the merges work with `</w>`. `preside|nt` and `arr|ived` are not morpheme boundaries, but the unseen `Kyrgyzstan` is now four known pieces. That is the trade: clean morphology for the ability to write down any word.

### A worked merge example

Corpus: `the tall man is taller than the tallest man`. Word types: `the` 2, `tall` 1, `man` 2, `is` 1, `taller` 1, `than` 1, `tallest` 1, each split into characters plus `</w>` (`t a l l e r </w>`).

**Count the pairs.** Six pairs tie at 3: `t+h` (2 from `the`, 1 from `than`), `t+a` (`tall`, `taller`, `tallest`; in `than` the `t` is followed by `h`), `a+l`, `l+l`, `a+n` (`man` twice, `than`) and `n+</w>`. `h+e`, `e+</w>`, `m+a` and `l+e` occur twice; every other pair once.

**Merge 1.** Breaking ties by the pair met first in the corpus gives `t+h → th`, so `the` becomes `th e </w>` and `than` becomes `th a n </w>`.

**Merge 2.** Recount: `t+a` is now the first pair at 3, so `t+a → ta`. Continuing the same way, merges 3 to 10 are `ta+l`, `tal+l`, `a+n`, `an+</w>` (count 3 each), then `th+e`, `the+</w>`, `m+an</w>` and `tall+e` (count 2 each). The corpus ends as:

```
the</w>  tall </w>  man</w>
i s </w>  talle r </w>  th an</w>
the</w>  talle s t </w>  man</w>
```

`the` and `man` became single symbols, as intended. But morphology says `tall+er` and `tall+est`, and BPE produced `talle+r` and `talle+s+t`: merging `tall+e` was the most frequent move at that point, it buried the morpheme boundary, and BPE never reconsiders a merge.

### What BPE bought in translation

Compared with word-level vocabularies (one entry per whole word), BPE has three major advantages: a much smaller **vocabulary** (less memory, better speed), better **translation quality**, and far fewer **out-of-vocabulary (OOV)** items. These benefits apply to **both** high- and low-resource language pairs.

In Sennrich et al.'s English to German experiments, the best neural systems used subwords (60K to 90K entries) and beat word-level systems (300K source, 500K target entries). Not every subword system won: **separate BPE** (merges per language) fell below the best word-level single model, and **joint BPE** (one set of merges for both languages) beat separate BPE; Sennrich et al.'s reason (outside the course material): a name is then split the same way on both sides, so its pieces map one to one. Character bigrams (a 50K vocabulary) did as well: they **tied joint BPE** as the best single model and gave the **best ensemble**. A **syntax-based statistical system** still beat every single neural model; only ensembles of 8 subword models beat it (the word-level ensemble did not).

## WordPiece: the same merges, a different score

The course only names WordPiece (Schuster and Nakajima, 2012); this section is from outside the course material. WordPiece builds its vocabulary like BPE, bottom-up by merging adjacent pairs, but scores a pair against how common its parts are:

> [!formula] Merge criteria
> $$\text{BPE:}\ \ \arg\max_{(a,b)} \text{count}(ab)$$
> $$\text{WordPiece:}\ \ \arg\max_{(a,b)} \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$$
>
> where:
> - $\text{count}(ab)$ is how often symbol $a$ is directly followed by symbol $b$
> - $\text{count}(a)$, $\text{count}(b)$ are how often each symbol occurs in total

The ratio measures how much more often $a$ and $b$ co-occur than if they were independent. Two individually common symbols, like `t` and `h`, score low even when `th` is common: WordPiece prefers pairs whose parts **rarely occur apart**.

> [!example] First merge on the same corpus
> Symbol counts: `t` 7, `a` 6, `l` 6, `e` 4, `h` 3, `n` 3, `m` 2, `s` 2, `i` 1, `r` 1, `</w>` 9.
>
> | pair | count | score |
> |---|---|---|
> | `t+h` | 3 | 3/21 = 0.14 |
> | `a+n` | 3 | 3/18 = 0.17 |
> | `e+r` | 1 | 1/4 = 0.25 |
> | `i+s` | 1 | 1/2 = 0.50 |
>
> **BPE** merges `t+h` (highest count, first of the six tied at 3). **WordPiece** merges `i+s`: it occurs once, but `i` occurs nowhere else. Its second merge is `e+r` (0.25, tied with `e+s` and met first), so WordPiece reaches the suffix `er` before it builds anything like `tall`.

Like BPE, WordPiece merges greedily and gives each word one segmentation; it is BERT's tokenizer. Its vocabulary marks **continuation** (`play ##ing`), where BPE's training marks the **end of a word** with `</w>`. A new word is segmented by repeatedly taking the **longest matching vocabulary entry** from the left (more in [[MNLP-L06 - Contextual Embeddings|L06]]).

## Unigram LM: prune down, score whole segmentations

BPE has two structural limitations:

- **greedy**: once a character sequence has been merged, it stays that way, so a locally best merge can block a better segmentation, as `tall+e` did;
- **deterministic**: a word is always segmented the same way, so there is nothing to sample and no second-best option.

**Unigram LM** (Kudo, 2018) addresses both. Google's **SentencePiece** toolkit implements it, alongside BPE.

> [!definition] Unigram LM
> - **considers all possible segmentations** of a word
> - **optimizes for the best global segmentation**
> - **supports multiple segmentations** (allowing sampling and n-best lists)
> - is **top-down**: it **starts with an existing large vocabulary that is shrunk to the desired size**

### The model

A unigram language model gives each subword $x$ in vocabulary $V$ a probability $p(x)$ and treats the pieces of a segmentation as independent:

$$P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i), \qquad \sum_{x \in V} p(x) = 1, \qquad \mathbf{x}^{\ast} = \arg\max_{\mathbf{x} \in S(X)} P(\mathbf{x})$$

where:
- $\mathbf{x} = (x_1, \dots, x_M)$ is one segmentation of the string $X$ into $M$ subwords
- $p(x_i)$ is the probability of subword $x_i$
- $S(X)$ is the set of all segmentations of $X$, and $\mathbf{x}^{\ast}$ is the most probable one

This is what "global" means: Unigram LM scores every complete segmentation and takes the best. Each extra piece multiplies in another probability below 1, so the model favours fewer pieces unless the pieces are much more probable, as in the `cats` example below.

### Building the vocabulary by pruning

```pseudo
Unigram LM vocabulary (Kudo 2018),
course version
Input: strings D, target vocab size k
V ← all substrings occurring more
    than once in D (not crossing words)
while |V| > k:
    fit unigram LM θ to D
    for t ∈ V:            // token 'loss'
        L_t ← p_θ(D) − p_θ'(D)
        (θ' = the LM without token t)
    remove min(|V| − k, ⌊α|V|⌋)
        tokens with the highest L_t
        // Kudo's paper: smallest (box)
    (α ∈ [0,1] is a hyperparameter)
fit final unigram LM θ to D
return V, θ
```

> [!warning] Which tokens are removed
> **What $L_t$ is.** $p_\theta(D)$ is the corpus likelihood with token $t$, $p_{\theta'}(D)$ without it, so $L_t$ is the likelihood **lost** by removing $t$: large for a token nothing else can express, small for one a cheap split can replace.
>
> **The exam answer.** The algorithm, presented as "Unigram LM (Kudo, 2018)", removes the tokens with the **highest** $L_t$: the ones whose removal hurts most. Expect "highest $L_t$" to be keyed true whether a statement names Unigram LM, Kudo, SentencePiece or nobody.
>
> **Background, outside the course material.** Kudo's paper and SentencePiece's code do the reverse: they remove the tokens with the **smallest** $L_t$, the ones the corpus misses least, and never remove single characters, so every string stays segmentable. Know it, but do not expect it to be the key.

The start is every substring occurring more than once within a word, typically millions of candidates. Kudo fits the model with EM (expectation maximisation), since each string's segmentation is hidden. Pruning is gradual, at most a fraction $\alpha$ per round with a refit in between, because each $L_t$ assumed all other tokens stay.

## Finding the best segmentation: dynamic programming

Unigram LM needs the most probable segmentation of each word. The **brute-force approach**, generating and scoring every segmentation, is exponential in word length:

> [!formula] Number of segmentations
> A string of $n$ characters has $2^{n-1}$ possible segmentations: each of the $n-1$ gaps between characters is either a cut or not.

Any approach needs two ingredients: **scores for segments** (units that are not split further) and a way to **combine the scores of sub-segmentations**. In the unigram model these are $p(x)$ and multiplication.

**Sequence cutting** is the same problem in the abstract. If pieces of length 1, 2, 3 and 4 are worth 1, 5, 8 and 9, a length-4 sequence is best cut `2+2` (10), beating whole (9), `1+3` (9), three pieces (7) and four (4). Neither extreme wins, and complete cuttings have to be compared, which dynamic programming does without listing them all.

> [!definition] Dynamic programming
> - **Solve each sub-problem only once.**
> - **Store the solution to each sub-problem.**
>
> It uses extra memory to save time, a **time-memory trade-off**. Two strategies: **top-down with memoization** (recursion that caches each result) and **bottom-up** (fill a table from small sub-problems to large). **Both have the same asymptotic run time.**

The segmentation algorithm is bottom-up. Cell $m[i,j]$ holds the score of the best segmentation of the span $c[i..j]$, and $s[i,j]$ where it splits (unset if the span is best kept whole):

```pseudo
Input: word w, probabilities p
Output: split points s, best scores m
n = length(w)
let m[1..n,1..n], s[1..n,1..n] be tables
let c[1..n] be the characters of w
for i = 1 to n
    m[i,i] = p(c[i])
for l = 2 to n          // span length
    for i = 1 to n-l+1
        j = i+l-1
        // option 1: keep span whole
        // (0 if not in V)
        m[i,j] = p(c[i:j])
        // option 2: split after k
        for k = i to j-1
            t = m[i,k] * m[k+1,j]
            if t > m[i,j]
                m[i,j] = t
                s[i,j] = k
return m and s
```

To read the answer out: if $s[1,n]$ is unset, the word is one piece; otherwise split at $k = s[1,n]$ and do the same for both halves.

Combining the best scores of two halves is valid **only because segments are independent** in the unigram model: the best cut of the left half does not depend on the right. Filling by increasing length means both halves are finished first. $O(n^2)$ cells with up to $n$ split points each give $O(n^3)$ time and $O(n^2)$ space. Real implementations add log probabilities, because long products underflow to zero.

> [!example] Segmenting `cats`
> Say $p(\texttt{cats}) = .0001$, $p(\texttt{cat}) = .003$, $p(\texttt{s}) = .05$. The split after `cat` scores $.003 \times .05 = .00015$, beating the whole word (the other splits use rarer pieces and score lower): **`cat | s`**. Two common pieces beat a rare whole word, which BPE cannot do once it has merged `cats`.

### More than one segmentation

**The Viterbi segmentation** (the search above) **only returns the most probable segmentation.** For more, **Yen's and Eppstein's algorithms** (k-best-path algorithms, run on the graph of possible cuts) **give n-best segmentations**, or you can **disallow certain sub-solutions** (forbid the winning split and search again). Because every segmentation has a probability, training can also sample a different segmentation each time a word is seen, so the model does not overfit to one split (Kudo's subword regularization). BPE, being deterministic, has nothing to sample from.

## SentencePiece: segmenting raw text losslessly

**Standard BPE implementations assume word boundary detection**: something has already split the text into words. **SentencePiece instead treats the input as a raw stream of Unicode characters, including spaces**, and **the space character is escaped as a visible symbol**, `▁`.

Why that matters shows when you undo a segmentation:

```
Raw text:   Hello world.
Tokenized:  [Hello] [world] [.]
```

Should this become `Hello world .`, `Helloworld.` or `Hello world.`? The tokens do not record where the spaces were, so a detokenizer guesses with per-language rules. **Languages without clear word boundaries** make it worse: Japanese `[こんにちは] [世界] [。]` must be joined with no spaces.

With the space kept as a symbol:

```
Raw text:   Hello▁world.
Tokenized:  [Hello] [▁wor] [ld] [.]
```

To desegment, join the pieces and turn `▁` back into a space: no rules, no language knowledge, and the original text comes back exactly, which is what **lossless** means. No language-specific pre-tokenizer is needed either. (Outside the course material: by default pieces still do not cross a space; the space is kept as `▁` instead of thrown away.)

> [!warning] SentencePiece is a toolkit
> SentencePiece implements **both** BPE and Unigram LM. What belongs to SentencePiece itself is the input handling: a raw stream including spaces, `▁` for the space, lossless desegmentation. "Trained with SentencePiece" does not say which algorithm was used.

## BPE against Unigram LM in practice

Bostrom and Durrett (2020) trained both on the same data at the same vocabulary size, then pretrained and fine-tuned identical language models on each, so any difference comes from the tokenizer. Their examples (`▁` is SentencePiece's escaped space, so it marks a word-initial piece):

| word | BPE | Unigram LM |
|---|---|---|
| tricycles | `▁t ric y cles` | `▁tri cycle s` |
| nanotechnology | `▁n an ote chn ology` | `▁nano technology` |
| corrupted | `▁cor rupted` | `▁corrupt ed` |
| 1848 | `▁184 8` | `▁1848` |

Unigram LM finds real morpheme boundaries (`corrupt|ed`, `cycle|s`). BPE produces frequency artefacts: it splits `nanotechnology` into generic letter clusters with no `nano`, and splits years into a meaningless prefix plus a digit. In Japanese, BPE glues grammatical particles onto content words (は onto 性, に onto 分類); Unigram LM keeps them apart. For 磁性は様々に分類がなされている。 ("Magnetism is classified in various ways."):

- BPE: 磁 | 性は | 様々 | に分類 | がなされている | 。
- Unigram LM: 磁 | 性 | は | 様々 | に | 分類 | がなされている | 。

How the two vocabularies differ at the same size:

- **The unigram LM produces longer segments on average**, in English and in Japanese. In English, BPE has more entries of length 3 to 5 and Unigram LM more of length 7 and up; in Japanese, BPE has more short entries (lengths 1 to 3) and Unigram LM more from length 4 up.
- **In English, the unigram LM uses its vocabulary space more effectively, with more tokens of moderate frequency.** Ranked by use, the two look alike until the last few thousand entries, where BPE's frequencies collapse: BPE fills the bottom of its vocabulary with near-unused tokens, each still costing an embedding.
- **What each over-produces** (English): BPE more **word-initial single capitals** (`▁H`, `▁K`), leftovers of capitalised words; Unigram LM more **suffixes and punctuation** (`s`, `ed`, `ing`, `ly`, `,`).
- **Unigram LM needs slightly fewer tokens** per word (lower fertility, defined below) and per word type in English.

**Against gold-standard segmentations** (CELEX2 for English morpheme boundaries, MeCab for Japanese word boundaries), Unigram LM beats BPE on precision, recall and F1 in both languages, by far more in English. English recall is low for both, probably because frequent words stay whole (`walked` misses `walk|ed`).

**Downstream**, with only the tokenizer changed, Unigram LM is ahead on every task, but on English only slightly: a bit over a point (1.1 to 1.4) on SQuAD (question answering) and MNLI matched (inference), and only a few tenths of a point on MNLI mismatched and CoNLL named-entity recognition. On **Japanese TyDi QA** (question answering) it is ahead by more than ten points. So the larger gold-standard gap (English) did not predict the larger downstream gap (Japanese).

## The unknown words that remain: byte-level BPE

Handling low-frequency words (including zero-frequency, that is, OOV) is one of the main motivations for subword segmentation. BPE and Unigram have **very low** OOV rates, but not zero:

- **typos** can still sometimes cause OOVs;
- **unknown characters** will result in OOVs: the base vocabulary holds only characters seen in training.

> [!definition] Byte-level BPE: guaranteeing zero OOV rates
> - **operate on raw UTF-8 bytes instead of Unicode characters**
> - **the base alphabet is fixed at 256 symbols**, so any input, in any script, can always be represented
> - **even unseen scripts can be encoded**, byte by byte, and then merged where patterns emerge

Training and inference are BPE's, run on bytes. (Outside the course material: GPT-2 and many later decoder-only models use it.)

The cost lands on non-Latin scripts. UTF-8 uses one byte for an ASCII character and two to four for most others (three for most Chinese, Japanese, Devanagari and Thai). Non-Latin, non-ASCII scripts cost **more bytes per character**, so additional merge operations are required just to get from bytes back to characters, resulting in **shorter subsegments**: each piece covers fewer characters in those scripts. The cause is UTF-8's encoding length, nothing about the language itself.

## Fertility: what segmentation costs a language

> [!definition] Fertility
> **Fertility measures the average number of segments a word is split into:**
> $$\text{fertility} = \frac{\text{number of subword tokens}}{\text{number of words}}$$
> A fertility of 1 means every word is a single token.

Fertility for a language is influenced by:

- the **maximum number of merge operations** (BPE) or **target vocabulary size** (Unigram LM): a bigger vocabulary means fewer splits;
- the **average length of the surface words** in the language;
- **morphological richness**: how many combinations of affixes are possible.

**Isolating, Latin-script languages** typically see fertility **close to 1** for common vocabulary words (an isolating language has about one morpheme per word; see [[MNLP-L03 - Morphology and Word Formation|L03]]). Agglutinative languages such as Finnish or Turkish string many affixes into one word, so they have far more distinct, rarer word forms, each split further.

**No single tokenization scheme dominates across all languages and scripts**, so whichever vocabulary a multilingual model picks, some languages end up with high fertility. High fertility costs those languages in three ways:

- the model **has to work harder** to reassemble a word's meaning from many small, less meaningful parts;
- commercial LLM APIs **price by token count**, so high-fertility languages **pay more** for the same amount of content;
- fixed **context windows fill up faster**.

## The methods side by side

| | builds | picks by | new word | input |
|---|---|---|---|---|
| BPE | merges up | pair count | replays merges | words or raw text† |
| WordPiece* | merges up | count ratio | longest match | split words |
| Unigram LM | prunes down | likelihood loss | best, n-best or sampled | words or raw text† |
| byte BPE | merges up | pair count | replays merges | UTF-8 bytes |

\* outside the course material. † raw text when run through SentencePiece.

**Morfessor** came first: segments chosen by MDL from unannotated words. Its search and new-word segmentation are not in the course; outside it, Morfessor's baseline method searches top-down, recursively splitting words in two, and segments a new word by finding its most probable split under the learned model. **SentencePiece** runs BPE or Unigram LM on raw text with `▁` for the space.

## What to be able to answer

Each has a practice question below except SentencePiece.

- Why word and character vocabularies fail; FSTs; MDL; the simplifications and gains of data-driven methods.
- BPE training and inference, and a merge trace by hand.
- WordPiece's score: why it merges `i+s` where BPE merges `t+h`.
- Unigram LM: the model, pruning, the course's and Kudo's rules.
- DP segmentation and the `cats` example.
- SentencePiece: raw stream, `▁`, lossless, a toolkit.
- Bostrom and Durrett: each direction, and where the gap is large.
- Byte-level BPE and fertility: who pays, and why.

## Exam questions

> [!exam]- Compare BPE and Unigram LM as subword segmentation methods: mechanism, segmentation quality and downstream effect.
> **Key points:** BPE bottom-up, greedy, one segmentation per word; Unigram LM top-down likelihood pruning, global best, n-best and sampling; Unigram LM closer to gold segmentations, far more so in English; downstream gains small in English (a few tenths on some tasks), large on Japanese TyDi QA; SentencePiece is the toolkit that implements both.
>
> 1. **Mechanism.** BPE (Sennrich et al., 2016) is **bottom-up**: start from characters and greedily merge the most frequent adjacent pair until the merge budget is spent. A merge is never undone, and each word has exactly one segmentation. Unigram LM (Kudo, 2018) is **top-down**: start from every substring occurring more than once, fit a unigram LM by EM, and prune tokens until the target size $k$ is reached (as given in MNLP, those with the highest loss $L_t$; Kudo's paper actually removes the smallest). It scores whole segmentations, $P(\mathbf{x}) = \prod_i p(x_i)$, takes the global best by dynamic programming, and can return n-best or sampled segmentations.
> 2. **Segmentation quality** (Bostrom and Durrett, 2020, same data and same vocabulary size): Unigram LM has higher precision, recall and F1 than BPE against CELEX2 (English morphology) and MeCab (Japanese word segmentation), with a far larger gap in English. Unigram LM recovers suffixes (`ed`, `s`); BPE over-produces word-initial single capitals and frequency artefacts such as `n|an|ote|chn|ology`.
> 3. **Vocabulary use:** Unigram LM produces longer segments on average (English and Japanese) and, in English, uses its vocabulary more effectively, with more tokens of moderate frequency; BPE spends the bottom of its vocabulary on near-unused tokens.
> 4. **Downstream** (identical pretraining and model, only the tokenizer differs): on English, Unigram LM is ahead by a bit over a point (under 1.5) on SQuAD and MNLI matched and by only a few tenths on MNLI mismatched and CoNLL NER; on Japanese TyDi QA it is ahead by more than ten points. The larger gold-standard gap (English) does not predict the larger downstream gap (Japanese).
>
> Losing marks: calling SentencePiece the algorithm (it is the toolkit that implements both); reading the low absolute English recall as "tokenizers fail at English" when the point is the comparison between the methods.

> [!exam]- Why does neural machine translation need subword segmentation? Argue from the weaknesses of word-level, character-level and rule-based (FST) alternatives.
> **Key points:** vocabulary bottleneck, representation depends on frequency and contexts; word-level: huge vocabularies, no sharing, unseen words become `<UNK>`; character-level: meaningless units, long sequences; FSTs: laborious, language dependent, no preference among ambiguities; subwords: 30K to 128K, near-zero OOV, learned from frequencies.
>
> - The vocabulary is one of the main bottlenecks of training an NMT system. How well a word is represented depends on its **frequency** and on the **number of different contexts** it occurs in, both properties of the training corpus.
> - Many word occurrences come from **word formation**: inflection (`tall`, `taller`, `tallest`) and compounding (`Donaudampfschifffahrtsgesellschaft`). A word-level vocabulary treats `tall` and `tallest` as unrelated entries.
> - **Word-level:** 300K to 500K entries and still OOVs, huge embedding and output matrices, every unseen word collapses to `<UNK>` (an unseen "Kyrgyzstan" becomes an unspecified place).
> - **Character-level:** near-zero OOV, but each symbol carries almost no meaning and sequences are long and slow.
> - **Hand-built FSTs:** accurate and efficient, but extremely laborious, totally language dependent, and with no preference among ambiguous analyses, so they do not scale to many languages.
> - **Subwords** (typically 30K to 128K merges): learned from frequencies with no annotation, vocabulary size tunable, near-zero OOV, moderate sequence length. The price is that pieces are not clean morphemes.

> [!exam]- Train BPE on the corpus `the tall man is taller than the tallest man` for 10 merges, breaking ties by first occurrence. Give the merges and comment on the result.
> **Key points:** six-way tie at 3, `t+h` met first; `than` does not count towards `t+a`; merge order ending in `tall+e`; final `talle r` and `talle s t`; the greedy merge destroys `tall|er` and is never undone.
>
> 1. Word types: `the` 2, `tall` 1, `man` 2, `is` 1, `taller` 1, `than` 1, `tallest` 1. Split each into characters and append `</w>`.
> 2. Initial counts: six pairs tie at 3: `t+h` (2 from `the`, 1 from `than`), `t+a`, `a+l`, `l+l`, `a+n`, `n+</w>`. `t+h` is met first, so it wins.
> 3. Merges in order: `t+h` (3), `t+a` (3), `ta+l` (3), `tal+l` (3), `a+n` (3), `an+</w>` (3), `th+e` (2), `the+</w>` (2), `m+an</w>` (2), `tall+e` (2).
> 4. Final corpus: `the</w>`, `tall </w>`, `man</w>`, `i s </w>`, `talle r </w>`, `th an</w>`, `the</w>`, `talle s t </w>`, `man</w>`.
> 5. Comment: the frequent words `the` and `man` became single tokens, as desired. But `taller` and `tallest` ended as `talle + r` and `talle + s + t` where the morphological split is `tall + er` and `tall + est`. The `tall + e` merge was locally the most frequent move and destroyed the morpheme boundary, and BPE never undoes a merge.
>
> Losing marks: counting `than` towards `t+a` (in `than` the `t` is followed by `h`).

> [!exam]- On the corpus `the tall man is taller than the tallest man`, which pair does WordPiece merge first, and how does that differ from BPE? Show the computation.
> **Key points:** WordPiece score count(ab) / (count(a) count(b)); symbol counts `t` 7, `h` 3, `i` 1, `s` 2, `e` 4, `r` 1; `t+h` scores 3/21, `i+s` scores 1/2; WordPiece merges `i+s`, BPE merges `t+h`; WordPiece prefers pairs whose parts rarely occur apart.
>
> 1. **BPE** merges the pair with the highest raw count. Six pairs tie at 3, and `t+h` is met first.
> 2. **WordPiece** merges the pair with the highest $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$. Symbol counts: `t` 7, `a` 6, `l` 6, `e` 4, `h` 3, `n` 3, `m` 2, `s` 2, `i` 1, `r` 1, `</w>` 9.
> 3. Scores: `t+h` $3/(7 \cdot 3) \approx 0.14$; `a+n` $3/(6 \cdot 3) \approx 0.17$; `l+l` $3/36 \approx 0.08$; `e+r` $1/(4 \cdot 1) = 0.25$; `i+s` $1/(1 \cdot 2) = 0.5$. WordPiece merges **`i+s`**, then `e+r` (0.25 after the recount, tied with `e+s` and met first).
> 4. Comment: `i+s` occurs once, but `i` occurs nowhere else, so the pair is maximally "surprising". `t` and `h` are each common, so `th` scores low despite its count. WordPiece therefore reaches the suffix `er` before building `tall`; both methods remain greedy, bottom-up and deterministic.
>
> (WordPiece is only named in MNLP; the score is how common implementations compute its likelihood criterion.)

> [!exam]- Segment `cats` by bottom-up DP with p(c)=.02, p(a)=.04, p(t)=.03, p(s)=.05, p(ca)=.002, p(at)=.0015, p(ts)=.0004, p(cat)=.003, p(ats)=.00005, p(cats)=.0001.
> **Key points:** each cell initialised with the unsplit probability; `ts` and `ats` split, `cat` stays whole; $s[1,4]=3$ gives `cat | s`, $P = 1.5\times10^{-4}$; a rare whole word loses to common pieces; valid because segments are independent.
>
> 1. **Length 1:** $m[i,i]$ is the character probability: 0.02, 0.04, 0.03, 0.05.
> 2. **Length 2:** `ca` stays whole, 0.002 (split gives 0.0008); `at` stays whole, 0.0015 (split 0.0012); `ts` splits, $0.03 \times 0.05 = 0.0015 > 0.0004$, so $s[3,4]=3$.
> 3. **Length 3:** `cat` stays whole, 0.003 (splits give 0.00003 and 0.00006); `ats` is best as `at|s`, $0.0015 \times 0.05 = 0.000075$, beating unsplit 0.00005 and `a|ts` 0.00006, so $s[2,4]=3$.
> 4. **Length 4:** unsplit `cats` 0.0001; $k=1$: $0.02 \times 0.000075 = 1.5\times10^{-6}$; $k=2$: $0.002 \times 0.0015 = 3\times10^{-6}$; $k=3$: $0.003 \times 0.05 = 1.5\times10^{-4}$. Winner $k=3$, $s[1,4]=3$.
> 5. **Read back:** $s[1,4]=3$ splits into $c[1..3]$ and $c[4]$; $s[1,3]$ is unset, so the result is `cat | s` with $P = 1.5\times10^{-4}$.
> 6. **Points to make:** the rare whole word `cats` loses to two common pieces, which BPE cannot do once it has merged `cats`. The morphologically correct split appears without any morphology in the model, because plural `s` is frequent. Combining the best scores of two halves is valid only because the unigram model treats segments as independent.

> [!exam]- Explain fertility and why one subword vocabulary serves languages unequally. Include byte-level BPE in your answer.
> **Key points:** fertility = subword tokens / words; drivers: vocabulary size, word length, morphological richness; no single scheme dominates, so some languages get high fertility; three costs: reassembly, API cost, context; byte-level BPE: zero OOV but multi-byte scripts get shorter subsegments, because of UTF-8 encoding length.
>
> - **Fertility** = subword tokens / words, the average number of segments per word. 1 means every word is one token.
> - **Drivers:** merge count or target vocabulary size (the one you control), average surface word length, morphological richness. Isolating Latin-script languages are close to 1 for common words; agglutinative languages (Finnish, Turkish) have many more, individually rarer word forms, which get split further.
> - **Why it is unequal:** no single tokenization scheme dominates across all languages and scripts, so whichever vocabulary a multilingual model picks, some languages get high fertility.
> - **Costs of high fertility:** (1) the model must reassemble meaning from many small parts; (2) commercial APIs price by token, so high-fertility languages pay more for the same content; (3) fixed context windows fill up faster.
> - **Byte-level BPE:** the 256-byte base alphabet guarantees zero OOV, but UTF-8 uses 1 byte for ASCII and 2 to 4 for most other characters (3 for most CJK, Devanagari and Thai). Those scripts spend merges just getting from bytes back to characters and end up with shorter subsegments, each covering fewer characters. The cause is encoding length, nothing about the language itself.

> [!exam]- Walk through Unigram LM vocabulary construction (Kudo, 2018). Which tokens are pruned each round?
> **Key points:** seed with all substrings occurring more than once, not crossing words; fit unigram LM (EM in Kudo's paper); $L_t$ = likelihood lost by removing $t$; prune at most a fraction $\alpha$ per round, refit; MNLP (labelled Kudo 2018): remove the highest $L_t$; background: Kudo's paper removes the smallest and never single characters.
>
> 1. Seed $V$ with all substrings that occur more than once in $D$ and do not cross word boundaries (millions of candidates: the top-down start).
> 2. While $|V| > k$: fit a unigram LM $\theta$ to $D$ (by EM in Kudo's paper, since each string's segmentation is hidden).
> 3. For each token $t$: $L_t = p_\theta(D) - p_{\theta'}(D)$, where $\theta'$ is the LM without $t$. This is the likelihood lost by removing $t$: large for a token nothing else can express.
> 4. Remove $\min(|V|-k, \lfloor\alpha|V|\rfloor)$ tokens, $\alpha \in [0,1]$ a hyperparameter, then refit. Pruning is gradual because each $L_t$ assumes all other tokens are still present.
> 5. Fit the final unigram LM and return $V$ and $\theta$.
>
> **Which tokens go.** As given in MNLP, under the label "Unigram LM (Kudo, 2018)", the tokens with the **highest** $L_t$ are removed, those whose removal hurts most; expect that to be keyed true whether a statement names Unigram LM, Kudo, SentencePiece or nobody. Background, outside the course material: Kudo's paper and SentencePiece's code remove those with the **smallest** $L_t$ and never remove single characters, so every string stays segmentable.

## Flashcards

> [!card]- Which two properties of the training data determine how well a word is represented in NMT?
> Its **frequency** and the **number of different contexts** it occurs in.

> [!card]- Which two word-formation processes produce many of the word occurrences a fixed word vocabulary handles badly?
> **Inflection** (`tall`, `taller`, `tallest`) and **compounding**.

> [!card]- What two kinds of unit can a subword be?
> A **character n-gram**, or a **morphologically meaningful unit** (language dependent).

> [!card]- What are the three weaknesses of hand-built FST morphological analyzers?
> Extremely **laborious** to produce, totally **language dependent**, **no preference** among ambiguous analyses.

> [!card]- Which data-driven morphological segmenter is based on the Minimum Description Length principle?
> **Morfessor** (Creutz and Lagus).

> [!card]- Minimum Description Length minimises the sum of which two costs?
> The cost of the **model itself** plus the cost for the model to **generate the data**.

> [!card]- Which three simplifications do current data-driven subword segmenters make?
> No **labelling** (a split only), no character **substitutions or deletions**, strictly **concatenative**.

> [!card]- What two practical properties do data-driven subword segmenters gain from their simplifications?
> **No annotation or linguistic knowledge** needed, and they can be **fitted to a desired vocabulary size**.

> [!card]- Which pair does BPE merge at each training step?
> The **most frequent** adjacent symbol pair, counted over word types weighted by frequency.

> [!card]- What three major advantages does BPE bring to NMT, compared with word-level vocabularies?
> Smaller **vocabulary**, better **translation quality**, far fewer **OOV items**.

> [!card]- What are the four properties of Unigram LM (Kudo)?
> Considers **all segmentations**, optimises the **best global** one, supports **multiple segmentations**, **top-down**.

> [!card]- How does Unigram LM score a segmentation $\mathbf{x} = (x_1, \dots, x_M)$?
> $P(\mathbf{x}) = \prod_i p(x_i)$: subwords are assumed **independent**.

> [!card]- What does the token loss $L_t$ measure in Unigram LM vocabulary construction?
> How much **corpus likelihood drops** if token $t$ is removed and its strings are re-segmented.

> [!card]- How many segmentations does a string of $n$ characters have?
> $2^{n-1}$: each of the $n-1$ gaps is either a **cut or not**.

> [!card]- What two ingredients are needed to score candidate segmentations?
> **Scores for segments** and a way to **combine** them; in the unigram model, $p(x)$ and multiplication.

> [!card]- What are the two principles of dynamic programming?
> Solve each sub-problem **only once** and **store** its solution: a time-memory trade-off.

> [!card]- In bottom-up DP segmentation, what does the cell $m[i,j]$ hold?
> The score of the **best segmentation of the span** $c[i..j]$.

> [!card]- How can you get more than the single Viterbi segmentation?
> **Yen's or Eppstein's** n-best algorithms, or **disallow** certain sub-solutions and search again.

> [!card]- What do standard BPE implementations assume about their input that SentencePiece does not?
> **Word boundary detection**: the text is already split into words.

> [!card]- What is the desegmentation problem?
> Tokens like `[Hello] [world] [.]` do not record **where spaces were**, so detokenizers guess per language.

> [!card]- Which of BPE and Unigram LM finds real morpheme boundaries in words like `corrupted` and `tricycles`?
> **Unigram LM** (`corrupt|ed`, `cycle|s`); BPE produces **frequency artefacts**.

> [!card]- In English, how do BPE and Unigram LM differ in how they use the bottom of their vocabulary?
> BPE fills it with **near-unused tokens**; Unigram LM keeps **more tokens of moderate frequency**.

> [!card]- What two things can still cause OOVs under BPE or Unigram LM?
> **Typos** (sometimes) and **unknown characters**.

> [!card]- What three factors influence fertility for a language?
> **Merge count or target vocabulary size**, average **surface word length**, **morphological richness**.

> [!card]- What three costs does high fertility bring a language?
> Harder **meaning reassembly**, higher **API cost**, faster **context exhaustion**.

> [!card]- Why do some languages end up with high fertility under a shared multilingual vocabulary?
> **No single tokenization scheme dominates** across all languages and scripts.

> [!card]- True or false: Unigram LM builds its vocabulary bottom-up, merging pairs until it reaches the desired size.
> **False.** It works top-down, pruning a large seed vocabulary to size $k$.

> [!card]- True or false: Applying a trained BPE model replays the learned merges in the order they were learned.
> **True.** The model is an ordered merge list.

> [!card]- True or false: BPE's advantages in vocabulary size, OOV rate and translation quality apply only to low-resource language pairs.
> **False.** They apply to both high- and low-resource pairs.

> [!card]- True or false: Data-driven subword segmenters account for character substitutions such as `city`/`cities`.
> **False.** They do not account for substitutions or deletions; they are strictly concatenative.

> [!card]- True or false: Unigram LM returns only a single segmentation per word.
> **False.** It supports multiple segmentations: n-best lists and sampling.

> [!card]- True or false: In Unigram LM vocabulary construction, the tokens with the highest loss $L_t$ are removed each round.
> **True.** As the algorithm is given for MNLP (labelled Kudo 2018); Kudo's paper and SentencePiece's code remove the smallest.

> [!card]- True or false: DP segmentation under a unigram model is valid because segments are treated as independent.
> **True.** The best cut of one half does not depend on the other.

> [!card]- True or false: In DP segmentation, each cell is initialised with the probability of the unsplit span.
> **True.** Split points then replace it only if they score higher.

> [!card]- True or false: Bottom-up DP has a better asymptotic run time than top-down DP with memoization.
> **False.** Both have the same asymptotic run time.

> [!card]- True or false: SentencePiece treats its input as a raw stream of Unicode characters, including spaces.
> **True.** The space is escaped as `▁`.

> [!card]- True or false: Byte-level BPE gives non-Latin scripts shorter subsegments because those languages are morphologically richer.
> **False.** Because their characters take more UTF-8 bytes.

> [!card]- True or false: At the same vocabulary size, Unigram LM produces longer segments on average than BPE.
> **True.** In both English and Japanese.

> [!card]- Multiple selection: which are true of BPE? (A) It is greedy: a merge is never undone. (B) It can sample alternative segmentations of a word. (C) Its only hyperparameter is the number of merges. (D) It starts from a large vocabulary and prunes it.
> **A and C.** B: BPE is deterministic. D: that is Unigram LM; BPE merges up from characters.

> [!card]- Multiple selection: which are true of SentencePiece? (A) It is an algorithm distinct from BPE and Unigram LM. (B) It escapes the space as a visible symbol `▁`. (C) It requires a language-specific pre-tokenizer. (D) Its segmentation is lossless.
> **B and D.** A: it is a toolkit implementing both. C: raw-stream input removes the pre-tokenizer.

> [!card]- Multiple selection: which are true of byte-level BPE? (A) Its base alphabet has 256 symbols. (B) It cannot encode a script unseen in training. (C) Scripts with more bytes per character get shorter subsegments. (D) It operates on Unicode characters.
> **A and C.** B: any script can be encoded byte by byte. D: it operates on raw UTF-8 bytes.

> [!card]- Multiple selection: which are true of fertility? (A) It is subword tokens divided by words. (B) It depends only on the vocabulary size. (C) High fertility fills fixed context windows faster. (D) Agglutinative languages tend to have lower fertility than isolating Latin-script languages.
> **A and C.** B: word length and morphological richness matter too. D: their many rare word forms get split further.

> [!card]- Multiple selection: which are true of Bostrom and Durrett's BPE against Unigram LM comparison? (A) Unigram LM wins English tasks by a large margin. (B) Unigram LM's advantage is much larger on Japanese TyDi QA. (C) Both tokenizers had the same vocabulary size. (D) BPE recovers more English morpheme boundaries.
> **B and C.** A: the English gains are small, under 1.5 points on every task. D: Unigram LM has higher precision and recall against CELEX2.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L03 - Morphology and Word Formation]] (morphemes, isolating and agglutinative languages)
- **Related concept:** [[Tokenization]]
- **Project:** [[MNLP - Mini Project]]
- **Source:** `subword.pdf`, Christof Monz, MNLP: Subword Segmentation, slides 1 to 24
- **Primary references:** Sennrich, Haddow and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* · Schuster and Nakajima (2012), *Japanese and Korean Voice Search* · Kudo (2018), *Subword Regularization* · Kudo and Richardson (2018), *SentencePiece* · Creutz and Lagus (2002), *Unsupervised Discovery of Morphemes* (the method later released as Morfessor) · Bostrom and Durrett (2020), *Byte Pair Encoding is Suboptimal for Language Model Pretraining*
