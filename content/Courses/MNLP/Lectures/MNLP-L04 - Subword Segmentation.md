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

> [!abstract] Overview
> Your translation system was trained with a fixed vocabulary. The test sentence is *"The president of France arrived in Kyrgyzstan."* and "Kyrgyzstan" was never in the training data. What does the model see?
>
> With a word-level vocabulary it sees `<UNK>`, and the sentence is now about the president of France arriving somewhere unspecified. With a character-level vocabulary it sees ten symbols with almost no individual meaning, and a sequence long enough to make the whole sentence expensive to encode. Neither is acceptable, and the fix that the entire modern NLP stack is built on is to stop asking which of the two you want and instead learn the unit size from data.
>
> This lecture is the practical counterpart to the morphology lecture that precedes it. Morphology says *what* the meaningful pieces of a word are. Subword segmentation says how to find pieces automatically, in any language, without a linguist, using nothing but frequency counts. The pieces you get are not morphemes, and a large part of this lecture is about measuring exactly how far off they are, and why that gap matters much more in some languages than in others.

## 1. The problem: a fixed vocabulary meets an open-ended language

> [!warning] The bottleneck
> One of the main bottlenecks of training an NMT system is the vocabulary.

The extent to which a word is correctly represented depends on two things:

- **its frequency** in the training data
- **the number of different contexts** in which it occurs

Both are properties of the training corpus, not of the language, so any fixed word list is a bet that the future looks like the past. It never quite does. Language keeps producing new surface forms, and most of them are not new roots: **many word occurrences are the result of word formations**, principally

- **inflections** (`tall`, `taller`, `tallest`)
- **compounding** (`Donaudampfschifffahrtsgesellschaft`)

A word-level vocabulary throws away exactly the information that would let you handle these. It has seen `tall` a thousand times and `tallest` twice, and it treats them as two unrelated integers.

> [!definition] Subword
> Instead of whole words, use **sub-words**, where a subword is either
>
> - a **character n-gram**, or
> - a **morphologically meaningful unit** (language dependent).
>
> The whole point of the data-driven methods in this lecture is that they aim at the first and are judged, unfairly but usefully, against the second.

### Why the two obvious extremes both fail

| Vocabulary | Size | OOV rate | Sequence length | Problem |
|---|---|---|---|---|
| **Word-level** | 300K to 500K and still not enough | high, and unbounded on new text | short (1 token per word) | huge embedding and softmax matrices, no sharing between `tall` and `tallest`, every unseen word collapses to `<UNK>` |
| **Character-level** | tens to a few hundred | essentially zero | very long | each symbol carries almost no meaning, the model must reassemble everything, long sequences are slow and burn context |
| **Subword** | typically 30K to 128K | near zero | moderate | a tunable dial between the two, learned from data |

The subword vocabulary is the compromise, and the size of that vocabulary is the dial. Section 10 is about what that dial costs you differently in different languages.

## 2. Rule-based morphological analyzers (FSTs)

Historically, morphological analyzers were often implemented as **finite-state transducers (FSTs)**, built by hand.

An FST has:

- **states**
- **transitions that "digest" a string and output a string**, written `input:output`

```
              a:x                b:y
    ( q0 ) ----------> ( q1 ) ----------> ( )

    transition q0 -> q1 consumes 'a' and emits 'x'
    transition q1 -> ...  consumes 'b' and emits 'y'
```

The **input is a morpheme and the output is an analysis**, for example `1st person, noun, singular, ...`.

> [!note] The trade-off that killed this approach
> **Strengths:** a very powerful framework, efficient, and able to handle cases and exceptions with great accuracy. If a linguist has encoded Finnish morphology properly, the analyzer is right about Finnish morphology.
>
> **Weaknesses:**
> - **extremely laborious to produce**, one expert-years-per-language cost
> - **totally language dependent**, nothing transfers to the next language
> - **in case of ambiguities, no preference** is expressed, you get a set of analyses with no ranking (probabilistic FSTs do exist, but the plain formalism gives you nothing)

For a course about multilinguality, the second point is fatal on its own. A method that costs one expert per language does not scale to a hundred languages.

## 3. Data-driven morphological analyzers

Instead of writing rules (transitions, in the case of FSTs) by hand, **learn morphological regularities automatically from data**.

The first approach was **Morfessor** (Creutz and Lagus, 2002), based on the **Minimum Description Length (MDL)** principle. MDL asks two questions at once:

1. **how much does it "cost" the model to generate the data?**
2. **how much does the model itself "cost"?**

> [!intuition] Why MDL is the right shape for this problem
> A vocabulary of whole words makes the data cheap to encode (one symbol per word) but the model enormous. A vocabulary of single characters makes the model tiny but the data expensive (many symbols per word). MDL says: minimise the sum. The optimum sits where reusable pieces exist, which is roughly where morphemes are, because morphemes are precisely the pieces a language reuses.

Current approaches follow similar principles, with three deliberate simplifications:

- they **do not label the data in any way**, so there is no actual morphological analysis, only a split
- they **do not account for any character substitutions or deletions**, so `run`/`ran` and `city`/`cities` get no special treatment
- they are **strictly concatenative**, so a word is exactly the concatenation of its segments and nothing else

In exchange you get two properties that matter enormously in practice:

- **no data annotation or linguistic knowledge required**
- they **can be fitted towards a desired vocabulary size**, which is also computationally handy, because vocabulary size is what determines the size of your embedding and output layers

## 4. Byte Pair Encoding (BPE)

The three approaches named in this lecture, all of which split words into segments **based on frequencies with no linguistic knowledge**:

| Method | Reference | Direction | Key property |
|---|---|---|---|
| **Byte Pair Encoding (BPE)** | Sennrich et al. (2016) | bottom-up (merge) | greedy, deterministic |
| **WordPieces** | Schuster et al. (2012) | bottom-up (merge) | merges by likelihood gain, not raw count |
| **SentencePiece (Unigram LM)** | Kudo (2018) | top-down (prune) | probabilistic, supports multiple segmentations |

### 4.1 The recipe

> [!definition] BPE training
> 1. **Split words into characters** (keep word boundaries) and collect frequencies.
> 2. **Consider all neighbouring occurrences** (adjacent symbol pairs) and collect frequencies.
> 3. **Merge the most frequent sequence** and repeat from (2).
>
> Stop when a maximum number of merge operations has been reached.

The number of merge operations is the only hyperparameter, and it is what controls the final vocabulary size.

**Example segmentation** from the lecture, with `@@` marking a piece that continues into the next token:

```
The president of France arrived in Kyrgyzstan .
The preside@@ nt of France arr@@ ived in K@@ yrg@@ yz@@ stan .
```

Read that carefully, because it is the whole lecture in one line. `The`, `of`, `France` and `in` are frequent enough to survive as whole words. `president` gets cut at `preside|nt`, which is not a morpheme boundary. `arrived` gets cut at `arr|ived`, which is close to one but on the wrong side. `Kyrgyzstan`, which is the word the system had never seen, is cut into four pieces and is therefore representable at all. That is the trade: you give up clean morphology and you buy the ability to write down any word.

### 4.2 The algorithm, exactly as given

This is the reference implementation from the slides, reproduced verbatim (it is Sennrich et al.'s own Algorithm 1).

```python
# Algorithm 1: Learn BPE operations
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
         'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
```

Line by line, the parts that are not obvious:

- `vocab` maps a **space-separated symbol sequence** to a **word-type frequency**. The words are types, not tokens: `'l o w </w>' : 5` means the type `low` occurred 5 times. This is why BPE training is cheap, it runs over the type vocabulary, not the corpus.
- `</w>` is the **end-of-word marker**. Without it, the word-final `est` in `highest` and the word-initial `est` in `establish` would be the same symbol, and after segmentation you could not tell where one word ends and the next begins. Attaching the marker lets a piece be word-final versus word-internal, and is what makes detokenization possible. (It is not what stops merges across words: each word type is a separate key in `vocab`, so no pair ever spans two words.)
- `get_stats` counts every adjacent pair, weighted by the word's frequency. A pair inside a word that occurs 5000 times counts 5000 times.
- `merge_vocab` rewrites `"a b"` as `"ab"` everywhere. The regex `(?<!\S)...(?!\S)` means "not preceded by a non-space and not followed by a non-space", which is how it makes sure it matches whole symbols and not a substring of a longer symbol.
- `max(pairs, key=pairs.get)` returns the **first** key attaining the maximum, and dict iteration order is insertion order. So ties are broken by whichever pair was encountered first while scanning the vocabulary.

> [!warning] The tie-break is implementation-specific
> When several pairs share the top count, this code picks the one it saw first. Other BPE implementations break ties lexicographically, or by the frequency of the constituent symbols. Two tokenizers trained on the same corpus with the same merge count can therefore produce different vocabularies. If you need reproducibility, pin the tokenizer library and its version, and do not assume that "BPE, 32k merges" identifies a tokenizer.

### 4.3 Worked example 1: the lecturer's trace

Corpus: `the tall man is taller than the tallest man`

Word types and frequencies: `the` 2, `tall` 1, `man` 2, `is` 1, `taller` 1, `than` 1, `tallest` 1.

**Step 0, split into characters and keep word boundaries:**

```
t h e </w>   t a l l </w>   m a n </w>   i s </w>   t a l l e r </w>
t h a n </w>   t h e </w>   t a l l e s t </w>   m a n </w>
```

**Step 1, count all adjacent pairs.** The lecture lists:

```
freq(t+h)=3,  freq(h+e)=2,  freq(t+a)=3,  freq(a+l)=3,  freq(l+l)=3,
freq(m+a)=2,  ...,  freq(i+s)=1, ...
```

Check `freq(t+h)=3` by hand: `the` contributes 2 (it occurs twice) and `than` contributes 1. Check `freq(t+a)=3`: `tall`, `taller` and `tallest`, once each. Note that `than` does **not** contribute to `t+a`, because in `than` the `t` is followed by `h`.

The full count, so you can see the ties:

| pair | freq | pair | freq | pair | freq |
|---|---|---|---|---|---|
| t+h | 3 | a+n | 3 | l+e | 2 |
| h+e | 2 | n+`</w>` | 3 | e+r | 1 |
| e+`</w>` | 2 | i+s | 1 | r+`</w>` | 1 |
| t+a | 3 | s+`</w>` | 1 | h+a | 1 |
| a+l | 3 | m+a | 2 | e+s | 1 |
| l+l | 3 | l+`</w>` | 1 | s+t, t+`</w>` | 1, 1 |

Six pairs tie at 3. The code takes the first one encountered, which is `t+h`, because `the` is the first word in the vocabulary.

**Merge operation 1, `t + h -> th`:**

```
th e </w>   t a l l </w>   m a n </w>   i s </w>   t a l l e r </w>
th a n </w>   th e </w>   t a l l e s t </w>   m a n </w>
```

Recount. The lecture lists `freq(th+e)=2, freq(t+a)=3, freq(a+l)=3, ...`. Max is still 3, and `t+a` is now first among the tied pairs.

**Merge operation 2, `t + a -> ta`:**

```
th e </w>   ta l l </w>   m a n </w>   i s </w>   ta l l e r </w>
th a n </w>   th e </w>   ta l l e s t </w>   m a n </w>
```

The lecture stops here. Continuing the same procedure, the next eight merges are:

| # | merge | count | resulting change |
|---|---|---|---|
| 3 | `ta + l -> tal` | 3 | `tall`, `taller`, `tallest` all become `tal l ...` |
| 4 | `tal + l -> tall` | 3 | `tall` is now a single symbol |
| 5 | `a + n -> an` | 3 | `man` (twice), `than` |
| 6 | `an + </w> -> an</w>` | 3 | word-final `an` becomes one symbol |
| 7 | `th + e -> the` | 2 | |
| 8 | `the + </w> -> the</w>` | 2 | `the` is now a single word-final symbol |
| 9 | `m + an</w> -> man</w>` | 2 | `man` is now a single word-final symbol |
| 10 | `tall + e -> talle` | 2 | shared prefix of `taller` and `tallest` |

**Final state of the corpus after 10 merges:**

```
the</w>   tall </w>   man</w>   i s </w>   talle r </w>
th an</w>   the</w>   talle s t </w>   man</w>
```

Two things are worth noticing, because they are exactly the criticisms in section 8.

First, the frequent whole words `the` and `man` were absorbed into single tokens, which is the desired behaviour. Second, the linguistically correct split of `taller` is `tall + er` and of `tallest` is `tall + est`, and BPE produced `talle + r` and `talle + s + t` instead. The greedy merge of `tall + e` was locally the most frequent move and it destroyed the morpheme boundary. BPE cannot undo it, because **once a character sequence has been merged, it stays that way**.

### 4.4 Worked example 2: tracing the code on the slide

The `vocab` in the slide's own code is the canonical example. Running it for its 10 merges prints:

| # | merge | count | vocabulary state after the merge |
|---|---|---|---|
| 0 | (initial) | | `l o w </w>` 5, `l o w e r </w>` 2, `n e w e s t </w>` 6, `w i d e s t </w>` 3 |
| 1 | `('e','s')` | 9 | `n e w es t </w>`, `w i d es t </w>` |
| 2 | `('es','t')` | 9 | `n e w est </w>`, `w i d est </w>` |
| 3 | `('est','</w>')` | 9 | `n e w est</w>`, `w i d est</w>` |
| 4 | `('l','o')` | 7 | `lo w </w>`, `lo w e r </w>` |
| 5 | `('lo','w')` | 7 | `low </w>`, `low e r </w>` |
| 6 | `('n','e')` | 6 | `ne w est</w>` |
| 7 | `('ne','w')` | 6 | `new est</w>` |
| 8 | `('new','est</w>')` | 6 | `newest</w>` |
| 9 | `('low','</w>')` | 5 | `low</w>` |
| 10 | `('w','i')` | 3 | `wi d est</w>` |

Where the counts come from at step 1: `e+s` appears in `newest` (6) and `widest` (3), total 9. So do `s+t` and `t+</w>`. The three-way tie is broken by scan order, and `e+s` is seen first.

Final state: `low</w>` 5, `low e r </w>` 2, `newest</w>` 6, `wi d est</w>` 3.

Notice that `lower` is still `low + e + r + </w>`, four tokens, even though `low` is now a single symbol. Ten merges is not enough to reach `er</w>`. This is the vocabulary-size dial in miniature.

### 4.5 BPE at inference time

> [!definition] Applying a trained BPE model
> **Split the input into characters and apply the merge operations in the same order they were learned.**

The learned model is therefore not a vocabulary, it is an **ordered list of merge rules**. Applying them out of order gives a different segmentation. This is also why BPE is **deterministic**: the same word always produces the same segmentation, regardless of the sentence it sits in.

```pseudo
Algorithm: BPE segmentation of one word (inference)
────────────────────────────────────────────────────
Input:  word w, ordered merge list M = [(a1,b1), (a2,b2), ..., (am,bm)]
Output: sequence of subword symbols

  symbols ← characters of w, with </w> appended
  for r = 1 to m:                      // in the ORDER THEY WERE LEARNED
      (a, b) ← M[r]
      i ← 1
      while i < length(symbols):
          if symbols[i] = a and symbols[i+1] = b:
              symbols[i] ← concat(a, b)   // apply the merge
              delete symbols[i+1]
          else:
              i ← i + 1
  return symbols
```

A practical implementation does not loop over all `m` merges per word. It keeps a rank table `rank[(a,b)] = r` and repeatedly applies the lowest-ranked applicable pair present in the word, which is equivalent and much faster, and it caches the result per word type.

### 4.6 Why it works: the three advantages

> [!tip] BPE has three major advantages
> - **significantly reduces the vocabulary size** (less memory, better speed)
> - **results in better translation quality**
> - **significantly reduces the number of out-of-vocabulary (OOV) items**
>
> These benefits apply to **both high- and low-resource language pairs**.

The evidence given in the lecture (Sennrich et al., English to German, BLEU on single systems and 8-way ensembles):

| name | segmentation | shortlist | vocab source | vocab target | BLEU single | BLEU ens-8 |
|---|---|---|---|---|---|---|
| syntax-based (Sennrich and Haddow, 2015) | | | | | 24.4 | |
| WUnk | - | - | 300 000 | 500 000 | 20.6 | 22.8 |
| WDict | - | - | 300 000 | 500 000 | 22.0 | 24.2 |
| C2-50k | char-bigram | 50 000 | 60 000 | 60 000 | **22.8** | **25.3** |
| BPE-60k | BPE | - | 60 000 | 60 000 | 21.5 | 24.5 |
| BPE-J90k | BPE (joint) | - | 90 000 | 90 000 | **22.8** | 24.7 |

Read the vocabulary columns first. `WUnk` and `WDict` are word-level systems carrying **300k source and 500k target** entries, and they are beaten by subword systems carrying **60k to 90k**. `WUnk` maps unknown words to `<UNK>`, `WDict` uses a back-off dictionary to copy or translate them, which is worth 1.4 BLEU on its own and shows how much damage the unknown-word problem does. `BPE-J90k` is the **joint** variant, where BPE is trained on the union of the two languages so that the same string is segmented identically on both sides, which helps the model copy names across.

> [!note] Where the field actually landed
> **Nowadays, BPE is typically used with 30K to 128K (multilingual) merge operations.** The larger end of that range is specifically a multilingual concession: a vocabulary that must cover many scripts needs more room before any single language gets a decent share of it.

## 5. WordPiece

The lecture name-checks **Schuster et al. (2012), Wordpieces** as one of the three common approaches and does not develop it further. Because it is the tokenizer of BERT and of a large fraction of encoder models you may pick up for the [[MNLP - Mini Project|mini project]], here is the part you need in order to tell it apart from BPE.

WordPiece is structurally identical to BPE, a bottom-up sequence of merges, and differs in **one line: which pair to merge**.

> [!formula] WordPiece merge criterion
> BPE merges the pair with the highest raw count:
> $$(a,b)^{\ast} = \arg\max_{(a,b)} \ \text{count}(ab)$$
>
> WordPiece merges the pair that most increases the likelihood of the training data under a unigram language model, which reduces to:
> $$(a,b)^{\ast} = \arg\max_{(a,b)} \ \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$$
>
> where:
> - $\text{count}(ab)$ is the frequency of the adjacent pair
> - $\text{count}(a)$, $\text{count}(b)$ are the frequencies of the two symbols individually

The denominator is the difference that matters. BPE will happily merge `t + h` because `th` is common, even though `t` and `h` are each common on their own and the pair tells you little. WordPiece divides that out and prefers pairs that are **surprisingly** frequent together, which is a pointwise-mutual-information-style criterion. In practice this pushes it slightly closer to morpheme-like pieces than plain BPE, without the top-down machinery of section 6.

The other visible difference is notation. WordPiece marks continuation rather than word end, so `playing` becomes `play`, `##ing`, where `##` means "this piece attaches to the left". BPE as presented here marks the opposite thing, word end, with `</w>`. The information content is the same, the strings are not, and mixing the two conventions is a classic way to break a pipeline.

## 6. SentencePiece Unigram LM

### 6.1 What is wrong with BPE

> [!warning] BPE's two structural limitations
> BPE is
> - **greedy**: once a character sequence has been merged, it stays that way. There is no lookahead and no undo, so an early merge that was locally optimal can block a globally better segmentation, exactly as `tall + e` did in section 4.3.
> - **deterministic**: a word is always segmented the same way. You cannot sample alternative segmentations, so you cannot use segmentation as a source of training-time regularisation, and you cannot hedge at inference.

**Unigram LM (Kudo, 2018)** is the answer to both. It is part of Google's **SentencePiece** model, and the other part of SentencePiece is BPE, so "SentencePiece" names the toolkit and not the algorithm.

> [!definition] Unigram LM, the four properties
> - **considers all possible segmentations** of a word
> - **optimizes for the best global segmentation**
> - **supports multiple segmentations** (allowing sampling and n-best lists)
> - is **top-down**: it starts with an existing large vocabulary that is shrunk to the desired size

That last property is the mirror image of BPE. BPE starts from characters and merges upward until it has enough symbols. Unigram LM starts from a bloated candidate set and prunes downward until it has few enough.

### 6.2 The model

Unigram LM assumes the subwords in a segmentation are independent:

> [!formula] Probability of a segmentation
> $$P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i) \qquad \text{subject to} \qquad \sum_{x \in V} p(x) = 1$$
>
> where:
> - $\mathbf{x} = (x_1, \dots, x_M)$ is one segmentation of the input string into $M$ subwords
> - $p(x_i)$ is the unigram probability of subword $x_i$
> - $V$ is the subword vocabulary
>
> The best segmentation of a string $X$ is then
> $$\mathbf{x}^{\ast} = \arg\max_{\mathbf{x} \in S(X)} P(\mathbf{x})$$
> where $S(X)$ is the set of all segmentations of $X$.

This is what "optimizes for the best global segmentation" means concretely: BPE picks a segmentation by replaying local decisions, Unigram LM scores whole segmentations and takes the best one.

### 6.3 The vocabulary construction algorithm

Reproduced verbatim from the slides:

```pseudo
Algorithm 2  Unigram LM (Kudo, 2018)
─────────────────────────────────────────────────────────────────
 1: Input: set of strings D, target vocab size k
 2: procedure UnigramLM(D, k)
 3:     V ← all substrings occurring more than
 4:           once in D (not crossing words)
 5:     while |V| > k do                      ▷ Prune tokens
 6:         Fit unigram LM θ to D
 7:         for t ∈ V do                      ▷ Estimate token 'loss'
 8:             L_t ← p_θ(D) − p_θ'(D)
 9:                 where θ' is the LM without token t
10:         end for
11:         Remove min(|V| − k, ⌊α|V|⌋) of the
12:            tokens t with highest L_t from V,
13:            where α ∈ [0,1] is a hyperparameter
14:     end while
15:     Fit final unigram LM θ to D
16:     return V, θ
17: end procedure
```

Reading it:

- **Line 3.** The seed vocabulary is every substring that occurs more than once and does not cross a word boundary. This is enormous, typically millions of candidates, which is why the algorithm is described as top-down: everything plausible is in, and the work is deciding what to throw out.
- **Line 6.** "Fit unigram LM $\theta$ to $D$" is an EM fit. The segmentation of each string is a latent variable, so E-step computes the expected counts of each subword over all segmentations of each string, and M-step re-normalises those counts into $p(x)$. This is why the pruning loop is expensive: it refits the model on every iteration.
- **Line 8.** $L_t$ is how much the corpus likelihood changes if token $t$ is removed from the vocabulary and every string that used it is re-segmented without it. A token that is always replaceable by a cheap split has a tiny $L_t$. A token that nothing else can express has a large one.
- **Lines 11 to 13.** Pruning is gradual, not one-shot: remove at most a fraction $\alpha$ of the current vocabulary per round, so the model can re-fit and the remaining pieces can absorb the work of the deleted ones. Removing exactly down to $k$ in one step would be much worse, because the losses $L_t$ are computed under a model that assumes all the other tokens are still present.

> [!warning] A sign convention in the printed algorithm that does not read correctly
> As written, $L_t = p_\theta(D) - p_{\theta'}(D)$ where $\theta'$ lacks token $t$. Removing a token cannot raise the likelihood, so $L_t \ge 0$, and a **large** $L_t$ means the token was **valuable**. Line 12 nonetheless says to remove the tokens with the **highest** $L_t$, which taken literally would prune the most useful pieces first.
>
> The intent in Kudo (2018), and what SentencePiece actually implements, is the opposite: keep the top-scoring pieces and drop the ones whose removal costs the least likelihood. Read line 12 as "remove the tokens with the smallest $L_t$", or equivalently define $L_t$ with the opposite sign as a per-token score to be maximised. I have reproduced the slide as printed, but do not implement it as printed.
>
> A second detail the printed algorithm leaves out: Kudo (2018) **never prunes single-character tokens**, so every string stays segmentable and pruning cannot create out-of-vocabulary items. Taken literally, lines 11 to 13 could remove a character.

### 6.4 Why you need dynamic programming: the brute-force baseline

> [!formula] Number of segmentations
> Given a string $s$ consisting of $n$ characters, there are
> $$2^{\,n-1}$$
> possible segmentations.
>
> The intuition: there are $n-1$ positions between characters, and each is independently either a cut or not a cut.

The **brute-force approach** is then:

1. Generate all $2^{n-1}$ possible segmentations of $s$
2. Score each segmentation
3. Select the best one

For a 15-character German compound that is 16 384 segmentations, per word, per occurrence. It is not usable.

To score anything at all you need two ingredients:

- **scores for segments**, that is, for units that are not further segmented
- a way to **combine scores for sub-segmentations**

In the unigram model those are $p(x_i)$ and multiplication respectively, and the fact that the combination is a product over independent parts is exactly what makes dynamic programming applicable.

### 6.5 The abstract version: sequence cutting

> [!example] Cut a sequence to maximise value
> Cut a sequence of length $n$ into several segments, such that the overall value is maximised.
>
> Values for segments:
>
> | length $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
> |---|---|---|---|---|---|---|---|---|---|---|
> | price $p_i$ | 1 | 5 | 8 | 9 | 10 | 17 | 17 | 20 | 24 | 30 |

For $n = 4$ there are $2^{4-1} = 8$ ways to cut, and the lecture draws all of them. Reproduced as ASCII, with the price of each piece written above it:

```
      9                 1     8            5      5           8      1
  ┌─┬─┬─┬─┐         ┌─┐┌─┬─┬─┐         ┌─┬─┐┌─┬─┐         ┌─┬─┬─┐┌─┐
  └─┴─┴─┴─┘         └─┘└─┴─┴─┘         └─┴─┘└─┴─┘         └─┴─┴─┘└─┘
     (a) 9            (b) 1+8 = 9        (c) 5+5 = 10       (d) 8+1 = 9

   1   1    5         1    5   1         5    1   1        1   1   1   1
  ┌─┐┌─┐┌─┬─┐        ┌─┐┌─┬─┐┌─┐       ┌─┬─┐┌─┐┌─┐        ┌─┐┌─┐┌─┐┌─┐
  └─┘└─┘└─┴─┘        └─┘└─┴─┘└─┘       └─┴─┘└─┘└─┘        └─┘└─┘└─┘└─┘
   (e) 1+1+5 = 7     (f) 1+5+1 = 7     (g) 5+1+1 = 7      (h) 4×1 = 4
```

> [!formula] Best segmentation
> **(c)**, cut as $2 + 2$, total revenue of **10**.

Note what this shows. The uncut sequence (a) scores 9, and every three-way and four-way cut scores worse still, so neither extreme wins. The optimum is an interior split, and finding it requires comparing whole configurations rather than making one local decision. That is precisely the situation in subword segmentation, with prices replaced by unigram probabilities and addition replaced by multiplication.

### 6.6 Dynamic programming

> [!definition] Dynamic programming
> - **Solve each sub-problem only once.**
> - **Store the solution to each sub-problem.**
>
> Dynamic programming uses additional memory to save computational time, an example of a **time-memory trade-off**.

There are **two dynamic programming strategies**:

- **top-down with memoization**: write the recursion naturally, cache each result the first time it is computed, return the cache afterwards
- **bottom-up**: fill a table in an order that guarantees every sub-result is ready before it is needed, here by increasing segment length

**Both strategies have the same asymptotic run time.** Bottom-up avoids recursion depth limits and has better cache behaviour; top-down only computes the sub-problems it actually needs. The algorithm below is bottom-up.

### 6.7 The DP segmentation algorithm

Reproduced verbatim from the slides:

```pseudo
Input: word w (character string), probabilities p
Output: segmentation s and subsegment costs

n = length(w)
let m[1..n,1..n] and s[1..n,1..n] be new tables
let c[1..n] be the characters of w

for i=1 to n
    m[i,i] = p(c[i])
for l=2 to n
    for i=1 to n-l+1
        j=i+l-1
        m[i,j] = p(c[i : j])
        for k=i to j-1
            t = m[i,k] * m[k+1,j]
            if t > m[i,j]
                m[i,j] = t
                s[i,j] = k

return m and s
```

What each piece is doing:

- `m[i,j]` is the **score of the best segmentation of the substring** `c[i..j]`. `s[i,j]` is the **split point** that achieved it, and is left unset when the best option was to leave `c[i..j]` unsplit.
- The base case `m[i,i] = p(c[i])` says a single character segments only as itself.
- The line `m[i,j] = p(c[i : j])` **initialises each cell with the no-split option**, the probability of the whole substring as one vocabulary item. If the substring is not in $V$ this is zero (or negative infinity in log space), and any split will beat it.
- The inner loop over `k` tries every way of cutting `c[i..j]` into a left part `c[i..k]` and a right part `c[k+1..j]`, and takes the product of their best scores. This is valid **only because the unigram model is independent across segments**: the best way to segment the left half does not depend on what happens in the right half.
- Filling by increasing length `l` guarantees that `m[i,k]` and `m[k+1,j]` are already final when `m[i,j]` is computed, since both cover shorter spans.

**Reconstructing the segmentation** is not in the printed algorithm, so here it is:

```pseudo
Algorithm: Read the segmentation back out of s
────────────────────────────────────────────────
Segments(i, j):
    if s[i,j] is unset:
        output c[i..j]              // this span is one subword
    else:
        k ← s[i,j]
        Segments(i, k)
        Segments(k+1, j)

call Segments(1, n)
```

**Complexity.** The table has $O(n^2)$ cells and each does $O(n)$ work, so this is $O(n^3)$ time and $O(n^2)$ space in the word length $n$. That is fine for words and would not be fine for sentences, which is one reason segmentation is done per word. The span table is more than the unigram model needs: because segments are independent, it is enough to keep one best score per end position, $\text{best}[j] = \max_{i<j} \text{best}[i] \cdot p(c[i{+}1..j])$, which is $O(n^2)$. If you also cap the maximum subword length at $L$ (real vocabularies do, since no piece is 40 characters long), this Viterbi lattice formulation over positions runs in $O(nL)$, which is what production implementations use. SentencePiece runs it over whole sentences, since it does no pre-tokenization (section 6.10).

> [!warning] Multiply probabilities and you will underflow
> The line `t = m[i,k] * m[k+1,j]` multiplies probabilities, and a long word multiplies many of them. In float32 this reaches zero quickly, at which point every candidate ties at zero and the argmax is meaningless. Work in log space: replace `*` with `+`, replace `p(...)` with `log p(...)`, initialise unknown substrings to `-inf`, and keep `>` as the comparison since the log is monotone. Everything else in the algorithm is unchanged.

### 6.8 Worked example: segmenting `cats`

Take $w = $ `cats`, so $n = 4$ and $c = [\texttt{c}, \texttt{a}, \texttt{t}, \texttt{s}]$, with this unigram model (substrings not listed have probability 0):

| segment | $p$ | segment | $p$ | segment | $p$ |
|---|---|---|---|---|---|
| `c` | 0.02 | `ca` | 0.002 | `cat` | 0.003 |
| `a` | 0.04 | `at` | 0.0015 | `ats` | 0.00005 |
| `t` | 0.03 | `ts` | 0.0004 | `cats` | 0.0001 |
| `s` | 0.05 | | | | |

**Base case, `l = 1`:**

```
m[1,1] = p(c) = 0.02
m[2,2] = p(a) = 0.04
m[3,3] = p(t) = 0.03
m[4,4] = p(s) = 0.05
```

**`l = 2`:**

| span | init `p(substring)` | candidate splits | winner |
|---|---|---|---|
| `ca` (1,2) | 0.002 | `k=1`: 0.02 × 0.04 = 0.0008 | **0.002**, no split |
| `at` (2,3) | 0.0015 | `k=2`: 0.04 × 0.03 = 0.0012 | **0.0015**, no split |
| `ts` (3,4) | 0.0004 | `k=3`: 0.03 × 0.05 = 0.0015 | **0.0015**, `s[3,4]=3` |

**`l = 3`:**

| span | init | candidate splits | winner |
|---|---|---|---|
| `cat` (1,3) | 0.003 | `k=1`: 0.02 × 0.0015 = 0.00003; `k=2`: 0.002 × 0.03 = 0.00006 | **0.003**, no split |
| `ats` (2,4) | 0.00005 | `k=2`: 0.04 × 0.0015 = 0.00006; `k=3`: 0.0015 × 0.05 = 0.000075 | **0.000075**, `s[2,4]=3` |

**`l = 4`:**

| span | init | candidate splits | winner |
|---|---|---|---|
| `cats` (1,4) | 0.0001 | `k=1`: 0.02 × 0.000075 = 0.0000015; `k=2`: 0.002 × 0.0015 = 0.000003; `k=3`: 0.003 × 0.05 = **0.00015** | **0.00015**, `s[1,4]=3` |

**The filled table** (rows are $i$, columns are $j$, blank means $j < i$):

```
        j=1        j=2        j=3        j=4
 i=1   2.0e-2     2.0e-3     3.0e-3     1.5e-4  ← s[1,4]=3
 i=2      ·       4.0e-2     1.5e-3     7.5e-5  ← s[2,4]=3
 i=3      ·          ·       3.0e-2     1.5e-3  ← s[3,4]=3
 i=4      ·          ·          ·       5.0e-2
```

**Read back:** `Segments(1,4)` finds `s[1,4] = 3`, so it splits into `Segments(1,3)` and `Segments(4,4)`. `s[1,3]` is unset, so `c[1..3] = cat` is emitted whole. `c[4..4] = s` is emitted.

> [!formula] Result
> $$\texttt{cats} \ \rightarrow \ \texttt{cat} \; | \; \texttt{s} \qquad P = 0.003 \times 0.05 = 1.5 \times 10^{-4}$$

Two observations. First, the model preferred `cat` + `s` over the single token `cats` (0.00015 against 0.0001), even though `cats` is in the vocabulary, because a rare whole word scores worse than two pieces that are each common. This is the behaviour BPE cannot produce, since BPE would have merged `cats` early and never reconsidered. Second, the correct morphological split fell out without any morphology in the model, purely because the English plural `s` is very frequent.

### 6.9 Beyond the single best segmentation

> [!note] n-best and sampled segmentations
> **The Viterbi segmentation only returns the most probable segmentation.** To get more:
>
> - **Yen's and Eppstein's algorithms** allow for **n-best segmentations** (they are general k-shortest-paths algorithms, and the segmentation lattice is a DAG, so they apply directly)
> - **alternatively, disallow certain sub-solutions**, that is, re-run the search with the winning split forbidden, which is the standard way of enumerating alternatives without a dedicated k-best algorithm

This is the machinery behind **subword regularization**: at training time, sample a segmentation from the n-best list instead of always taking the Viterbi one, so the model sees `taller` as `tall|er` sometimes and `talle|r` other times and cannot overfit to one arbitrary segmentation. It is free data augmentation and it is unavailable to BPE, which has no distribution to sample from. (BPE-dropout later retrofitted a similar trick onto BPE by randomly skipping merges.)

### 6.10 SentencePiece as a preprocessing choice

> [!definition] Raw-stream input
> **Standard BPE implementations assume word boundary detection**, that is, they assume something has already split the input into words, usually a whitespace or language-specific tokenizer.
>
> **SentencePiece instead treats the input as a raw stream of Unicode characters, including spaces**, and **the space character is escaped as a visible symbol**.

This is a smaller-sounding decision with two large consequences. It removes the need for a language-specific pre-tokenizer, which is the last piece of per-language engineering in the pipeline and the thing that makes Japanese and Chinese awkward. And it makes the process lossless, which is section 7.

## 7. Lossless subword segmentation

> [!example] The desegmentation problem
> ```
> Raw text:   Hello world.
> Tokenized:  [Hello] [world] [.]
> ```
>
> **How should this segmentation be desegmented?**
>
> ```
> Hello world .
> Helloworld.
> Hello world.
> ```
>
> All three are consistent with the token sequence. Nothing in `[Hello] [world] [.]` records that there was a space before `world` and not before `.`, so the detokenizer has to guess, and it guesses with hand-written per-language rules about punctuation and spacing.

The problem gets worse, not better, for **languages without clear word boundaries**:

```
Raw text:   こんにちは世界。   (Hello world.)
Tokenized:  [こんにちは] [世界] [。]
```

Japanese does not put spaces between words at all, so a detokenizer that inserts spaces between tokens produces something wrong, and one that never inserts spaces cannot handle English. There is no single rule.

> [!tip] The fix
> **Add a special white space character** (SentencePiece uses `▁`, U+2581, rendered as a low underscore-like mark):
>
> ```
> Raw text:   Hello▁world.
> Tokenized:  [Hello] [▁wor] [ld] [.]
> ```
>
> Now whitespace is part of the token string. Desegmentation becomes `''.join(tokens).replace('▁', ' ')`, with no rules and no language knowledge, and it is exactly invertible. This property is what "lossless" means, and it is why `▁` shows up glued to the front of word-initial pieces in every SentencePiece vocabulary you will ever inspect.

Note also that `▁wor` and `ld` in the example straddle what a human would call the word boundary of `world`. That is legal here precisely because the boundary is encoded in the characters rather than in the tokenizer's assumptions.

## 8. BPE against Unigram LM, empirically

The comparison in the lecture is **Bostrom and Durrett (2020)**, who trained both tokenizers to the same vocabulary size on the same data and then pretrained identical models on each.

### 8.1 Example segmentations

`▁` marks a word-initial piece.

| Original | BPE | Unigram LM |
|---|---|---|
| furiously | `▁fur` `iously` | `▁fur` `ious` `ly` |
| tricycles | `▁t` `ric` `y` `cles` | `▁tri` `cycle` `s` |
| nanotechnology | `▁n` `an` `ote` `chn` `ology` | `▁nano` `technology` |
| corrupted | `▁cor` `rupted` | `▁corrupt` `ed` |
| 1848 and 1852, | `▁184` `8` `▁and` `▁185` `2,` | `▁1848` `▁and` `▁1852` `,` |

And the longer one:

```
Original:    Completely preposterous suggestions
BPE:         ▁Comple t ely  ▁prep ost erous  ▁suggest ions
Unigram LM:  ▁Complete ly   ▁pre post er ous  ▁suggestion s
```

Go through them, because the pattern is consistent. Unigram LM recovers `ious|ly`, `cycle|s`, `corrupt|ed`, `Complete|ly`, `suggestion|s`, `pre|post`, all real morpheme boundaries. BPE produces `t|ely`, `prep|ost|erous`, `n|an|ote|chn|ology`, which are frequency artefacts. The `nanotechnology` case is the clearest: BPE has burned its early merges on generic high-frequency character clusters and has no symbol for `nano`, while Unigram LM, which selected its vocabulary by usefulness rather than by merge history, does.

The numeric example is a practical warning. BPE splits `1848` as `184` + `8` and `1852` as `185` + `2`, so the model has to reconstruct the year from a shared meaningless prefix. Unigram LM keeps both years whole. If your project touches numbers, dates or code, check what your tokenizer does to them.

**Japanese:**

```
Original     磁性は様々に分類がなされている。
BPE          磁 | 性は | 様々 | に分類 | がなされている | 。
Unigram LM   磁 | 性 | は | 様々 | に | 分類 | がなされている | 。
Gloss        magnetism | (top.) | various ways | in | classification | is done | .
Translation  Magnetism is classified in various ways.
```

BPE glues the topic particle は onto 性, and glues the particle に onto 分類. Those are grammatical function words being absorbed into content words, which is worse than an arbitrary split inside a word, because it destroys a unit the model needs as a unit. Unigram LM separates them correctly.

### 8.2 Token length distributions

**English**, both vocabularies at the same size, counting how many vocabulary entries have each length:

```
 tokens
  4000 ┤█                                          █ BPE
       ┤█                                          ▓ Unigram LM (both overlap in teal)
  3000 ┤█        █
       ┤█     ▓  █  █
  2000 ┤█     ▓  ▓  ▓  ▓  ▓
       ┤█  ▓  ▓  ▓  ▓  ▓  ▓  ▓
  1000 ┤█  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓
       ┤█  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓
     0 ┼──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15   token length
```

The readable numbers, approximated off the plot:

| length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BPE | 4250 | 1280 | 2700 | 3050 | 2520 | 1950 | 1520 | 1080 | 700 | 450 | 260 | tail |
| Unigram LM | 4250 | 1280 | 2220 | 2410 | 2110 | 1950 | 1800 | 1420 | 1030 | 700 | 380 | longer tail |

They are identical at length 1, since both must keep the full character alphabet, and (as read off the plot) at length 2. BPE bulges at lengths 3 to 5 and Unigram LM bulges from 7 upward.

> [!formula] Lecture conclusion (English)
> - **The unigram LM produces longer segments on average.**
> - **The unigram LM uses its vocabulary space more effectively, with more tokens of moderate frequency.**

The second bullet comes from the companion plot, token frequency (log scale, $10^2$ to $10^7$) against token frequency rank (1 to 20 000):

```
 freq
 10^7 ┤
 10^6 ┤▚
 10^5 ┤ ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
 10^4 ┤                  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▚▄        ← BPE falls off here (~rank 15.5k)
      ┤                                      ▀▀▀▀▚    ← Unigram LM holds ~10^4 to ~rank 17k
 10^3 ┤                                           ▚
 10^2 ┤                                            ▚▄▄
      ┼───────────────────────────────────────────────
       1                                         20000   rank
```

The two curves are indistinguishable for the first ~15 000 ranks. Then BPE's curve collapses while Unigram LM's holds a plateau near $10^4$ for another couple of thousand ranks before falling. In plain terms: **BPE spends the bottom of its vocabulary on near-useless tokens, and Unigram LM does not.** Vocabulary slots are paid for in embedding parameters whether they are used or not, so this is wasted capacity.

**Japanese**, same experiment:

| length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| BPE | 4350 | 7900 | 4150 | 1950 | 1030 | 440 | 200 | 100 |
| Unigram LM | 4060 | 7280 | 3830 | 2470 | 1370 | 600 | 250 | 100 |

Same shape of conclusion, on a much shorter axis (the distribution peaks at length 2 and dies by 9, versus 15 for English, because a Japanese character carries far more information than a Latin one).

> [!formula] Lecture conclusion (Japanese)
> **The unigram LM produces longer segments on average.**

### 8.3 What each method over-produces (English)

| More frequent in **BPE** | More frequent in **Unigram LM** |
|---|---|
| `▁H` `▁L` `▁M` `▁T` `▁B` | `s` `.` `,` `ed` `d` |
| `▁P` `▁C` `▁K` `▁D` `▁R` | `ing` `e` `ly` `t` `▁a` |

This table is the whole argument in twenty tokens. BPE's characteristic extra tokens are **single capital letters at the start of a word**, the debris left when a capitalised word cannot be merged into anything (`▁K` in `K@@ yrg@@ yz@@ stan`, from section 4.1). Unigram LM's characteristic extra tokens are **English suffixes**: `s`, `ed`, `ing`, `ly`. One method is producing junk and the other is producing morphology.

| | BPE | Unigram LM |
|---|---|---|
| Tokens per word **type** | 4.721 | 4.633 |
| Tokens per word | 1.343 | 1.318 |

Both numbers favour Unigram LM, but only slightly. Note that this is the fertility measure of section 10 applied to English: 1.318 versus 1.343 tokens per word is a ~2% difference in sequence length, which is real but is not where the interesting variation lives. Section 10 is.

### 8.4 Segmentation quality against gold standards

Segmentations are scored against a linguistic reference: **CELEX2** for English morphology, **MeCab** for Japanese word segmentation.

| Method | English (w.r.t. CELEX2) Precision | Recall | F1 | Japanese (w.r.t. MeCab) Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| BPE | 38.6% | 12.9% | 19.3% | 78.6% | 69.5% | 73.8% |
| Uni. LM | **62.2%** | **20.1%** | **30.3%** | **82.2%** | **72.8%** | **77.2%** |

> [!warning] Do not read these as "tokenizers are bad at English"
> The English numbers are low in absolute terms for both methods, and the recall numbers especially so: BPE recovers 12.9% of CELEX2's morpheme boundaries and Unigram LM 20.1%. That is expected, because neither method is trying to find morphemes, and because most English word tokens are frequent enough to stay whole, so the tokenizer predicts no boundary at all where CELEX2 has one (*walked* as one token misses *walk|ed*).
>
> What is meaningful is the **ratio**. Unigram LM is roughly 1.6x BPE on English F1 and about 1.05x on Japanese F1. The gap between the two methods is far larger in English than in Japanese. The two columns do not measure the same thing, though: the English reference is morpheme boundaries inside space-delimited words, the Japanese reference is word boundaries in unspaced text. And the downstream gap (section 8.5) runs the other way, about one point in English against 12.3 on Japanese TyDi QA, so a larger gold-standard F1 gap does not predict a larger task gap.

### 8.5 Does it matter downstream?

Identical pretraining, identical model, only the tokenizer differs:

| Model | SQuAD 1.1 (dev) EM | SQuAD F1 | MNLI (dev) Acc. (m) | Acc. (mm) | CoNLL NER Dev F1 | Test F1 | Japanese TyDi QA (dev) EM | F1 |
|---|---|---|---|---|---|---|---|---|
| Ours, BPE | 80.6 ± .2 | 88.2 ± .1 | 81.4 ± .3 | 82.4 ± .3 | 94.0 ± .1 | 90.2 ± .0 | 41.4 ± 0.6 | 42.1 ± 0.6 |
| Ours, Uni. LM | **81.8 ± .2** | **89.3 ± .1** | **82.8 ± .2** | **82.9 ± .2** | **94.3 ± .1** | **90.4 ± .1** | **53.7 ± 1.3** | **54.4 ± 1.2** |
| BERT_BASE | 80.5 | 88.5 | 84.6 | 83.4 | 96.4 | 92.4 | - | - |

On the English tasks Unigram LM wins consistently but by roughly one point, which is a real effect and a modest one. On **Japanese TyDi QA it wins by 12.3 EM and 12.3 F1**, which is not modest, and which is the number to remember from this lecture. The `BERT_BASE` row is a reference point trained on different data and is not a controlled comparison, which is why it wins on MNLI and NER; it is there to show the authors' models are in a sensible range.

> [!tip] The multilingual moral
> The tokenizer is a hyperparameter that is invisible in English benchmarks and dominant outside them. A paper that reports a one-point English gain from a modelling change, and never mentions its tokenizer, may be reporting a tokenizer effect. If you evaluate multilingually, you have to control for segmentation.

## 9. Out-of-vocabulary items

> [!definition] The motivation, restated
> **Handling low-frequency words (including zero-frequency, that is, OOV) is one of the main motivations for subword segmentation.**

**BPE and Unigram have very low OOV rates**, but not zero:

- **typos can still sometimes cause OOVs**
- **unknown characters will result in OOVs**

The second is the real one. A character-based vocabulary contains the characters it saw during training. Unicode has far more than that, and a document containing an emoji, a rare CJK character, or a script the tokenizer never saw produces a genuine unknown symbol with nothing to fall back to.

### Byte-level BPE

> [!definition] Byte-level BPE: guaranteeing zero OOV rates
> - **Operate on raw UTF-8 bytes instead of Unicode characters.**
> - **The base alphabet is fixed at 256 symbols**, so any input, in any script, can always be represented.
> - **Even unseen scripts can be encoded**, byte by byte, and then merged where patterns emerge.

The argument is airtight and is why GPT-2 and most decoder-only models since use it: every file on earth is a byte sequence, there are 256 bytes, put all of them in the vocabulary and the OOV problem is definitionally solved. The base alphabet is also *smaller* than a Unicode-character alphabet, which frees vocabulary slots for merges.

> [!warning] The bill arrives in the multilingual case
> **Non-Latin, non-ASCII scripts cost more bytes per character.** UTF-8 gives ASCII one byte per character, Latin-with-diacritics and Cyrillic and Greek two, most CJK and Devanagari and Thai three, and many emoji four.
>
> Therefore **additional merge operations are required just to get from bytes back to characters, resulting in shorter subsegments** for those scripts. A merge budget that buys English a vocabulary of words and morphemes buys Hindi a vocabulary that is still partly reassembling individual characters. The same nominal vocabulary size delivers a much weaker tokenizer to some languages than to others, and the mechanism is UTF-8's encoding length, not anything about the language.

## 10. Fertility, and what segmentation costs in a given language

> [!definition] Fertility
> **Fertility measures the average number of segments a word is split into.**
> $$\text{fertility} = \frac{\text{number of subword tokens}}{\text{number of words}}$$
>
> Fertility of 1.0 means every word is a single token. Higher is worse for the model, in the senses listed below.

This is exactly the "Tokens per word" row of section 8.3, where English scored 1.343 for BPE and 1.318 for Unigram LM.

**Fertility for a language is influenced by a number of factors:**

- the **maximum number of merge operations** (BPE) or **target vocabulary size** (Unigram LM), the one factor you control
- the **average lengths of the surface words** in a language
- **morphological richness**, that is, how many combinations of affixes are possible

**Isolating, Latin-script languages typically see fertility close to 1 for common vocabulary words.** That is the best case, and English sits in it. An agglutinative language such as Finnish, Turkish or Hungarian, where one surface word encodes what English spreads over five, has vastly more distinct word forms, each individually rarer, so each gets split further.

Four consequences, in increasing order of how annoying they are:

1. **Modelling.** **When a word is chopped into many small, less meaningful parts, the model has to work harder to reassemble its meaning from parts.** The semantic content that English gets for free in one embedding lookup, Turkish has to compose across six positions of attention.
2. **No universal answer.** **No single tokenization scheme dominates across all languages and scripts.** A vocabulary tuned for one language family is a compromise for the others, and a multilingual vocabulary is a compromise for all of them. This is why multilingual models push to the 128K end of the merge range.
3. **Money.** **Commercial LLM APIs price by token count**, so **high-fertility languages pay more for the same amount of content.** Identical text in English and in Telugu is not the same number of tokens, and the speaker of the lower-resourced language is charged more for it.
4. **Capacity.** **Fixed context windows fill up faster for high-fertility languages.** A 128k-token context is a different amount of actual document depending on what language the document is in.

> [!tip] The connection back to the course
> Points 3 and 4 are the sharpest concrete statement in this course so far of what "English-centric NLP" costs. It is not only that models are worse in other languages. It is that the pricing, the context budget and the per-token compute are all quietly worse too, and all of it traces back to a merge table learned from a corpus that was mostly English.

## Key Takeaways

> [!tip] Exam Focus
> Six things to be able to state cold.
>
> 1. **Why subwords at all.** Vocabulary is the bottleneck in NMT. Word-level gives a 300k to 500k vocabulary that still has OOVs; character-level gives near-zero OOVs but meaningless units and long sequences. Subwords are the tunable middle, at 30K to 128K merges.
> 2. **BPE training, in three steps.** Split into characters keeping word boundaries and count; count all adjacent pairs; merge the most frequent pair and repeat, until the merge budget is spent. **BPE inference:** split into characters and replay the merges **in the order they were learned**.
> 3. **The two properties of BPE that motivate everything after it:** greedy (a merge is never undone) and deterministic (one segmentation per word, no sampling).
> 4. **Unigram LM.** $P(\mathbf{x}) = \prod_i p(x_i)$, all segmentations considered, best one chosen globally, multiple segmentations available, **top-down** (start large, prune with EM refits down to $k$). Contrast the direction with BPE, which is bottom-up.
> 5. **Why DP.** There are $2^{n-1}$ segmentations of an $n$-character string, so brute force is out. The bottom-up table `m[i,j]` = best score for span $i..j$, initialised with the unsplit probability, then maximised over split points $k$, gives $O(n^3)$.
> 6. **Fertility** = subword tokens per word, and the four consequences: harder composition for the model, no scheme dominating all languages, API pricing penalties, faster context exhaustion.
>
> The most likely comparison question is **BPE against Unigram LM**, and the answer has three layers: mechanism (greedy merge from below versus likelihood pruning from above), segmentation quality (English F1 19.3 against 30.3 versus CELEX2), and downstream effect (about one point on English tasks, **12.3 EM on Japanese TyDi QA**).

> [!warning] The distinction to get right
> **SentencePiece is not an algorithm.** It is Google's toolkit, and it implements both BPE and Unigram LM. What is specific to SentencePiece is the *input handling*: raw Unicode stream including spaces, whitespace escaped as `▁`, no language-specific pre-tokenizer, lossless desegmentation. "We used SentencePiece" does not tell you which algorithm was used, and you should ask.

> [!note] For the mini project
> Practical decisions this lecture should drive in [[MNLP - Mini Project|the mini project]], where [[Tokenization]] choices will be load-bearing whether or not they are the stated topic:
>
> - **Report the tokenizer, the algorithm and the vocabulary size** as experimental settings, not as an afterthought. "SentencePiece" alone is under-specified.
> - **If you compare across languages, measure fertility per language** and report it next to your results. A difference in sequence length is a confound for anything that depends on sequence length, which is most things.
> - **If you compare tokenizers, hold the vocabulary size fixed.** Comparing 32k BPE against 64k Unigram LM measures the vocabulary size, not the algorithm.
> - **Look at your actual segmentations** before you look at your metrics. Ten minutes printing the tokenization of fifty real inputs catches the numbers-split-into-digits and particles-glued-to-nouns failures that would otherwise show up as an unexplained score.
> - **A tokenizer comparison is a legitimate and cheap project.** It needs no model training beyond fine-tuning, the baselines are published, and the multilingual angle is built in.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L03 - Morphology and Word Formation]] (the morphology lecture directly precedes this one and supplies the notion of morpheme used throughout)
- **Related concept:** [[Tokenization]]
- **Project:** [[MNLP - Mini Project]]
- **Source:** `MNLP_subword.pdf`, Christof Monz, MNLP: Subword Segmentation, slides 1 to 24
- **Primary references:** Sennrich, Haddow and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* · Schuster and Nakajima (2012), *Japanese and Korean Voice Search* · Kudo (2018), *Subword Regularization* · Kudo and Richardson (2018), *SentencePiece* · Creutz and Lagus (2002), *Unsupervised Discovery of Morphemes* (the method later released as Morfessor) · Bostrom and Durrett (2020), *Byte Pair Encoding is Suboptimal for Language Model Pretraining*
