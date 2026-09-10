---
type: lecture
course: 5204MNLP6Y
week: 2
lecture: 3
status: complete
topics:
  - Words, vocabularies and lexicons
  - Word embeddings and CBOW
  - Vocabulary short-lists and the <unk> token
  - Word segmentation and word boundaries
  - Forward and backward maximum matching
  - BMES character labeling with HMMs and Viterbi
  - Inflection, derivation and compounding
  - Morphemes, affixes and non-concatenative morphology
  - Morphological typology
  - Why morphology breaks word-level NLP
---

# MNLP-L03: Morphology and Word Formation

> [!abstract] Overview
> Your model has seen *moved* ten thousand times and *hammered* never. Internally *move* is index 2863 and *moved* is index 87542: two integers with nothing in common except that a lookup table happens to hold both. What is the model supposed to do when *hammered* shows up?
>
> That is the whole lecture in one question. Words are not atoms, they have internal structure, and every system that treats a word as an unanalysable index throws that structure away. For English you get away with it, mostly. For Turkish, Finnish, Russian or Inuktitut you do not, because a single lemma there surfaces as thousands of distinct forms and a fixed vocabulary sized for English blows up or silently maps most of the language to `<unk>`.
>
> The lecture runs in two halves that look unrelated and are not. First, where words come from at all: how you find word boundaries when the writing system does not mark them. Second, what is inside a word: morphemes, inflection, derivation, compounding, and the four-way typology of how languages package them. Both halves land on the same conclusion, which is the setup for the subword segmentation lecture that follows: word-level vocabularies are the wrong unit, and something between the character and the word has to carry the load.

## 1. What is a word, and where do vocabularies come from

The lecture starts by assuming the problem away, then spends the rest of the hour taking the assumption back.

Assume words are either separated by white space (after punctuation splitting), or the output of an explicit word segmentation process for languages that do not mark word boundaries at all (Chinese, Thai, and others). That process is called **tokenization** or **word segmentation**, in contrast to **sub-word segmentation**, which is a different problem covered in the next lecture. Given tokenized text, you can collect every word that occurs and call the result your vocabulary.

Three terms that get used interchangeably and should not be:

| Term | What it is |
|---|---|
| **Vocabulary** | The inventory of words in a language, or in a corpus, or known to a person |
| **Lexicon** | A store of the meanings and functions of words, not just the word forms |
| **Dictionary** | A written version of a lexicon, published or digital |

So a vocabulary is a list. A lexicon is a list plus what the entries mean and do. Your NLP system almost always has the first and rarely has the second.

### The design questions nobody answers cleanly

Once you decide to build a vocabulary you immediately face a set of questions that the slides pose without resolving, because they genuinely do not have clean answers:

- Should it contain all words ever seen in a corpus?
- Should it be language-specific?
- Should it contain typos and misspellings?
- Should it contain very rare words?
- Should it contain all variations of a word (*book*, *books*, *booking*, *booked*), or just *book*?
- Should it be limited in size?

> [!intuition] Why this list is the whole lecture in disguise
> Every one of these questions is a question about morphology. *book / books / booking / booked* is one lemma with four inflected and derived surface forms. Whether they get four vocabulary slots or one is exactly the trade-off between vocabulary size and the model's ability to see that they are related. Typos and rare words are the same trade-off from the other side: they are forms you cannot afford a slot for, and if you refuse them a slot you need some way to still represent them.

## 2. Word meanings, and why embeddings replaced hand-built resources

What does a word mean? The classical answer defines meaning through relations between words:

| Relation | Definition | Example |
|---|---|---|
| **Synonyms** | Same meaning | *purchase* :: *acquire* |
| **Hyponyms** | is-a | *car* :: *vehicle* |
| **Meronyms** | part-whole | *wheel* :: *car* |
| **Antonyms** | Opposites | *small* :: *large* |

These are explicit, qualitative relations, and getting them requires hand-crafted resources such as WordNet. Three problems, all fatal at scale:

- **Expensive.** Every relation is entered by a human.
- **Incomplete.** No hand-built resource ever covers a real corpus.
- **Language-specific.** You cannot reuse the English one for Finnish, so the cost multiplies by the number of languages, which for this course is the whole point.

The alternative is to learn relations automatically, and to make them **quantitative** rather than qualitative, so that instead of asserting *car* is-a *vehicle* you can compute $\text{sim}(\textit{car}, \textit{vehicle}) > \text{sim}(\textit{car}, \textit{tree})$.

### Word embeddings

> [!definition] Word embeddings
> **Distributed representations** of words as vectors that are:
> - **low dimensional**: e.g. 512, against a vocabulary size $|V|$ that may be hundreds of thousands
> - **dense**: no zeros, unlike a one-hot vector or a count vector
> - **continuous**: $c_w \in \mathbb{R}^m$
> - **learned by performing a task**, specifically a prediction task, rather than being counted or hand-assigned

The historically important approach is **Word2Vec** (Mikolov et al.), which is really two approaches: **Continuous Bag of Words (CBOW)** and **Skip-Gram**. The lecture develops CBOW and only names Skip-Gram. For completeness, Skip-Gram is the mirror image: instead of predicting the centre word from its context, it predicts each context word from the centre word.

See also [[Word Embeddings]] for the general concept note.

## 3. CBOW

> [!definition] The CBOW task
> Given a position $t$ in a sentence, the $n$ words to its left $\{w_{t-n}, \dots, w_{t-1}\}$ and the $m$ words to its right $\{w_{t+1}, \dots, w_{t+m}\}$, predict the word at position $t$.
>
> `the man X the road`, with `X = ?`

The slide writes the right context as $\{w_{t+1}, \dots, w_{t+n}\}$ while calling its size $m$, which is a slip: since typically $n = m$ it makes no difference in practice.

This looks like $n$-gram language modelling with $n = (\text{LM order} - 1)$ and $m = 0$, and the resemblance is the point of comparison, but it is not the same task. A language model predicts the next word from the left context only. CBOW sees both sides, which is fine because CBOW is not trying to be a language model. It is trying to produce good embeddings, and the prediction task is only a means to that end.

The architecture is a feed-forward neural network with three deliberate choices:

- **Focus on learning the embeddings themselves**, not on prediction quality
- **Simple network**, because depth would spend capacity on the task instead of the representation
- **Bring the embedding/projection layer closer to the output**, so the embedding does more of the work
- Typically $n = m$, with $n \in \{2, 5, 10\}$

### Architecture

Reproducing the slide figure: four context words go into the input layer, their vectors are summed in the projection layer, and that single summed vector predicts the centre word.

```
      Input Layer            Projection Layer       Output Layer

  w_{n-2} ┌─────┐
          │     │──────┐
          └─────┘      │
  w_{n-1} ┌─────┐      │        ┌───────┐
          │     │──────┤        │  Sum  │
          └─────┘      ├──────► │       │ ──────► ┌─────┐
                       │        │       │         │     │  w_n
  w_{n+1} ┌─────┐      │        └───────┘         └─────┘
          │     │──────┤
          └─────┘      │
  w_{n+2} ┌─────┐      │
          │     │──────┘
          └─────┘
```

> [!intuition] Why it is called "bag of words"
> The projection layer *sums* the context vectors. Summation is commutative, so the order of the context words is destroyed: `the man X the road` and `road the X man the` produce the same projection. The context is a bag, not a sequence. That is a real loss of information, and it is the price paid for a very cheap model that trains on enormous corpora. See [[Bag of Words]] for the same idea in retrieval.

## 4. Vocabulary short-lists, and the `<unk>` problem

Here is where the vocabulary questions from section 1 stop being philosophical.

In neural models, all weight matrices are of **fixed size**, and that includes input embeddings and the output layer. The output layer in particular contributes significantly to the memory footprint of the computational graph, because it is a $|V| \times m$ matrix plus a softmax over $|V|$ at every single position.

And $|V|$ is not small. The slides give the numbers that matter for this course:

| Language | Realistic vocabulary size |
|---|---|
| English | ~200K words |
| Russian | ~1M words |

That five-to-one ratio is not because Russian speakers know five times more concepts. It is because Russian marks case on every noun, so each lemma appears in many more distinct surface forms. This is the first appearance of the lecture's central point.

> [!definition] Short-list
> The quick fix: **ignore rare words and keep only the $n$ most frequent**. Every rare word is mapped to a single special token, `<unk>`.
>
> Typical short-list sizes: 10K, 50K, 100K, sometimes 200K words.

> [!warning] The disadvantage
> All rare words receive **equal probability** in a given context, because they are all literally the same token to the model. The model cannot prefer *hammered* over *doomscrolled* in `She ___ the nail`, since both are `<unk>`. Frequency-based truncation does not degrade gracefully, it collapses the entire tail into one undifferentiated symbol.

## 5. The pivot: contextual information is not enough

Distributional models do a genuinely good job on related words. Given enough sentences like

```
She moved the desk into her office.
He moved the table into the office.
                 ...
They transported the table into the room.
```

the model learns that *desk* and *table* belong in similar contexts, and their embeddings end up close. Internally *desk* and *table* are just indices, say 11364 and 4527, and the *training data* pulls those two rows of the embedding matrix together.

Now push on it:

- What about *move* (2863) and *moved* (87542)? Nothing in the indices says they are related. The model has to learn that relationship from scratch, from data, for every single inflected pair.
- What about *transports* (9056) and *transported* (32871)? Same problem, again from scratch.
- What about **unseen or rare words** such as *hammered*? There is no data. Context cannot help, because the word never appeared in any context.

> [!tip] The claim the rest of the lecture defends
> **Word-internal structure (morphology) can help induce meaning for unseen words.** If the model knew that *hammered* decomposes as *hammer* + *-ed*, and it already has a good embedding for *hammer* and knows what *-ed* does to a verb, it could construct a representation for a word it has never seen. Contextual information and word-internal information are complementary sources, and word-level indexing uses only the first.

## 6. Two problems, one preprocessing step

For languages that put white space around words, tokenization is easy, with some exceptions: `Mr. Smith` should not become `Mr . Smith`, because the period is part of the abbreviation and not a sentence boundary.

Beyond that, two separate problems hide under the word "segmentation", and confusing them is a standard error:

| Problem | What it is | Name | Where it is solved |
|---|---|---|---|
| **First** | Languages without explicit word boundaries (Chinese, Japanese, Thai, Burmese). Boundaries must be established before any further processing at all | **Word segmentation** / tokenization | This lecture, sections 7 to 9 |
| **Second** | Morphologically rich languages blow up fixed vocabularies with huge inflectional possibilities | **Sub-word segmentation** | Next lecture |

Both are about deciding where to cut a character string. They differ in what the pieces are meant to be: whole words in the first case, units below the word in the second.

## 7. Finding word boundaries

Word segmentation must happen **before** POS tagging, parsing, or anything else, which makes it a hard dependency for the entire pipeline and a task in its own right, with its own benchmarks (the SIGHAN bake-offs).

It is not trivial, because the same character string often segments in more than one valid way.

> [!example] The ambiguous string
> 研究生活动
>
> | Segmentation | Reading |
> |---|---|
> | 研究生 \| 活动 | graduate-student activities |
> | 研究 \| 生活 \| 动 | research on life activities |
>
> Both are grammatical. Nothing in the characters themselves decides between them.

The space of all possible segmentations of a string of length $n$ is huge, but only a few of them are actually valid.

> [!formula] Size of the search space
> A string of $n$ characters has $n-1$ internal gaps, and each gap is independently either a boundary or not:
> $$\#\text{segmentations} = 2^{\,n-1}$$
> For a 20-character sentence that is $2^{19} = 524{,}288$ candidate segmentations. Enumerating them is not an option, which is why every method below is either greedy or a dynamic program.

Two families of approaches: **rule-based**, and **data-driven / statistical**.

## 8. Rule-based segmentation: maximum matching

Assume a pre-compiled vocabulary $V$ of known words, then walk through the character string trying to **match the longest known word at each position**. Greedy, simple, and surprisingly strong as a baseline: still used as a preprocessing step or as a fast fallback in production pipelines.

It can be applied in either direction, and the two directions often disagree on exactly the ambiguous strings you care about.

### Forward Maximum Matching (FMM)

```pseudo
Algorithm: Forward Maximum Matching (FMM)
──────────────────────────────────────────────────────────────
Input:  sentence S (character string), vocabulary V,
        max word length L
Output: list of segmented words

i ← 0
output ← []
while i < length(S):
    for l ← min(L, length(S) − i) down to 1:
        w ← S[i : i+l]                 // candidate substring
        if w ∈ V or l == 1:
            append w to output
            i ← i + l
            break
return output
```

Two details that matter. The inner loop counts **down** from the longest allowed length, so the first hit is by construction the longest match. The `or l == 1` clause is the escape hatch: if no substring of any length is in $V$, the single character is emitted anyway so the algorithm always terminates. That escape hatch is also where out-of-vocabulary words go to die, see the limitations below.

Complexity is $O(n \cdot L)$ dictionary lookups for a string of length $n$ and maximum word length $L$.

> [!example] FMM on 研究生活
> $V$ contains: 研究, 研究生, 生活, 生, 活
>
> | Step | Position | Longest match | Remaining |
> |---|---|---|---|
> | 1 | `i=0` | 研究生活 → 研究生 (3 chars, ∈ V) | 活 |
> | 2 | `i=3` | 活 (1 char, ∈ V) | nil |
>
> **Return: 研究生 \| 活**

### Backward Maximum Matching (BMM)

```pseudo
Algorithm: Backward Maximum Matching (BMM)
──────────────────────────────────────────────────────────────
Input:  sentence S (character string), vocabulary V,
        max word length L
Output: list of segmented words

i ← length(S)
output ← []
while i > 0:
    for l ← min(L, i) down to 1:
        w ← S[i−l : i]                 // candidate substring
        if w ∈ V or l == 1:
            prepend w to output
            i ← i − l
            break
return output
```

Structurally identical to FMM with three changes: start at the end, take the substring ending at $i$ rather than starting at $i$, and **prepend** rather than append so the output stays in reading order.

> [!example] BMM on 研究生活
> $V$ contains: 研究, 研究生, 生活, 生, 活
>
> | Step | Position | Longest match | Remaining |
> |---|---|---|---|
> | 1 | `i=4` | 研究生活 → 生活 (2 chars, ∈ V) | 研究 |
> | 2 | `i=2` | 研究 (2 chars, ∈ V) | nil |
>
> **Return: 研究 \| 生活**

> [!warning] Same string, same dictionary, different answer
> FMM returns 研究生 \| 活. BMM returns 研究 \| 生活. Nothing changed except the direction of the scan. This is not a bug in either algorithm, it is the ambiguity of the string leaking through a greedy decision rule, and it is the entire motivation for what comes next.

### Bidirectional matching

Run both, then compare:

- If both segmentations **agree**, accept the result with high confidence. This is cheap and covers most strings.
- If they **disagree**, apply a tie-breaking heuristic. Common choices: prefer the segmentation with **fewer total words**, or with **fewer single-character words**, both of which are proxies for less fragmentation.

On the example above, FMM gives 2 words and BMM gives 2 words, so the "fewer words" heuristic ties and the "fewer single-character words" heuristic prefers BMM's 研究 \| 生活, which has none, over FMM's 研究生 \| 活, which has one.

> [!warning] Why maximum matching is fundamentally limited
> - **It assumes a precompiled vocabulary.** Compiling a reliable, good-coverage vocabulary is labour-intensive, and it has to be done per language.
> - **Genuinely new words cannot be matched at all.** Names, neologisms, typos: none of them are in $V$, so none of them can ever be recovered.
> - **No mechanism to exploit data.** The algorithm cannot get better by seeing more text. This is the direct motivation for statistical methods.
> - **OOV words break silently.** Anything not in the dictionary gets chopped into single characters or greedily mismatched, and nothing in the output signals that this happened.
> - **Ambiguity is genuine, not a matching artifact.** Some strings really do have multiple correct segmentations depending on meaning and context, so no direction-based heuristic can be right in general. You need something that weighs evidence.

## 9. Statistical word boundary detection

### Reformulate as character labeling

The key move: stop thinking about where to cut, and start thinking about what role each character plays. Label every character with one of four tags.

> [!definition] The BMES tag set
> | Tag | Name | Meaning |
> |---|---|---|
> | **B** | Begin | First character of a multi-character word |
> | **M** | Middle | Interior character of a word with 3 or more characters |
> | **E** | End | Last character of a multi-character word |
> | **S** | Single | A character that is itself a complete one-character word |
>
> Once every character has a label, the segmentation **falls out automatically**: insert a word boundary after every **E** and every **S**.

> [!example] BMES on 研究生活
> 研究生活 → 研/B 究/E 生/S 活/S → **研究 \| 生 \| 活**
>
> Note that this is a *third* answer, different from both FMM (研究生 \| 活) and BMM (研究 \| 生活). Same four characters, three methods, three segmentations. The point is not that one of them is obviously right, it is that the labeling formulation lets you *learn* which is right from data instead of legislating it.

Reformulated this way, the problem is ordinary **sequence labeling**, and any sequence labeling method applies.

### HMMs

> [!definition] HMM for word boundaries
> There is a **hidden** sequence of tags (B/M/E/S) that generated the **observed** sequence of characters. The model has two kinds of parameters:
>
> - **Transition probabilities** $P(\text{tag}_i \mid \text{tag}_{i-1})$: how likely one tag is to follow another (bigrams here). For example, B is very likely to be followed by M or E, and **cannot** be followed by another B.
> - **Emission probabilities** $P(\text{character}_i \mid \text{tag}_i)$: how likely a given character is to be "produced" by a given tag. For example, 究 might have a high probability of being tagged E.

The joint probability of a character sequence $c_{1:n}$ and a tag sequence $t_{1:n}$ is the product of the two:

> [!formula] HMM decoding objective
> $$\hat{t}_{1:n} \;=\; \arg\max_{t_{1:n}} \; \prod_{i=1}^{n} P(t_i \mid t_{i-1}) \; P(c_i \mid t_i)$$
> where:
> - $c_i$ is the $i$-th character of the input string
> - $t_i \in \{B, M, E, S\}$ is the tag assigned to it
> - $P(t_i \mid t_{i-1})$ is a transition probability, with $t_0$ a sentence-start symbol
> - $P(c_i \mid t_i)$ is an emission probability

Ideally this requires **labeled data**, that is a segmented corpus, so the probabilities can be estimated from counts with simple maximum likelihood estimates:

$$P(t_i \mid t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})} \qquad\qquad P(c \mid t) = \frac{C(t, c)}{C(t)}$$

At inference time, given an unsegmented character string, find the most probable tag sequence given the model.

> [!intuition] What the transition matrix encodes for free
> The BMES definitions make most transitions structurally impossible, and the transition probabilities learn this from data without being told:
>
> | From \ To | B | M | E | S |
> |---|---|---|---|---|
> | **B** | no | yes | yes | no |
> | **M** | no | yes | yes | no |
> | **E** | yes | no | no | yes |
> | **S** | yes | no | no | yes |
>
> A word must open with B, continue with zero or more M, and close with E. So B and M can only be followed by something *inside* the same word, and E and S can only be followed by something that *starts a new word*. Half the transition matrix is effectively zero, which is why even a first-order model does well here.

### Viterbi

Trying and scoring all possible labelings is computationally intractable, for the $2^{n-1}$ reason from section 7 (here it is $4^n$ tag sequences, worse still). Use the **Viterbi algorithm**, based on [[Dynamic Programming|dynamic programming]]: store intermediate results instead of recomputing them.

```pseudo
Algorithm: Viterbi for BMES tagging
──────────────────────────────────────────────────────────────
for i = 1 to n (each character position):
    for each tag t in {B, M, E, S}:
        score[i][t] = max over previous tag t' of
                      ( score[i-1][t'] × P(t | t') × P(char_i | t) )
        keep a backpointer to whichever t' achieved that max
backtrace from the highest-scoring final tag to recover the full tag sequence
```

> [!formula] The Viterbi recursion
> $$\text{score}[i][t] \;=\; \max_{t'} \; \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$$
> where:
> - $\text{score}[i][t]$ is the probability of the **best possible path** that ends at position $i$ with tag $t$
> - $t'$ ranges over the tags at position $i-1$
>
> Keeping **backpointers** avoids having to recompute whole paths: you store which predecessor won, then walk the pointers backwards from the best final cell.

> [!formula] Complexity
> $$O(n \times |T|^2)$$
> $n$ positions, and at each position every one of $|T|$ tags is compared against every one of $|T|$ predecessors. With $|T| = 4$ this is $16n$, which is linear in the sentence length. Compare that with $4^n$ for brute force.

The trellis, drawn out for the first three characters, with one column per position and one row per tag. Every cell keeps a single best incoming edge:

```
            研            究            生            活
        ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
   B    │ score  │──►│ score  │──►│ score  │──►│ score  │
        └────────┘╲ ╱└────────┘╲ ╱└────────┘╲ ╱└────────┘
   M    ┌────────┐ ╳ ┌────────┐ ╳ ┌────────┐ ╳ ┌────────┐
        │ score  │╱ ╲│ score  │╱ ╲│ score  │╱ ╲│ score  │
        └────────┘   └────────┘   └────────┘   └────────┘
   E    ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
        │ score  │   │ score  │   │ score  │   │ score  │
        └────────┘╲ ╱└────────┘╲ ╱└────────┘╲ ╱└────────┘
   S    ┌────────┐ ╳ ┌────────┐ ╳ ┌────────┐ ╳ ┌────────┐
        │ score  │╱ ╲│ score  │╱ ╲│ score  │╱ ╲│ score  │
        └────────┘   └────────┘   └────────┘   └────────┘

   every cell = max over 4 incoming edges, weighted by
   P(tag | prev_tag) × P(character | tag), plus one backpointer
```

### What HMMs buy, and what they still cannot do

> [!tip] Advantages of HMMs over FMM and BMM
> - **No fixed dictionary of possible words is required.** The model works over characters, so an unseen word is not a special case, it is just a character sequence with a plausible tag path.
> - **Ambiguity is resolved with probabilistic evidence** rather than direction-based tie-breaking heuristics.
> - **Trainable directly from any segmented corpus**, even one from a different domain.

> [!warning] Shortcomings for word boundary detection
> - **The first-order Markov assumption is narrow.** A tag depends only on the immediately preceding tag and the current character. Higher orders are possible but more involved, and the parameter count grows as $|T|^{k+1}$.
> - **No longer-range context.** The model cannot directly exploit the word two positions back, or a whole surrounding phrase. This is the standard motivation for **conditional random fields (CRFs)**, which condition on arbitrary features of the whole input.
> - **Unsupervised training degrades quality.** HMMs *can* be trained fully unsupervised with Baum-Welch, an instance of EM, but quality drops.
>
> Any neural sequence labeling approach (LSTMs, Transformers) can be used for word boundary detection instead, and in practice is.

## 10. Types of morphological change

Second half of the lecture. We now have words. What is inside them?

The **stem**, **root** or **lemma** of a word defines its core meaning. Three processes operate on it.

> [!definition] Inflection
> **Retains the part of speech**, but changes grammatical features such as number and tense.
>
> - tense: *see, saw, seen*; *bake, baked, baking*
> - number: *book, books*; *see, sees*
>
> *see* and *saw* are both verbs. *book* and *books* are both nouns. Inflection never gives you a new dictionary entry, it gives you another form of the same one.

> [!definition] Derivation
> **Changes the part of speech and/or the meaning.**
>
> - change in POS: *happy* (adj) → *happiness* (noun) → *happily* (adv)
> - change in meaning: *function* → *disfunction*; *perfect* → *imperfect*
>
> Derivation produces what a lexicographer would call a new word, with its own dictionary entry.

> [!definition] Compounding
> **Combines several words into one word.**
>
> - noun + noun: German *Tisch* + *Bein* = *Tischbein* (table leg)
> - adjective + verb: German *sicher* + *gehen* = *sichergehen* (sure + go = "ensure", "make sure")
>
> Both examples are German, and that is not an accident: German writes compounds as single orthographic words where English writes them as phrases. *Tischbein* is one token, *table leg* is two. The morphology is the same, the tokenization consequences are not.

> [!warning] Productivity is the thing that hurts
> Morphological processes are **very productive**: they can be applied iteratively and in new combinations, and this leads to a large increase in vocabulary size.
>
> *doomscrolling, doomscrolled, doomscrolls, doomscroller*
>
> This word did not exist a few years ago. It is already a compound (*doom* + *scroll*) carrying inflection (*-ing*, *-ed*, *-s*) and derivation (*-er*). Four vocabulary slots, or more, for a lemma no corpus from 2015 contains. No fixed vocabulary can be closed over a productive process, which is the formal reason OOV never goes away.

> [!warning] Spelling on the slides
> The slides write *disfunction* and *disfunctional*. Standard English is *dysfunction* / *dysfunctional*, from Greek *dys-*, not Latin *dis-*. The morphological point (a negating prefix changing meaning without changing POS) is unaffected, but do not reproduce the spelling in a report.

## 11. Inflectional morphology in detail

### English verbs

Five forms, and the same five slots for regular and irregular verbs:

| Form | *see* | *call* | *try* | *speak* | *send* | *change* |
|---|---|---|---|---|---|---|
| Present tense | see | call | try | speak | send | change |
| Simple past | saw | called | tried | spoke | sent | changed |
| Past participle | seen | called | tried | spoken | sent | changed |
| Present participle | seeing | calling | trying | speaking | sending | changing |
| 3rd person singular | sees | calls | tries | speaks | sends | changes |

Note how little the regular verbs distinguish: for *call*, simple past and past participle are the same string, so the paradigm has 5 slots but only 4 distinct forms. English inflection is impoverished, which is exactly why English-shaped assumptions travel badly.

### English nouns

- **Number** (singular vs plural): *book, books*; *house, houses*
- **Case**, on pronouns only: *he*, *him* (accusative), *his* (possessive), *them* (accusative + plural), *their* (possessive + plural)

That "pronouns only" is the crucial limitation. English nouns do not mark case at all, so an English-trained intuition has no slot for it.

### Other languages are much richer

| Language | Feature | Examples |
|---|---|---|
| **Russian** | More cases, and cases marked on **all** nouns, not just pronouns | see *knigami* in section 14 |
| **German** | **Mood**: *kommt* (indicative), *käme* (subjunctive), *komm* (imperative) | one verb, three moods, three forms |
| **Russian, Czech, Polish** | **Aspect**: Russian *sdelat'* (perfective, completed) vs *delat'* (imperfective, ongoing) | a distinction English has no inflectional marking for at all |

> [!intuition] Why this multiplies rather than adds
> Inflectional features are independent slots, so the number of forms of a lemma is the **product** of the slot sizes, not their sum. English nouns have 2 numbers and no case, so 2 forms. Add a six-way case system and you have 12. Add possessive marking with six persons and you have 72, from one lemma, before any derivation. This is the arithmetic behind English 200K versus Russian 1M from section 4.

## 12. Derivational morphology in detail

Grouped by what the affix does:

**English nominalization** (make a noun):

| Pattern | Example |
|---|---|
| verb + *-ation* | *derive* + *-ation* = *derivation* |
| verb + *-er* | *kill* + *-er* = *killer*; *bake* + *-er* = *baker* |
| adjective + *-ness* | *happy* + *-ness* = *happiness* |

**English adjectivization** (make an adjective):

| Pattern | Example |
|---|---|
| verb + *-able* | *accountable*, *reasonable* |
| noun + *-al* | *parental*; *colony* + *-al* = *colonial*; *office* + *-al* = *official* |

**English negation** (change the meaning, keep the POS):

| Prefix | Examples |
|---|---|
| *un-* | *unseen*, *unheard* |
| *mis-* | *misjudge*, *misappropriation* |
| *dis-* | *disfunctional* (standard spelling: *dysfunctional*) |
| *im-* | *implausible* |
| *in-* | *indifferent* |

> [!tip] Two things worth noticing
> **Spelling changes at the boundary.** *colony* + *-al* is not *colonyal*, it is *colonial*. *office* + *-al* is *official*, not *officeal*. The morpheme is regular, the surface string is not, so a naive "strip the suffix and look up the stem" analyser fails on exactly these cases.
>
> **The negation prefixes are not interchangeable.** *im-* before labials (*implausible*), *in-* elsewhere (*indifferent*): they are conditioned variants of the same underlying morpheme. A model that treats *im-* and *in-* as unrelated strings misses that they do the same job.

## 13. Morphemes

> [!definition] Morpheme
> The **smallest meaning-carrying parts of words**.
>
> *disproportionally* = *dis* + *proportion* + *al* + *ly* = prefix + stem + suffix + suffix

One word, four morphemes, each contributing something: negation, the core meaning, a POS change to adjective, a POS change to adverb.

| Term | Bound or free | Definition |
|---|---|---|
| **Root** | Free | The morpheme that determines the basic meaning of a word, and can stand alone |
| **Affix** | Bound | Attached to a root to change meaning or grammatical function, and cannot stand alone |

The distinction between **free** morphemes (which are words on their own: *proportion*, *book*, *see*) and **bound** morphemes (which are not: *dis-*, *-al*, *-ly*, *-ed*) is the one that matters for segmentation, because bound morphemes are precisely the pieces that will never appear as standalone tokens in a corpus.

### Kinds of affix, by position

| Affix | Position | Example |
|---|---|---|
| **Prefix** | Attaches to the front. In English, mostly only one per word | *dis-* in *disproportionally* |
| **Suffix** | Attaches to the end. There can be more than one | *-al* and *-ly* in *disproportionally* |
| **Infix** | Inserted in the middle of a word. More common in other languages | *passerby* + *-s-* = *passersby* |
| **Circumfix** | Attaches to the front and back **simultaneously** | German *ge*-seh-*en* (past participle of *sehen*) |

The circumfix is the interesting one for NLP: *gesehen* cannot be analysed by stripping a prefix or a suffix independently, because *ge-* and *-en* are one morpheme in two pieces. Neither half means "past participle" on its own.

> [!warning] The infix example is atypical
> *passerby → passersby* is really plural marking on the head noun *passer* inside a compound, not infixation in the strict sense. English has almost no true infixes. Languages with productive infixation (Tagalog, for instance) place a genuine morpheme inside the root. The slide's own hedge, "more common in other languages", is doing a lot of work.

### Concatenative versus non-concatenative

All the examples above are **concatenative**: morphemes are strung together, and the word is the concatenation of its parts. This is the case that string-splitting handles.

**Non-concatenative** morphology changes the root itself. The lecture's example is **apophony**, a vowel change:

| German | Singular | Plural |
|---|---|---|
| book | *Buch* | *Bücher* |
| house | *Haus* | *Häuser* |

The plural is not *Buch* + suffix. The stem vowel *u* becomes *ü*, and only then is a suffix added. English does the same in *foot / feet*, *sing / sang / sung*.

> [!warning] Why this is the hardest case for any segmenter
> Every method in this course, from maximum matching to byte-pair encoding, works by **cutting a string into contiguous pieces**. Non-concatenative morphology cannot be expressed that way, because the change is *inside* the root rather than *between* morphemes. There is no cut that separates "book" from "plural" in *Bücher*. Keep this in mind next lecture, when subword segmentation is presented as the practical solution: it is a solution to the concatenative case.

## 14. Morphological typology

Languages can be grouped by the way they realize morphological processes. Four types, and this classification is the single most likely thing on the exam from this lecture.

> [!definition] The four types
> **Isolating** (Chinese, Vietnamese)
> - Words tend not to change form at all
> - Grammatical relations come from **word order**
> - **Particles** express additional modification
>
> **Agglutinative** (Turkish, Finnish)
> - Words are built by stringing morphemes together, potentially involving character modifications
> - **Each morpheme contributes one clear piece of meaning**, and boundaries are visible
>
> **Fusional** (Russian, Spanish, Arabic)
> - A **single suffix simultaneously encodes several grammatical features** that often cannot be cleanly separated
>
> **Polysynthetic** (Inuktitut, Mohawk)
> - Incorporate multiple grammatical roles simultaneously (verb, subject, object)
> - Incorporate various modifiers (negation, tense, mood, instrument, location)
> - All fused or agglutinated into a **single word**

Summary, with the axis that actually distinguishes them:

| Type | Morphemes per word | Are boundaries separable? | Languages | Signature |
|---|---|---|---|---|
| **Isolating** | About one | Not applicable, nothing to separate | Chinese, Vietnamese | Grammar lives in word order and particles |
| **Agglutinative** | Many | Yes, cleanly | Turkish, Finnish | One morpheme, one meaning |
| **Fusional** | Few | No, features are fused | Russian, Spanish, Arabic | One morpheme, several meanings |
| **Polysynthetic** | Very many | Partly, and fused as well | Inuktitut, Mohawk | One word, one sentence |

Isolating and polysynthetic are the two ends of a scale (how much meaning is packed into one word). Agglutinative and fusional sit in the middle and are distinguished by a different question (can you cut the packed meaning apart), not by how much is packed.

### Isolating: Chinese

我 昨天 看 了 一 本 书 ("I read a book yesterday")

| 我 | 昨天 | 看 | 了 | 一 | 本 | 书 |
|---|---|---|---|---|---|---|
| wǒ | zuótiān | kàn | le | yī | běn | shū |
| I | yesterday | see/read | PFV | one | CL | book |

- 看 (*kàn*) **never changes form** regardless of tense or aspect. Past is not marked on the verb, it is expressed by the particle 了 (perfective aspect, glossed PFV) and by the adverb 昨天 "yesterday".
- Word order is straight **SVO** (subject-verb-object), and that order is what carries the grammatical relations.
- CL is a **classifier**, 本 (*běn*), obligatory between a numeral and a noun of this class. English has nothing comparable except in phrases like "two *sheets* of paper".

### Isolating: Vietnamese

Tôi đã đi học ("I went to study")

| Tôi | đã | đi | học |
|---|---|---|---|
| tôi | đã | đi | học |
| I | PST | go | study |

- Neither *đi* nor *học* changes form. Instead *đã* is a **PST particle** (past / completed action).
- Note what this means for a tokenizer: every morpheme is already its own whitespace-delimited token. Isolating languages are the easy case for word-level vocabularies, and the hard case for anything that expects tense to be recoverable from the verb.

### Agglutinative: Turkish

*evlerinizden* ("from your houses")

| ev | ler | iniz | den |
|---|---|---|---|
| house | plural | your | from |

- One Turkish word corresponds to a four-word English phrase.
- Every suffix is a separate, identifiable unit with exactly one job. You can read the word left to right like a small sentence.

### Agglutinative: Finnish

*taloissamme* ("in our houses")

| talo | i | ssa | mme |
|---|---|---|---|
| house | plural | inessive | our |

- The **inessive** is a locative case meaning "in". Finnish has five other locative cases: **elative** = "out of", **illative** = "into", **allative** = "to", and others.
- Each suffix is clearly separable and **reusable across many other nouns**, which is what makes the type learnable: a segmenter that discovers *-ssa* once can apply it everywhere.
- Some vowel and consonant changes may be required during attachment (note *talo* + *-i-* here), so the boundaries are clean at the level of morphemes but not always at the level of characters.

### Fusional: Russian

*knigami* ("with books")

| knig | ami |
|---|---|
| book | instrumental.plural.feminine |

- One suffix, *-ami*, fuses **case** (instrumental), **number** (plural) and **gender** (feminine) into a single unsegmentable morpheme.
- There is no substring of *-ami* that means "plural". That is the definition of fusional, and it is why a purely string-based segmenter cannot recover the features even in principle.

### Fusional: Spanish

*hablé* ("I spoke")

| habl | é |
|---|---|
| speak | 1pers.singular.preterite |

- One suffix, *-é*, fuses **person** (1st), **number** (singular), **tense** (preterite, that is past) and **aspect** (preterite, that is completed).
- Four features, one vowel.

### Fusional: Arabic

*yaktubu* ("he writes")

| Root | Pattern | Result |
|---|---|---|
| *ktb* (C1 C2 C3) | *ya-* C1 C2 *-u-* C3 *-u* | *ya-k-t-u-b-u* = *yaktubu* |

- The root is three **consonants**, *k-t-b*, carrying the core meaning "write". The pattern is a template of prefixes, vowels and slots that the consonants are poured into.
- One pattern fuses **person** (3rd), **gender** (male) and **aspect** (not completed).
- The vowels in this pattern are **not written down** in regular Arabic orthography, only spoken. So the written form gives you the consonantal skeleton and the reader supplies the rest.

> [!warning] Why Arabic is the worst case for a naive tokenizer
> This is **non-concatenative** morphology at industrial scale. The morphemes are interleaved, not concatenated, so no set of cuts recovers them. On top of that, the written form omits the very vowels that carry person, gender and aspect, so the information is not merely hard to segment, it is not in the character string at all. A subword tokenizer trained on Arabic learns consonant clusters and has no way to represent the pattern as a unit.

### Polysynthetic: Inuktitut

*Qangatasuukkuvimmuuriaqalaaqtunga* ("I will have to go to the airport")

| Morpheme | Meaning |
|---|---|
| *qangata-* | fly |
| *-suukkuvik* | thing that flies habitually → airport |
| *-mut* | to (allative case) |
| *-uq-* | go to |
| *-riaqaq-* | have to |
| *-laaq-* | future |
| *-tunga* | 1st person singular |

- **One word corresponds to an entire English sentence.** A verb, its subject, its object, tense and modal meaning, all fused together.
- The surface form is not the concatenation of the pieces listed: *-k-mut-uq* becomes *-mmuu-*. Phonological rules at the boundaries chew up the morphemes so that the written word cannot be split back into them by string matching.
- Note the internal derivation: *qangata-* "fly" plus *-suukkuvik* "thing that does X habitually" gives "airport". The word contains its own etymology, productively.

> [!tip] What this means for vocabulary size
> In an isolating language, the number of word types is roughly the number of morphemes. In a polysynthetic language, the number of word types is closer to the number of *sentences*, because a word is a sentence. There is no vocabulary size that covers Inuktitut. Not 200K, not 1M, not any number. This is the reductio that the whole lecture is building towards.

## 15. Morphology and NLP: why English-shaped systems break

The payoff slide, and the bridge to the rest of the course.

**Morphology is a common process for most languages.** Rich morphology can encode exactly the things that English, and other morphologically poor languages, encode through word order. Turkish *evlerinizden* and English "from your houses" carry the same information: one language puts it in suffixes, the other in separate words and their arrangement. Neither is more complex, but only one of them is friendly to a whitespace tokenizer.

**Word-level vocabularies blow up or become extremely restrictive.** These are the two horns, and there is no third option:

- A single lemma can surface as **thousands of word forms** (also called *surface forms*). Keep them all and $|V|$ explodes, taking the embedding matrix and the output softmax with it.
- A **fixed vocabulary sized for English** will produce massive **out-of-vocabulary (OOV)** rates for other languages. Keep it small and most of the language becomes `<unk>`, which by section 4 means every rare form gets the same probability.

> [!warning] The multilingual version of the problem
> Now put both in one model, which is what this course is about. A shared vocabulary across languages has to serve English, where 50K types covers most text, and Finnish, where it does not come close. Whatever size you pick, the morphologically rich languages get fragmented into meaningless pieces while English words stay whole. That asymmetry in how many tokens a sentence costs is a bias baked into the tokenizer before a single parameter is trained.

**These problems are the main motivation for morphological analysers.** But:

- Morphological analysis is **highly language dependent**, so you need one per language, and building it needs linguistic expertise per language. This is the WordNet problem from section 2 all over again.
- Modern approaches **build on data-driven methods** rather than hand-written rules.
- The **practical compromise: subword tokenization.** Do not try to find the linguistically correct morphemes. Find statistically useful pieces, in a language-agnostic way, from data.

> [!tip] The handoff
> Subword tokenization is a compromise and the lecture says so. It does not recover *knig* + *ami*, and it certainly does not recover *k-t-b* + pattern. What it does is guarantee a fixed vocabulary with zero OOV rate, by falling back to characters when nothing longer fits. Whether the pieces it finds line up with morphemes is an empirical question, and mostly the answer is "somewhat, in the languages that are concatenative". That is the subject of the next lecture.

## 16. Where the lecture lands

- **Words are not atoms.** Word-level indices throw away the relationship between *move* and *moved*, and leave you with nothing at all for *hammered*.
- **Word boundaries are not given.** For Chinese, Japanese, Thai and Burmese they have to be computed, greedily (FMM/BMM, cheap but dictionary-bound and direction-dependent) or probabilistically (BMES labeling with an HMM plus Viterbi, trainable and ambiguity-aware, but limited by the Markov assumption).
- **Inside a word are morphemes**, combined by inflection (same POS, different features), derivation (new POS or new meaning) and compounding (several words into one), and all three are productive, so the vocabulary is never closed.
- **Languages differ systematically** in how much they pack into one word and whether the packing is separable: isolating, agglutinative, fusional, polysynthetic.
- **Therefore fixed word-level vocabularies fail** outside morphologically poor languages, and the practical fix is subword segmentation.

## Key Takeaways

> [!tip] Exam Focus
> Five things to be able to do cold:
>
> 1. **Run FMM and BMM by hand** on a short character string given a vocabulary, and say why they disagree. The 研究生活 example is the canonical one: FMM gives 研究生 \| 活, BMM gives 研究 \| 生活. Know the `or l == 1` fallback and what it does to OOV words.
> 2. **State the BMES scheme** and convert a tag sequence to a segmentation (boundary after every E and S) and back. Know why B cannot follow B.
> 3. **Write the Viterbi recursion** $\text{score}[i][t] = \max_{t'} \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$, say what a backpointer is for, and give the complexity $O(n |T|^2)$ against $|T|^n$ for brute force.
> 4. **Distinguish inflection from derivation** with a one-line test: does the part of speech survive? *see / saw* is inflection, *happy / happiness* is derivation. Add compounding as the third process.
> 5. **Name the four typological classes with one example language and one worked example each.** Turkish *ev-ler-iniz-den* for agglutinative and Russian *knig-ami* for fusional are the minimal pair: both pack several features into suffixes, only one of them lets you cut the suffixes apart.

> [!warning] The distinction people lose marks on
> **Agglutinative is not "more morphemes", it is "separable morphemes".** Fusional languages can be just as morphologically dense. The question is whether a single affix carries a single feature (agglutinative: Turkish *-ler* is plural and nothing else) or several fused features (fusional: Russian *-ami* is instrumental *and* plural *and* feminine, with no way to divide it). If you define the classes by density you will misclassify Spanish.

> [!warning] The other one
> **Word segmentation and subword segmentation are different problems.** The first finds word boundaries in a script that does not mark them (Chinese). The second splits words that are perfectly well delimited into smaller pieces (Finnish). They use different methods for different reasons, and this lecture only solves the first one.

## Links

- **Course:** [[MNLP - Overview|Course overview]] · [[MNLP - Mini Project]]
- **Previous:** [[MNLP-L01 - Overview]]
- **Next:** Morphological analysis and subword segmentation, which takes the OOV problem from section 15 and solves it statistically
- **Concepts:** [[Word Embeddings]] · [[Tokenization]] · [[Bag of Words]] · [[Dynamic Programming]]
- **Source:** `MNLP_morphology.pdf` (Christof Monz, Morphology: Word Formation, 34 slides)
