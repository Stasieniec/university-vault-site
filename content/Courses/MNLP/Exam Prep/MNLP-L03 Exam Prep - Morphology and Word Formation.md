---
type: exam-prep
course: 5204MNLP6Y
lecture: 3
status: complete
---

# MNLP-L03 Exam Prep - Morphology and Word Formation

> [!abstract] What the exam can ask here
> A block on this lecture most likely tests word segmentation (forward and backward maximum matching, bidirectional matching, BMES tagging with an HMM and Viterbi), the line between inflection, derivation and compounding, affix types, and the four morphological types with their example languages. The traps sit in directions and conditions: which scan gives which result, which way the emission probability conditions, what the HMM does and does not need, which type a language belongs to, and what counts as concatenative.
> Lecture: [[MNLP-L03 - Morphology and Word Formation|lecture note]]. Numbers: [[MNLP - Cheat Sheet|cheat sheet]].

## Word segmentation and subword segmentation

- Languages **without explicit word boundaries** (Chinese, Japanese, Thai, Burmese) need word segmentation, also called **tokenization**.
- Word segmentation must happen **before** POS tagging, parsing or anything else.
- It is a task in its own right, with its **own benchmarks** (SIGHAN bake-offs).
- **Sub-word segmentation** answers a different problem: morphologically rich languages **blow up fixed vocabularies** with their inflected forms.
- The same character string often has **more than one valid** segmentation: 研究生活动 is "graduate-student activities" or "research on life activities".
- The space of possible segmentations is **huge**, and only a **few** are valid.
- Even with spaces there are exceptions: *Mr. Smith* should **not** become *Mr . Smith*.
- Approaches are **rule-based** or **data-driven (statistical)**.

> [!warning] Traps
> - *"Word segmentation is needed mainly for morphologically rich languages such as Finnish."* False: it is for scripts without word boundaries; rich languages need subword segmentation.
> - *"Word segmentation can be run after POS tagging to correct its output."* False: it must come before any further processing.

## Maximum matching: forward, backward, bidirectional

- Requires a **pre-compiled vocabulary** of known words; the algorithm also takes a **maximum word length** $L$.
- At each position it takes the **longest** known word: **greedy**.
- **Simple and surprisingly strong** as a baseline, still used as preprocessing or a **fast fallback** in production pipelines.
- **Forward** Maximum Matching scans **left to right**; **Backward** scans **right to left** and **prepends** each word.
- A single character is **always accepted** (`l == 1`), so the scan always moves on.
- The two **often disagree** on ambiguous strings. 研究生活 with $V$ = {研究, 研究生, 生活, 生, 活}: FMM gives **研究生 | 活**, BMM gives **研究 | 生活**.
- **Bidirectional** matching runs both: if they **agree**, accept with **high confidence**.
- If they **disagree**, apply a **tie-breaking heuristic**: fewer total words, or fewer single-character words (a proxy for less fragmentation).
- Still limited by the **precompiled vocabulary**: compiling a good-coverage one is labour-intensive, and genuinely new words (names, typos) **cannot be matched at all**.
- Out-of-vocabulary words **break silently**: chopped into single characters or greedily mismatched.
- **No mechanism to exploit data**, which motivates statistical methods.
- The ambiguity is **genuine**: some strings have several correct segmentations depending on meaning and context.

> [!warning] Traps
> - *"Given the same vocabulary, FMM and BMM always produce the same segmentation."* False: they often disagree on ambiguous strings.
> - *"Bidirectional matching resolves disagreements using probabilistic evidence."* False: it uses a tie-breaking heuristic; probabilistic evidence is the HMM's advantage.
> - *"Maximum matching gets better as it sees more text."* False: it has no mechanism to exploit data.
> - *"Maximum matching marks words it cannot match as unknown."* False: they break silently.
> - *"Running both directions removes the dependence on a precompiled vocabulary."* False: bidirectional matching is still limited by it.
> - *"FMM/BMM disagreement is purely an artefact of greedy matching."* False: the ambiguity is genuine.

## BMES character tagging

- Segmentation becomes a **character-labelling** problem.
- **B**: first character of a **multi-character** word. **M**: interior character of a word with **3 or more** characters. **E**: last character of a multi-character word. **S**: a character that is a **one-character word**.
- A word boundary goes **after every E and every S**.
- 研/B 究/E 生/S 活/S gives 研究 | 生 | 活.
- **Several** sequence labelling approaches can solve it: HMMs, CRFs, and **any** neural sequence labeller (LSTMs, Transformers).

> [!warning] Traps
> - *"A word boundary is inserted after every B tag."* False: after every E and S.
> - *"M marks the second character of a two-character word."* False: a two-character word is B E; M needs 3 or more characters.

## The HMM for word boundaries

- A **hidden** sequence of tags (B/M/E/S) generated the **observed** sequence of characters.
- **Transition** probabilities $P(\text{tag}_i \mid \text{tag}_{i-1})$: tag **bigrams**.
- B is very likely followed by M or E, and **cannot** be followed by another B.
- **Emission** probabilities $P(\text{character}_i \mid \text{tag}_i)$: the **character given the tag**, how likely a tag "produces" a character.
- Training **ideally** uses **labelled** data: simple **maximum likelihood** estimates from counts.
- It **can** be trained **fully unsupervised** with **Baum-Welch** (an instance of EM), but **quality drops**.
- Inference: the most probable tag sequence for an unsegmented string.

**Gains over FMM and BMM:**
- **No fixed dictionary** of words required: it works on characters.
- Resolves ambiguity with **probabilistic evidence** instead of tie-breaking heuristics.
- Trainable from **any** segmented corpus, **even** one from a **different domain**.

**Shortcomings:**
- First-order Markov assumption: a tag depends **only** on the **immediately preceding tag** and the **current character**.
- Higher orders are **possible** but more involved.
- **Cannot directly** exploit longer-range context (the word two positions back, a whole phrase): this motivates **CRFs**.

| | maximum matching | HMM |
|---|---|---|
| needs a word dictionary | yes | no, characters |
| ambiguity settled by | tie-breaking heuristic | probabilistic evidence |
| learns from data | no | yes, segmented corpus |

> [!warning] Traps
> - *"The emission probability is the probability of a tag given a character."* False: it is the character given the tag.
> - *"In the HMM, the characters are the hidden states."* False: the tags are hidden, the characters observed.
> - *"An HMM segmenter cannot be trained without a segmented corpus."* False: Baum-Welch trains it unsupervised.
> - *"Unsupervised Baum-Welch training gives the same quality as supervised training."* False: quality drops.
> - *"An HMM segmenter needs a precompiled dictionary of words."* False: that is maximum matching.
> - *"An HMM segmenter must be trained on in-domain data."* False: any segmented corpus, even from a different domain.
> - *"A first-order HMM can use the word two positions back."* False: no longer-range context; CRFs address that.
> - *"HMMs beyond first order are impossible."* False: possible, but more involved.

## Viterbi decoding

- Trying and scoring **all** labellings is **intractable**.
- Viterbi uses **dynamic programming**: it stores intermediate results.
- $\text{score}[i][t]$ = probability of the **best path ending at position $i$ with tag $t$**.
- $\text{score}[i][t] = \max_{t'} \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$
- **Backpointers** avoid recomputing whole paths; backtrace from the **highest-scoring final tag**.
- Complexity $O(n \times |T|^2)$: **linear** in the number of characters $n$.

> [!warning] Traps
> - *"Viterbi enumerates and scores every tag sequence, keeping the best."* False: that is intractable; Viterbi keeps only the best path per cell.
> - *"Viterbi runs in $O(n^2 \times |T|)$."* False: $O(n \times |T|^2)$.
> - *"Each cell stores the sum over all paths ending in that tag."* False: the probability of the best path (a max).

## Inflection, derivation, compounding

- The **stem, root or lemma** defines a word's core meaning.
- **Inflection** **retains** the POS and changes features such as number and tense: *see, saw, seen*; *book, books*.
- **Derivation** changes the POS **and/or** the meaning: *happy, happiness, happily* (POS); *function, disfunction*, *perfect, imperfect* (meaning only).
- **Compounding** combines several words into one: *Tisch + Bein = Tischbein* (noun + noun), *sicher + gehen = sichergehen* (**adjective + verb**).
- Morphological processes are **very productive**: applied **iteratively** and in **new combinations**, giving a **large increase** in vocabulary size (*doomscrolling, doomscrolled, doomscrolls, doomscroller*).
- English verbs: present, simple past, past participle, present participle, 3rd person singular.
- English nouns inflect for **number**; **case** only on **pronouns** (*he, him, his, them, their*).
- **Russian**: more cases, marked on **all** nouns.
- **German** marks **mood**: *kommt* indicative, *käme* **subjunctive**, *komm* **imperative**.
- **Slavic** languages mark **aspect**: Russian *sdelat'* **perfective** (completed), *delat'* **imperfective** (ongoing).
- English derivation: **nominalization** (*-ation*, *-er*: *baker*, *-ness*), **adjectivization** (*-able*, *-al*: *colonial*, *official*), **negation** (*un-, mis-, dis-, im-, in-*).

> [!warning] Traps
> - *"Derivation always changes the part of speech."* False: *perfect/imperfect* changes only the meaning.
> - *"imperfect is an inflection of perfect, since the POS stays the same."* False: derivation can keep the POS.
> - *"Compounding only combines nouns."* False: *sichergehen* is adjective + verb.
> - *"English marks case on all nouns."* False: only on pronouns; Russian marks it on all nouns.
> - *"German komm is the subjunctive."* False: imperative; *käme* is subjunctive.
> - *"Russian delat' is the perfective form."* False: *sdelat'* is perfective, *delat'* imperfective.
> - *"-er in baker is an inflectional suffix."* False: derivation (nominalization).

## Morphemes and affixes

- A **morpheme** is the **smallest meaning-carrying** part of a word: *dis + proportion + al + ly* = prefix + stem + suffix + suffix.
- A **root** is a **free** morpheme that determines the basic meaning.
- An **affix** is a **bound** morpheme that changes meaning or grammatical function.
- **Prefix**: front; in English **mostly only one**.
- **Suffix**: end; there **can be more than one**.
- **Infix**: inserted in the middle (*passerby + -s- = passersby*), **more common in other languages**.
- **Circumfix**: attaches to front and back **simultaneously** (German *gesehen*).
- These are all **concatenative**. **Apophony** is **non-concatenative**: German *Buch → Bücher*, *Haus → Häuser*.

> [!warning] Traps
> - *"A root is a bound morpheme."* False: free; affixes are bound.
> - *"English words typically take several prefixes but at most one suffix."* False: the reverse.
> - *"Apophony (Buch → Bücher) is concatenative morphology."* False: non-concatenative.
> - *"In gesehen, ge- and -en are two independent affixes."* False: one circumfix, attached simultaneously.

## Morphological types

| type | how | languages |
|---|---|---|
| isolating | words **tend not** to change form; **word order**, particles | Chinese, Vietnamese |
| agglutinative | morphemes strung together, **one meaning each** | Turkish, Finnish |
| fusional | **one suffix, several features**, **often** not separable | Russian, Spanish, Arabic |
| polysynthetic | verb, subject, object, modifiers in **one word** | Inuktitut, Mohawk |

- **Isolating**: words tend **not to change form**; grammatical relations come from **word order**; **particles** add modification.
- Chinese 我 昨天 看 了 一 本 书: 看 **never** changes form for tense or aspect; word order is **SVO**. Vietnamese *Tôi đã đi học*: *đi*, *học* unchanged; *đã* is a **past** (completed-action) particle.
- **Agglutinative**: morphemes strung together, **potentially with character modifications**; each contributes **one clear piece** of meaning.
- Turkish *ev-ler-iniz-den* = house-plural-your-from: **one** Turkish word for a **four-word** English phrase.
- Finnish *talo-i-ssa-mme* = house-plural-inessive-our. **Inessive** is a locative case meaning "**in**"; elative = out of, illative = into, allative = to.
- Agglutinative suffixes are **clearly separable** and **reusable** across many other nouns; some vowel or consonant changes **may** be needed when attaching.
- **Fusional**: a **single** suffix **simultaneously** encodes several features that **often cannot be cleanly separated**.
- Russian *knig-ami* "with books": *-ami* fuses **case, number and gender** (instrumental, plural, feminine) into one unsegmentable morpheme.
- Spanish *habl-é* "I spoke": *-é* fuses **person, number, tense and aspect**.
- Arabic *yaktubu* "he writes": root *ktb* + pattern *ya-C1C2-u-C3-u*; the pattern fuses person, gender and aspect, and its vowels are **not written** in regular Arabic.
- **Polysynthetic**: multiple grammatical roles (verb, subject, object) and modifiers (negation, tense, mood, instrument, location), **fused or agglutinated** into a single word.
- Inuktitut *Qangatasuukkuvimmuuriaqalaaqtunga* = an **entire** English sentence ("I will have to go to the airport"); morphemes change shape where they meet (*-k-mut-uq* becomes *-mmuu-*).

> [!warning] Traps
> - *"Arabic is an agglutinative language."* False: fusional (root and pattern fuse features).
> - *"Finnish is a fusional language."* False: agglutinative, like Turkish.
> - *"In agglutinative languages, morphemes never change their characters when attached."* False: vowel or consonant changes may be required.
> - *"Isolating languages mark tense by inflecting the verb."* False: the verb does not change; particles and word order do the work.
> - *"The Finnish inessive case means 'into'."* False: inessive is "in"; illative is "into".
> - *"In a fusional language, each suffix encodes exactly one grammatical feature."* False: that describes agglutinative languages.

## Vocabularies, short-lists and why word-level models break

- A **vocabulary** is the inventory of words; a **lexicon** stores their **meanings and functions**; a **dictionary** is a **written** version of a lexicon.
- In neural models **all** weight matrices are **fixed size**, **including** input embeddings and output layers.
- The output layer contributes **significantly** to the memory footprint.
- Vocabularies run far larger for Russian than for English (sizes on the cheat sheet).
- **Short-list**: keep only the $n$ **most frequent** words; all rare words map to one token, `<unk>`.
- Short-list disadvantage: **all rare words receive equal probability** in a given context.
- Internally words are **just indices**: *move* and *moved* are unrelated numbers; an unseen word such as *hammered* gets nothing.
- Word-internal structure (morphology) **can help induce meaning for unseen words**.
- **Most** languages have morphology; rich morphology encodes inside the word what English encodes through **word order**.
- A single lemma can surface as **thousands** of word forms.
- A fixed vocabulary sized for English gives **massive OOV** rates for other languages.
- Morphological analysis is **highly language dependent**; the practical compromise is **subword tokenization**.

> [!warning] Traps
> - *"A lexicon is a list of the word forms of a language."* False: that is a vocabulary; a lexicon stores meanings and functions.
> - *"A short-list keeps the rarest words and drops the frequent ones."* False: it keeps the most frequent.
> - *"With a short-list, rare words keep distinct probabilities."* False: all become `<unk>`, with equal probability.
> - *"Morphological analysers are language independent, so one serves all languages."* False: highly language dependent.

## Word meaning, embeddings and CBOW

- Word meaning as relations: **synonyms** (*purchase :: acquire*), **hyponyms** (*car :: vehicle*, **is-a**), **meronyms** (*wheel :: car*, **part-whole**), **antonyms** (*small :: large*).
- Hand-crafted resources such as WordNet are **expensive**, **incomplete** and **language-specific**.
- Word embeddings are **low dimensional**, **dense** (no zeros), **continuous**, and **learned by performing a task**.
- **Word2Vec** has two approaches: **CBOW** and **Skip-Gram**.
- CBOW predicts the word at position $t$ from the $n$ words to its **left** and the $m$ words to its **right**.
- It looks like n-gram language modelling with $m = 0$.
- A **simple feed-forward** network that **focuses on learning the embeddings themselves**: the projection layer **sums** the context embeddings, and the embedding layer sits **closer to the output**.
- Typically $n = m$, with $n \in \{2, 5, 10\}$.

> [!warning] Traps
> - *"wheel :: car is a hyponym relation."* False: meronym (part-whole); *car :: vehicle* is the hyponym.
> - *"Word embeddings are sparse, high-dimensional vectors."* False: dense and low dimensional.
> - *"CBOW predicts the centre word from the left context only."* False: from both sides.
> - *"Hand-crafted resources like WordNet are language independent."* False: language-specific.

## Practice questions

**1.** Forward Maximum Matching (FMM) and Backward Maximum Matching (BMM) segment 研究生活 with the vocabulary {研究, 研究生, 生活, 生, 活}. Which of the following statements are true? (multiple answers possible)
- [A] FMM returns 研究生 | 活.
- [B] BMM also returns 研究生 | 活, since both use the same vocabulary.
- [C] If disagreements are broken by preferring fewer single-character words, bidirectional matching picks the BMM result.
- [D] A name missing from the vocabulary is flagged as unknown in the output.

> [!example]- Answer
> **A and C.** B: BMM returns 研究 | 生活. C is true because 研究 | 生活 has no single-character word and 研究生 | 活 has one. D: out-of-vocabulary words break silently, chopped into single characters or mismatched.

**2.** Word boundary detection is cast as BMES tagging with a first-order HMM. Which of the following statements are true? (multiple answers possible)
- [A] The characters are the hidden states and the BMES tags are observed.
- [B] The emission probability is the probability of a character given its tag.
- [C] Because it works on characters, the HMM needs no fixed dictionary of words.
- [D] The model can condition each tag on the whole surrounding phrase.

> [!example]- Answer
> **B and C.** A: the tags are hidden and the characters observed. D: a tag depends only on the preceding tag and the current character; longer-range context is what motivates CRFs.

**3.** Which of the following statements about training and decoding an HMM word segmenter are true? (multiple answers possible)
- [A] Viterbi decoding runs in $O(n \times |T|^2)$ time.
- [B] Viterbi scores every possible tag sequence and keeps the best one.
- [C] Training with Baum-Welch on unsegmented text gives the same quality as training on a segmented corpus.
- [D] The HMM can be trained on a segmented corpus from a different domain.

> [!example]- Answer
> **A and D.** B: scoring every labelling is intractable; Viterbi stores the best path per position and tag. C: fully unsupervised Baum-Welch training works, but quality drops.

**4.** Which of the following statements about word formation are true? (multiple answers possible)
- [A] Inflection retains the part of speech of a word.
- [B] Derivation always changes the part of speech of a word.
- [C] German *gesehen* contains a circumfix.
- [D] German *Buch → Bücher* is an example of concatenative morphology.

> [!example]- Answer
> **A and C.** B: derivation changes the POS and/or the meaning; *perfect → imperfect* keeps the POS. D: apophony is non-concatenative.

**5.** Which of the following statements about morphological typology are true? (multiple answers possible)
- [A] In agglutinative languages, each morpheme contributes one clear piece of meaning.
- [B] Arabic is agglutinative, since root and pattern are strung together.
- [C] In isolating languages, grammatical relations come from word order.
- [D] In Russian *knig-ami*, the suffix *-ami* encodes case, number and gender at once.

> [!example]- Answer
> **A, C and D.** B: Arabic is fusional; the pattern fuses person, gender and aspect into the word.

**6.** Which of the following statements about vocabularies in NLP systems are true? (multiple answers possible)
- [A] A lexicon stores the meanings and functions of words.
- [B] A short-list keeps the most frequent words and maps all others to `<unk>`.
- [C] With a short-list, rare words keep distinct probabilities in a given context.
- [D] Morphological analysis is largely language independent, which makes it the practical fix for vocabulary blow-up.

> [!example]- Answer
> **A and B.** C: all rare words become `<unk>` and receive equal probability. D: morphological analysis is highly language dependent; the practical compromise is subword tokenization.
