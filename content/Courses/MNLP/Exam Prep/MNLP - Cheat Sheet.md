---
type: exam-prep
course: 5204MNLP6Y
status: complete
---

# MNLP - Cheat Sheet

The two-sided sheet for the exam (Thu 2026-10-22, 09:00, see [[MNLP - Exam Analysis]]). Print version: `Assets/MNLP-Cheat-Sheet.pdf`, two A4 sides, rebuilt from this note with `docs/superpowers/tools/cheat_sheet_pdf.cjs`. Each entry gives what a thing is, then **+** advantages, **−** disadvantages, and **vs** the thing it is confused with; each topic ends with traps (plausible false statements with the correction). Results are directions only, no benchmark numbers. Where Monz's slide and the paper disagree, "exam answer: slide" marks the slide's version. Checked against Monz's slide decks and the lecture notes: [[MNLP-L01 - Overview|L01]], [[MNLP-L02 - Multilinguality and Writing Systems|L02]], [[MNLP-L03 - Morphology and Word Formation|L03]], [[MNLP-L04 - Subword Segmentation|L04]], [[MNLP-L05 - Static Embeddings|L05]], [[MNLP-L06 - Contextual Embeddings|L06]], [[MNLP-L07 - Crosslingual NLP|L07]]. <!-- print-header: MNLP cheat sheet (Monz), exam 22 Oct 2026 -->

## L01 Overview

### Scenarios, tasks, deep learning, LLMs
- **Multilingual** (training not language-specific): NER for many languages, parsing (universal features), classification across languages.
- **Crosslingual** (knowledge transfer): MT (make understandable), QA (extract), unlearning (manipulate), reasoning (combine).
- **Tasks**: QA returns the answer (IR: ranked documents, even a whole question chopped into keywords); **sentiment** pos/neg or fine-grained (X likes Y about Z), needs NER, sentence and document level; **MT** typically per sentence; **summarisation** one or several documents; **NER** much harder outside English.
- **Before deep learning**: one approach per application; ML generative (Bayesian) and discriminative; hand-built features (parse trees syntactic, named entities semantic) that ML only weighs.
- **Neural**: state of the art for all NLP. **+** strong performance, little or no feature engineering, few network types, so insights transfer to vision, speech. **−** needs large data, errors hard to trace, can fail spectacularly.
- **LLMs**: originated in NLP; do tasks end to end, substitute or supplement humans (annotation, LLM as a judge). Most are English-centric (Internet data); Llama 3.1, DeepSeek better on high-resource languages; dedicated multilingual (Aya-Expanse) tend to lag behind.

!! **Traps:** "unlearning combines information" (false: manipulates).

## L02 Multilinguality and writing systems

### Resource tiers
- **English dominance**: most data, tools, benchmarks, papers; other languages get by-products, translated English resources, less so dedicated effort. Pipelines assume closed vocabularies; language independence usually untested.
- **Coverage** follows commercial interest, military relevance.
- **High resource**: large labelled and unlabelled data, many tools: English, Mandarin, [big gap] French, Spanish, Arabic, Russian.
- **Medium**: limited labelled, reasonable unlabelled, some tools: German, Italian, Dutch, Korean, Japanese.
- **Low**: no labelled, limited unlabelled, no tools: remaining ≈7,000.

!! **Traps:** "coverage depends mainly on number of speakers" (false: commercial, military). "Japanese is high resource" (false: medium). "Arabic is medium" (false: high). "low resource: limited labelled data, some tools" (false: no labelled, no tools).

### English-first assumptions

| assumption | broken by |
|---|---|
| whitespace between words | Chinese, Japanese, Thai, Lao, Khmer, Burmese |
| word order = relations | case marking: Russian, Finnish, Korean |
| one word = one meaning | agglutinative: Finnish, Turkish, Hungarian; compounding: German, Dutch, Swedish |
| left to right | Arabic, Hebrew, Urdu, Persian |

- **Thai**: spaces end clauses or sentences (no full stop). **Right-to-left** in Unicode: stored left to right, rendered right to left. **Chinese**: verb never marks tense (去 go/went); past from sentence-final particle 了.

!! **Traps:** "Thai uses spaces between words" (false: ends clauses). "Turkish is the compounding example" (false: agglutinative). "Unicode stores Arabic right to left" (false: rendered).

### Writing systems

| type | one symbol = | examples |
|---|---|---|
| alphabetic | consonant or vowel; small inventory | Latin, Cyrillic, Hangul |
| syllabic | whole syllable | Hiragana, Katakana |
| logographic | word or morpheme; little sound | Hanzi, Kanji |
| alphasyllabic | consonant + inherent vowel | Devanagari |

- **Arabic**: root only, vowels omitted. **Hangul**: letter shapes encode articulatory features. **Script is not language**: Arabic script also Persian, Urdu; Serbian, Kurdish, Uzbek, Punjabi use two scripts. Own script: Armenian, Georgian, Greek.

!! **Traps:** "Kanji is syllabic" (false: logographic). "logographs carry much sound information" (false: little).

### Code pages and Unicode
- **Code pages** (Latin-1, Shift-JIS, GBK, KOI8-R): ASCII's 128 (or 256) byte codes read per script. **−** wrong decoding: gibberish; one script per document.
- **Unicode**: one universal character set; each character of the set gets one codepoint `U+hex`. Defines meaning, not bytes (the encoding stores it). Superset of ASCII (U+0000 to U+007F), so English-only Latin-1 is compatible. U+0000 to U+10FFFF: 1.1M+ codepoints in 17 planes of 65,536 (slide typo 65,356).
- **Planes**: 0 BMP (most modern languages, most common CJK, Private Use Area); 1 historic and some modern scripts, music, emoji; 2, 3 more CJK; 15, 16 private use.

!! **Traps:** "Unicode specifies the bytes" (false: the encoding does). "emoji are in the BMP" (false: plane 1). "all CJK is in the BMP" (false: the most common).

### Codepoints and normalisation
- **Codepoint**: abstract number, unit of identity. **Character/grapheme**: what a reader perceives as one unit, one or more codepoints. **Glyph**: stylistic rendering. café: NFC `U+00E9` (4 codepoints) or NFD `e`+`U+0301` (5); naive comparison fails.
- **Forms**: C composed, D decomposed; NFC, NFD canonical, non-lossy; NFKC, NFKD compatibility, lossy. **NFD**: base + combining marks; keeps semantic and visual equivalence. **NFKD**: most basic parts, splits ligatures; styling not reconstructable (ſ → s, kept by NFC/NFD). **Use** NFD to compare rendered text; NFKD for search indexing, NLP analysis (not necessarily generation).
- **Crawled text** mixes NFC and NFD: dedup, search, string equality, prefix matching go wrong. Fix: normalise consistently (e.g. decompose, then compose).

!! **Traps:** "NFD is lossy" (false: NFKD is). "use NFKD to compare rendered text" (false: NFD). "NFC splits into base + marks" (false: NFD). "crawled Unicode is consistently normalised" (false).

### UTF-8, UTF-16, UTF-32
- **UTF-8**: 1 to 4 bytes per codepoint, ASCII compatible; web, Linux, NLP tools; most common. **UTF-16**: 2 or 4, not ASCII compatible; Java, JavaScript, Windows. **UTF-32**: exactly 4, not ASCII compatible; rare, easy indexing, memory-hungry.
- **UTF-8 byte count**: up to U+007F 1 byte (7 bits); U+07FF 2 (11); U+FFFF 3 (16); U+10FFFF 4 (21). Beyond the BMP: 4 bytes in UTF-8 and UTF-16. Lead bits: `0` single byte (ASCII); `11` first byte, number of 1s = length; `10` continuation. A 1, é 2, 中 (most CJK) 3, emoji 4.

!! **Traps:** "UTF-16 is ASCII compatible" (false). "UTF-16 uses exactly 2 bytes" (false: 2 or 4). "a byte starting 10 begins a sequence" (false: continuation). "most CJK takes 2 bytes in UTF-8" (false: 3).

### Bytes, breaking and other pitfalls
- **Byte inflation** (bytes/char): hello 1.0, café 1.25, 中文 3.0. Byte-level tokenisers: more bytes, splitting tends to be more aggressive; can create non-existing characters.
- **Breaking**: bytes can split a codepoint; codepoints can split base from diacritic; graphemes ideal but very language-specific. Mixed strategies misalign spans (SQuAD-style QA) and cross-corpus counts. **Turkish** lowercase(I) = ı. Codepoints not always alphabetical; Swedish å, ä, ö after z; German ä → ae. Punctuation (“” «») hand-mapped.

!! **Traps:** "more bytes, less splitting" (false). "codepoint breaking keeps base and diacritic together" (false). "grapheme breaking is language-independent" (false).

## L03 Morphology and word formation

### Vocabulary, lexicon, short-lists
- **Vocabulary**: word inventory. **Lexicon**: meanings and functions of words. **Dictionary**: written lexicon.
- **WordNet**: hand-crafted (hyponym *car :: vehicle* is-a, meronym *wheel :: car* part-whole); expensive, incomplete, language-specific.
- **Fixed size**: all weight matrices, input embeddings and output layer included; output layer costs significant memory.
- **Short-list**: keep the $n$ **most frequent** words (10K to 200K), map all rare words to `<unk>`. **−** all rare words get **equal probability** in a given context. Words are indices (*move*/*moved* unrelated); morphology helps unseen words.

!! **Traps:** "a lexicon is the word inventory" (false: vocabulary; lexicon = meanings, functions). "a short-list keeps rare words distinct" (false: all become `<unk>`).

### Word segmentation: maximum matching
- **Word segmentation / tokenisation**: boundaries where none are written (Chinese, Japanese, Thai, Burmese); **before** POS tagging, parsing, anything else. Ambiguity is **genuine**. **vs subword segmentation:** for rich morphology.
- **FMM** (left to right) / **BMM** (right to left, prepends): precompiled vocabulary $V$, max word length $L$; take the **longest** word in $V$; a single character is **always accepted**. Greedy. **+** simple, strong baseline, fast fallback. **Often disagree** on ambiguous strings: 研究生活, $V$ = {研究, 研究生, 生活, 生, 活}: FMM 研究生 | 活, BMM 研究 | 生活.
- **Bidirectional**: run both; agree = accept with high confidence; disagree = **tie-breaking heuristic**: fewer words or fewer single-char words (word count ties above; single-char picks BMM).
- **Limits** (bidirectional too): labour-intensive vocabulary; new words (names, typos) **cannot be matched**, **break silently** (single characters or mismatched); **no mechanism to exploit data**.

!! **Traps:** "FMM and BMM give the same output for the same vocabulary" (false: often disagree). "bidirectional matching uses probabilistic evidence" (false: heuristic; HMM does). "unknown words are flagged" (false: break silently). "maximum matching improves with more data" (false).

### BMES tagging with an HMM and Viterbi
- **BMES**: per character. **B** first of multi-char word, **M** interior (words of **3+** chars), **E** last of multi-char word, **S** one-char word. Boundary **after every E and S**. Solvers: HMM, CRF, any neural labeller.
- **HMM**: **hidden** tags generate **observed** characters. Transition $P(t_i \mid t_{i-1})$ (tag bigrams; B never followed by B). Emission $P(c_i \mid t_i)$: **character given tag**. Ideally labelled data (MLE counts); **can** train unsupervised (Baum-Welch, EM), **quality drops**.
- **Viterbi**: dynamic programming (scoring all labellings intractable): $s[i][t] = \max_{t'} s[i-1][t']\,P(t \mid t')\,P(c_i \mid t)$ = **best** path ending at $i$ with tag $t$; backpointers. $O(n \times |T|^2)$.
- **HMM vs FMM/BMM**: **+** **no fixed dictionary** (characters); ambiguity resolved by **probabilistic evidence**; trainable from **any** segmented corpus, **even another domain**. **−** first-order: tag depends only on **preceding tag and current character** (higher orders possible, more involved); **no longer-range context**: motivates **CRFs**.

!! **Traps:** "emission = tag given character" (false: character given tag). "characters are hidden" (false: tags). "Viterbi scores every tag sequence" (false: best path per cell). "$O(n^2 \times |T|)$" (false: $O(n \times |T|^2)$). "HMM needs in-domain data" (false).

### Word formation and morphemes
- **Inflection**: **retains** POS, changes number, tense, case, mood, aspect (*see/saw*). **Derivation**: changes POS **and/or** meaning (*happy/happiness*; *perfect/imperfect* keeps POS). **Compounding**: several words into one (*Tischbein* N+N, *sichergehen* adj+verb). **All three very productive** (iterative, new combinations): vocabulary grows a lot. English case **only on pronouns**, Russian on **all** nouns.
- **Morpheme**: smallest meaning-carrying part. **Root**: **free**. **Affix**: **bound**: **prefix** (English: mostly one), **suffix** (can be several), **infix** (*passersby*), **circumfix** (*ge-seh-en*, front and back simultaneously).
- **Concatenative** (all affixes above) **vs non-concatenative**: **apophony** *Buch → Bücher*.

!! **Traps:** "derivation always changes the POS" (false: and/or meaning). "compounding only combines nouns" (false: *sichergehen*). "a root is bound" (false: free). "apophony is concatenative" (false).

### Morphological typology

| type | how | languages |
|---|---|---|
| isolating | words tend not to change form; word order, particles | Chinese, Vietnamese |
| agglutinative | morphemes strung together, one clear meaning each | Turkish, Finnish |
| fusional | one affix or pattern, several features, often not separable | Russian, Spanish, Arabic |
| polysynthetic | verb, subject, object, modifiers in one word | Inuktitut, Mohawk |

- **Agglutinative vs fusional**: **separability**. Agglutinative joins may change characters; Russian *-ami* fuses case, number, gender.
- **Word-level NLP breaks**: a lemma can have **thousands** of forms, **massive OOV** with an English-sized vocabulary; analysers **highly language dependent**, so **subword tokenisation**.

!! **Traps:** "Arabic is agglutinative" (false: fusional). "Finnish is fusional" (false: agglutinative). "agglutinative morphemes never change form" (false).

## L04 Subword segmentation

### Why subwords, FSTs and Morfessor
- **Vocabulary**: one of the main bottlenecks of NMT; a word's representation depends on its frequency **and** number of distinct contexts. **Whole words −** huge vocabularies; short-lists map rare words to `<unk>`.
- **Subword**: character n-gram or morphologically meaningful unit (language dependent). Handling rare and OOV words is one main motivation.
- **FST** (historical, hand-built): morpheme in, analysis out. **+** powerful, efficient, cases and exceptions with great accuracy. **−** extremely laborious, totally language dependent, no preference among ambiguous analyses (probabilistic FSTs exist).
- **Morfessor** (Creutz and Lagus 2002): first data-driven; **MDL** = cost of generating the data **plus** cost of the model. **Current methods** (BPE, WordPiece, Unigram LM): frequency-based, no annotation or linguistic knowledge, no labels, strictly concatenative (no character substitutions or deletions). **+** fitted to a desired vocabulary size.

!! **Traps:** "FSTs handle exceptions poorly" (false: great accuracy). "MDL minimises model size" (false: model plus data cost). "data-driven segmenters handle character deletions" (false: strictly concatenative).

### BPE and WordPiece
- **BPE training**: characters, word boundaries kept (`</w>`); merge the **most frequent** adjacent pair until a **maximum number of merges** (the only hyperparameter; 30K to 128K, high end multilingual). **Greedy** (once merged, stays merged), **deterministic** (one segmentation per word).
- **BPE inference**: characters, then the learned merges **in learned order**. **+** much smaller vocabulary (memory, speed), better translation, far fewer OOVs, for **both** high- and low-resource pairs.
- **Sennrich EN→DE**: best single: char-bigram **ties** joint BPE; best ensemble: char-bigram; per-language BPE single **below** best word-level; syntax-based beats **every** single model, only subword ensembles beat it.
- **WordPiece** (Schuster and Nakajima 2012): **bottom-up** like BPE, and **predates** it. Non-initial characters prefixed `##`; merge the pair with highest $\text{count}(a,b)/(\text{count}(a)\,\text{count}(b))$ until the target vocabulary size. **Inference**: left to right **greedy longest match** (no merge replay); no match at a position: `[UNK]` (L06 table: whole word).

!! **Traps:** "at inference BPE merges the input's most frequent pair" (false: replays learned merges). "WordPiece merges the most frequent pair" (false: count ratio).

### Unigram LM and DP segmentation
- **Unigram LM** (Kudo 2018, in SentencePiece): considers **all** segmentations, best **global** one, **multiple** (sampling, n-best), **top-down** (large vocabulary shrunk to target).
- **Pruning**: start $V$ = substrings occurring more than once, **not crossing words**. While $|V|>k$: fit $\theta$; $L_t = p_\theta(D) - p_{\theta'}(D)$ ($\theta'$ without $t$); remove $\min(|V|-k, \lfloor\alpha|V|\rfloor)$ tokens with **highest** $L_t$ (exam answer: slide; Kudo's paper: lowest); $\alpha\in[0,1]$ hyperparameter.
- **Unigram inference**: **Viterbi** gives only the best; n-best via **Yen's/Eppstein's** algorithms or disallowing sub-solutions.
- **Brute force**: $2^{n-1}$ segmentations, each scored. **DP**: solve each sub-problem once, store it: **extra memory to save time**. Top-down memoization and bottom-up: **same asymptotic** run time. $m[i,j]$ = $p$(whole span), replaced by $m[i,k]\cdot m[k+1,j]$ (**product**; segments independent) if larger.

!! **Traps:** "starts from single characters" (false: substrings occurring more than once). "all surplus tokens removed at once" (false: at most $\lfloor\alpha|V|\rfloor$ per round, refit between). "Viterbi returns n-best" (false). "bottom-up DP is asymptotically faster" (false: same). "DP saves time and memory" (false: spends memory).

### SentencePiece, OOV and byte-level BPE
- **SentencePiece**: a **toolkit** (BPE **and** Unigram LM). Standard BPE assumes word boundary detection; SentencePiece reads a **raw Unicode stream including spaces**, space escaped as `▁`. **+** **lossless** desegmentation, languages without word boundaries.
- **OOV**: BPE and Unigram LM **very low**, not zero: typos **sometimes**, unknown characters **will**.
- **Byte-level BPE**: BPE on **UTF-8 bytes**, base alphabet fixed at **256**: **zero OOV guaranteed**, even unseen scripts. **−** non-Latin, non-ASCII scripts: more bytes per character, extra merges, **shorter** subsegments.

!! **Traps:** "BPE and Unigram LM guarantee zero OOV" (false: byte-level only). "byte-level BPE works on Unicode characters" (false: bytes). "SentencePiece is only Unigram LM" (false: BPE too). "non-ASCII scripts get longer subsegments" (false: shorter).

### BPE, WordPiece, Unigram LM compared

| | BPE | WordPiece | Unigram LM |
|---|---|---|---|
| builds | merges up | merges up | prunes |
| criterion | pair count | count ratio | likelihood |
| new word | replay | longest match | Viterbi/sample |
| fallback (L06) | char/byte | word → `[UNK]` | char |
| marker | `@@`, `</w>` | `##` | `▁` = space |
| output | one | one | one or sampled |
| used in | GPT, RoBERTa, XLM | BERT, mBERT | T5, XLM-R, ALBERT |

### BPE vs Unigram LM (Bostrom and Durrett 2020)
- **Segments**: Unigram LM closer to morphemes, **longer** (English **and** Japanese), more moderate-frequency tokens (English); BPE more word-initial capitals (`▁H`), Unigram LM more suffixes (*s, ing, ed*); in Japanese BPE attaches particles to words.
- **Results**: vs gold segmentations Unigram LM wins precision, recall **and** F1 in **both**, gap **much larger in English**. Downstream ahead on **every** task: **slightly** in English, **a lot** in Japanese.

### Fertility
- **Fertility**: average segments per word; depends on merges (BPE) or vocabulary size (Unigram LM), word length, morphological richness. Isolating, Latin-script: **typically close to 1** for common words. No scheme dominates across all languages and scripts. **High fertility −** harder to reassemble meaning, **pays more** under per-token pricing, context fills **faster**.

!! **Traps:** "isolating languages have high fertility" (false: near 1). "high-fertility languages fill context more slowly" (false: faster).

## L05 Static embeddings

### Count vectors and embeddings
- **Count vectors** (distributional semantics): context-word counts in a window (or sentence, document); context and target vocabularies may be the same; cosine. High-dimensional, discrete, sparse. **+** unsupervised; interpretable dimensions. **−** sparse despite stemming; no signal vs noise criterion.
- **Embeddings**: low-dimensional, dense, continuous, learned by a prediction task, usually as a by-product. **Representation learning**: embeddings are the main objective.

!! **Traps:** "embeddings are high-dimensional and sparse" (false: count vectors).

### Word2Vec: CBOW and skip-gram
- **CBOW**: centre word from context; $h = \frac{1}{2n} W w_C$ (average; figure says SUM); one hidden layer, **no non-linearities**, simpler than PNLM; $n \in \{2,5,10\}$.
- **Both**: $\text{softmax}(W'h)$. $W'$ **not necessarily** $W^\top$. Embedding: typically $W$, or $W^\top + W'$.

| | CBOW | skip-gram |
|---|---|---|
| predicts | centre from context | each context word from centre |
| hidden | average of context | centre embedding $W w_I$ |
| outputs | one | $2n$, shared $W'$ |
| assumes | order ignored | context words independent |

!! **Traps:** "skip-gram predicts the centre word" (false: CBOW). "CBOW has a non-linearity" (false). "Word2Vec requires $W' = W^\top$" (false). "embeddings only from $W'$" (false: typically $W$).

### Negative sampling
- **Negative sampling**: full output layer too costly, so classify whether a word occurs in the input word's context: **1 positive** + **$k$ negatives** (same $w$, noise word as $c$); $p(D{=}1) = \sigma(v_c \cdot v_w)$.
- **Parameters**: $k$ = **5 to 20**; unigram$^{\mathbf{0.75}}$ (biases towards **rarer** words); context **typically 2 to 5**; **more data → smaller** context, fewer negatives.

!! **Traps:** "negatives drawn uniformly" (false). "0.75 shifts mass to frequent words" (false: to rarer). "more data needs more negatives" (false). "negative sampling uses a softmax over the vocabulary" (false: sigmoid of dot product).

### GloVe
- **GloVe**: combines global co-occurrence statistics (count-based, tends to do worse) with local prediction-based updating (each window an independent event, still tends to win).
- **Objective**: $\sum f(X_{ij})(w_i\cdot\tilde w_j + b_i + \tilde b_j - \log X_{ij})^2$, weighted least squares on log counts. $X$ built in **one pass**; training sees only $X$. **Final vector $w_i + \tilde w_i$.**
- **$f$**: $(x/x_{max})^\alpha$ below $x_{max}$, else 1; $x_{max} = 100$, $\alpha = 0.75$. De-emphasises very rare (noisy) **and** very frequent pairs: rises from zero, **caps at 1**.

!! **Traps:** "GloVe treats windows as independent events" (false: Word2Vec). "count-based outperforms prediction-based" (false). "$f$ gives frequent pairs the lowest weight" (false: capped at 1). "GloVe subsamples frequent words" (false: Word2Vec; GloVe uses $f$).

### Evaluating embeddings
- **Intrinsic**: embeddings alone; **extrinsic**: downstream (MT, IR). **Qualitative**: examples; **quantitative**: overall score. **Word similarity**: **Spearman** with human rankings. **WS-353 = relatedness**, **SimLex-999 = similarity**. **Analogies**: **accuracy**, incl. **syntactic**. **Visualisation** (PCA; t-SNE hyperparameters matter a lot): **complements, does not substitute** quantitative or extrinsic evaluation.

!! **Traps:** "SimLex-999 annotates relatedness" (false: WS-353). "similarity scored by accuracy" (false: Spearman).

<!-- side 2 -->

## L05 Static embeddings (cont.)

### fastText
- **fastText**: remedies one opaque vector per word type (morphologically rich languages suffer most, no OOV fallback): skip-gram with negative sampling; **centre word = sum of character n-gram vectors** (n default **3 to 6**, `<` `>` boundaries) **plus whole word** `<w>`. **Context vectors stay per word.** Same gradient to every n-gram. **Returns the n-gram vectors**; **OOV words get vectors.**
- **Results**: similarity best or tied except **English WS-353** (CBOW wins); gains in morphologically rich languages and English rare words. Analogies: gains on **syntactic**; **semantic** barely change, drop for German.

!! **Traps:** "context words also built from n-grams" (false). "cannot represent unseen words" (false). "largest gains on semantic analogies" (false: syntactic).

### Crosslingual embeddings
- **Shared space hypothesis**: train separately, jointly or on aligned data, alignment needed either way. Similar relational structure, **approximate isomorphism** (same cloud shape; orientation, scale may differ). Only a hypothesis. If true, **one general linear map** aligns them (linear because analogies are already linear in one space).
- **Supervision**: **supervised** thousands of seed pairs (top frequent words; words only, no phrases); **semi-supervised** hundreds, bootstraps by confidence; **unsupervised** none, adversarial.

!! **Traps:** "spaces differ only by a rotation" (false: general linear map). "semi-supervised uses thousands" (false: hundreds).

### Supervised mapping: Procrustes, orthogonal
- **Mikolov et al. (2013)**: $\min_W \sum_i \lVert Wx_i - y_i\rVert^2$, **Procrustes** (exam answer: slide; Conneau: only the orthogonal SVD one); exact closed form exists, used **SGD** (scales better, memory-friendlier). Worse for rarer words. **−** stretches, shears; overfits the seed dictionary; Euclidean training vs cosine retrieval (equal only for unit vectors).
- **Xing et al. (2015)**: **orthogonal** $W^\top W = I$, unit vectors; rotations and reflections only, preserves distances and angles; exact by **SVD**. Better at every dimension; unconstrained **worsens 300 to 700 dims** (equal dims), orthogonal slightly improves.

!! **Traps:** "orthogonal maps can rescale" (false). "unconstrained improves with dimension" (false, equal dims). "orthogonal needs gradient descent" (false: SVD).

### Hubs and CSLS
- **Hubness**: few vectors are NN of many points regardless of similarity; high-dimensional geometry, **not specific to embeddings**, **worse with more dimensions**.
- **CSLS**: $2\cos(x,y) - r_T(x) - r_S(y)$; $r_T(x)$ = mean cosine of $x$ to its $K$ NN in the **target** space, $r_S(y)$ in the **source** space. **Penalises** hubs; **no parameter tuning**; beats plain NN on every pair.

!! **Traps:** "CSLS boosts candidates near many points" (false). "$r_T$ uses the source space" (false).

### Unsupervised alignment: MUSE
- **Generator**: $W$, one $d \times d$ **linear** matrix, deliberately (a deep one could scramble which word maps where); starts as identity (or random orthogonal); **re-orthogonalised** after each update.
- **Discriminator**: MLP, label smoothing; **50k most frequent** words; alternating steps, other side frozen; generator uses **flipped labels**.
- **Model selection**: not adversarial loss; mean cosine of CSLS NN of **10k most frequent** source words. **Refinement** ("semi-supervised" on slide, no human dictionary): **mutual** CSLS NN among frequent words, SVD, repeat.
- **Results**: on par or better than supervised for English-Spanish, -French, -German; well short for -Russian, -Chinese.

!! **Traps:** "generator is a deep network" (false). "checkpoint by discriminator loss" (false). "refinement needs a seed dictionary" (false). "always worse than supervised" (false).

### Where unsupervised fails (Søgaard et al. 2018)
- **Marking** of grammar: on dependent (possessor), on head (verb), both (double), neither (zero).
- **Fails**: English-**Estonian, Finnish, Greek** (mixed/double marking): grammar spread over many rarer forms, **non-isomorphic**. **Identity baseline** (identically spelled words): beats adversarial on **every English pair**; **Estonian-Finnish** is the exception. **Not data size**: far more Finnish data, still zero.
- **Domain mismatch**: adversarial fails completely, even English-Spanish; identity degrades but survives. **Model**: window, n-gram changes barely matter; **skip-gram on one side, CBOW on the other fails**.
- **Eigenvector similarity**: NN graph; Laplacian $A - D$ (exam answer: slide; standard $D - A$); largest $n\%$ eigenvalues (bigger = denser connectivity); $\Delta$ = sum of squared differences. Bigger $\Delta$ = less isomorphic, no rotation fixes it; no alignment needed. **One direction only**: low $\Delta$ does not prove isomorphism. Highest for the failing pairs.

!! **Traps:** "a bigger Finnish corpus fixes it" (false). "fails for every Finnish pair" (false). "identity beats adversarial on every pair" (false: every English pair). "low $\Delta$ shows isomorphism" (false). "window size breaks alignment" (false: skip-gram/CBOW mismatch does).

## L06 Contextual embeddings

### Static and contextual representations
- **Static embedding**: one vector per word type. Static from Word2Vec (limited window) and also as a **by-product** of a deep task model (classification, MT) reading the whole sentence.
- **Contextual representation**: the hidden states above a word, different per occurrence. **Balance**: enough context to tell occurrences apart, not so much that the word's own contribution gets unclear. **Strategies**: right layer(s), right task(s), residual layers.

!! **Traps:** "embeddings learned inside an MT model are contextual" (false: the embedding layer is static). "more context always gives a better word representation" (false: the word's contribution gets unclear).

### Pre-training and fine-tuning
- **Pre-training**: model of wider context (LSTM, Transformer **or CNN**) on a general task with very large data (word prediction); updates $\theta$. **Fine-tuning**: adds task model $\theta'$ for the real task; updates $\theta'$ **and maybe** $\theta$. **Feat** (feature extraction): encoder frozen, only the head trained.
- **Encoder-decoder**: original Transformer (MT, summarisation, QA). **Encoder-only**: predicts its own noised input (purest form: auto-encoder), **BERT**. **Decoder-only**: next-word prediction, **all** current LLMs; can frame encoder-decoder tasks.

!! **Traps:** "fine-tuning always updates $\theta$" (false: $\theta'$ always, $\theta$ maybe). "MT requires an encoder-decoder" (false: decoder-only can frame it). "contextual pre-training requires a Transformer" (false: LSTM, CNN too).

### BERT pre-training
- **BERT**: Transformer **encoder**. Pre-training **self-supervised** (missing words), task relevance **not required**; fine-tuning **supervised**, works with small data. Input = token + segment ($E_A$/$E_B$) + position embeddings.
- **MLM**: tokens **selected** with $p_{mask}=0.15$; of those **80%** `[MASK]`, **10%** random, **10%** unchanged. **Loss at all selected positions**, only there. Masks drawn once in pre-processing (**static**).
- **`[CLS]`**: not tied to a word, **never masked**, represents the **entire** input; loss from a classification task (NSP). **NSP**: **50%** B follows A, **50%** B random **from the corpus**.
- **Sizes**: base 110M, 12 layers, 768; large 340M, 24 layers, 1024; **16 heads both** (exam answer: slide; paper: 12 for base). BookCorpus + English Wikipedia, WordPiece.

!! **Traps:** "15% of tokens are replaced by `[MASK]`" (false: 80% of the 15%). "loss only at `[MASK]` positions" (false: all selected). "`[CLS]` is masked like other tokens" (false). "NSP negatives use the preceding sentence" (false: random).

### BERT fine-tuning and whole-word masking
- **Fine-tuning head**: small-ish, **randomly initialised**; word tasks on **top encoder layers**, classification on the **`[CLS]` top layer**; BERT updated or not. **2 to 4 epochs**.
- **NLI**: entails / contradicts / no relationship, 3-way. **XNLI**: crowd-sourced test/dev pairs **translated** into 14 languages. **SQuAD**: **start/end span**; **one** start vector $S$, **one** end $E$, span maximises $T_i\cdot S+T_j\cdot E$, $i\le j$.
- **Whole-word masking**: mask all pieces of a word; **rate unchanged**. Pieces carrying **morphology** (tense, number, case) are **easier** to predict than stems. Helps BERT-Large, cased and uncased.

!! **Traps:** "head initialised from the pre-trained model" (false: random). "whole-word masking raises the masking rate" (false). "a start/end vector per position" (false: one shared). "suffix pieces are harder than stems" (false: easier).

### mBERT and language sampling
- **mBERT**: **Same architecture and loss** as BERT (MLM + NSP), 104 concatenated Wikipedias. NSP pairs **within one language**, no mixing in a sequence; a **batch can** mix languages. **Lacks**: crosslingual loss, parallel data (none intentionally), language-ID embedding, language-specific parameters (adapters), alignment loss.
- **Shared vocabulary**: WordPiece (L04) **~120k** (English BERT 30k); BPE or Unigram LM work **similarly well**. ~178M vs 110M parameters: **only the vocabulary**. Chinese, kanji, **hanja** split into characters first. The $\alpha$ smoothing (below) also applies to subword frequencies; still favours **high-resource and Latin script**; low-resource, morphologically rich: **higher fertility**. **Uncased** variant lowercases and **strips accents** (hurts diacritic languages).
- **Sampling** $p_l=n_l^\alpha/\sum_{l'}n_{l'}^\alpha$, $n_l$ = resource size, $\alpha$ = importance (smoothing; $1/T$ in L07).

| $\alpha$ | effect |
|---|---|
| 1 | proportional: English, large Wikipedias dominate |
| 0.7 | up-weights low-resource, down-weights high |
| 0 | uniform: tiny Wikipedias oversampled, memorised |

!! **Traps:** "mBERT has a language-ID embedding" (false). "NSP pairs mix languages" (false). "more parameters because more layers" (false: vocabulary). "$\alpha=0$ is proportional" (false: uniform). "$\alpha=0.7$ down-weights low-resource" (false).

### Zero-shot crosslingual transfer findings
- **Zero-shot**: task data in A improves the task in B with **no** task data in B.
- **Pires 2019** (NER, POS, 16 languages): zero-shot decent, below in-language, **asymmetric** (en→nl beats nl→en). **Cross-script works**; pairs with Japanese drop (maybe typology). Jaccard overlap of **named-entity** wordpieces: English BERT rises with it, **mBERT high even at low overlap**. From SVO, SVO targets beat SOV; more shared WALS features, better POS. **Translation retrieval**: **peaks in middle layers**.
- **Wu and Dredze 2019** (from English): XNLI translate-train **beats** zero-shot mBERT, which loses even to XLM MLM-only; NER **beats** Xie et al.; POS **below** Kim et al. **Freezing** lowest $k$ layers ($k$ = 0 is embeddings): $k\le6$ mostly neutral or **helps**; $k=9$ hurts NLI, NER, POS (MLDoc gains); no fine-tuning (Feat) below best fine-tuned. **Overlap vs zero-shot** per task: NER near-perfect correlation, XNLI **none**, others positive.
- **K et al. 2020** (bilingual BERT, XNLI): **fake English** (zero overlap) costs **little**: overlap not the main cause. **Permuting words** in pre-training only: significant drop, transfer **not eliminated**. **Depth** has the largest architecture impact (parameters ~constant): in-language saturates, zero-shot keeps rising. **Mixed-language** premise/hypothesis: below **both** monolingual pairs.
- **Rajaee and Monz 2024** (mBERT, XNLI 15×15): **across < within on average for every language** (single English-hypothesis cells can beat within); same-language premise best per hypothesis language; foreign premise + **English hypothesis** beats the reverse. **Hypothesis**: transfer of **heuristics can contribute**.
- **What drives transfer**: shared vocabulary **plays a role, not the main cause**; structure matters (permutation hurts, does not eliminate); deeper = **more language-independent**; **no single factor**.

!! **Traps:** "mBERT transfers poorly across scripts" (false). "retrieval best at the top layer" (false: middle). "removing subword overlap eliminates transfer" (false: small cost). "heuristics are the main cause" (false: hypothesis that they can contribute). "shared vocabulary plays no role" (false).

### XLM-R
- **XLM-R** (Conneau 2020): RoBERTa-style mBERT. **Drops NSP** (little use, sometimes hurts): MLM only. **Dynamic masking** (new mask each pass) **vs static** (once, in pre-processing). **CC-100** (filtered CommonCrawl, **100** languages, ~2 orders of magnitude more data). **250k** Unigram LM (SentencePiece), double mBERT's, better fertility. Still **no crosslingual signal**.
- **XLM-R vs mBERT** within and across: see L07.

!! **Traps:** "XLM-R covers more languages" (false: 100 vs 104). "XLM-R keeps NSP" (false: MLM only).

## L07 Crosslingual NLP

### Multilingual vs crosslingual, parallel data
- **Multilingual model** (mBERT, XLM-R): one model and subword vocabulary, MLM context **within the same language**. **Nothing explicitly links languages**: predicting $W$ in $L$ gains nothing from unrelated context in $L'$. mBERT sees no translation pair; alignment is an **emergent side effect**.
- **Crosslingual knowledge transfer**: fine-tune on a task in A, test in B. Zero-shot results **seem to support** it emerging without crosslingual signal; crosslingual tasks **cast doubt**.
- **Parallel data**: **meaning-equivalent** sentence pairs. **vs comparable data:** same **topic**, not fully parallel, a **sliding scale** (news, Wikipedia).
- **Sources**: UN, EU, news, **OPUS**; crawl (**document, then sentence alignment**, or straight from Common Crawl). Equivalence: dictionary overlap, relative length (old); **sentence embeddings** (recent). **vecalign**: LASER matching at massive scale.

!! **Traps:** "joint training on many languages gives an explicit crosslingual signal" (false: MLM stays in one language). "comparable data is translation pairs" (false: parallel).

### Pretrained models compared

| model | data | objective | vocab; language ID |
|---|---|---|---|
| mBERT | Wikipedia | MLM + NSP | WordPiece ~120k |
| XLM-R | CC-100 | MLM | Unigram 250k |
| LASER | parallel | **translate** | BPE; $L_{id}$ to decoder |
| XLM | Wiki + parallel | MLM or CLM + **TLM** | BPE; lang. embedding |
| InfoXLM | CC-100 + parallel | MMLM + **TLM** + **XLCo** | XLM-R's |
| BART | English | denoising | |
| mBART | monolingual, many | denoising | ID token `<En>` |

- **Bold** = crosslingual signal; no language ID unless listed. XLM from scratch; InfoXLM from XLM-R.

### LASER
- **LASER**: encoder-decoder trained to translate on parallel data. **Encoder**: shared BPE embeddings, **BiLSTM** stack, **max pooling** into one sentence embedding; never told the input language. **Decoder** (LSTM): sentence embedding + previous token + **language ID $L_{id}$** of the output; later **discarded**.

!! **Traps:** "LASER's encoder gets the input language ID" (false: decoder, output language). "LASER uses `[CLS]`" (false: max pooling).

### XLM: CLM, MLM, TLM
- **CLM**: next word from prefix. **MLM**: masked word from context (BERT). **TLM**: MLM's loss on sentence + translation concatenated, masks in **both** halves, so a masked word can be read off its translation. **Positions restart at 0** in the second sentence; **language embeddings** mark each half.
- **TLM is the crosslingual objective**, always paired: **MLM + TLM** or **CLM + TLM**. **NSP dropped**.
- **XNLI**: **translate-train** (MT training set), **translate-test** (MT test into English), **zero-shot** (English only). MLM + TLM beats MLM in **every** language (least for English). Zero-shot beats translate-test **on average only** (translate-test wins every European language, tr, ar). **Translate-train best on average**.
- **Nepali LM** gains more from **Hindi** (related) than English; Hindi + English together best.

!! **Traps:** "XLM keeps NSP" (false). "TLM is used alone" (false: with MLM or CLM). "TLM positions continue from the first sentence" (false: restart at 0). "zero-shot beats translate-test in every language" (false: on average).

### InfoXLM
- **InfoXLM**: **initialised from XLM-R** plus XLM's parallel-data ideas; three losses, **equally weighted**. **MMLM**: like MLM, but instead of directly optimising the ground-truth loss uses **negative sampling** (exam answer: slide; paper: plain MLM). **TLM** as in XLM, but **no language embeddings**. **XLCo**: `[CLS]` vectors of a parallel pair, **classify whether parallel**, negatives = **random sentences**.
- **TLM vs XLCo:** TLM aligns **tokens**, XLCo **sentences**; XLM is token level only.

!! **Traps:** "InfoXLM is trained from scratch" (false). "InfoXLM uses language embeddings" (false). "XLCo predicts masked words from the translation" (false: TLM).

### Within vs across languages
- **Within vs across**: fine-tuned on English; input parts in one language or two. **Every model lower across**. XLM-R beats mBERT within on both tasks (on average); across, better on XNLI but **worse on QA** (works mainly with an **English question**). **InfoXLM**: slightly better within, much better across, **smallest gap**, biggest gain on QA. Grids **not symmetric**.

!! **Traps:** "XLM-R beats mBERT across languages on both tasks" (false: worse on QA). "InfoXLM gains mainly within a language" (false: across).

### Seq2seq and NMT
- **Seq2seq vs sequence labelling**: **no isomorphic** $x_t$, $y_t$ relation; lengths **may differ**. $p(y_t \mid Y_{<t}, X)$: $X$ = encoder output, $Y_{<t}$ = decoder prefix.
- **Sutskever 2014**: LSTM encoder and decoder, decoder initialised with the **last** encoder state: **information bottleneck**; Transformer's **source-target context** layer (cross-attention) removes it.
- **NMT quality**: data **amount, variation, relevance**. **86%** of directions poor (limited parallel data); generalises not very well beyond training data, **not at all** beyond specific directions. Small data: NMT **far below** phrase-based, ahead only when large.

!! **Traps:** "seq2seq assumes equal lengths" (false). "Sutskever's decoder attends to every encoder state" (false: last state).

### Parent-child transfer
- **Transfer learning** = basically **fine-tuning**: big pair (**parent**), then continue on low-resource pair (**child**). **Works best if**: **identical target**, **related source**, **decoder frozen**, shared vocabulary.
- **Zoph**: **not shared**: source vocabulary, source embeddings; **shared**: target language, all else; freeze target-specific.
- **Freezing**: all frozen = zero; **source embeddings** biggest jump; source RNN, target RNN, attention keep helping; unfreezing **target embeddings hurts** (slide's "decoder frozen").
- **Relatedness**: French parent beats German for a Spanish child; reshuffled-vocabulary French still gains: **structure transfers**. **Kocmi and Bojar**: shared vocabulary, **unrelated** parent helps as much, even with a different target. Reversed (small parent): little or **hurts**. **Aji**: **inner layers** carry most; **embeddings alone worse than random init** on average.
- **Fine-tuning** (sample exam, no official key): starts from general-domain parameters; more in-domain updates help, then hurt; suffers catastrophic forgetting.

!! **Traps:** "Zoph's parent and child share source embeddings" (false). "unfreezing target embeddings helps" (false). "with a shared vocabulary only a related parent helps" (false).

### Multilingual NMT
- **Johnson 2017**: **one system** for all directions; transfer **in parallel** (parent-child: in stages). **English-centric** (xx→en, en→xx). **Target language tag** (`<2es>`) prepended to the source. **All parameters shared**; one WordPiece vocabulary; source, target, output embeddings shared; directions mixed within a batch.
- **Many-to-one**, **one-to-many**: training; **many-to-many: testing only** (zero-shot). Vs bilingual: many-to-one better everywhere (small); one-to-many **mixed**. **Signs of transfer**: zero-shot quality; beating the single-direction baseline.
- **Transfer** (better from sharing) **vs interference** (worse from competing for capacity). **Arivazhagan**: low-resource gain **far more into English** than out; high-resource **lose** (no multilingual model beats bilingual); **All→All** weakest multilingual everywhere.
- **Sampling**: proportional ($T$=1): high-resource dominate; uniform ($T$=100): they suffer; best: over-sample low, under-sample high. $p_l \propto n_l^{1/T}$ (L06's $\alpha=1/T$):

| $T$ | high | medium | low |
|---|---|---|---|
| 1 | best | worst | worst, below bilingual |
| 5 | middle | best | near $T$=100 |
| 100 | worst | middle | best |

- **Zero-shot**: entirely transfer; more training languages help **in most cases**; **still lagging**.

!! **Traps:** "the tag names the source" (false: target). "uniform sampling hurts low-resource" (false: high). "higher $T$ moves towards proportional" (false: uniform).

### BART
- **BART**: **bidirectional encoder** reads corrupted text, **autoregressive decoder** rebuilds the original; monolingual data. **vs BERT** (encoder only), **GPT** (decoder only).

| noise | source | learn |
|---|---|---|
| token masking | → `[MASK]` | tokens |
| token deletion | removed, unmarked | tokens + **where** |
| text infilling | span → **one** `[MASK]` | span + **how many** |
| sentence permutation | shuffled | order |
| document rotation (slide: permutation) | random start | real start |

- **Infilling best** single noise on almost every task; deletion beats masking on generation; rotation, permutation alone poor.
- **Fine-tuning**: classification from the **last hidden time step**; span: **begin and end** labels; MT: random-init **source encoder replaces the embedding layer** of BART's encoder (**frozen or updated**; source vocabulary may differ). **Frozen slightly below** baseline, **tuned above**.

!! **Traps:** "infilling uses one `[MASK]` per token" (false: per span). "BART's decoder is bidirectional" (false). "the new encoder replaces BART's whole encoder" (false: embedding layer).

### mBART
- **mBART**: **concatenated monolingual** data; BART objective **within the same language**, **no crosslingual signal**. **Language ID token ends the encoder input, starts the decoder input**. **Fine-tuned directly** on parallel data.
- **vs random init**: better on **every** pair, both directions; gains **largest** low-to-medium; **near zero** with almost no data (10K pairs).

!! **Traps:** "mBART is pretrained on parallel data" (false). "mBART needs a new encoder for MT" (false: BART does). "mBART rescues pairs with almost no data" (false).
