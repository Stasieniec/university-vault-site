---
type: exam-prep
course: 5204MNLP6Y
status: complete
---

# MNLP - Cheat Sheet

The two-sided sheet for the exam (Thu 2026-10-22, 09:00, see [[MNLP - Exam Analysis]]). The print version is `Assets/MNLP-Cheat-Sheet.pdf`, two A4 pages, rebuilt from this note with `docs/superpowers/tools/cheat_sheet_pdf.cjs`. <!-- print-header: MNLP cheat sheet (Monz, UvA), exam Thu 22 Oct 2026. key = the answer the exam most likely expects where Monz's lecture and the paper differ. -->
Everything here comes from the lecture notes and `docs/mnlp-cheat-sheet-numbers/`. Where Monz's lecture and the paper behind it disagree, **key** marks the answer the exam most likely expects. Lectures: [[MNLP-L01 - Overview|L01]], [[MNLP-L02 - Multilinguality and Writing Systems|L02]], [[MNLP-L03 - Morphology and Word Formation|L03]], [[MNLP-L04 - Subword Segmentation|L04]], [[MNLP-L05 - Static Embeddings|L05]], [[MNLP-L06 - Contextual Embeddings|L06]], [[MNLP-L07 - Crosslingual NLP|L07]].

## L01 Overview

- Text: unstructured + discrete; speech, video: unstructured + continuous; tables: structured. **IR**: keywords in, **ranked documents** out. **QA**: well-formed question in, **the answer** (or a short list) out.
- **NER** (persons, organisations, dates, locations): seems trivial for English (still silly mistakes), much harder for other languages. **Sentiment** requires NER; sentence and document level. **Summarisation**: single or multiple documents, traditionally IR term weights. **MT**: ~50% of web pages not English; typically one sentence at a time; can fail spectacularly.
- **Before deep learning**: one approach per application; ML only **weighs** hand-built features (POS, parse trees, named entities, argument roles). **Neural**: state of the art for all applications, little or no feature engineering, few network types; needs large data, errors hard to trace. Insights transfer between AI fields.
- **LLMs** originated in NLP; used end to end (QA, summarisation, MT) and to substitute or supplement humans (annotation, evaluating outputs, dialog). Data: en 51%, zh 10%, de 7%, es 4%, ja 4%, other 24%, so English-centric. General multilingual models (Llama 3.1, DeepSeek) do better on high-resource languages; dedicated multilingual ones (Aya-Expanse) tend to lag behind.
- **Multilingual**: one system for many languages, not language-specific: NER for many languages, language-independent parsing, **classification across languages (multilingual despite "across")**. **Crosslingual**: knowledge crosses languages: MT, crosslingual QA, crosslingual unlearning, crosslingual reasoning.

## L02 Multilinguality and writing systems

- Resources are English: labelled and unlabelled data, toolkits, benchmarks. Non-English ones arise as **by-products** (translations, reviews), as **translations of English resources**, less so by dedicated effort. "Language independent" claims are usually untested.
- 7,000+ languages, 100+ with over 10M speakers. **Tier follows commercial interest and military relevance**, not speakers. High: en, Mandarin, *big gap*, fr, es, ar, ru. Medium (limited labelled, reasonable unlabelled, some tools): de, it, **nl (about 24M speakers)**, ko, ja. Low (no labelled, limited unlabelled, no tools): the other ~7,000.
- 我不知道她去哪儿了: 1 whitespace token (en: 6); 去 has no tense, past from sentence-final 了; SVO. Russian *kniga*: 12 cells (6 cases × 2 numbers), 9 distinct forms. Finnish *onnellisimmillanikin*: 6 morphemes = 4 English words.

| English-first assumption | broken by |
|---|---|
| whitespace separates words | zh, ja, Thai, Lao, Khmer, Burmese (Thai spaces end clauses) |
| word order encodes relations | case-rich: ru, fi, ko |
| one word = one meaning | agglutinative fi, tr, hu; compounding de, nl, sv |
| left to right | ar, he, ur, fa: stored left to right, rendered right to left |

- **Alphabetic** (one sound per symbol, small inventory): Latin, Cyrillic, **Hangul** (shapes encode articulation). **Syllabic**: Hiragana, Katakana. **Logographic** (words/morphemes, little sound): Hanzi, Kanji. **Devanagari alphasyllabic** (consonant + inherent vowel). **Arabic: consonants only**, vowels omitted. Japanese mixes three.
- Script is not language. Arabic script: ar, fa, ur, Pashto, Sorani. Cyrillic: ru, be, sr. Devanagari: hi, mr, ne. Ge'ez: Amharic, Tigrinya. Own scripts: Armenian, Georgian, Greek, Sinhala, Khmer. Two scripts: Serbian (Cyr/Lat), Kurdish (Lat/Arab), Uzbek (Lat/Cyr), Punjabi (Gurmukhi/Arabic).
- **ASCII** 1963, 128 codes. **Code pages** (Latin-1, Shift-JIS, GBK, KOI8-R) reinterpret the same bytes per script: wrong table gives gibberish, usually no error; no mixing scripts.
- **Unicode**: one codepoint per character, `U+hex`; defines meaning, not bytes; superset of ASCII (U+0000 to U+007F), so ASCII files are valid UTF-8 (Latin-1 é = byte E9 is not). U+0000 to U+10FFFF = 17 planes × 65,536 = 1,114,112 (slide typo "65,356"). Plane 0 BMP: most modern languages, common CJK; 1: historic scripts, music, **emoji**; 2, 3: more (rare, historic) CJK; 14: tags, variation selectors; 15, 16: private use; 4 to 13 unassigned.
- **Codepoint** = number; **character** = perceived unit, one or more codepoints; **glyph** = rendering; perceived character = **grapheme**. café NFC 4 codepoints (U+00E9), NFD 5 (e + U+0301): look identical, compare unequal.

| | composed | decomposed |
|---|---|---|
| canonical, lossless | NFC | NFD |
| compatibility (K), lossy | NFKC | NFKD |

- K forms also merge stylistic variants (ligatures): long s ſ survives NFC/NFD, becomes plain s in NFKC/NFKD. **NFD to compare rendered text; NFKD for search indexing and NLP analysis, not necessarily generation.** Crawled data mixes forms even when all Unicode, so **dedup, search, string equality, prefix matching** silently fail; fix: consistent normalisation in the pipeline (decompose, then compose).

| | bytes per codepoint | ASCII compatible |
|---|---|---|
| UTF-8 | 1 to 4 | yes |
| UTF-16 | 2 or 4 | no |
| UTF-32 | exactly 4 | no |

- UTF-8: ≤U+007F `0xxxxxxx` (7 bits); ≤U+07FF `110xxxxx 10xxxxxx` (11); ≤U+FFFF `1110xxxx`+2 (16); ≤U+10FFFF `11110xxx`+3 (21). Leading 1s = length of whole sequence; `10` = continuation; `0` = ASCII. é = C3 A9; 中 U+4E2D = E4 B8 AD; U+1F600 = F0 9F 98 80. Beyond the BMP: 4 bytes in UTF-8 and UTF-16.
- **Byte inflation** = UTF-8 bytes / characters: hello 1.0, café 1.25, 中文 3.0. More bytes, more aggressive splitting; byte splits can produce non-existent characters.
- Breaking: bytes can split codepoints; codepoints can split base from diacritic; ideal is graphemes, but those are language-specific. Mixing strategies misaligns **span annotation** (SQuAD offsets) and **counts across corpora**.
- Turkish lowercase(I) = ı. Codepoint order is not alphabetical (é after z). Swedish å ä ö **after z**; German ä = ae, **interleaved**. Split punctuation off words, match it by category `\p{P}`; quote normalisation must be mapped by hand.

## L03 Morphology and word formation

- **Vocabulary** = list of forms; **lexicon** = meanings and functions; **dictionary** = written lexicon.
- Embeddings: low-dimensional (e.g. 512 vs |V| in the hundreds of thousands), dense, continuous, learned by a task. **CBOW**: predict $w_t$ from $n$ left and $m$ right words ($m=0$ would be a language model); projection sums (L05 formula: averages) context embeddings, order lost; **no non-linear hidden layer**; $n=m$, $n \in \{2,5,10\}$.
- **Short-list**: keep the most frequent words (10K to ~200K), the rest become `<unk>`, so **all rare words get equal probability**. en ~200K vs ru ~1M words (about 5:1).
- **Word segmentation** (zh, ja, Thai, Burmese) must come before POS/parsing; **subword segmentation** splits delimited words of rich languages. $2^{n-1}$ segmentations (20 chars: 524,288).
- **FMM** (left to right) / **BMM** (right to left, prepend): take the longest word in $V$; a single character is always accepted; $O(n \cdot L)$ lookups. 研究生活 with $V$ = {研究, 研究生, 生活, 生, 活}: FMM 研究生\|活, BMM 研究\|生活. **Bidirectional**: agree, accept; disagree, prefer **fewer words** or **fewer single-character words** (picks BMM here). Limits: needs a precompiled vocabulary (new words break **silently**), cannot exploit data, ambiguity is in the language.
- **BMES**: B begin, M middle (words of 3+ chars), E end, S single; boundary after every E and S. After B or M only M, E; after E or S only B, S (8 of 16 transitions impossible). HMM: transitions $P(t_i \mid t_{i-1})$, emissions $P(c_i \mid t_i)$ (**character given tag**); $\hat t = \arg\max \prod P(t_i \mid t_{i-1}) P(c_i \mid t_i)$; MLE from counts. **Viterbi** $s[i][t] = \max_{t'} s[i-1][t'] P(t \mid t') P(c_i \mid t)$ + backpointers, $O(n|T|^2)$ = 16n vs $4^n$. Gains: **no dictionary**, probabilistic disambiguation, trainable from any segmented corpus. Lacks: narrow Markov assumption (tag depends on previous tag and current char), no long-range context (so CRFs, neural); unsupervised Baum-Welch (EM) works but quality drops.
- **Inflection** keeps POS, changes features (*see/saw*, *book/books*). **Derivation** changes POS and/or meaning (*happiness*; *perfect/imperfect*). **Compounding** joins words (*Tischbein* N+N, *sichergehen* Adj+V). Very productive. English nouns inflect for number only (case on pronouns). German marks mood (*kommt/käme/komm*); Slavic aspect (*sdelat'*/*delat'*). Features multiply: thousands of forms per lemma.
- Morpheme = smallest meaning-carrying part. Root = free; affix = bound. Prefix; suffix (several possible); infix (*passersby*); circumfix (*ge-seh-en*, one morpheme). **Non-concatenative**: apophony *Buch/Bücher*, Arabic root + pattern. String cutting captures only concatenative morphology.

| type | how | examples |
|---|---|---|
| isolating | no form change; word order, particles | zh 了, vi *đã* |
| agglutinative | separable, one meaning each | tr *ev-ler-iniz-den*, fi *talo-i-ssa-mme* |
| fusional | one affix, several features | ru *knig-ami*, es *habl-é*, ar *yaktubu* |
| polysynthetic | a sentence in one word | Inuktitut, Mohawk |

- Agglutinative vs fusional: the **defining** difference is separability; more morphemes per word follows from it. *-ami*: course gloss case + number + gender.
- Word vocabularies explode or give massive OOV; analysers are language-dependent; **subword tokenization is the compromise**. Characters: very few OOV, **not zero** (unseen characters); only byte level removes it.

## L04 Subword segmentation

- Word vocabularies (300K to 500K) still hit `<UNK>`; characters carry little meaning, long sequences. **FSTs**: accurate, efficient; extremely laborious, totally language dependent, no preference among analyses (probabilistic FSTs rank).
- **Morfessor** (Creutz and Lagus 2002): **MDL** = cost of generating the data + cost of the model. Modern methods: no labels, no character substitutions or deletions, strictly concatenative; need no annotation and fit a chosen vocabulary size.

| | BPE | WordPiece | Unigram LM | byte BPE |
|---|---|---|---|---|
| builds | merges up | merges up | prunes down | merges up |
| score | $c(ab)$ | $\frac{c(ab)}{c(a)c(b)}$ | loss $L_t$ | $c(ab)$ |
| new word | replay merges in order | greedy longest match | Viterbi, n-best, sampled | replay merges |
| unknown | char fallback | whole word `[UNK]` | char fallback | none: 256 bytes |
| marker | `</w>` (shown `@@`) | `##` continuation | `▁` (SentencePiece) | |
| one split? | yes, greedy | yes, greedy | no | yes |
| used in | GPT, RoBERTa, XLM | BERT, mBERT, Johnson NMT | T5, XLM-R | |

- **BPE**: chars + `</w>`, count adjacent pairs over word types × frequency (never across words), merge the most frequent, repeat until the merge count (only hyperparameter; vocab = chars + merges; **30K to 128K** typical, multilingual at the top). On *the tall man is taller than the tallest man*, merging tall+e gives *talle r*, *talle s t*: the boundary is buried and **never reconsidered**. WordPiece merges i+s first (1/2), then e+r (1/4).
- BPE gains: smaller vocab, better translation, far fewer OOV, **for high- and low-resource pairs**. Sennrich 2016 en→de BLEU (single / 8-ensemble): words WUnk 20.6/22.8, WDict 22.0/24.2; char bigrams C2-50k 22.8/25.3; separate BPE-60k 21.5/24.5; **joint** BPE-J90k 22.8/24.7; syntax SMT 24.4. Separate BPE < best word-level single; joint > separate; SMT beats every single NMT, only 8-ensembles beat it.
- **Unigram LM**: $P(\mathbf x)=\prod_i p(x_i)$, $\mathbf x^* = \arg\max_{\mathbf x \in S(X)} P(\mathbf x)$. All segmentations, global optimum, multiple segmentations (sampling, n-best), **top-down**. Start: all substrings occurring more than once (within words); loop: fit LM (EM), $L_t = p_\theta(D) - p_{\theta'}(D)$ = likelihood **lost** without $t$, remove $\min(|V|-k, \lfloor\alpha|V|\rfloor)$ tokens with the **highest $L_t$** (**key**: the course's rule). Kudo's paper and SentencePiece remove the **lowest** $L_t$ and keep single chars.
- **[[Dynamic Programming|DP]]**: solve each sub-problem once and store it (time-memory trade-off); top-down memoisation and bottom-up have the **same asymptotic run time**. $m[i,j]$ = best of keeping span whole or $\max_k m[i,k]\,m[k{+}1,j]$; $O(n^3)$ time, $O(n^2)$ space; valid **because segments are independent**; use log probs. *cats*: $p$(cats) .0001 < cat .003 × s .05 = .00015, so cat\|s. Cutting values 1, 5, 8, 9 for lengths 1 to 4: best 2+2 = 10. Viterbi gives only the best; **Yen's, Eppstein's** give n-best; or disallow sub-solutions; sampling = subword regularization.
- **SentencePiece** is a **toolkit** (BPE and Unigram LM): raw Unicode stream including spaces, space escaped as `▁`, **lossless** desegmentation, no pre-tokenizer. Standard BPE assumes word boundaries are detected.
- **Bostrom and Durrett** (same data, same vocab size): Unigram finds morphemes (▁tri cycle s, ▁nano technology, ▁corrupt ed, ▁1848 vs BPE ▁184 8; keeps Japanese particles apart). Unigram has **longer segments on average** (en and ja), uses vocab space better in en (BPE's tail collapses near rank 15.5k), fewer tokens per word. BPE over-produces word-initial capitals (▁H), Unigram suffixes and punctuation.

| BPE vs Unigram LM | BPE | Unigram |
|---|---|---|
| CELEX2 en P/R/F1 | 38.6/12.9/19.3 | 62.2/20.1/30.3 |
| MeCab ja F1 | 73.8 | 77.2 |
| fertility en (tokens/type) | 1.343 (4.721) | 1.318 (4.633) |
| SQuAD 1.1 EM/F1 | 80.6/88.2 | 81.8/89.3 |
| MNLI m/mm | 81.4/82.4 | 82.8/82.9 |
| CoNLL NER test F1 | 90.2 | 90.4 |
| TyDi QA ja EM/F1 | 41.4/42.1 | 53.7/54.4 |

- Unigram ahead on every task: slightly in en, **+12.3 in Japanese**. The bigger gold gap (en) did **not** predict the bigger downstream gap (ja).
- BPE/Unigram OOV: **very low, not zero** (typos, unknown characters). **Byte-level BPE**: raw UTF-8 bytes, **base alphabet 256**, zero OOV, unseen scripts encodable. UTF-8 bytes/char: ASCII 1; Latin with diacritics, Cyrillic, Greek 2; most CJK, Devanagari, Thai 3; many emoji 4. So non-Latin scripts need extra merges just to rebuild characters: **shorter subsegments**, caused by UTF-8 length, not the language.
- **Fertility** = subword tokens / words (1 = every word one token). Depends on merge count / vocab size, average word length, morphological richness. Isolating Latin-script languages ≈ 1. **No scheme dominates all languages.** High fertility: model works harder, **APIs charge more**, context fills faster.

## L05 Static embeddings

- Distributional hypothesis (Firth 1957). **Count vectors**: high-dim, discrete, sparse, interpretable; no criterion for signal vs noise. **Embeddings**: low-dim, dense, continuous, learned by a task; dims not interpretable. $\cos(u,v) = \frac{u \cdot v}{\|u\|\|v\|}$ ignores length.
- **CBOW**: $h = \frac{1}{2n} W \sum_{i \ne t} w_i$ (average), $\hat y = \text{softmax}(W'h)$; **no non-linearities**; $W' \ne W^\top$ (not necessarily shared); cross entropy, SGD. Embedding: $W$, or $W^\top + W'$.
- **Skip-gram**: $h = W w_I$; $2n$ outputs, context words **independent given the input word**: $\prod_{i \ne t} p(w_i \mid w_I)$; all outputs share $W'$, same distribution.
- **[[Negative Sampling|Negative sampling]]**: 1 positive + $k$ negatives from a noise distribution, a classification task; cost $k+1$ dot products instead of $|V|$.
$$\arg\max_\theta \sum_{(w,c)\in\mathcal D}\log\sigma(v_c\cdot v_w) + \sum_{(w,c)\in\mathcal D'}\log\sigma(-v_c\cdot v_w)$$
- A negative keeps $w$ and swaps $c$; one step updates $v_w$ and the $k+1$ output vectors only. $k$ = **5 to 20**; noise $P(w)^{0.75}$ (towards rarer words); window **2 to 5**; **more data, smaller window and fewer negatives**.
- **GloVe** (Pennington 2014): global co-occurrence statistics + local prediction-based updating. $J=\sum f(X_{ij})(w_i\cdot\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$, weighted least squares on log counts. $f(x)=(x/x_{max})^{\alpha}$ if $x<x_{max}$, else 1; $x_{max}=100$, $\alpha=0.75$; **caps at 1** (frequent pairs still get the top weight); $f(0)=0$, zero cells drop. One corpus pass builds $X$; AdaGrad; final vector $w+\tilde w$. Word2Vec treats each window as an independent event.
- Evaluation: intrinsic/extrinsic × qualitative/quantitative. Similarity scored by **Spearman** (rank). **WS-353 = relatedness** (bread/butter high); **SimLex-999 = strict similarity**. Analogies by accuracy: $v_{king}-v_{man}+v_{woman}$, nearest by cosine excluding inputs. PCA linear; t-SNE non-linear, **hyperparameters have substantial impact**; plots complement, never substitute, quantitative evaluation.
- **fastText** (Bojanowski 2017) = skip-gram + NS where the **centre word** = sum of its char n-gram vectors (**3 to 6** chars, `<` `>` boundary markers, plus the whole word); **context vectors stay per word**; every n-gram gets the same gradient; OOV words get vectors. Similarity (Spearman ×100, sg to fastText): de GUR350 61 to 70, ru HJ 59 to 66, en Rare Words 43 to 47; **only loss en WS-353** (71 vs cbow 73); n-gram vectors for OOV add up to 6. Analogies: **syntactic** gains (cs 52.8 to 77.8, de +11.9, it +11.2, en +4.8); **semantic barely change, de drops** 66.5 to 62.3.
- **Shared space hypothesis**: separately trained languages encode similar relational structure; **approximate isomorphism**; if true, a **general linear map** aligns them. Signal: **supervised** = seed dictionary (thousands); **semi-supervised** = hundreds + bootstrapping; **unsupervised** = none, adversarial. Seed dictionaries are limited to words; sources: human dictionaries, word-aligned data, identical spellings.
- **Mikolov 2013**: $W^*=\arg\min_W\sum_i\|Wx_i-y_i\|^2$, closed form $W=YX^\top(XX^\top)^{-1}$ (needs $\ge d_1$ independent pairs); Mikolov used **SGD**. en→es P@1: matrix 33%, edit distance 13%, co-occurrence 19%, matrix + edit distance 43% (en→cs 27, 29). Rises with monolingual data (9 at 2×10⁷ words, 53 at 2.5×10¹⁰, flat above 2×10⁹); drops for rarer words (53 to 40).
- Unconstrained $W$: stretches, shears, overfits the seed dictionary; Euclidean training vs cosine retrieval differ unless unit-normalised. **Xing 2015 orthogonal**: $W^\top W=I$, unit vectors; only rotations and reflections; $\|Wx-y\|^2 = 2-2\cos$. **Procrustes**: $U\Sigma V^\top=\text{SVD}(YX^\top)$, $W=UV^\top$, exact. en→es P@1 unconstrained / orthogonal: 300d 30.43/38.99, 500d 25.76/39.91, 700d 20.69/41.04: unconstrained **worse as dimension grows**, orthogonal better at every dimension. "Procrustes" for Monz = the general least-squares problem (true in the lecture's sense); in Conneau's tables = orthogonal SVD.
- **Hubs**: a few vectors are nearest neighbours of many points; high-dimensional geometry, **worse with more dimensions**; NN is not symmetric. **CSLS**$(x,y)=2\cos(x,y)-r_T(x)-r_S(y)$, $r$ = mean cosine to $K$=10 nearest neighbours in the other space; $r_S(y)$ penalises hubs; $r_T(x)$ is constant for a fixed $x$; **no parameter tuning**. Procrustes NN to CSLS: en-es 77.4 to 81.4, fr-en 76.1 to 82.4, ru-en 58.2 to 63.7; CSLS wins every pair, about level with inverted softmax.
- **MUSE** (Conneau 2018): generator = **linear $d\times d$ $W$, no non-linearity** (a deep one could scramble which word lands where); discriminator MLP 2 × 2048, ReLU, input dropout, **label smoothing 0.2**; top **50k** words. Alternate discriminator and $W$ steps, each with the other frozen; $W$ trained on **flipped labels**; re-orthogonalise $W\leftarrow(1+\beta)W-\beta(WW^\top)W$, $\beta$ = 0.01. Model selection: **not the adversarial loss**; mean cosine of CSLS nearest neighbours of the top **10k** source words. Refinement: synthetic dictionary of **mutual** CSLS nearest neighbours among frequent words, Procrustes by SVD, repeat. MUSE overall = unsupervised; Conneau call refinement semi-supervised.
- en-es: adversarial 69.8, +CSLS 75.7, +refinement 79.1, both 81.7. Unsupervised / supervised Procrustes-CSLS: en-es 81.7/81.4, en-fr 82.3/81.1, en-de 74.0/73.5 (close pairs: match or slightly beat), en-ru 44.0/51.7, en-zh 32.5/42.7 (distant: well short).
- **Søgaard, Ruder, Vulić 2018**. Marking: dependent, head, double (both), zero (neither), mixed. Conneau's languages: mostly dependent-marking, none agglutinative, none with 10+ cases. Added: et, fi (mixed, agglutinative, 10+ cases), el (double, fusional, 3), hu (dependent, agglutinative, 10+), pl, tr.

| P@1 | adversarial | identical words | Δ |
|---|---|---|---|
| EN-ES | 81.89 | 82.62 | 2.07 |
| EN-ET | **0.00** | 31.45 | 6.61 |
| EN-FI | **0.09** | 28.01 | 7.33 |
| EN-EL | **0.07** | 42.96 | 5.01 |
| EN-HU | 45.06 | 46.56 | 3.27 |
| EN-PL | 46.83 | 52.63 | 2.56 |
| EN-TR | 32.71 | 39.22 | 3.14 |
| ET-FI | **29.62** | 24.35 | 3.98 |

- Identical-word supervision beats adversarial on **every pair with English**; ET-FI is the one pair where adversarial wins. **Marking type alone does not decide** (French is mixed-marking and aligns well); failing languages combine it with rich morphology. Cause: grammar spread over more, rarer word forms, so **non-isomorphic** spaces.
- **Eigenvector similarity**: nearest-neighbour graph adjacency $A$, Laplacian $L=D-A$ (lecture: $A-D$), keep the largest $n\%$ eigenvalues, $\Delta=\sum_i(\lambda^{(1)}_i-\lambda^{(2)}_i)^2$. Bigger $\Delta$ = less isomorphic; needs no alignment; highest for the three failing pairs (en-et, en-fi, en-el). **One way only: a low $\Delta$ does not prove isomorphism.**
- Not data size: Finnish WaC (1.7B words) still EN-FI 0.0 (Wikipedia fi 12M vs es 363M). en-hu: nouns 26.87, verbs 25.44 worst. **Domain** (EuroParl, Wikipedia, EMEA): same domain both work; cross-domain identical degrades (4.84 to 25.48), **adversarial collapses (0.0 to 0.13), even en-es**. **Algorithm**: es skip-gram variants 81.89 to 80.15 (barely matters); **es CBOW vs en skip-gram 0.00 to 0.13**.

## L06 Contextual embeddings

- Contextual trade-off: enough context to tell occurrences apart, not so much that the word's own contribution washes out. Remedies: **right layer(s), right training task(s), residual connections**.
- **Pre-train** $M_{C,\theta}$ on a big general task; **fine-tune** with head $M_{R,\theta'}$ updating $\theta'$ **and maybe** $\theta$. Only $\theta'$ = feature extractor; both = full fine-tuning (BERT default). Encoder-decoder (Vaswani 2017: MT, summarisation, QA); encoder-only (predicts its own noised input, auto-encoder: BERT); decoder-only (next word; **all current LLMs**).
- **BERT** (Devlin 2019): Transformer encoder, bidirectional; pre-training **self-supervised** (MLM + NSP), fine-tuning supervised. Input = token + segment ($E_A$/$E_B$) + position embeddings. **Select 15%**; of those **80% `[MASK]`, 10% random word, 10% unchanged**; **loss only at the selected positions, all three kinds**. Static masking (once, in pre-processing). `[CLS]`: no word, never masked, represents the whole sequence, loss from a classification task (NSP: 50% IsNext, 50% random NotNext). Tagging/span heads on $T_i$; classification on $C$. **NLI**: premise entails, contradicts or is neutral to the hypothesis (3-way); premise = segment A. XNLI = English NLI test/dev translated into 14 languages. Fine-tuning: batch 16 or 32, **2 to 4 epochs**. SQuAD: only new parameters start $S$, end $E\in\mathbb R^H$; $P_{start}(i)=\text{softmax}_i(T_i\cdot S)$; span maximises $T_i\cdot S+T_j\cdot E$, $i\le j$.
- BERT-base 110M, 12 layers, hidden 768; large 340M, 24 layers, 1024, 16 heads. **Heads, base: 12 in Devlin; lecture table says 16 for both (key: 16 if the statement comes from the table).** Pre-trained on BookCorpus + en Wikipedia.
- **WordPiece**: bottom-up like BPE; score $\frac{c(a,b)}{c(a)c(b)}$ (parts that rarely occur apart). hug 10, pug 5, pun 12, bun 4, hugs 5: BPE merges `##u ##g` (20); WordPiece `##g ##s` (5/100 = .05 vs 1/36). Inference: **greedy longest match, does not replay merges**; an unmatchable char makes the **whole word `[UNK]`**.
- **Whole-word masking**: mask all pieces of a word, **rate unchanged** (~15%); suffix pieces are easy to complete, stems hard. BERT-Large SQuAD F1 91.0 to 92.8, MultiNLI 86.05 to 87.07 (uncased; cased gains too).
- **mBERT**: GitHub 2018, **no paper**; same architecture and loss as BERT (MLM + NSP); **no crosslingual loss, no parallel data (intentionally), no language-ID embedding, no language-specific parameters, no alignment loss**; 104 Wikipedias; NSP pairs and sequences within one language (a batch can mix); **one shared WordPiece vocab ~120k** (BERT 30k); ~178M parameters, the extra **entirely from the embedding matrix**. CJK characters split into single chars first. Sampling smoothing also applied to subword frequencies. Vocab still favours high-resource and **Latin script**; low-resource and rich languages get higher fertility. Uncased strips accents.
- **Sampling** $p_l=\frac{n_l^{\alpha}}{\sum_{l'}n_{l'}^{\alpha}}$: $\alpha$=1 proportional (big languages dominate); $\alpha$=0 uniform (tiny ones memorised); **mBERT $\alpha$ = 0.7**. Ratio $n_1/n_2$ becomes $(n_1/n_2)^\alpha$, ranks kept. en 1000, sw 10: $\alpha$=1 $p_{sw}$=.0099; $\alpha$=0.7 $p_{sw}$=.038 (100:1 becomes ~25:1); $\alpha$=0: 0.5 each. **$\alpha=1/T$**: raising $T$ flattens; $T$=5 is $\alpha$=0.2; $\alpha$=0.7 is $T\approx1.4$.
- **Zero-shot crosslingual transfer**: fine-tune in A, test in B with no B task data (off-diagonal cells).
- **Pires 2019** (NER, POS, 16 langs): off-diagonal decent but below in-language; **asymmetric**: NER en→nl 77.36 (nl→nl 89.86) vs nl→en 65.46. **Cross-script works**: POS hi→ur 85.9, ur→hi 91.1, en→bg 87.1; Japanese drops (en→ja 49.4), possibly typology. **Entity overlap**: English BERT near 0 at low overlap, rises roughly linearly; **mBERT roughly flat**: not string matching. Word order (POS): SVO→SVO 81.55 vs SVO→SOV 66.52; from SOV both ≈ 64. Adjective-noun order: smaller effect. More shared WALS features, better transfer, mBERT far above en BERT. **Translation retrieval** (mean hidden states without `[CLS]`/`[SEP]`, one mean difference vector, $\ell_2$ NN): inverted U, **peak in layers 6 to 8 of 12** (71 to 76%), falls in the top layers.
- **Wu and Dredze 2019** (English fine-tuning, 39 langs). MLDoc zero-shot: mBERT 74.5 vs Artetxe and Schwenk (parallel data) 74.9; in-language 91.2. **XNLI**: mBERT zero-shot 66.3; translate-train 71.6; beats X-LSTM 65.6; loses to Artetxe and Schwenk 70.2, **XLM MLM-only 71.5**, XLM MLM+TLM 75.1 (X-LSTM and XLM chose models on target-language dev). Weakest: sw 50.4, th, ur, hi, tr. **NER**: 74.03, **beats** dedicated Xie et al. 67.13; zh collapses. **POS**: 84.3, **does not match** Kim et al. with 320 target sentences 89.9.
- Freezing layers $\le k$ ($k$ = 0 is embeddings): **Feat** (no fine-tuning) below best on every task (MLDoc −14.8 most, POS −1.9 least). Freezing low layers helps or is neutral; best $k$: NER 0 or 3, POS 3, MLDoc and XNLI 6. Freezing to 9 hurts NER most (67.3 vs best 74.3), POS, XNLI; **not MLDoc** (77.2 vs 74.5 unfrozen).
- Wordpiece overlap vs score, type-level R: **NER 0.99**, MLDoc 0.8, POS 0.58, parsing 0.5, **XNLI −0.036** (token level 0.36, non-significant). Correlation is not cause.
- **K et al. 2020** (bilingual B-BERT, XNLI). **Fake English** (Unicode shifted, zero overlap) costs 0.5 to 1.4 (es 1.4, hi 0.5, ru 0.7): overlap is not the main cause. **Word permutation** in pre-training only: significant drop, **not eliminated**; full permutation −8.4 es, −16.5 hi, −12.1 ru. **Depth: largest architecture effect** (parameters held ~133M to 139M): in-language saturates by ~6 layers, zero-shot keeps rising; gap 21.6 (1 layer) to 11.3 (24). **Mixed-language input** below both monolingual conditions; target premise + fake-English hypothesis beats the reverse; Hindi lowest mixed but also lowest baseline (similar drops).
- **Rajaee and Monz 2024**: across$(L)=\frac{1}{2(N-1)}(\sum_{L'\ne L}A_{LL'}+\sum_{L'\ne L}A_{L'L})$, $N$ = 15. **Across < within for every language**; en highest within. For any hypothesis language, the same-language premise is best. **English hypothesis helps** (foreign premise + en hypothesis beats en premise + foreign hypothesis; for de, ur, hi, sw, th it even beats the own-language hypothesis). sw hypothesis barely above chance (33%), th a bit better. Related pairs good (ru-bg, es-fr, hi-ur). **Heuristics hypothesis**: transfer of heuristics can contribute; SNLI full premise-hypothesis overlap = entailment 94.7%; across languages overlap ≈ 0, so the shortcut says "not entailment" (an interpretation).
- **XLM-R** (Conneau 2020), a **RoBERTa-style extension of mBERT**: **drop NSP** (little use, sometimes hurts), **dynamic masking** (new mask each pass; static = once at data level), **CC-100** (2.5 TB, 100 langs, ~2 orders of magnitude more data), **250k Unigram LM (SentencePiece) vocab** (better fertility, bigger embeddings). Still **no crosslingual signal**. Within: beats mBERT on both tasks (+8.5 XNLI, +7.6 QA; en QA tie). Across: better on XNLI, **worse on QA** (table in L07). Thai QA within 66.5 vs 40.0, across 29.6 vs 26.1. mBERT degrades smoothly; XLM-R: own language or English question.
- **Conclusions**: shared vocabulary plays a role, **not the main cause**; structure matters (permutation hurts, does not eliminate); **deeper = more language-independent**; **no single factor explains transfer**.

## L07 Crosslingual NLP

- Multilingual models: joint training, one model, one vocab, **context always within one language**: nothing links languages. **Parallel** = meaning-equivalent pairs; **comparable** = same topic, degree of parallelism on a sliding scale (news on one event, Wikipedia on one entity). Sources: news (Xinhua), UN, EU, bilingual countries' sites; **OPUS**. Crawl: document then sentence alignment, or sentences straight from Common Crawl. Old equivalence measures: dictionary overlap, relative length.

| model | pre-training data | crosslingual signal | notes |
|---|---|---|---|
| mBERT | 104 Wikipedias | none | MLM+NSP, WordPiece 120k, static mask |
| XLM-R | CC-100, 100 langs | none | MLM only, dynamic mask, Unigram 250k |
| LASER | parallel | translation decoder | encoder kept |
| XLM | Wikipedia + parallel | TLM (token level) | from scratch, BPE, language emb. |
| InfoXLM | CC-100 (94) + parallel | TLM + XLCo (sentence) | init XLM-R, no language emb. |
| BART | English mono | none | seq2seq denoising |
| mBART | multilingual mono | none | same-language denoising |

- **LASER** (Artetxe and Schwenk 2019): encoder-decoder trained to translate on parallel data. Encoder: BPE, **one embedding table for all languages**, BiLSTM stack, **max-pooled** to one fixed-size vector, **never told the input language**. Decoder LSTM gets the sentence vector (initial state via $W$ and concatenated at every step) + previous token + **language ID $L_{id}$** of the output. Decoder discarded. **vecalign** = LASER similarity + search that scales to massive data.
- **XLM** (Lample and Conneau 2019): CLM (next word), MLM, **TLM** = MLM on a sentence and its translation concatenated; **same loss, different input**; **positions restart at 0** in the second sentence; **language embeddings** mark the halves; **TLM always combined with MLM or CLM**; **NSP dropped**; parallel data from OPUS, the 15 XNLI languages paired with English.
- XNLI settings: **translate-train** (MT the training set into each language), **translate-test** (MT test sets into English), **zero-shot** (English only).

| XLM on XNLI (avg, 15 langs) | acc |
|---|---|
| translate-train MLM+TLM | 76.7 |
| zero-shot MLM+TLM | 75.1 |
| translate-test MLM+TLM | 74.2 |
| zero-shot MLM | 71.5 |
| LASER (Artetxe and Schwenk) zero-shot | 70.2 |

- **TLM helps every language** (+3.6 avg): most vi +4.9, tr +4.7, ar +4.6, zh +4.6; least en +1.8, fr, ru +2.2. Zero-shot beats translate-test **only on average**: translate-test wins every European language except en (tie), plus tr and ar; zero-shot wins vi, th, zh, hi, sw, ur. Translate-train best on average (**ur the exception**). Zero-shot XLM beats mBERT and LASER in every reported language.
- Nepali LM perplexity: alone 157.2, +en 140.1, **+hi 115.6** (related, same script), +both 109.3. XLM's input embeddings align translations better than MUSE and Concat fastText.
- **InfoXLM** (Chi 2021): initialised from XLM-R, further pre-trained; 14 English-centric pairs (~42 GB); XLM-R's SentencePiece vocab; **no language embeddings**. Loss = MMLM + TLM + XLCo, **equally weighted**. **MMLM**: Monz "like MLM, but uses **negative sampling**" (**key: true**; the paper: plain MLM softmax read as contrastive). **XLCo**: `[CLS]` of a parallel pair, classify parallel or not, **random sentences as negatives** ([[Contrastive Learning|contrastive]], InfoNCE). TLM aligns tokens, XLCo sentences.

| within / across (gap) | XNLI acc | XSQuAD F1 |
|---|---|---|
| mBERT | 65.7 / 54.5 (11.2) | 64.4 / 44.2 (20.2) |
| XLM-R | 74.2 / 64.8 (9.4) | 72.0 / **36.8** (35.2) |
| InfoXLM | 75.4 / 70.3 (5.1) | 73.8 / 64.5 (9.3) |

- XNLI: 15 langs, 225 combinations, 3-way, chance 33.3%. XSQuAD: en + 10 translations, 121. **Every model drops within to across.** InfoXLM vs XLM-R: within only +1.2/+1.8, across **+5.5/+27.7**; smallest gap on both. **English is the best partner** (en hypothesis gives the best mixed pair); **Swahili the weakest** (InfoXLM lifts it most); grids **not symmetric**; mBERT weak on Thai even within. XLM-R QA works across mainly with an **English question** (en context + ar question 38.1, ar context + en question 59.2); non-English pairs collapse (as low as 14.6). InfoXLM: no cell collapses (min 51.7). QA exposes it more: word-level matching.
- **Seq2seq** drops two sequence-labelling assumptions: **no isomorphic positions, no equal length**. *Hiermit hörte sie nicht auf* / *She did not stop with this*: 6 vs 7 tokens, one-to-many (*Hiermit*), many-to-one (*hörte ... auf*). $p(\mathbf Y\mid\mathbf X)=\prod_t p(y_t\mid\mathbf Y_{<t},\mathbf X)$, $\mathbf X$ = encoder output. $\mathbf z$ = source; $\mathbf x$ = `<s>` + target (read); $\mathbf y$ = target + `</s>` (predicted); $y_i = x_{i+1}$; teacher forcing. **Sutskever 2014**: LSTMs, $h^{dec}_0=h^{enc}_n$, **bottleneck**. Transformer decoder layer: masked self-attention (target context), **cross-attention over all top encoder outputs** (removes bottleneck), feed-forward; residuals.
- NMT quality depends on data **amount, variation, relevance**. **Schwenk 2019: 86% of directions poor**; English directions far best; no generalisation beyond specific directions. **Zoph 2016**: under 2M tokens, NMT 6.9 to 12.7 BLEU below syntax-based SMT. **Koehn and Knowles 2017** (data doubles per step): at ~0.4M words NMT 1.6, phrase-based 16.4, +big LM 21.8; NMT improves faster; passes phrase-based at **~1.2 to 2.4×10⁷ words**, phrase-based + big LM at **~1 to 2×10⁸**.
- **Transfer = basically fine-tuning**: train parent (big pair), **continue training** on child. **Monz's conditions for best transfer**: identical target language; related source languages; **decoder parameters frozen** (**key: true**; the experiment says only the **target embeddings**); shared vocabulary. Zoph setup: parent fr→en 300M tokens, child uz→en 1.8M, dropout 0.5; **source vocab and embeddings not shared** (uz words take arbitrary fr rows); target and all else shared. Gains: Hausa +4.5, Turkish +5.6, Uzbek +3.7, **Urdu +8.6** (5.2 to 13.8; smallest corpus gains most); with tricks beats SBMT only for Hausa (24.0 vs 23.7).

| uz→en, trainable (cumulative) | BLEU |
|---|---|
| nothing | 0.0 |
| + source embeddings (biggest jump) | 7.7 |
| + source RNN | 11.8 |
| + target RNN | 14.2 |
| + attention (best: all but target emb.) | **15.0** |
| + target input embeddings | 14.7 |
| + target output embeddings | 13.7 |

- **Relatedness**: es→en child: fr parent 31.0, de 29.8, none 16.4; uz with fr parent 15.0. **French'** (vocab consistently reshuffled): 13.3 to 20.0 with fr parent: **structure transfers**. **Kocmi and Bojar 2018** (shared vocab): en→et child 17.03; parents en→fi 19.74, **en→cs 20.41, en→ru 20.09**: unrelated parents (and different targets) help as much; **size direction matters**: smaller parent helps little or hurts (parent et→en, child fi→en 23.95 vs 24.40 child alone; parent en→sk, child en→cs 22.99 vs 23.48). **Aji 2020** (my, id, tr→en): all transferred 21.7; inner layers only 18.3 (about half the gain); **embeddings only 13.7 < scratch 14.5**; embeddings on top of inner add more; en→de parent ≈ de→en parent.
- Exam Q3.1 (fine-tune general to in-domain): **A true** (init from general params); B false ("continuously": more adaptation helps, then hurts); C false (catastrophic forgetting).
- **Johnson 2017 multilingual NMT**: **many-to-one, one-to-many: training; many-to-many: testing only** (English-centric data, so non-English pairs are zero-shot = direction unseen in training). Transfer shows as zero-shot performance and beating the bilingual baseline. **Target language tag** prepended to the source (`<2es>`), source language never named; **all parameters shared**; one WordPiece vocab (32k, 64k); source, target, output embeddings shared; directions mixed within a batch; balancing by weighted sampling. **Many-to-one beats bilingual in every direction** (+0.05 to +1.27); **one-to-many mixed** (+0.90 to −2.11): target languages compete in the decoder; balancing moves the loss around.
- **Transfer** = better from sharing; **interference** = worse from competing for capacity. **Arivazhagan 2019** (103 langs):

| BLEU (high/med/low) | into English | out of English |
|---|---|---|
| bilingual | 37.61/31.41/21.63 | 29.34/17.50/11.72 |
| Any→En / En→Any | 36.61/33.66/**30.56** | 28.75/17.32/12.98 |
| All→All ($T$=5) | 33.85/30.25/26.96 | 28.03/16.91/12.75 |

- Low-resource: **+8.93 into English, +1.26 out**; high-resource lose everywhere (**no multilingual model beats bilingual on high**); medium +2.25 into, −0.18 out; **All→All weakest in every group**.
- **Temperature** $p_l\propto(n_l/\sum n)^{1/T}$, $\alpha=1/T$. **$T$=1 proportional**: best for high-resource, **wrecks low** (en→any low 6.24 vs 11.72 bilingual). **$T$=100 effectively uniform**: best low, worst high. **$T$=5 compromise**: within 0.12/0.36 of uniform on low, 0.83/0.60 better on high, best on medium.
- Zero-shot, 10 vs 102 languages: be→ru 36.28 to 50.26, **yi→de 8.97 to 20.00** (more than doubles), de→fr 11.15 to 14.24, hi→fi 2.98 to 8.76, ru→fi 6.02 to 9.06; **fr→zh 15.07 to 11.83 (worse)**. Depends on which languages; still behind supervised.
- **BART**: **bidirectional encoder** reads a corrupted document, **autoregressive decoder** rebuilds the original (cross-attention); denoising autoencoder; NLL loss; pre-trained on **English only**. On `A B C . D E .`: token masking `A _ C . _ E .`; deletion `A . C . E .` (must find where); **infilling** `A _ . D _ E .` (one mask per span, must find how many; Poisson $\lambda$ = 3, length-0 spans); sentence permutation `D E . A B C .`; document rotation `C . D E . A B`.
- Results: **text infilling best single** (SQuAD 90.8); deletion beats masking on every generation task; rotation (77.2) and permutation weak alone; **infilling + permutation best CNN/DM perplexity** (5.41); left-to-right LM best on ELI5, worst on SQuAD (76.7). Large BART **matches RoBERTa** on SQuAD (F1 94.6 both), **best on summarisation**, most on XSum (R1 45.14 vs RoBERTaShare 41.45).
- BART fine-tuning: **classification** from the decoder's last hidden state (same input to both); **span**: label tokens, predict begin/end; **MT**: a **randomly initialised source encoder replaces BART's encoder embedding layer**; BART encoder can be frozen or updated; source vocab may differ. ro→en BLEU: baseline 36.80, **fixed BART 36.29 (worse)**, tuned BART 37.96 (+1.16).
- **mBART** (Liu 2020): one model on concatenated monolingual data; BART objective **within the same language (no crosslingual signal)**; noise = **infilling + sentence permutation**; **language ID token ends the encoder input and starts the decoder input**. Fine-tuned **directly** on parallel data, whole model, no new encoder: source tag ends encoder input, **target** tag starts the decoder. mBART25 vs random init: **better on every pair, both directions**; average +7.2/+4.4; **largest gains at a few hundred thousand pairs** (en-vi 133K +12.5/+10.6, en-tr 207K +10.3/+8.3); **10K (en-gu) stays near zero** (0.3); en-kk 91K 0.8 to 7.4; gains **mostly** shrink in the millions (en-lv 4.5M +3.7/+3.0; but en-hi 1.56M +12.6/+6.6).
