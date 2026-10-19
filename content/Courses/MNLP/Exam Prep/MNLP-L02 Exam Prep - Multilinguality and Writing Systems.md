---
type: exam-prep
course: 5204MNLP6Y
lecture: 2
status: complete
---

# MNLP-L02 Exam Prep - Multilinguality and Writing Systems

> [!abstract] What the exam can ask here
> Two kinds of statement: which languages break which English-first assumption (and which tier or script a language belongs to), and how Unicode works underneath (normalization forms, UTF encodings, bytes against codepoints against graphemes). The Unicode half is where the one-word flips live: lossy against non-lossy, NFD against NFKD, "ASCII compatible", "exactly", "continuation byte", "language-specific".
> [[MNLP-L02 - Multilinguality and Writing Systems|Lecture note]]; Unicode counts and byte examples are on the [[MNLP - Cheat Sheet|cheat sheet]].

## English-centric resources and the resource tiers

- **Most** NLP resources are in English: labelled data, unlabelled data, toolkit coverage, benchmarks; most publications focus on English.
- Non-English resources arise as a **naturally occurring by-product** (translations, reviews) or as **translations of English resources**, **less so** as a dedicated effort.
- Coverage by NLP resources depends on **commercial interest** and **military relevance**.
- There are over 7,000 spoken languages: a **long tail**, with more than 100 languages over 10M speakers.
- **High resource** (large labelled and unlabelled data, many tools): English, Chinese (Mandarin), then a **big gap**, then French, Spanish, Arabic, Russian.
- **Medium resource** (limited labelled, reasonable unlabelled, some tools): German, Italian, **Dutch**, Korean, **Japanese**.
- **Low resource** (**no** labelled data, limited unlabelled, **no** tools): all remaining languages.
- Standard pipelines encode **properties of English**: whitespace tokenization, fixed word order, **closed vocabularies**.
- Language independence claims are **usually untested** outside English (or a very small subset).
- This is **understandable**: resource generation costs time and money.

> [!warning] Traps
> - *"Coverage by NLP resources depends mainly on the number of speakers."* False: commercial interest and military relevance.
> - *"Non-English resources are mostly created by dedicated efforts."* False: by-products and translations; less so dedicated efforts.
> - *"Japanese and German are high-resource languages."* False: medium.
> - *"Arabic is a medium-resource language."* False: high, after the big gap.
> - *"Low-resource languages have limited labelled data and some tools."* False: no labelled data and no tools.

## Where languages differ from English

- Analysis runs from small to large: **characters** (character set size, writing systems), **words** (morphology, lexical semantics), **clauses and sentences** (syntax, sentence semantics), **paragraphs** (discourse structure, **anaphora**, **ellipsis**).
- Chinese has **(almost) no spaces** between words.
- In 我不知道她去哪儿了, the verb 去 carries **no tense marking**; past tense comes from the **sentence-final particle** 了.
- Both the main clause (I NEG know) and the embedded clause (she go where) are **SVO**.
- A **morpheme** is the smallest unit of meaning **or grammatical function**: *dogs* has two (dog, -s).
- The **root** (base) carries the primary meaning (*friend* in *unfriendly*).
- **Affixes** are prefixes **and** suffixes that modify a root's meaning or grammatical role (*un-*, *-ly*).
- English has **limited** morphological complexity; Russian **more complex** (case and number on *kniga*); Finnish **very complex** (*onnellisimmillanikin*, "even at my happiest").

> [!warning] Traps
> - *"Chinese marks past tense on the verb."* False: through the sentence-final particle 了.
> - *"The embedded clause in the Chinese example is SOV."* False: both clauses are SVO.
> - *"Affixes are suffixes added to the end of a root."* False: prefixes and suffixes.
> - *"A morpheme is the smallest unit of meaning, so grammatical endings such as -s are not morphemes."* False: meaning or grammatical function.
> - *"Anaphora and ellipsis are sentence-level phenomena."* False: the slides place them at paragraph (discourse) level.

## The English-first assumptions

- **Words are separated by whitespace**: not true for Chinese, Japanese, **Thai**, Lao, Khmer, Burmese.
- Thai uses spaces to **end clauses or sentences**; it has **no dedicated full stop**.
- **Grammatical relations are encoded by word order**: not true for languages with rich morphology, **especially case marking**: Russian, Finnish, **Korean**.
- **One word = one unit of meaning**: not true for **agglutinative** languages (Finnish, Turkish, Hungarian) and **compounding** languages (German, Dutch, Swedish).
- **Text flows left to right**: not true for Semitic languages and languages in Arabic script: Arabic, Hebrew, Urdu, Persian.
- In Unicode such text is **written left to right but rendered right to left**.

| assumption | broken by |
|---|---|
| whitespace | Chinese, Japanese, Thai, Lao, Khmer, Burmese |
| word order | case marking: Russian, Finnish, Korean |
| one word, one meaning | agglutinative: Finnish, Turkish, Hungarian; compounding: German, Dutch, Swedish |
| left to right | Arabic, Hebrew, Urdu, Persian |

> [!warning] Traps
> - *"Thai uses spaces to separate words."* False: to end clauses or sentences.
> - *"German breaks the whitespace assumption."* False: compounding breaks one word = one meaning.
> - *"Korean encodes grammatical relations mainly by word order."* False: case marking.
> - *"Turkish breaks the assumption through compounding."* False: Turkish is listed as agglutinative.
> - *"Unicode stores Arabic text right to left."* False: written left to right, rendered right to left.

## Types of writing system

- **Alphabetic**: each symbol is a **consonant or vowel sound**; typically a **small inventory**. Latin, Cyrillic, **Hangul**.
- **Syllabic**: each symbol is a **whole syllable** (vowel and consonant combined). Japanese **Hiragana and Katakana**.
- **Logographic**: each symbol represents **words or morphemes directly**, with **little sound information**. Chinese Hanzi, Japanese **Kanji**.
- Han: **one symbol, one morpheme**.
- Devanagari is **alphasyllabic**: consonant **plus inherent vowel**.
- Arabic writes the **root only**; **vowels are omitted**.
- Korean letter shapes **encode articulatory features**.

| system | symbol stands for | examples |
|---|---|---|
| alphabetic | one sound | Latin, Cyrillic, Hangul |
| syllabic | a syllable | Hiragana, Katakana |
| logographic | word or morpheme | Hanzi, Kanji |

> [!warning] Traps
> - *"Japanese Kanji is a syllabic writing system."* False: logographic; Hiragana and Katakana are syllabic.
> - *"Hangul is syllabic."* False: the slides list it as alphabetic.
> - *"Logographic symbols carry a lot of sound information."* False: typically little.
> - *"Alphabetic systems typically have a large inventory of symbols."* False: small.
> - *"Arabic script writes vowels explicitly."* False: vowels are omitted.

## Script and language

- One writing system serves **many languages**: Latin (most European languages, also several African, Turkic and Asian ones), Arabic (Arabic, **Persian, Urdu, Pashto, Sorani**), Cyrillic (Russian, Belarusian, **Serbian**), Devanagari (Hindi, Marathi, Nepali), **Ethiopic (Ge'ez)** (Amharic, Tigrinya).
- **Language-specific** writing systems: Armenian, Georgian, **Greek**, Sinhala, Khmer.
- Languages with **more than one** writing system: Serbian (Cyrillic and Latin), Kurdish (Latin and Arabic), Uzbek (Latin and Cyrillic), Punjabi (**Gurmukhi** and Arabic).

> [!warning] Traps
> - *"Arabic script is used only for the Arabic language."* False: also Persian, Urdu, Pashto, Sorani.
> - *"Amharic is written in Arabic script."* False: Ethiopic (Ge'ez).
> - *"Serbian is written only in Cyrillic."* False: Cyrillic and Latin.
> - *"Greek script is shared by several languages."* False: listed as language-specific.

## ASCII and code pages

- ASCII (1963) assigned **128 codes**: enough for English letters, digits, punctuation.
- Code pages use the **same 128 (or 256) codes** in a byte as ASCII and **interpret them differently** depending on language or script (Latin-1, Shift-JIS, GBK, KOI8-R).
- Disadvantage: text encoded in one script and decoded in another **leads to gibberish**.
- Disadvantage: **cannot use multiple scripts** in the same document.
- Unicode removes the need for writing-system-specific code pages.
- Converting a code page to UTF-8 with `iconv -f GB2312 -t UTF-8` **fails if the source encoding is not correct**.

> [!warning] Traps
> - *"Code pages extend ASCII by using more than 256 codes per byte."* False: same 128 or 256 codes, interpreted differently.
> - *"Code pages allow several scripts in one document."* False: one document cannot use multiple scripts.

## Unicode and codepoints

- Unicode is a **single, universal character set**, maintained by the Unicode Consortium.
- Every character in every supported script gets **exactly one unique number**: its **codepoint**, written U+number (U+0041 for "A").
- Unicode **defines meaning, not bytes**: a codepoint is an **abstract identifier**; storing it as bytes is an **encoding** issue.
- Unicode is a **superset of ASCII**: U+0000 to U+007F are identical to ASCII.
- That is why **English-only** Latin-1 documents are Unicode compatible.
- The code space runs U+0000 to U+10FFFF, organised into **17 planes**. The slide's "65,356 codepoints per plane" is a typo for 65,536.
- Plane 0, the **Basic Multilingual Plane**: most modern languages, including the most common CJK ideographs; symbols such as currency; **a Private Use Area**.
- Plane 1 (Supplementary Multilingual Plane): **non-CJK** historic scripts, modern scripts, symbols such as musical notation, **emoji**.
- Planes 2 and 3: more CJK ideographs, mostly historical, uncommon or variants. Planes 4 to 13: **unassigned**. Plane 14: tags and variation selectors. Planes 15 and 16: private use.

> [!warning] Traps
> - *"Unicode specifies how each codepoint is stored as bytes."* False: that is the encoding's job (UTF-8, UTF-16, UTF-32).
> - *"Unicode and ASCII assign different codes to English letters."* False: U+0000 to U+007F are identical.
> - *"All CJK ideographs are in the Basic Multilingual Plane."* False: the most common are; more sit in planes 2 and 3.
> - *"Emoji are in the Basic Multilingual Plane."* False: plane 1.
> - *"Private use codepoints exist only in planes 15 and 16."* False: the Basic Multilingual Plane also has a Private Use Area.

## Codepoint, character, glyph

- **Codepoint**: an abstract number (U+00E9), Unicode's **unit of identity**.
- **Character**: what a reader **perceives** as one unit; may be built from **several codepoints**.
- **Glyph**: a **stylistic rendering** of a character.
- The same rendered word can have **different codepoint sequences**: café as U+0063 U+0061 U+0066 U+00E9 (**NFC**), or with U+0065 U+0301 (**NFD**).
- A **naive string similarity check will not match** both sequences.
- Same for ã: NFC is one codepoint (U+00E3), NFD is a + combining tilde (U+0061 U+0303). In JavaScript the two compare **unequal** (`===` gives false) and have **length 1 against 2**.
- A reader sees a **grapheme**, represented by **one or more** codepoints, each encoded in **1 to 4 bytes**.

> [!warning] Traps
> - *"Every character a reader perceives corresponds to exactly one codepoint."* False: it may be several.
> - *"A glyph is Unicode's unit of identity."* False: the codepoint is; a glyph is a rendering.
> - *"The NFC and NFD forms of café are equal under a naive string comparison."* False: different codepoint sequences.
> - *"The NFC and NFD forms of ã have the same string length."* False: 1 codepoint against 2.

## Normalization forms

- **NFC**: canonical **composition** (é as one codepoint). **NFD**: canonical **decomposition** (e + combining accent).
- NFD splits characters into **base characters and combining marks** and **preserves semantic and visual equivalence**.
- NFKD (compatibility decomposition) breaks characters into their **most basic parts**, **including splitting ligatures**.
- NFD is **non-lossy**: decompose and compose again without losing styling or formatting details.
- NFKD is **lossy**: it normalizes visual styling to the base form, which **cannot be reconstructed** during composition.
- Example: the long s ſ with a dot above and below keeps its ſ under NFC and NFD; NFKC and NFKD turn it into a **plain s** with the dots.
- Use **NFD** to **compare rendered text**.
- Use **NFKD** for **search indexing or NLP analysis**, **but not necessarily generation**.
- Crawled resources can **mix forms even when all in Unicode** (some NFC, some NFD).
- **Deduplication, search, string equality and prefix matching** assume identical codepoint sequences, so they become **inaccurate** without proper normalization.
- Fix: **always** perform **consistent** normalization in the pipeline, e.g. first decompose, then compose.
- Python's `unicodedata` can **normalize** a string into a given form **and test** whether it is in one.

| | composed | decomposed |
|---|---|---|
| canonical, non-lossy | NFC | NFD |
| compatibility, lossy | NFKC | NFKD |

> [!warning] Traps
> - *"NFD is lossy."* False: non-lossy; NFKD is lossy.
> - *"NFKD can be recomposed into the original characters."* False: the styling cannot be reconstructed.
> - *"Use NFKD to compare rendered text."* False: NFD.
> - *"NFKD is the recommended form for generation."* False: for search indexing and analysis, not necessarily generation.
> - *"NFC splits characters into base characters and combining marks."* False: NFD does; NFC composes.
> - *"NFKD leaves ligatures intact."* False: it splits them.
> - *"Text crawled in Unicode is already consistently normalized."* False: it can mix NFC and NFD.

## UTF-8, UTF-16, UTF-32

- **UTF-8**: **1 to 4** bytes per codepoint; **ASCII compatible**; used by the web, Linux, most file formats, virtually all modern NLP tooling.
- **UTF-16**: **2 or 4** bytes; **not** ASCII compatible (at least 2 bytes per character); Java, JavaScript strings, Windows internals, .NET.
- **UTF-32**: **exactly 4** bytes; **not** ASCII compatible; **rare** in practice; simple for indexing but **memory-hungry**.
- UTF-8 is **by far the most common** form and what people usually mean by "Unicode".
- A byte starting with **0** is a **single byte** (the 128 ASCII characters).
- A byte starting with **11** is the **first byte** of a multi-byte sequence; the **number of 1s** gives its length.
- A byte starting with **10** is a **continuation byte**.
- In UTF-8, "A" takes 1 byte, é 2, most CJK 3, an emoji 4.

| encoding | bytes | ASCII compatible |
|---|---|---|
| UTF-8 | 1 to 4 | yes |
| UTF-16 | 2 or 4 | no |
| UTF-32 | exactly 4 | no |

> [!warning] Traps
> - *"UTF-16 is ASCII compatible."* False: at least 2 bytes per character.
> - *"UTF-16 uses exactly 2 bytes per codepoint."* False: 2 or 4.
> - *"UTF-32 uses 1 to 4 bytes per codepoint."* False: exactly 4.
> - *"UTF-32 is the most common form because it is simple to index."* False: rare, memory-hungry; UTF-8 is most common.
> - *"A UTF-8 byte starting with 10 begins a multi-byte sequence."* False: it is a continuation byte; 11 begins one.
> - *"Most Chinese characters take 2 bytes in UTF-8."* False: most CJK takes 3.

## Bytes and tokenizers

- Byte-level tokenizers are affected by how characters are represented **at the byte level**, **not as codepoints**.
- Byte inflation $= \frac{\text{UTF-8 bytes}}{\text{characters}}$: "hello" lowest, "café" higher, "中文" highest (factors on the cheat sheet).
- **The more bytes** are needed, **the more aggressively** tokens tend to be split.
- Byte-level splitting can result in **non-existing characters**.

> [!warning] Traps
> - *"Byte-level tokenizers are independent of how characters are encoded in bytes."* False: they are affected by it.
> - *"The more bytes a script needs, the less its tokens are split."* False: the more aggressively they are split.
> - *"Byte-level splitting always yields valid characters."* False: it can produce non-existing characters.

## Where to break a string

- **Byte-level** breaking can **mess up codepoints**, which can consist of multiple bytes.
- **Codepoint** breaking is also not ideal: it can break a character into segments (**base and diacritic**).
- **Ideally** break at the **grapheme** level (what the reader perceives as a character), but this is **very language-specific**.
- **Mixing** breaking strategies causes misalignments: **span annotation** (SQuAD-style QA) and **counting across corpora**.
- Any string breaking or length computation must be **consistent across documents**.

> [!warning] Traps
> - *"Codepoint-level breaking is ideal because a codepoint is what the reader perceives as a character."* False: the grapheme is; codepoint breaking can split base and diacritic.
> - *"Grapheme breaking is language-independent."* False: very language-specific.
> - *"Mixing breaking strategies only matters for byte-level tokenizers."* False: it misaligns span annotation and counts across corpora.

## Case, sorting and punctuation

- Lowercasing: English lowercase(I) = i; **Turkish** lowercase(I) = **ı** (Turkish has İ / i).
- **Codepoint order** does **not necessarily** match alphabetical order: accented letters' codepoints are not necessarily close to their base letter.
- Languages use **different alphabetical orders**: Swedish puts å, ä, ö **after z**; German treats ä as ae, ö as oe, ü as ue.
- Punctuation **adjacent to words** is not part of them, so it is split off.
- Punctuation is often covered as a **general category** (`\p{P}`).
- Punctuation normalization (“...” against «...» against 「...」) has **no simple solution**: mappings must be **encoded manually**.

> [!warning] Traps
> - *"Lowercasing works the same in every language that uses Latin script."* False: Turkish lowercase(I) is ı.
> - *"Sorting by codepoint gives alphabetical order."* False: accented letters can be far from their base letter.
> - *"Punctuation normalization can be solved by a simple universal mapping."* False: mappings must be encoded by hand.

## Practice questions

**1.** Which of the following statements about Unicode normalization are true? (multiple answers possible)
- [A] NFD splits characters into base characters and combining marks
- [B] NFKD is non-lossy, since decomposed characters can always be composed again
- [C] NFKD is suitable for search indexing
- [D] NFD should be used for generation targets because it is lossy

> [!example]- Answer
> **A and C.** B: NFKD is lossy; the styling cannot be reconstructed. D: NFD is non-lossy, and the slides recommend it for comparing rendered text.

**2.** Which of the following statements about Unicode encodings are true? (multiple answers possible)
- [A] UTF-8 is ASCII compatible
- [B] UTF-16 uses exactly 2 bytes per codepoint
- [C] UTF-32 is simple for indexing but memory-hungry
- [D] In UTF-8, a byte starting with 10 is the first byte of a multi-byte sequence

> [!example]- Answer
> **A and C.** B: 2 or 4 bytes. D: a byte starting with 10 is a continuation byte; 11 starts a sequence.

**3.** Which languages break the assumption that grammatical relations are encoded by word order? (multiple answers possible)
- [A] Korean
- [B] Thai
- [C] Russian
- [D] Dutch

> [!example]- Answer
> **A and C.** Both have rich morphology with case marking. B breaks the whitespace assumption. D breaks one word = one meaning through compounding.

**4.** Which of the following statements about writing systems are true? (multiple answers possible)
- [A] Japanese Kanji is a syllabic writing system
- [B] Logographic symbols typically carry little sound information
- [C] Devanagari is alphasyllabic: a consonant carries an inherent vowel
- [D] Serbian is written in both Cyrillic and Latin

> [!example]- Answer
> **B, C and D.** A: Kanji is logographic; Hiragana and Katakana are syllabic.

**5.** Which of the following statements about bytes, codepoints and breaking are true? (multiple answers possible)
- [A] Byte-level tokenizers tend to split text more aggressively the more bytes a character needs
- [B] Codepoint-level breaking keeps a base letter and its diacritic together
- [C] Grapheme-level breaking is ideal but very language-specific
- [D] Mixing breaking strategies can misalign span annotations in SQuAD-style QA

> [!example]- Answer
> **A, C and D.** B: codepoint breaking can split a character into base and diacritic.

**6.** Which of the following statements about NLP resources are true? (multiple answers possible)
- [A] Resource coverage depends on commercial interest and military relevance
- [B] Dutch is a high-resource language
- [C] Non-English resources mostly come from dedicated efforts
- [D] Claims of language independence are usually untested outside English

> [!example]- Answer
> **A and D.** B: Dutch is medium resource. C: they come mostly as by-products or translations of English resources, less so dedicated efforts.
