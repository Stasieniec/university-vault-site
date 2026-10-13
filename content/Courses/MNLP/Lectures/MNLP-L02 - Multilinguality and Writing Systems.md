---
type: lecture
course: 5204MNLP6Y
week: 1
lecture: 2
status: complete
topics:
  - English-centric NLP and the resource tiers
  - Typological diversity across languages
  - Morphology, from English to Russian to Finnish
  - The four English-first assumptions
  - Types of writing system
  - Script versus language
  - Code pages and mojibake
  - Unicode, codepoints, planes
  - Normalization forms NFC, NFD, NFKC, NFKD
  - UTF-8, UTF-16, UTF-32
  - Byte inflation and tokenization
  - Grapheme breaking and punctuation handling
  - Lowercasing and sorting pitfalls
---

# MNLP-L02: Multilinguality and Writing Systems

> [!abstract] In short
> Standard NLP pipelines split on whitespace, rely on word order, treat a word as one unit of meaning and assume text runs left to right. Those are properties of English, and each fails for a large group of languages. A second set of problems sits lower down, where text becomes numbers and bytes.
>
> What a reader sees as one character (a grapheme) can be several Unicode codepoints, each stored as 1 to 4 bytes in UTF-8. The same visible text can have different codepoint sequences, so pipelines must normalize consistently, and the normalization form (NFC, NFD, NFKC, NFKD) decides what information survives.
>
> No sample exam question covers this lecture. Expect statements on exact conditions: which normalization forms are lossy, which encodings are ASCII compatible, which languages break which assumption.

## NLP mostly means English NLP

Most NLP resources are in English: **annotated (labelled) training data, unannotated training data, toolkit coverage** (dictionaries, analyzers) and **benchmarks and competitions**. Most publications also focus on English.

Non-English resources come about mostly in two ways, and less often in a third:

- **as a naturally occurring by-product** of something else, such as translations or product reviews, which nobody made for NLP;
- **as translations of existing (annotated) English resources**, so the annotation scheme and content stay English-shaped;
- **less so as a dedicated effort**, the route that would actually close the gap.

The result is that standard pipelines encode assumptions that are **properties of English, not of language in general**: whitespace tokenization, fixed word order, closed vocabularies. **Claims of language independence are usually untested** outside English (or a very small set of languages). This is **understandable**: creating resources costs time and money.

### Resource tiers follow commercial and military interest

There are **over 7,000 spoken languages**. Most have few speakers, but more than a hundred have over ten million each. How well a language is covered by NLP resources does not follow its speaker numbers: it depends on **commercial interest** and **military relevance**. Christof Monz (the lecturer) sorts languages into three tiers:

**High resource:** large labelled and unlabelled data, many tools. English and Mandarin Chinese, then a **big gap**, then French, Spanish, Arabic and Russian. All six are high resource, but the tier holds two very different groups.

**Medium resource:** limited labelled data, reasonable unlabelled data, some tools. German, Italian, Dutch, Korean, Japanese.

**Low resource:** no labelled data, limited unlabelled data, no tools. All of the roughly 7,000 remaining languages, which, on these tiers, includes most of the 100+ languages with over ten million speakers.

## Languages differ in ways English hides

### One sentence, two languages

> [!example] How many words is this?
> **English:** I don't know where she went.
>
> **Chinese:** 我不知道她去哪儿了。 Chinese has (almost) no spaces, so splitting on whitespace returns the whole sentence as one token.
>
> | 我 | 不 | 知道 | 她 | 去 | 哪儿 | 了 | 。 |
> |---|---|---|---|---|---|---|---|
> | wǒ | bù | zhīdào | tā | qù | nǎr | le | |
> | I | NEG | know | she | go | where | SFP | PUNC |

(NEG: negation; SFP: sentence-final particle.)

- **The verb 去 ("go") carries no tense.** English changes the verb ("go" against "went"); Chinese leaves it alone.
- **Past tense comes from the sentence-final particle 了**, at the end of the clause, away from the verb. (Outside the course material: 了 marks a completed event, so [[MNLP-L03 - Morphology and Word Formation|L03]] glosses it as perfective.)
- **Both clauses are SVO** (subject, verb, object): the main clause "I NEG know" and the embedded clause "she go where". Word order matches English here; word boundaries and the place of past-time marking do not.

### Morphology: what is inside a word

Words have parts: *unfriendly* is *un-* + *friend* + *-ly*, and each part adds meaning.

> [!definition] Morphology
> **Morphology** is the study of the internal structure of words and how they are formed from smaller units of meaning.
> - **Morpheme:** the smallest unit of meaning or grammatical function. *dogs* has two: *dog* (the base meaning) and *-s* (plural).
> - **Root / base:** the core part of a word that carries the primary meaning (*friend* in *unfriendly*).
> - **Affix:** a prefix (added to the front) or suffix (added to the back) that modifies a root's meaning or grammatical role (*un-* and *-ly* in *unfriendly*).

Three languages show the range, from limited to very complex morphology.

**English has limited morphology.** From *happy* you get *happier, happiest, unhappy, unhappier, unhappiest, happiness, unhappiness*, and little more. That is why closed (fixed-size) word vocabularies seemed a reasonable design.

**Russian has more complex morphology.** The noun *kniga* ("book") has twelve slots, six cases (grammatical roles such as subject, possessor, instrument) times singular and plural, but only nine distinct forms. So some forms fill several slots: *knígi* is "of the book", "the books" and "books" as direct object, and the case cannot always be read off the string.

**Finnish has very complex morphology.** It is **agglutinative**: morphemes are glued on one after another, each adding one piece of meaning, so a single word can carry what English spreads over a phrase.

> [!example] Building "even at my happiest" in Finnish
> *onni* ("happiness") + *-llinen* ("having the quality of") gives *onnellinen*, "happy"; with the stem *onnellis-* and comparative *-empi*, *onnellisempi* is "happier". Then *onnellisimmillanikin*, "even at my happiest", is:
>
> ```
> onnellis-  stem, "happy"
> -imm-      superlative
> -i-        plural oblique-case marker
> -lla       adessive case, "at/on/in"
> -ni        "my" (1st person possessive)
> -kin       "even"
> ```
>
> One word for four English ones.

A fixed word-level vocabulary has almost certainly never seen *onnellisimmillanikin*, and Finnish produces combinatorially many words like it, so more data does not fix this. That motivates subword segmentation ([[MNLP-L04 - Subword Segmentation|L04]]); morphology itself is [[MNLP-L03 - Morphology and Word Formation|L03]].

## The English-first assumptions

**1. Words are separated by whitespace.** False for **Chinese, Japanese, Thai, Lao, Khmer and Burmese**, which have no word delimiter. Thai does use spaces, but to end **clauses or sentences** (it has no dedicated full stop), so a Thai space does not mark a word boundary.

**2. Grammatical relations are encoded by word order.** False for languages with **rich morphology, especially case marking**: **Russian, Finnish, Korean**. Who did what to whom is marked on the nouns themselves, so word order is more flexible and a word's position says much less about its role.

**3. One word = one unit of meaning.** False for **agglutinative** languages (**Finnish, Turkish, Hungarian**) and for languages with **compounding** (**German, Dutch, Swedish**). German *Bundesverfassungsgericht* is one token holding three concepts: federal, constitutional, court.

**4. Text flows from left to right.** False for **Semitic languages and languages in Arabic script**: **Arabic, Hebrew, Urdu, Persian**, which are read right to left. Unicode text in these languages is **written (stored) left to right but rendered right to left**: codepoints sit in the file in reading order and are displayed from the right, so the first character of the string is the rightmost on screen.

## Writing systems: what a symbol stands for

Writing systems differ in what one symbol stands for: a sound, a syllable or a whole word or morpheme.

> [!definition] Types of writing system
> - **Alphabetic:** each symbol represents a **consonant or vowel sound**; typically a **small inventory** of symbols. Latin, Cyrillic, Hangul.
> - **Syllabic:** each symbol represents a **whole syllable**, a combination of consonant and vowel. Japanese Hiragana and Katakana.
> - **Logographic:** each symbol represents **words or morphemes directly**, with typically **little sound information**. Chinese Hanzi, Japanese Kanji.

One practical difference for NLP is inventory size: an alphabet has tens of symbols, a logographic script thousands, which a character-level vocabulary must budget for.

| Script | Word | Reading |
|---|---|---|
| Latin | water | water |
| Devanagari | पानी | pani |
| Han | 水 | shui |
| Arabic | كتب | ktb |
| Korean | 한글 | Hangul |

**Latin** spells sounds with letters and **Korean Hangul** is alphabetic too (each letter a consonant or vowel), with **letter shapes that encode articulatory features** (how the sound is produced); **Han** uses one symbol per morpheme (logographic). Devanagari and Arabic fit none of the three types cleanly:

**Devanagari** (Hindi) is **alphasyllabic**: each consonant letter carries an **inherent vowel**. (Outside the course material: an attached vowel sign can replace it, so पानी is two units, *pā* + *nī*.)

**Arabic** writes the **consonant root only; vowels are omitted**: letters for sounds, but only the consonants. In examples from outside the course material, كتب (k-t-b, "write") can be read *kataba* ("he wrote"), *kutiba* ("it was written") or *kutub* ("books"). The writing is lossier than the language, so the model must resolve the ambiguity from context; no tokenizer can recover vowels that were never written.

A single Japanese sentence can mix three systems: in アムステルダムはオランダの首都です ("Amsterdam is the capital of the Netherlands"), the names are Katakana, 首都 ("capital") is Kanji, the rest Hiragana.

## Script is not language

**One writing system, many languages:**
- **Latin:** English, German, French, most European languages (except some Eastern European ones), and also several African, Turkic and Asian languages
- **Arabic:** Arabic, Persian, Urdu, Pashto, Sorani (Central Kurdish)
- **Cyrillic:** Russian, Belarusian, Serbian
- **Devanagari:** Hindi, Marathi, Nepali
- **Ethiopic (Ge'ez):** Amharic, Tigrinya

**Language-specific writing systems:** Armenian, Georgian, Greek, Sinhala, Khmer.

**One language, more than one writing system:** Serbian (Cyrillic and Latin), Kurdish (Latin and Arabic), Uzbek (Latin and Cyrillic), Punjabi (Gurmukhi and Arabic).

> [!tip] What each direction breaks
> **Detecting the script does not identify the language.** Latin script narrows almost nothing; Arabic script still leaves Arabic, Persian, Urdu, Pashto and Sorani.
>
> **Knowing the language does not fix the script.** A Serbian corpus can contain Cyrillic and Latin text, even in the same document. A model trained on one script and tested on the other shares almost no letters with the test data, though the language is the same.

## Before Unicode: ASCII and code pages

From here on, the question is how text in all these scripts becomes numbers and bytes, which is where multilingual pipelines break silently.

**ASCII** (1963) assigned **128 codes**: enough for English letters, digits and punctuation.

**Code pages** extended this by reinterpretation. They use the same 128 (or 256) codes in a byte as ASCII and **interpret them differently depending on the language or script**: Latin-1 for Western Europe, Shift-JIS for Japanese, GBK for Chinese, KOI8-R for Russian, and many more. (Outside the course material: Shift-JIS and GBK use two bytes for most characters.)

Two disadvantages:

- **Gibberish when text encoded in one script is decoded in another.** The bytes are often legal in both tables, so usually no error is raised (the result is called mojibake, a term from outside the course material).
- **No mixing of scripts in one document**, since a document is read with one table.

## Unicode: one number per encoded character

> [!definition] Unicode and codepoints
> **Unicode** is a **single, universal character set**, maintained by the Unicode Consortium. Every character it encodes, in every supported script, gets **exactly one unique number**, its **codepoint**, written `U+number` in hexadecimal: `U+0041` is Latin "A", `U+4E2D` is the Han character 中. What a reader sees as one character can be a sequence of several codepoints (next section).

One code space for every script removes both code-page problems: no table to guess, and one document can mix any scripts.

- **Unicode defines meaning, not bytes.** A codepoint is an abstract identifier for a character. How to store it as bytes on disk or in memory is a separate encoding question, answered by UTF-8, UTF-16 or UTF-32.
- **Unicode is a superset of ASCII:** `U+0000` to `U+007F` are identical to ASCII. So a pure ASCII file is already valid UTF-8 (UTF-8 keeps ASCII's bytes). A Latin-1 file with accented letters is not: Latin-1 stores é as the single byte E9, which UTF-8 rejects.

**Scale.** The code space runs from `U+0000` to `U+10FFFF`, over 1.1 million possible codepoints, in **17 planes of $2^{16} = 65{,}536$ codepoints each**:

- **Plane 0, the Basic Multilingual Plane** (BMP): most modern languages, including the most common CJK (Chinese, Japanese, Korean) ideographs.
- **Plane 1:** historic scripts, some modern scripts, symbols such as musical notation, **emoji**.
- **Planes 2 and 3:** more CJK ideographs, mostly historical or rare, so not every CJK character is in the BMP.
- **Plane 14:** tags and variation selectors; **planes 15 and 16:** private use (plane 0 also has a private use area); planes 4 to 13 unassigned.

## Codepoint, character, glyph

> [!definition] Codepoint, character, glyph
> - **Codepoint:** one abstract number, such as `U+00E9`; Unicode's unit of identity.
> - **Character:** what a reader perceives as one unit. It may be built from **several codepoints**.
> - **Glyph:** a stylistic rendering of a character (how a font draws it).
>
> What a reader perceives as one character is also called a **grapheme**.

So a codepoint is always one number, and one perceived character is one or more codepoints: é can be the single codepoint `U+00E9`, or e followed by a combining accent.

> [!example] Two identical-looking "café"
> ```
> café in NFC: 4 codepoints
> U+0063 U+0061 U+0066 U+00E9
> café in NFD: 5 codepoints
> U+0063 U+0061 U+0066 U+0065 U+0301
> ```
> In NFC, é is one **precomposed** codepoint; in NFD, e plus a **combining** acute accent. They render identically but compare unequal: **a naive string comparison will not match them.** The same holds for ã: in JavaScript, `'ã' === 'ã'` is false, though both display as ã (lengths 1 and 2).

## Normalization: choosing one sequence for the same text

**Normalization** rewrites text into a standard codepoint sequence. The four forms come from two choices: **composed** (C: precomposed codepoints where possible) or **decomposed** (D: base character plus combining marks), and **canonical** (merges only sequences for the same character) or **compatibility** (K: also merges stylistic variants, such as a ligature and its letters).

| | Composed | Decomposed |
|---|---|---|
| **Canonical** (lossless) | NFC | NFD |
| **Compatibility** (lossy) | NFKC | NFKD |

> [!example] Where compatibility forms lose information
> Take long s (ſ, an old form of s) with a dot above, followed by a combining dot below: ẛ̣.
> - **NFC:** ẛ + combining dot below (2 codepoints: no precomposed character has both dots)
> - **NFD:** ſ + dot below + dot above (3 codepoints)
> - **NFKC:** ṩ, a plain s with both dots (1 codepoint)
> - **NFKD:** plain s + dot below + dot above (3 codepoints)
>
> The canonical forms keep the long s ſ, because canonically ſ and s are different characters. The compatibility forms replace it with plain s, and nothing can bring it back.

> [!definition] The decomposition forms and when to use them
> - **NFD (canonical decomposition):** splits characters into **base characters and combining marks**; preserves semantic and visual equivalence. **Non-lossy**: you can decompose and compose again without losing styling or formatting details.
> - **NFKD (compatibility decomposition):** breaks characters down into their **most basic parts, including splitting ligatures**. **Lossy**: it normalizes visual styling to the base form, which cannot be reconstructed when composing.
> - **Use NFD to compare rendered text.**
> - **Use NFKD for search indexing or NLP analysis**, but not necessarily for generation.

Why: for analysis, merging variants helps (the ligature ﬁ should match "fi" in a search index, ① should count as 1). For generation it can hurt: a model trained on NFKD targets never sees, and so cannot produce, the characters that were folded away.

Outside the course material: for a plain equality check NFC works too, because NFD and NFC agree on which strings are equivalent. The course's rule for comparing rendered text is NFD.

### Normalizing crawled data

**Resources crawled from the internet can mix normalization forms even when they are all in Unicode**: some pages NFC, others NFD.

Many preprocessing steps **assume identical codepoint sequences**: **deduplication**, **search**, **string equality and prefix matching**. Without proper normalization they become **inaccurate**, silently: deduplication keeps two copies that differ only in normalization form.

**The easy fix: always perform consistent normalization as part of the pipeline**, for example first decompose and then compose (canonical decomposition then composition ends in NFC). In Python, the `unicodedata` module can **normalize** a string into a given form and **test** whether it is already in one:

```python
import unicodedata as ud

a = "caf\u00e9"    # NFC: 4 codepoints
b = "cafe\u0301"   # NFD: 5 codepoints

a == b                         # False
ud.is_normalized("NFD", b)     # True
ud.normalize("NFD", a) == b    # True
```

Text that is not in Unicode at all, but in a code page such as GB2312 or KOI8-R, is converted on the way in, for example with `iconv -f GB2312 -t UTF-8`. This **fails if the source encoding is not correct**, which is useful: a failure is better than silent mojibake. Outside the course material: iconv only fails on bytes that are invalid in the stated encoding. That is common for multi-byte GB2312, but in a single-byte code page such as KOI8-R every byte is legal, so a wrong guess converts silently into mojibake.

## Storing codepoints as bytes: UTF-8, UTF-16, UTF-32

| Encoding | Bytes per codepoint | ASCII compatible |
|---|---|---|
| **UTF-8** | 1 to 4 | yes |
| **UTF-16** | 2 or 4 | no: at least 2 |
| **UTF-32** | exactly 4 | no: exactly 4 |

**UTF-8 is by far the most common** (the web, Linux, NLP tooling), and it is what people usually mean by "Unicode". UTF-16 is used by Java, JavaScript and Windows; UTF-32 is rare, simple to index but memory-hungry.

### How UTF-8 works

UTF-8 uses more bytes for larger codepoints. The **number of leading 1s in the first byte gives the length of the whole sequence**, the first byte included (`1110xxxx` starts a 3-byte sequence, so two bytes follow), and every following byte carries 6 bits of the codepoint.

> [!definition] The UTF-8 byte rules
> - A byte starting with **`0`** is a **single byte**, encoding one of 128 characters: ASCII.
> - A byte starting with **`11`** is the **first byte of a multi-byte sequence**; the **number of leading 1s** gives the length of the whole sequence.
> - A byte starting with **`10`** is a **continuation byte** inside a multi-byte sequence.

```pseudo
Encode codepoint c in UTF-8
───────────────────────────
c ≤ U+007F    0xxxxxxx
c ≤ U+07FF    110xxxxx 10xxxxxx
c ≤ U+FFFF    1110xxxx 10xxxxxx
              10xxxxxx
c ≤ U+10FFFF  11110xxx + three
              10xxxxxx bytes
```

The payload widths are 7, 11 (5+6), 16 (4+6+6) and 21 (3+6+6+6) bits. Write c in binary, pad it with zeros on the left to the payload width, and fill the x positions from left to right.

> [!example] 中, U+4E2D
> 0x4E2D in binary is `100111000101101`, 15 bits. That is more than the 2-byte form's 11, so use the 3-byte form with 16 payload bits. Pad to `0100111000101101` and split 4 + 6 + 6: `0100 | 111000 | 101101`.
>
> Byte 1 = `1110` + `0100` = `11100100`
> Byte 2 = `10` + `111000` = `10111000`
> Byte 3 = `10` + `101101` = `10101101`, so 中 is `E4 B8 AD`.

By the same rules, A (`U+0041`) is one byte identical to ASCII, é (`U+00E9`) two bytes, most CJK characters three, and a plane-1 emoji such as `U+1F600` four. Codepoints beyond plane 0 need more than 16 bits, so they take 4 bytes in both UTF-8 and UTF-16.

Since no byte is ambiguous about its role, from any point in a UTF-8 stream you find the next character start by skipping continuation bytes (outside the course material this is called self-synchronisation).

## What bytes do to tokenizers: byte inflation

**Byte-level tokenizers** (vocabulary built over UTF-8 bytes, see [[MNLP-L04 - Subword Segmentation|L04]]) are affected by how characters are represented as bytes, not as codepoints.

> [!formula] Byte inflation
> $$\text{inflation}(s) = \frac{\lvert \text{UTF-8 bytes}(s) \rvert}{\lvert \text{characters}(s) \rvert}$$
>
> where:
> - $\lvert \text{UTF-8 bytes}(s) \rvert$: the length of string $s$ once encoded as UTF-8
> - $\lvert \text{characters}(s) \rvert$: the number of characters a reader perceives

For example, "hello" is 5 bytes for 5 characters, 1.0x; "café" in NFC is 5 bytes for 4 characters (é takes two), 1.25x; "中文" is 6 bytes for 2 characters, 3.0x.

Two consequences:

- **The more bytes are needed, the more aggressively text tends to be split into tokens**: more pieces per character.
- **Byte-level splitting can produce non-existent characters.** Cut a 3-byte character after its second byte and neither piece is a valid character.

Outside the course material: in practice CJK text often costs more tokens for the same content, depending on how much CJK the tokenizer was trained on.

## Where to break a string

Splitting a string can happen at three levels, each with a cost:

- **Byte-level breaking can split codepoints**, which can consist of several bytes, producing fragments that are not characters.
- **"Smarter" codepoint breaking is not ideal either**: it can break one character into pieces, for example separating a base letter from its diacritic.
- **Ideally, break at the grapheme level** (what the reader perceives as a character), but grapheme boundaries are **very language-specific**.

```
"café" in NFD, split three ways
bytes:      63|61|66|65|CC|81
codepoints: c |a |f |e |◌́
graphemes:  c |a |f |é
```

CC and 81 on their own are not characters, and at the codepoint level the accent is stranded.

**Mixing breaking strategies leads to misalignments**:

- **Span annotation systems** (SQuAD-style question answering stores answers as character offsets): offsets counted one way and read another give shifted answers.
- **Any counting across corpora** with different breaking strategies: counts are no longer comparable.

So breaking and length computation must be **consistent across documents**.

> [!tip] The Unicode stack
> A reader sees a **grapheme**, which is represented by **one or more codepoints**, each encoded in **1 to 4 bytes** (UTF-8). For é in NFD:
> ```
> reader sees      é               1 grapheme
> Unicode numbers  U+0065 U+0301   2 codepoints
> tokenizer eats   65 CC 81        3 bytes
> ```

## Other pitfalls: case, sorting, punctuation

**Lowercasing is language-dependent.** In English, lowercase(I) = i. In Turkish, lowercase(I) = ı (dotless i), because Turkish has two pairs: dotted İ / i and dotless I / ı. Lowercasing Turkish with English rules changes the word.

**Codepoint order is not alphabetical order**: accented letters' codepoints are not necessarily near their base letter's. é is `U+00E9`, far after z (`U+007A`), so a codepoint sort puts "émile" after "zoe".

**Languages use different alphabetical orders.** Swedish puts å, ä, ö **after z**, as separate letters. German sorts ä as ae, ö as oe, ü as ue, **interleaved** with the plain letters.

**Punctuation.** Defining a word is hard, but **punctuation adjacent to a word is not part of it**, so tokenizers split it off. Some also expand contractions (*He'd* to *He would*), another normalization decision and a lossy one, since *He'd* can also mean *He had*:

```
He'd say "I promise you!", but
then disappear.
→ He ' d say " I promise you ! " ,
  but then disappear .
→ He would say " I promise you ! " ,
  but then disappear .
```

Unicode gives every character a **general category** (letter, number, punctuation, and so on), so punctuation is matched by category rather than a hand-written list: `\p{P}` covers punctuation in every script, and `re.sub(r'(\p{P})', r' \1 ', s)` puts spaces around each mark. (Outside the course material: Python's built-in `re` lacks `\p{P}`; the third-party `regex` module has it.) A class such as `[.!?]` misses the Japanese 。, the Bengali ।, the Amharic ።.

**Punctuation normalization** (“...” against «...» against 「...」) has **no simple solution: the mappings must be encoded by hand**. It comes up in quotes, end-of-sentence markers, question marks and single quotes.

## What to be able to answer

- The English-first assumptions and the languages that break each (Thai spaces end clauses).
- What sets a language's resource tier; the three tiers; why "language independent" claims are suspect.
- Alphabetic, syllabic, logographic; the Devanagari, Arabic and Hangul cases.
- Script and language in both directions.
- The two disadvantages of code pages, and what Unicode fixes.
- Codepoint, character, glyph, grapheme; why two "café" strings fail to match.
- The four normalization forms: which are lossy, which for rendered text, which for search.
- What breaks silently without normalization, and the fix.
- UTF-8, UTF-16, UTF-32: bytes and ASCII compatibility; the UTF-8 byte rules; encoding by hand.
- Byte inflation and its two consequences.
- Byte, codepoint and grapheme breaking; where mixing them misaligns.
- Turkish lowercasing, Swedish and German sorting, punctuation handling.

## Exam questions

> [!exam]- State the four English-first assumptions built into standard NLP pipelines. For each, give languages where it fails and what goes wrong concretely.
> **Key points:** whitespace word boundaries fail for Chinese, Japanese, Thai; word-order grammar fails for case-marking Russian, Finnish, Korean; one word one meaning fails for agglutinative and compounding languages; left-to-right fails for Arabic, Hebrew, Urdu, Persian.
>
> 1. **Words are separated by whitespace.** Fails for Chinese, Japanese, Thai, Lao, Khmer, Burmese: there is no word delimiter, so whitespace tokenization returns one token for a whole Chinese sentence. Thai uses spaces to end **clauses or sentences** and has no dedicated full stop, so a space does not mark a word boundary.
> 2. **Grammatical relations are encoded by word order.** Fails for languages with rich morphology, especially case marking (Russian, Finnish, Korean): who did what to whom is marked on the noun, word order is more flexible, and position says much less about a word's role.
> 3. **One word = one unit of meaning.** Fails for agglutinative languages (Finnish, Turkish, Hungarian) and compounding languages (German, Dutch, Swedish): *Bundesverfassungsgericht* is one token holding federal + constitutional + court.
> 4. **Text flows left to right.** Fails for Semitic languages and languages in Arabic script (Arabic, Hebrew, Urdu, Persian): the text is stored left to right but rendered right to left, so the first character in the string is the rightmost on screen.

> [!exam]- Argue that "script is not language" holds in both directions, and explain what each direction breaks in an NLP pipeline.
> **Key points:** one script serves many languages (Latin, Arabic); script detection does not identify the language; one language uses several scripts (Serbian, Uzbek); language does not fix the script, so train and test data can share almost no letters.
>
> - **One script, many languages:** Latin (English, German, French, most European languages, several African, Turkic and Asian languages), Arabic (Arabic, Persian, Urdu, Pashto, Sorani), Cyrillic (Russian, Belarusian, Serbian), Devanagari (Hindi, Marathi, Nepali), Ethiopic (Amharic, Tigrinya). So **detecting the script does not identify the language**: Latin narrows almost nothing, and Arabic script still leaves five candidates.
> - **One language, many scripts:** Serbian (Cyrillic and Latin), Kurdish (Latin and Arabic), Uzbek (Latin and Cyrillic), Punjabi (Gurmukhi and Arabic). So **knowing the language does not tell you the script**: a Serbian corpus can mix both, even in one document, and a model trained on one script and tested on the other shares almost no letters with the test data, though the language is the same.
> - Some scripts are language-specific (Armenian, Georgian, Greek, Sinhala, Khmer).
> - Also: one sentence can mix several writing systems (Japanese uses Katakana, Kanji and Hiragana together), so a single "script of this text" label can fail.

> [!exam]- Why is the English-centric bias of NLP structural, and why do claims of "language independence" deserve suspicion?
> **Key points:** resources exist mostly in English; non-English data are by-products or translations of English resources; coverage follows commercial and military interest, not speaker numbers; pipelines encode English properties; language independence is rarely tested.
>
> - Most resources are in English: annotated data, unannotated data, toolkit coverage (dictionaries, analyzers), benchmarks. Most publications also focus on English.
> - Non-English resources arise mostly as **by-products** (translations, product reviews) or as **translations of English annotated resources** (so the annotation scheme stays English-shaped), less so as dedicated efforts.
> - Coverage tracks **commercial interest and military relevance**, not speaker numbers. Over 7,000 languages exist. The high-resource tier is English and Mandarin, then, after a big gap, French, Spanish, Arabic and Russian. German, Italian, Dutch, Korean and Japanese are medium resource. The rest are low resource, which on these tiers includes most languages with over ten million speakers.
> - Standard pipelines encode properties of English (whitespace tokenization, fixed word order, closed vocabularies) as if they were properties of language.
> - "Language independent" claims are usually untested outside English or a very small set of languages.
> - The bias is understandable, since resource creation costs time and money.

> [!exam]- Compare alphabetic, syllabic and logographic writing systems and explain what each, plus the Devanagari, Arabic and Hangul cases, implies for NLP.
> **Key points:** alphabetic: sounds, small inventory; syllabic: whole syllables; logographic: words or morphemes, thousands of symbols; Devanagari alphasyllabic (consonant + inherent vowel); Arabic writes the consonant root and omits vowels, so the model must disambiguate; Hangul letter shapes encode articulation.
>
> - **Alphabetic:** each symbol is a consonant or vowel sound, small inventory (Latin, Cyrillic, Hangul).
> - **Syllabic:** each symbol is a whole syllable (Hiragana, Katakana).
> - **Logographic:** each symbol is a word or morpheme directly, little sound information (Chinese Hanzi, Japanese Kanji).
> - **Inventory size** sets the cost of a character: tens of symbols for an alphabet, thousands for a logographic script, which a character-level vocabulary must budget for.
> - **Devanagari** is alphasyllabic: a consonant letter carries an inherent vowel.
> - **Arabic** writes the consonant root and omits vowels, so one written form can stand for several words (كتب can be read *kataba*, *kutiba* or *kutub*, examples from outside the course material). The writing is lossier than the language and the model must resolve the ambiguity from context. No tokenizer can recover vowels that were never written.
> - **Hangul** is alphabetic, and its letter shapes encode articulatory features.

> [!exam]- Explain the difference between a grapheme, a codepoint and a byte, and show how confusing them corrupts tokenization, deduplication and span annotation.
> **Key points:** grapheme is the perceived character; one or more codepoints per grapheme; 1 to 4 UTF-8 bytes per codepoint; NFC/NFD mismatch silently breaks dedup and search; mixed breaking conventions misalign spans.
>
> - **Grapheme:** what the reader sees as one character. **Codepoint:** Unicode's abstract number, one or more per grapheme. **Byte:** UTF-8 stores each codepoint in 1 to 4 bytes.
> - Example: NFD é is 1 grapheme, 2 codepoints (U+0065 U+0301), 3 bytes (65 CC 81).
> - **Tokenization:** byte-level tokenizers see the byte stream. Byte-level splits can produce non-existent characters; codepoint splits can strand a diacritic from its base letter.
> - **Deduplication, search, equality, prefix matching** assume identical codepoint sequences, so NFC and NFD copies of the same text silently fail to match.
> - **Span annotation** (SQuAD-style QA): offsets recorded under one breaking convention and read under another give shifted answers. Counts across corpora with different conventions are not comparable.
> - Fix: normalize consistently in the pipeline, and keep breaking and length computation consistent across documents.

> [!exam]- Crawled multilingual data mixes normalization forms. Which Unicode normalization form would you use for search indexing and for comparing rendered text, and is NFKD suitable for generation targets? Why?
> **Key points:** canonical lossless, compatibility lossy; NFD for comparing rendered text; NFKD for search indexing and analysis; NFKD not necessarily for generation; normalize consistently.
>
> - The four forms form a grid: canonical (lossless) NFC and NFD; compatibility (lossy) NFKC and NFKD; composed (C) against decomposed (D).
> - **Comparing rendered text:** NFD, which splits into base characters plus combining marks and preserves semantic and visual equivalence. It is non-lossy, so you can decompose and recompose. (Outside the course material: NFC also gives a valid equality check, but the expected answer is NFD.)
> - **Search indexing or NLP analysis:** NFKD. Its lossiness helps: the ligature ﬁ and the pair fi become the same, ① becomes 1.
> - **Generation:** NFKD is not necessarily suitable. A model trained on NFKD targets never sees, so cannot produce, the folded-away characters (for example the long s ſ becomes plain s).
> - Whatever you choose, apply it consistently in the pipeline (for example canonical decomposition then composition, which ends in NFC), otherwise dedup, search and equality checks are silently inaccurate.
>
> Losing marks: getting the analysis against generation rule backwards, or claiming NFKD is reversible.

> [!exam]- Explain byte inflation for byte-level tokenizers and compute it for "hello", "café" and "中文". What are its consequences?
> **Key points:** UTF-8 bytes over perceived characters; 1.0x, 1.25x (café in NFC), 3.0x; more bytes means more aggressive splitting; byte splits can produce non-existent characters.
>
> - $\text{inflation}(s) = \lvert\text{UTF-8 bytes}(s)\rvert \,/\, \lvert\text{characters}(s)\rvert$, characters meaning what a reader perceives.
> - "hello": 5 bytes / 5 = **1.0x**. "café" (NFC, é is 2 bytes): 5 / 4 = **1.25x**. "中文" (3 bytes each): 6 / 2 = **3.0x**.
> - More bytes per character means text tends to be split more aggressively, into more pieces per character; byte splits can also produce fragments that are not characters at all.
> - Outside the course material: in practice CJK text often costs more tokens for the same content, depending on the tokenizer's training data.

> [!exam]- Encode U+4E2D (中) in UTF-8 by hand, stating which byte rules you use.
> **Key points:** 15 bits exceed 11, so 3-byte form; 1110xxxx 10xxxxxx 10xxxxxx; result E4 B8 AD; lead and continuation byte rules.
>
> - 0x4E2D = `100111000101101`: 15 bits, more than the 11 payload bits of the 2-byte form, so use the 3-byte form `1110xxxx 10xxxxxx 10xxxxxx` with $4 + 6 + 6 = 16$ payload bits.
> - Pad to 16 bits and split 4 + 6 + 6: `0100 | 111000 | 101101`.
> - Bytes: `11100100 10111000 10101101` (hex E4 B8 AD).
> - Rules: a byte starting `0` is a single byte (ASCII); a byte starting `11` begins a multi-byte sequence, with the number of leading 1s giving the length of the whole sequence (here 3); a byte starting `10` is a continuation byte.
> - Because no byte is ambiguous about its role, a reader can find the next character start from any offset by skipping `10xxxxxx` bytes.

## Flashcards

> [!card]- According to Monz, what determines how well a language is covered by NLP resources?
> **Commercial interest** and **military relevance**.

> [!card]- Is speaker count a good predictor of how well a language is resourced for NLP?
> **No.** Coverage follows commercial and military interest; on Monz's tiers most languages with over ten million speakers are low resource.

> [!card]- Which languages make up Monz's high-resource tier?
> English and Mandarin, then after a **big gap** French, Spanish, Arabic, Russian.

> [!card]- How do most non-English NLP resources come about?
> As **by-products** (translations, reviews) or **translations of English resources**; less often as dedicated efforts.

> [!card]- Why do claims that an NLP method is "language independent" deserve suspicion?
> They are usually **untested** outside English or a very small set of languages.

> [!card]- How does Chinese mark past tense in 我不知道她去哪儿了 ("I don't know where she went")?
> With the **sentence-final particle** 了; the verb 去 itself is unchanged.

> [!card]- What does Russian *knígi* (of the book, the books, books as object) show for NLP?
> One surface form fills several cells, so the **case cannot be read off the string**.

> [!card]- What does it mean for a language such as Finnish to be agglutinative?
> Morphemes are **glued on in sequence**, each adding one piece of meaning.

> [!card]- Why can more training data not fix a fixed word-level vocabulary for Finnish?
> Agglutination yields **combinatorially many** word forms, so unseen ones always remain. This motivates **subword segmentation**.

> [!card]- Which four English-first assumptions are built into standard NLP pipelines?
> Words separated by **whitespace**; grammar encoded by **word order**; **one word = one meaning**; text flows **left to right**.

> [!card]- Which languages break the assumption that words are separated by whitespace?
> Chinese, Japanese, Thai, Lao, Khmer, Burmese.

> [!card]- Why do word-position features carry little information in Russian, Finnish or Korean?
> Grammatical relations are marked by **case on the noun**, so word order is flexible.

> [!card]- Which two kinds of language break the assumption "one word = one unit of meaning"?
> **Agglutinative** languages (Finnish, Turkish, Hungarian) and **compounding** languages (German, Dutch, Swedish).

> [!card]- In Arabic or Hebrew text, is the first character of the Unicode string the leftmost one on screen?
> **No.** Text is stored left to right in reading order but **rendered right to left**.

> [!card]- Which type of writing system has each symbol represent a consonant or vowel sound?
> **Alphabetic** (Latin, Cyrillic, Hangul), typically a small inventory.

> [!card]- What does each symbol represent in a syllabic writing system such as Hiragana or Katakana?
> A **whole syllable**, a combination of consonant and vowel.

> [!card]- What does each symbol represent in a logographic writing system such as Chinese Hanzi?
> A **word or morpheme** directly, with little sound information.

> [!card]- How does writing-system type affect a character-level vocabulary?
> Alphabets need **tens** of symbols; logographic scripts need **thousands**, which the vocabulary must budget for.

> [!card]- What is Devanagari called, where each consonant letter carries an inherent vowel?
> **Alphasyllabic**.

> [!card]- What does Arabic script leave out of a word such as كتب (ktb)?
> The **vowels**: only the consonant root is written.

> [!card]- Why is Arabic's vowel-less writing a problem no tokenizer can fix?
> The vowels were **never written**, so one written form can stand for several words; context must disambiguate.

> [!card]- What do the letter shapes of Korean Hangul encode, beyond the sound itself?
> **Articulatory features**: how the sound is produced.

> [!card]- Why can a single "script of this text" label fail for a Japanese sentence?
> One Japanese sentence mixes **Katakana, Kanji and Hiragana**.

> [!card]- Does detecting the script of a text identify its language?
> **No.** Latin serves many languages; Arabic script covers Arabic, Persian, Urdu, Pashto, Sorani.

> [!card]- Does knowing a text's language tell you its script?
> **No.** Serbian is written in Cyrillic and Latin, sometimes in the same document.

> [!card]- What happens to a model trained on Cyrillic Serbian and tested on Latin Serbian?
> It shares **almost no letters** with the test data, although the language is the same.

> [!card]- How many codes did ASCII define?
> **128**: enough for English letters, digits and punctuation.

> [!card]- How did code pages extend ASCII?
> By **reinterpretation**: the same 128 or 256 byte codes, read differently per language or script.

> [!card]- What are the two disadvantages of code pages?
> Gibberish when decoded with the **wrong table**; **one script per document**.

> [!card]- Why does decoding text with the wrong code page usually raise no error?
> The bytes are often **legal in both tables**, so you silently get wrong characters (mojibake).

> [!card]- What is Unicode?
> A **single universal character set** giving every character it encodes, in every supported script, one unique number, its **codepoint** (`U+number`).

> [!card]- Does Unicode define how characters are stored as bytes?
> **No.** Unicode defines codepoints; storing them as bytes is a separate **encoding** question (UTF-8, UTF-16, UTF-32).

> [!card]- Which two code-page problems does Unicode remove?
> No **table to guess** when decoding, and one document can **mix any scripts**.

> [!card]- How does Unicode relate to ASCII?
> Unicode is a **superset**: `U+0000` to `U+007F` are identical to ASCII.

> [!card]- How many codepoints does each Unicode plane contain?
> **65,536** ($2^{16}$), in 17 planes.

> [!card]- What follows from a codepoint lying outside plane 0, the Basic Multilingual Plane?
> It needs **more than 16 bits**, so it takes four bytes in UTF-8 and UTF-16, as emoji do.

> [!card]- Is every CJK character in the Basic Multilingual Plane (plane 0)?
> **No.** Planes 2 and 3 hold more CJK ideographs, mostly rare or historical.

> [!card]- What is the difference between a codepoint, a character and a glyph?
> Codepoint: abstract **number**. Character: what a reader **perceives**, possibly several codepoints. Glyph: a stylistic **rendering**.

> [!card]- Why does a naive comparison fail on two identical-looking "café" strings?
> One is NFC (é **one precomposed codepoint**), the other NFD (e **plus combining accent**): different sequences, lengths, hashes.

> [!card]- How does the Unicode stack run from what a reader sees down to what a byte-level tokenizer eats?
> A **grapheme**; one or more **codepoints**; 1 to 4 **bytes** per codepoint.

> [!card]- Which Unicode normalization forms are lossy?
> The **compatibility** forms NFKC and NFKD; the canonical forms NFC and NFD are lossless.

> [!card]- What does the K in NFKC and NFKD stand for?
> **Compatibility**: these forms also merge stylistic variants such as ligatures.

> [!card]- What does NFKD do that NFD does not?
> Breaks characters into their **most basic parts**, e.g. splitting ligatures, normalizing visual styling away.

> [!card]- Which normalization form should you use to compare rendered text?
> **NFD** (non-lossy, preserves semantic and visual equivalence).

> [!card]- Which form do you end in after canonical decomposition followed by composition?
> **NFC** (compatibility decomposition then composition gives NFKC).

> [!card]- Which normalization form should you use for search indexing or NLP analysis?
> **NFKD**.

> [!card]- Why is NFKD's lossiness useful for search indexing?
> It unifies variants: the ligature **ﬁ matches fi**, ① becomes 1.

> [!card]- Why is NFKD not necessarily suitable for generation targets?
> A model trained on them **never sees, so cannot produce**, the characters NFKD folded away.

> [!card]- What happens to the long s ſ under NFD, and under NFKD?
> NFD **keeps** ſ; NFKD replaces it with plain **s**, irreversibly.

> [!card]- How many bytes per codepoint does UTF-8 use?
> **1 to 4**, variable.

> [!card]- Which Unicode encoding uses exactly 4 bytes for every codepoint?
> **UTF-32**: simple for indexing, memory-hungry, rare in practice.

> [!card]- Which of UTF-8, UTF-16 and UTF-32 is ASCII compatible?
> Only **UTF-8**; the other two use at least two bytes per character.

> [!card]- How many UTF-8 bytes does a common CJK character such as 中 take?
> **3**.

> [!card]- In UTF-8, what does a byte starting with `0` signal?
> A complete **single-byte** character (ASCII).

> [!card]- In UTF-8, what does a byte starting with `11` signal?
> The **first byte** of a multi-byte sequence; its number of leading 1s gives the **whole sequence's** length.

> [!card]- What is a UTF-8 byte of the form `10xxxxxx`?
> A **continuation byte** inside a multi-byte sequence.

> [!card]- Why does 中 (U+4E2D, 15 significant bits) need the 3-byte UTF-8 form?
> Its bits exceed the **11 payload bits** of the two-byte form; the three-byte form carries sixteen.

> [!card]- Why can a reader find character boundaries from any offset in a UTF-8 stream?
> No byte is ambiguous about its role: skip `10xxxxxx` continuation bytes to reach a character start.

> [!card]- How is byte inflation defined for byte-level tokenizers?
> **UTF-8 bytes** of a string divided by its number of **perceived characters**.

> [!card]- Which has the highest byte inflation: "hello", "café" or "中文"?
> **中文**: 3 bytes per character, against 1 per character for "hello".

> [!card]- How does higher byte inflation affect byte-level tokenization?
> Text tends to be split **more aggressively**: more pieces per character.

> [!card]- Which preprocessing steps silently go wrong when crawled text mixes NFC and NFD?
> **Deduplication**, **search**, string equality and prefix matching: all assume identical codepoint sequences.

> [!card]- What is the fix for crawled data that mixes normalization forms?
> **Consistent normalization** in the pipeline, e.g. first decompose, then compose.

> [!card]- What does `iconv -f GB2312 -t UTF-8` do if the source encoding is not correct?
> It **fails**, which beats silent mojibake.

> [!card]- What goes wrong when breaking a string at the byte level?
> Multi-byte codepoints get split into fragments that are **not characters** at all.

> [!card]- What goes wrong when breaking a string at the codepoint level?
> A base letter can be **separated from its combining diacritic** (e + ◌́).

> [!card]- Why is grapheme-level breaking hard?
> Grapheme boundaries are **very language-specific**.

> [!card]- Where does mixing string-breaking strategies cause misalignment bugs?
> **Span annotation** (SQuAD-style offsets shift) and **counting across corpora**.

> [!card]- Why does lowercasing the letter I go wrong for Turkish?
> Turkish lowercase I is dotless **ı**; dotted İ pairs with i. English rules change the word.

> [!card]- Why can strings not be sorted alphabetically by codepoint?
> Accented letters' codepoints are **not necessarily near** their base letter's.

> [!card]- How do Swedish and German differ in sorting å, ä, ö?
> Swedish puts them **after z** as separate letters; German **interleaves** them (ä as ae).

> [!card]- Why is punctuation normalization (“...” against «...» against 「...」) hard?
> There is **no simple solution**: the mappings between traditions must be **encoded by hand**.

> [!card]- How should a multilingual tokenizer match punctuation?
> By **Unicode general category P** (`\p{P}`), not a hand-written class like `[.!?]`, which misses 。 and ।.

> [!card]- True or false: Unicode defines both each character's codepoint and how it is stored as bytes.
> **False.** Unicode defines codepoints only; bytes are the job of an encoding such as UTF-8.

> [!card]- True or false: How well a language is covered by NLP resources depends mainly on its number of speakers.
> **False.** It depends on commercial interest and military relevance.

> [!card]- True or false: UTF-8 is ASCII compatible.
> **True.** `U+0000` to `U+007F` are encoded as single bytes identical to ASCII.

> [!card]- True or false: UTF-32 is ASCII compatible.
> **False.** It uses exactly 4 bytes for every codepoint, ASCII included.

> [!card]- True or false: UTF-16 encodes codepoints outside plane 0 in 4 bytes.
> **True.** They need more than 16 bits; plane-0 codepoints take 2 bytes.

> [!card]- True or false: NFKD normalization is lossless.
> **False.** Compatibility forms are lossy (ſ becomes s); only the canonical forms NFC and NFD are lossless.

> [!card]- True or false: NFKD normalization makes the ligature ﬁ and the pair fi identical.
> **True.** Compatibility decomposition splits ligatures, which is why NFKD suits search indexing.

> [!card]- True or false: The form to use for comparing rendered text is NFD.
> **True.** NFD is non-lossy and preserves semantic and visual equivalence; NFKD is for search and analysis.

> [!card]- True or false: Code pages allow only one script per document.
> **True.** A document is read with one table, so scripts cannot be mixed.

> [!card]- True or false: Detecting a text's script is enough to identify its language.
> **False.** Arabic script alone covers Arabic, Persian, Urdu, Pashto and Sorani.

> [!card]- True or false: NFC and NFD versions of "café" render identically but are unequal as strings.
> **True.** Same appearance, different codepoint sequences (4 against 5 codepoints).

> [!card]- True or false: Deduplicating a crawled corpus that mixes NFC and NFD raises an error on the mismatched copies.
> **False.** It fails silently: both copies are kept and the numbers are just wrong.

> [!card]- True or false: Grapheme boundaries are language-specific.
> **True.** They are very language-specific, which makes the ideal grapheme-level breaking hard.

> [!card]- True or false: Byte-level tokenizers tend to split text more aggressively in scripts that need more UTF-8 bytes per character.
> **True.** More bytes per character means more pieces per character.

> [!card]- True or false: In UTF-8, the number of leading 1s in the first byte gives the length of the whole sequence.
> **True.** The first byte is included: `1110xxxx` starts a 3-byte sequence, so two bytes follow.

> [!card]- True or false: In Unicode, every character a reader perceives has exactly one codepoint.
> **False.** A perceived character (grapheme) can be several codepoints, such as e plus a combining accent.

> [!card]- True or false: Whitespace tokenization works for Thai because Thai uses spaces between words.
> **False.** Thai spaces end clauses or sentences, not words.

> [!card]- Multiple selection: which languages break the assumption that words are separated by whitespace? (A) Thai (B) Russian (C) Khmer (D) German
> **A and C.** Russian breaks the word-order assumption; German breaks one-word-one-meaning through compounding.

> [!card]- Multiple selection: which are true of script and language? (A) Serbian is written in Cyrillic and Latin (B) Arabic script identifies the Arabic language (C) Armenian has a language-specific script
> **A and C.** B is false: Arabic script also writes Persian, Urdu, Pashto and Sorani.

> [!card]- Multiple selection: which are true of Unicode normalization? (A) NFC and NFD are lossless (B) NFKD is the recommended form for generation targets (C) NFKD maps the long s ſ to s (D) NFD merges a base letter and combining mark into one codepoint
> **A and C.** B is false: NFKD is for analysis, not necessarily generation. D is false: NFD decomposes; NFC composes.

> [!card]- Multiple selection: which are true of UTF-8? (A) It uses 1 to 4 bytes per codepoint (B) The number of leading 1s in a lead byte gives the sequence length (C) Emoji fit in 3 bytes (D) From a random offset you cannot tell whether you are at a character start
> **A and B.** C is false: plane-1 emoji need 4 bytes. D is false: continuation bytes start with `10`, so a character start is recognisable.

> [!card]- Multiple selection: which are true of string breaking? (A) Byte-level breaking can produce fragments that are not characters (B) Codepoint-level breaking always keeps a diacritic with its base letter (C) Grapheme boundaries are language-specific (D) Mixing breaking strategies across corpora is harmless
> **A and C.** B is false: e and a combining accent are two codepoints and can be split. D is false: spans misalign and counts become incomparable.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L01 - Overview]]
- **Next:** [[MNLP-L03 - Morphology and Word Formation]]
- **Project:** [[MNLP - Mini Project]]
- **Related concept:** [[Tokenization]]
- **Source:** `multilingual-nlp.pdf`, 33 slides, Christof Monz, Informatics Institute, University of Amsterdam
- **Further reading (recommended in the lecture):** Joel Spolsky (2003), *The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets*; Nikita Prokopov (2023), *The Absolute Minimum Every Software Developer Must Know About Unicode in 2023* (both blog posts)
