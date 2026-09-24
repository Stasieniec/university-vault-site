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
---

# MNLP-L02: Multilinguality and Writing Systems

> [!abstract] Overview
> You split on whitespace, you cap the vocabulary, you lowercase everything, and you measure string length with `len()`. Which languages have you just quietly destroyed?
>
> The answer is most of them, and the interesting part is that none of those four decisions looks like a decision. They look like what text *is*. This lecture is the argument that they are not properties of language, they are properties of English written in Latin script, and that every one of them fails somewhere in the 7,000 languages that exist.
>
> The lecture then works downwards to the layer where the damage actually happens: bytes. A grapheme is what a reader sees, a codepoint is what Unicode identifies, a byte is what your tokenizer eats, and these three are not in one-to-one correspondence. Getting that stack wrong is how you end up deduplicating a corpus that silently keeps two copies of every accented word.

## 1. The premise: "NLP" usually means "English NLP"

Almost every resource that makes NLP work exists first, and often only, in English.

- **Annotated/labeled training data**
- **Unannotated training data**
- **Coverage of existing toolkits** (dictionaries, analyzers)
- **Benchmarks/competitions**

Non-English resources do exist, but look at *how* they come into being, because the mechanism determines the quality:

- As a **naturally occurring by-product**: translations, product reviews. Nobody made them for NLP, they just happen to be there.
- As a **translation of existing (annotated) English resources**. The annotation scheme is then English-shaped even when the text is not.
- **Less so as a dedicated effort**, which is the category that would actually fix the problem.

And **most publications focus on English**, so the methods literature inherits the same bias as the data.

## 2. Language diversity and the resource tiers

There are **over 7,000 spoken languages**. The distribution is a long tail, but the head is not tiny: **more than 100 languages have over 10 million speakers**.

The degree to which a language is covered by NLP resources is not a function of how many people speak it. It depends on:

- **Commercial interest**
- **Military relevance**

Monz's three tiers:

| Tier | What exists | Languages |
|---|---|---|
| **High resource** | Large labeled and unlabeled data, many tools | English, Chinese (Mandarin), then a big gap, then French, Spanish, Arabic, Russian |
| **Medium resource** | Limited labeled and reasonable unlabeled data, some tools | German, Italian, Dutch, Korean, Japanese (EU and Asian languages) |
| **Low resource** | No labeled data, limited unlabeled data, no tools | All remaining roughly 7,000 languages |

> [!warning] The gap is inside the top tier, not just below it
> The slide puts an explicit "[big gap]" between {English, Mandarin} and {French, Spanish, Arabic, Russian}. So even "high resource" is two very different things. Dutch, spoken by roughly 24 million people in one of the richest countries on earth, is only medium resource. Speaker count is a bad predictor and money is a good one.

## 3. One sentence, two languages

> [!example] How many words is this?
> **English:** I don't know where she went.
>
> **Chinese:** 我不知道她去哪儿了。
>
> The English sentence has 6 whitespace-separated tokens. The Chinese sentence has **no spaces at all** (well, almost no spaces), so whitespace tokenization returns exactly one token for the whole sentence.

The gloss, which is where the real trouble shows up:

| 我 | 不 | 知道 | 她 | 去 | 哪儿 | 了 | 。 |
|---|---|---|---|---|---|---|---|
| wǒ | bù | zhīdào | tā | qù | nǎr | le | |
| I | NEG | know | she | go | where | SFP | PUNC |

Three observations from the lecture:

1. **There is no tense marking on the verb** 去. English distinguishes "go" from "went" by changing the verb. Chinese does not touch the verb at all.
2. **Past tense comes from the sentence-final particle** 了 (SFP, sentence-final particle). The tense information is not in the verb, it is at the end of the clause, arbitrarily far from it.
3. **Both the matrix clause (I NEG know) and the embedded clause (she go where) are SVO.** So word order is not the difference here, and a naive "Chinese is exotic, English is normal" story does not survive contact with the data.

Notice what point 2 does to a model that assumes morphological features live on the word they modify. In Chinese, the tense of the embedded verb is carried by a particle that a tokenizer will happily treat as an unrelated token.

Notice also that the 9 Chinese characters correspond to **7 words plus punctuation**, so the character-to-word ratio is not 1:1 either. 知道 (know) and 哪儿 (where) are two characters each.

## 4. Morphology: word formation

> [!definition] Morphology
> **Morphology** in linguistics is the study of the internal structure of words and how they are formed from smaller units of meaning.

The vocabulary you need for the rest of the course:

| Term | Definition | Example |
|---|---|---|
| **Morpheme** | The smallest unit of meaning or grammatical function in a language | *dogs* has two morphemes: *dog* (base meaning) and *-s* (plural) |
| **Root / base** | The core part of a word that carries the primary meaning | *friend* in *unfriendly* |
| **Affix** | Prefixes (added to the front) and suffixes (added to the back) that modify a root's meaning or grammatical role | *un-* and *-ly* in *unfriendly* |

The lecture then runs the same concept, "happy", through three languages at three levels of morphological complexity.

### 4.1 English has limited morphological complexity

happy, happier, happiest, unhappy, unhappier, unhappiest, happiness, unhappiness

That is 8 surface forms from one root. A vocabulary of a few tens of thousands of word types covers English acceptably, which is exactly why closed vocabularies felt like a reasonable design decision for so long.

### 4.2 Russian has more complex morphology

One noun, *kniga* ("book"), inflects for six cases and two numbers:

| Form | Case and number | Meaning |
|---|---|---|
| kníga | nom.sg | a/the book |
| knígi | gen.sg | of the book |
| knígye | dat.sg | to the book |
| knígu | acc.sg | a book as direct object |
| knígoy | inst.sg | with the book |
| knígye | prep.sg | about/in a book |
| knígi | nom.pl | the books |
| knig | gen.pl | of the books |
| knígam | dat.pl | to books |
| knígi | acc.pl | books as direct object |
| knígami | inst.pl | with the books |
| knígakh | prep.pl | about/in books |

Twelve cells, but only **nine distinct forms**. *knígye* covers both dat.sg and prep.sg, and *knígi* covers gen.sg, nom.pl and acc.pl at once. This syncretism cuts both ways for NLP: fewer types to learn, but the surface form is ambiguous between grammatical functions, so you cannot read the case off the string.

### 4.3 Finnish has very complex morphology

Finnish is **agglutinative**: morphemes stack, each contributing one piece of meaning, and the word grows without bound.

> [!example] onnellinen = "happy"
> `onni` ("happiness") `+ -llinen` (adjective-forming suffix, roughly "having the quality of")
>
> Literally something like "having happiness", so "happy".

> [!example] onnellisempi = "happier"
> `onnellinen` has an oblique stem `onnellis` (adjectives ending in *-nen* often swap this for *-s* before suffixes).
>
> `onnellis- + -empi` (*-mpi* is a comparative suffix, plus a linking vowel).

> [!example] onnellisimmillanikin = "even at my happiest"
> Six morphemes in one orthographic word:
>
> ```
> onnellis  -imm-  -i-   -lla   -ni     -kin
> │         │      │     │      │       │
> │         │      │     │      │       └─ "even"
> │         │      │     │      └───────── 1st person singular possessive ("my")
> │         │      │     └──────────────── adessive case ("at/on/in")
> │         │      └────────────────────── plural oblique-case marker
> │         └───────────────────────────── superlative marker
> └─────────────────────────────────────── stem, "happy"
> ```

The English translation "even at my happiest" needs four words, one of them a preposition. Finnish needs one token. Any model with a fixed word-level vocabulary will simply have never seen *onnellisimmillanikin*, and there are combinatorially many more like it, so the problem cannot be fixed by collecting more data. This is the motivation for subword modelling, which the course picks up in the next deck.

## 5. Four assumptions English lets you get away with

This is the core of the lecture and the most likely exam material. Each row is an assumption that is invisible in English pipelines and false somewhere important.

| Common assumption | Where it fails | Concretely |
|---|---|---|
| **Words are separated by whitespace** | Chinese, Japanese, Thai, Lao, Khmer, Burmese | No word delimiter at all. Thai additionally uses spaces to end **clauses or sentences** and has no dedicated full stop, so a space means the opposite of what your tokenizer thinks it means |
| **Grammatical relations are encoded by word order** | Languages with rich morphology, especially case marking: Russian, Finnish, Korean | Who did what to whom is marked on the noun, so word order is free and position-based features carry much less information |
| **One word = one unit of meaning** | Agglutinative languages (Finnish, Turkish, Hungarian); compounding languages (German, Dutch, Swedish) | *Bundesverfassungsgericht* = federal + constitutional + court, one token, three concepts |
| **Text flows from left to right** | Semitic languages and languages in Arabic script: Arabic, Hebrew, Urdu, Persian | Rendering runs right to left |

> [!warning] The directionality trap
> **Unicode is written left to right but rendered right to left.** The order of codepoints in the file is *logical* order, and the visual order you see on screen is produced by the bidirectional algorithm at render time.
>
> The consequence: "the first character of the string" and "the leftmost character on screen" are different characters in Arabic and Hebrew. Any code that assumes they are the same, character offsets, span annotation, truncation, will be wrong in a way that is invisible to a developer who cannot read the script.

## 6. Why these assumptions are inherent, not accidental

- The **overwhelming majority of NLP research and resources target a small handful of languages**.
- **Standard pipelines encode assumptions that are properties of English, not of language in general**: whitespace tokenization, fixed word order, closed vocabularies.
- **Language independence claims are usually untested outside English**, or outside a very small subset of languages. A method is declared language-independent because nobody checked.
- This is also **understandable**, because resource generation costs time and money. The bias is structural rather than malicious, which is precisely why it does not fix itself.

## 7. From small to large

The lecture organises the whole course by unit size, from characters up to paragraphs.

```
  Paragraphs           discourse structure, anaphora, ellipsis
       ▲
  Clauses / sentences  grammar or syntax, sentence semantics
       ▲
  Words                word formation (morphology), lexical meaning/semantics
       ▲
  Characters           size of the character set, writing systems
```

This lecture lives at the bottom two rungs. Everything above them inherits whatever you get wrong down here.

## 8. Writing systems

### 8.1 The same sentence, eleven scripts

The slide shows "Amsterdam is the capital of the Netherlands" written in eleven different scripts. Reproduced in full, because the visual variety *is* the argument:

| Script | Language | Sentence |
|---|---|---|
| Armenian | Armenian | Ամստերդամը Նիդերլանդների մայրաքաղաքն է: |
| Ethiopic (Ge'ez) | Amharic | አምስተርዳም የኔዘርላንድስ ዋና ከተማ ናት። |
| Japanese (Kanji, Katakana, Hiragana together) | Japanese | アムステルダムはオランダの首都です。 |
| Cyrillic | Russian | Амстердам — столица Нидерландов. |
| Georgian (Mkhedruli) | Georgian | ამსტერდამი ნიდერლანდების დედაქალაქია. |
| Arabic | Pashto | امستردام د هالنډ پلازمېنه ده. |
| Canadian Aboriginal syllabics | Inuktitut | ᐋᒻᔅᑕᑕᒻ ᐊᖓᔪᖅᑳᖃᕐᕕᐅᕗᖅ ᓂᑕᓚᓪᒥ. |
| Latin | Somali | Amsterdam waa caasimadda Netherlands. |
| Bengali | Bengali | আমস্টারডাম নেদারল্যান্ডসের রাজধানী। |
| Thai | Thai | อัมสเตอร์ดัมเป็นเมืองหลวงของประเทศเนเธอร์แลนด์ |
| Latin | Vietnamese | Amsterdam là thủ đô của Hà Lan. |

Things worth noticing while looking at that list:

- The **Japanese line mixes three writing systems in one sentence**: アムステルダム and オランダ are Katakana, 首都 is Kanji, は, の and です are Hiragana. A single "which script is this text in" label cannot describe it.
- The **Thai line has no spaces anywhere**, not even between words, and no full stop at the end.
- The **sentence terminator is a different character in almost every row**: `:` in Armenian (as reproduced here it is an ASCII colon; the proper Armenian full stop is `։`, U+0589), `።` in Amharic, `。` in Japanese, `।` in Bengali, nothing at all in Thai, and `.` in the Russian, Georgian, Pashto, Inuktitut, Somali and Vietnamese lines. A regex looking for `[.!?]` finds the end of only six of these eleven sentences.
- **Two rows are Latin script but different languages** (Somali and Vietnamese), and Vietnamese piles diacritics on top: *thủ đô của* carries tone marks and a stroked *đ*.

### 8.2 Types of writing system

> [!definition] Alphabetic
> - Each symbol represents a **consonant or vowel sound**
> - Typically a **small inventory** of symbols
> - Examples: Latin, Cyrillic, Hangul

> [!definition] Syllabic
> - Symbols represent **whole syllables**, that is, combinations of vowels and consonants
> - Examples: Japanese Hiragana / Katakana

> [!definition] Logographic
> - Each symbol represents **words or morphemes directly**
> - Typically **little sound information**
> - Examples: Chinese Hanzi, Japanese Kanji

The practical consequence is inventory size, and therefore what a "character" costs you. An alphabetic script needs tens of symbols. A logographic script needs thousands, and a model with a character-level vocabulary has to budget for that.

### 8.3 Five worked examples

The lecture then shows the same idea concretely, one word per system:

| System | Example | Reading | What the symbols encode |
|---|---|---|---|
| **Latin** | water | water | Alphabetic, sounds spelled out with letters |
| **Devanagari** | पानी | pani | Alphasyllabic, consonant plus an inherent vowel |
| **Han** | 水 | shui | Logographic, one symbol is one morpheme |
| **Arabic** | كتب | ktb | Root only, vowels are omitted |
| **Korean** | 한글 | Korean (Hangul) | Letter shapes encode articulatory features |

Two of these have standard typological names that the slide does not use, and it is worth attaching them because the exam-relevant taxonomy in the reading uses them:

- The **Devanagari** pattern, a consonant carrying an inherent vowel that other marks override, is what typologists call an **abugida**. पानी is *pā* + *nī*, two akshara, not *p* + *ā* + *n* + *ī*: each is a consonant letter (प *pa*, न *na*) whose inherent *a* is overridden by a vowel sign (ा *ā*, ी *ī*).
- The **Arabic** pattern, consonantal skeleton written and vowels left out, is what typologists call an **abjad**. كتب is the root k-t-b, which underlies *kataba* ("he wrote"), *kitāb* ("book"), *maktab* ("office") and many more. The written form كتب itself can be read *kataba* ("he wrote"), *kutiba* ("it was written") or *kutub* ("books"), and does not tell you which. (*kitāb* and *maktab* are spelled differently, كتاب and مكتب, because long vowels and the *m-* prefix are written.)

> [!warning] Why the abjad case is nasty for NLP
> The Arabic writing system is systematically **lossier than the language**. A single written form maps to several distinct words, so ambiguity that other languages resolve in the orthography has to be resolved by the model from context. This is not a tokenizer bug you can fix, it is information that was never in the input.

**Korean Hangul** is the odd one out and is included to break the neat taxonomy. It is alphabetic, each *jamo* is a consonant or a vowel, but the jamo are grouped into square syllable blocks, and the letter shapes themselves are designed to depict the position of the tongue and lips. So it is alphabetic by symbol-to-sound mapping and syllabic by visual arrangement.

### 8.4 Script is not language

This is the distinction the lecture wants you to hold, because tooling constantly conflates the two.

**One writing system, many languages:**

| Writing system | Languages |
|---|---|
| **Latin** | English, German, French, most European languages except some eastern European ones, but also several African languages, Turkic and Asian languages |
| **Arabic** | Arabic, Persian, Urdu, Pashto, Sorani, ... |
| **Cyrillic** | Russian, Belarusian, Serbian, ... |
| **Devanagari** | Hindi, Marathi, Nepali, ... |
| **Ethiopic (Ge'ez)** | Amharic, Tigrinya, ... |

**Language-specific writing systems:** Armenian, Georgian, Greek, Sinhala, Khmer, ...

**Languages with more than one writing system:**

| Language | Writing systems |
|---|---|
| Serbian | Cyrillic and Latin |
| Kurdish | Latin and Arabic |
| Uzbek | Latin and Cyrillic |
| Punjabi | Gurmukhi and Arabic |

> [!tip] The two consequences you should be able to state
> 1. **Detecting the script does not identify the language.** Seeing Latin script narrows nothing. Seeing Arabic script leaves you choosing between Arabic, Persian, Urdu, Pashto and Sorani, which are not even all in the same family.
> 2. **Detecting the language does not tell you the script.** A Serbian corpus can contain both Cyrillic and Latin text, sometimes in the same document. If you train on one and test on the other, your model sees no shared characters at all, despite it being the same language.

## 9. Encoding writing systems: ASCII and code pages

Before Unicode, the answer to "how do I put a script in a file" was: pick a table and hope everyone else picked the same one.

**ASCII (1963)** assigned **128 codes**, which is enough for English letters, digits, and punctuation. That is the entire design brief, and it worked because in 1963 the requirement was English.

**Code pages** extended this by reinterpretation rather than by expansion:

- Use the same **128 (or 256) codes in a byte** as ASCII
- **Interpret the encoding differently**, depending on the language or script

| Code page | Covers |
|---|---|
| **Latin-1** | Western Europe |
| **Shift-JIS** | Japanese |
| **GBK** | Chinese |
| **KOI8-R** | Russian |
| ... | ... |

The two disadvantages, both fatal:

- **Gibberish if text encoded in one script is decoded in a different one.** The bytes are legal in both tables, so nothing errors, you just get the wrong characters. This is mojibake.
- **You cannot use multiple scripts in the same document.** One document, one table, one script.

```
  bytes on disk:   D0 9F D1 80 D0 B8 D0 B2 D0 B5 D1 82
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  decoded as UTF-8              decoded as Windows-1252
        │                               │
      Привет                        ÐŸÑ€Ð¸Ð²ÐµÑ‚

  Same bytes. No error raised. One of these is your training data.
```

(Strict Latin-1 maps bytes `9F`, `80` and `82` to invisible C1 control characters, so the visible string above is what Windows-1252, the Microsoft superset of Latin-1, produces. That is also what most real mojibake looks like.)

## 10. Unicode

> [!definition] Unicode
> Unicode is a **single, universal character set**, maintained by the Unicode Consortium (`https://unicode.org/`), in which **every character, in every supported script, gets exactly one unique number**.

That unique number is called a **codepoint**, written as `U+number`:

- `U+0041` for the Latin letter "A"
- `U+4E2D` for the Han character 中

Two properties that carry most of the practical weight:

- **Unicode defines meaning, not bytes.** A codepoint is an abstract identifier. How to store it as bytes on disk or in memory is a separate **encoding** question, answered by UTF-8, UTF-16 or UTF-32 (section 13).
- **Unicode is a superset of ASCII.** `U+0000`–`U+007F` are identical to ASCII. This is also the reason English-only Latin-1 documents are Unicode compatible for free, and the reason the migration ever happened at all.

### 10.1 Scale

- The full Unicode code space covers **over 1.1 million possible codepoints**
- **150,000+ codepoints are currently assigned**
- **168 scripts** are represented
- The whole set is organised into **17 planes**, spanning `U+0000`–`U+10FFFF`, each plane containing 65,536 codepoints

> [!warning] Typo on the slide
> The slide says each plane contains **65,356** codepoints. The real figure is $2^{16} = 65{,}536$, which is also what makes $17 \times 65{,}536 = 1{,}114{,}112$, the "over 1.1 million" number on the previous slide. Learn 65,536.

### 10.2 The planes

| Plane | Range | Name | Contents |
|---|---|---|---|
| 0 | `0000`–`FFFF` | Basic Multilingual Plane (BMP) | Support for most modern languages including the most common CJK Unified Ideographs; symbols such as currency; Private Use Area |
| 1 | `10000`–`1FFFF` | Supplementary Multilingual Plane (SMP) | Non-CJK ideographic historic scripts; modern scripts; symbols such as musical notation; emoji |
| 2 | `20000`–`2FFFF` | Supplementary Ideographic Plane (SIP) | Additional CJK Unified Ideographs, mostly historical, uncommon, or variants |
| 3 | `30000`–`3FFFF` | Tertiary Ideographic Plane (TIP) | CJK Unified Ideographs, mostly historical, that are not encoded in the BMP or SIP |
| 4-13 | `40000`–`DFFFF` | unassigned | |
| 14 | `E0000`–`EFFFF` | Supplementary Special-purpose Plane (SSP) | Tags and variation sequence selectors |
| 15-16 | `F0000`–`10FFFF` | Supplementary Private Use Area planes (SPUA-A/B) | Private Use Area |

The practical reading: **everything you normally handle is in plane 0**, and anything outside plane 0 needs more than 16 bits. That single fact is what makes UTF-16 awkward and what makes emoji four bytes.

### 10.3 A spread of examples

The slide shows a grid of ten characters with their codepoints, chosen to span the space:

| Character | Codepoint | What it is |
|---|---|---|
| a | `U+0061` | Basic Latin, ASCII range |
| b | `U+0062` | Basic Latin |
| c | `U+0063` | Basic Latin |
| d | `U+0064` | Basic Latin |
| e | `U+0065` | Basic Latin |
| ã | `U+00E3` | Latin-1 Supplement, a precomposed accented letter |
| & | `U+0026` | ASCII punctuation |
| ⻩ | `U+2EE9` | CJK Radicals Supplement, the "yellow" radical, visually a form of 黄 |
| ❁ | `U+2741` | Dingbats, an ornamental florette |
| (dog face) | `U+1F436` | Emoji, plane 1 (SMP) |

The point of the grid is that "character" covers all of this. Letters, accented letters, punctuation, a CJK radical that looks like a character but is a separate codepoint, a printer's ornament, and a pictograph, all in one namespace with one numbering scheme.

## 11. Codepoint is not character is not glyph

> [!definition] The three levels
> - **Codepoint**: an abstract number, for example `U+00E9`. Unicode's unit of identity.
> - **Character**: what a reader perceives as one unit. May be built from **several codepoints**.
> - **Glyph**: a stylistic rendering of a character.

The failure this sets up:

> [!example] Two identical-looking "café"
> ```
> café    U+0063 U+0061 U+0066 U+00E9              (4 codepoints)
> café    U+0063 U+0061 U+0066 U+0065 U+0301       (5 codepoints)
> ```
>
> The first sequence is **NFC** (Normalization Form Canonical **Composition**): é is the single precomposed codepoint `U+00E9`.
>
> The second is **NFD** (Normalization Form Canonical **Decomposition**): e (`U+0065`) followed by COMBINING ACUTE ACCENT (`U+0301`).
>
> They render identically. **A naive string similarity check will not match both sequences.** They are not equal, they do not have the same length, and they do not hash to the same value.

### 11.1 The composition and decomposition figure

The slide works two examples through the normalization forms. First the easy one:

```
                  Source
                    ã
        LATIN SMALL LETTER A WITH TILDE
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
      ┌─────┐              ┌─────┐
      │ NFC │              │ NFD │
      │  ã  │              │ a ◌̃ │
      │U+00E3│             │U+0061 U+0303│
      └─────┘              └─────┘
```

Then the one that shows why there are four forms and not two:

```
                       Source
                         ẛ̣
      LATIN SMALL LETTER LONG S WITH DOT ABOVE
            AND COMBINING DOT BELOW
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
   ┌───────────────┐          ┌──────────────────────┐
   │      NFC      │          │         NFD          │
   │    ẛ    ◌̣     │          │   ſ     ◌̣      ◌̇     │
   │ U+1E9B U+0323 │          │U+017F U+0323 U+0307  │
   └───────────────┘          └──────────────────────┘
   ┌───────────────┐          ┌──────────────────────┐
   │     NFKC      │          │        NFKD          │
   │       ṩ       │          │   s     ◌̣      ◌̇     │
   │     U+1E69    │          │U+0073 U+0323 U+0307  │
   └───────────────┘          └──────────────────────┘
```

Read the bottom row carefully. The **canonical** forms (NFC, NFD) keep the long s ſ, because ſ and s are different characters as far as canonical equivalence is concerned. The **compatibility** forms (NFKC, NFKD) throw the long s away and give you a plain `s` (`U+0073`). That is information destroyed, deliberately.

## 12. Types of decomposition

- **NFD (Canonical Decomposition):** splits characters into **base characters and combining marks**, and preserves semantic and visual equivalence.
- **NFKD (Compatibility Decomposition):** breaks characters down into their **most basic parts**, including splitting ligatures.
- **NFD is non-lossy.** You can decompose and then compose without losing styling or formatting details.
- **NFKD is lossy**, because it normalizes visual styling to its base form, and that cannot be reconstructed during composition.
- **Use NFD if you want to compare rendered text.**
- **Use NFKD for search indexing or NLP analysis**, but not necessarily for generation.

> [!warning] Typo on the slide
> The slide labels NFD as "Normalization Form **C** - Canonical Decomposition". NFD is Normalization Form **D**. The **K** in NFKD stands for compatibility (from the German *Kompatibilität*, since **C** was already taken by composition).

The full grid, which is the shape worth memorising:

| | Composed | Decomposed |
|---|---|---|
| **Canonical** (reversible) | **NFC** | **NFD** |
| **Compatibility** (lossy) | **NFKC** | **NFKD** |

The last bullet is the practical rule and it is easy to get backwards. For **analysis**, being lossy is a feature: you *want* the ligature ﬁ and the pair fi to be the same thing when indexing, and you want ① to become 1. For **generation**, being lossy is a bug: if you NFKD your training targets, your model can never learn to produce the character the user actually typed.

## 13. Unicode bytes: UTF-8, UTF-16, UTF-32

There is one Unicode character set but several ways to serialise codepoints into bytes.

| Encoding | Bytes per codepoint | ASCII compatible | Used by |
|---|---|---|---|
| **UTF-8** | 1 to 4 | Yes | The web, Linux, most file formats, virtually all modern NLP tooling |
| **UTF-16** | 2 or 4 | No, every character takes at least 2 bytes | Java, JavaScript strings, Windows internals, .NET |
| **UTF-32** | Exactly 4 | No, every character takes exactly 4 bytes | Rare in practice. Simple for indexing, but memory-hungry |

**UTF-8 is by far the most common form and is what people typically mean when they say "unicode."**

### 13.1 Worked examples

The four cases from the slide, from one byte to four:

```
  A          U+0041      1 byte
  01000001
  (basic Latin, identical to 7-bit ASCII)

  é          U+00E9      2 bytes
  11000011 10101001
  (Latin-1 Supplement, accented Latin)

  中         U+4E2D      3 bytes
  11100100 10111000 10101101
  (Han Chinese, most CJK falls in this range)

  (grinning face)  U+1F600      4 bytes
  11110000 10011111 10011000 10000000
  (emoji, Supplementary Multilingual Plane)
```

### 13.2 How codepoints are represented

The UTF-8 encoding table from the slide:

```
  Unicode code points     Encoding
  ───────────────────     ────────
  U+000000–U+00007f       0xxxxxxx

  U+000080–U+0007ff       110yyyxx
                          10xxxxxx

  U+000800–U+00ffff       1110yyyy
                          10yyyyxx
                          10xxxxxx

  U+010000–U+10ffff       11110zzz
                          10zzyyyy
                          10yyyyxx
                          10xxxxxx
```

The three rules that make this self-synchronising:

- A byte starting with **`0`** is a **single byte** (encoding 128 characters, that is, ASCII).
- A byte starting with **`11`** is the **first byte of a multi-byte sequence**, and the number of leading `1`s indicates how long the sequence is.
- A byte starting with **`10`** is a **continuation byte** in a multi-byte sequence.

> [!intuition] Why this design is good
> No byte value is ambiguous about its role. Drop into a UTF-8 stream at a random offset and you can tell immediately whether you are at a character start or mid-character, and you can walk backwards to the start by skipping `10xxxxxx` bytes. That is why UTF-8 survives truncation, concatenation and byte-level search in a way that code pages never could.

### 13.3 Deriving the encodings

Worth doing once by hand, because the exam can ask you to.

> [!example] é, U+00E9
> $\text{0xE9} = 233 = \texttt{11101001}_2$, which is 8 significant bits. That exceeds the 7 bits of the 1-byte form, so use the 2-byte form, which carries $5 + 6 = 11$ payload bits.
>
> Pad to 11 bits and split as 5 + 6: `00011 | 101001`
>
> Byte 1 = `110` + `00011` = `11000011`
> Byte 2 = `10` + `101001` = `10101001`
>
> Which matches the slide exactly.

> [!example] 中, U+4E2D
> $\text{0x4E2D} = \texttt{0100111000101101}_2$, 15 significant bits (written out to 16 with a leading zero), which exceeds the 11 of the 2-byte form, so use the 3-byte form, which carries $4 + 6 + 6 = 16$ payload bits.
>
> Split as 4 + 6 + 6: `0100 | 111000 | 101101`
>
> Byte 1 = `1110` + `0100` = `11100100`
> Byte 2 = `10` + `111000` = `10111000`
> Byte 3 = `10` + `101101` = `10101101`

> [!example] U+1F600, GRINNING FACE
> $\text{0x1F600}$ needs 17 bits, so use the 4-byte form, which carries $3 + 6 + 6 + 6 = 21$ payload bits.
>
> Pad to 21 bits: `000011111011000000000`, split as 3 + 6 + 6 + 6: `000 | 011111 | 011000 | 000000`
>
> Byte 1 = `11110` + `000` = `11110000`
> Byte 2 = `10` + `011111` = `10011111`
> Byte 3 = `10` + `011000` = `10011000`
> Byte 4 = `10` + `000000` = `10000000`

## 14. Unicode and tokenization: byte inflation

**Byte-level tokenizers are affected by how characters are represented at the byte level, not as codepoints.** The model never sees your careful codepoint reasoning, it sees the UTF-8 stream.

> [!formula] Byte inflation
> $$\text{inflation}(s) = \frac{\lvert \text{UTF-8 bytes}(s) \rvert}{\lvert \text{characters}(s) \rvert}$$
>
> where:
> - $\lvert \text{UTF-8 bytes}(s) \rvert$ is the length of $s$ once encoded as UTF-8
> - $\lvert \text{characters}(s) \rvert$ is the number of characters a reader perceives

The three worked cases from the slide:

| String | Bytes | Characters | Inflation |
|---|---|---|---|
| "hello" | 5 | 5 | 1.0x |
| "café" | 5 | 4 | 1.25x |
| "中文" | 6 | 2 | 3.0x |

Two consequences:

- **Byte inflation:** the more bytes are needed, the more aggressively tokens tend to be split. A script that costs 3 bytes per character gets chopped into more pieces than one that costs 1, for the same amount of meaning.
- **Byte-level splitting can result in non-existing characters.** Cut a 3-byte sequence after byte 2 and neither piece is a valid character. The model is then learning over fragments that no reader would recognise.

> [!warning] This is a fairness problem, not just an efficiency one
> The Latin-script user gets roughly 1 byte per character, the CJK user gets 3. The same sentence therefore costs the CJK user more tokens, which means a shorter effective context window, more compute for identical content, and a tokenizer whose units are less linguistically meaningful. The English-first assumption from section 6 has followed us all the way down to the byte layer.

## 15. Normalization in practice

**Resources crawled from the internet, even when they are in Unicode, can mix different forms.** Some pages are NFC, others are NFD, and nothing announces which.

Many preprocessing steps **assume identical codepoint sequences**:

- **Deduplication**
- **Search**
- **String equality, prefix matching**

If sequences are not normalized properly, **these steps will be inaccurate**. Note the failure mode: not a crash, just silently wrong numbers. Your deduplicated corpus keeps two copies of a document that differ only in normalization form, and your reported corpus size is a lie.

**The easy fix: always perform consistent normalization as part of the pipeline**, for example first decompose and then compose.

In Python, use `unicodedata`:

- It can **normalize** strings into a given form
- It can **test** whether a string is already in a given normalization form (for example NFD)

```python
import unicodedata

a = "café"          # NFC:  c a f é           -> 4 codepoints
b = "café"         # NFD:  c a f e + U+0301  -> 5 codepoints

a == b                                  # False
len(a), len(b)                          # (4, 5)

unicodedata.is_normalized("NFC", a)     # True
unicodedata.is_normalized("NFC", b)     # False

unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)   # True
```

For material that is not in Unicode at all but in some code page (GB2312, KOI8-R, ...), convert it on the way in:

```
iconv -f GB2312 -t UTF-8
```

This **fails if the from-encoding is not correct**, which is the useful part. A failure is a signal. Silently decoding with the wrong code page is what produces mojibake in the first place.

## 16. Where to break?

Once you accept that a character is not a byte and not always a codepoint, "split this string" stops being a well-defined instruction. Three candidate levels, all flawed:

- **Byte-level breaking can mess up codepoints**, which can consist of multiple bytes. You produce fragments that are not characters at all.
- **"Smarter" codepoint breaking is also not ideal**, because it can break a character into segments, for example separating a base letter from its diacritic. `e` and `◌́` are two codepoints but one character.
- **Ideally break at the grapheme level**, that is, what the reader perceives as a character. But grapheme boundaries are **very language-specific**, so this is the expensive option.

```
  "café"  breaking at three levels

  bytes:       63 | 61 | 66 | 65 | CC | 81     6 units, the last two are not characters
  codepoints:  c  | a  | f  | e  | ◌́           5 units, the accent is stranded
  graphemes:   c  | a  | f  | é                4 units, what the reader sees
```

**Mixing breaking strategies leads to misalignments**, which is where the bugs actually bite:

- **Span annotation systems** (SQuAD-style QA). The answer span is recorded as character offsets under one convention and read back under another, so the extracted answer is shifted or truncated.
- **Any counting across corpora with multiple breaking strategies.** Length statistics, token counts and coverage numbers computed under different conventions are not comparable, and nothing warns you.

The rule to take away: pick one convention and enforce it across every corpus, tool and metric in the pipeline. Consistency matters more than which level you pick.

## 17. Other pitfalls

**Lowercasing is not language-independent.** In English `lowercase(I) = i`, but in Turkish `lowercase(I) = ı`, because Turkish has two distinct letter pairs, dotted `İ / i` and dotless `I / ı`. So `.lower()` on Turkish text with the wrong locale changes which word you are looking at. This is the classic Turkish-I problem.

**Codepoint order does not necessarily correspond to alphabetic order.** The codepoints of accented letters are not necessarily close to their base letter's codepoint, so sorting by codepoint puts å nowhere near a.

**Different languages use different lexicographic (alphabetical) orders**, so there is no single correct sort even for one script:

- **Swedish:** a, b, c, ..., x, y, z, å, ä, ö. The accented vowels come **after** z, as separate letters at the end of the alphabet.
- **German:** ä sorts as ae, ö as oe, ü as ue. The accented vowels are **interleaved** with the plain ones.

Same three characters, two languages, two incompatible orderings. Sorting is therefore a locale-dependent operation, not a string operation.

**Punctuation normalization** is a problem of its own, because the same function is served by different characters in different traditions:

| Function | English | French / Russian | Japanese |
|---|---|---|---|
| Quotation | “...” | «...» | 「...」 |

## 18. Punctuation handling

While it is not trivial to define what a word is, one thing you can agree on is that **punctuation adjacent to a word is not part of the word**.

> [!example] Tokenizing punctuation
> ```
> He'd say "I promise you!", but then disappear.
>
> → He ' d say " I promise you ! " , but then disappear .
>
> → He would say " I promise you ! " , but then disappear .
> ```
>
> Step 1 separates every punctuation mark into its own token. Step 2 additionally **expands the contraction**, turning `He ' d` into `He would`, which is a further normalization decision and a lossy one.

Punctuation is usually handled as a **general category** rather than as an enumerated list of characters. On the slide, in Python:

```python
re.sub(r'(\p{P})', r' \1 ', string)
```

> [!warning] This does not run in Python's standard library
> The `\p{P}` Unicode-property syntax is **not supported by Python's built-in `re` module**. It works in the third-party `regex` package:
>
> ```python
> import regex
> regex.sub(r'(\p{P})', r' \1 ', string)
> ```
>
> The idea on the slide is right and important, which is to match the **Unicode general category** P (punctuation) rather than a hand-written character class. A hand-written class like `[.,!?]` is exactly the English-first assumption again: it silently misses `。`, `،`, `।` and `።`.

**Punctuation normalization** (“...” vs «...» vs 「...」) has **no simple solution**, and the mappings **must be manually encoded**. It occurs in:

- Quotes
- End-of-sentence markers
- Question marks
- Single quotes

## 19. Unicode recap

> [!tip] The stack, in one sentence
> A reader sees a **grapheme**, which is represented by **one or more codepoints**, where each codepoint is encoded in **1 to 4 bytes**.

```
  what the reader sees      é                       1 grapheme
                            │
  what Unicode identifies   U+0065  U+0301          2 codepoints  (NFD)
                            │       │
  what your tokenizer eats  65      CC 81           3 bytes       (UTF-8)
```

The rest of the recap slide:

- **Unicode can encode over a million characters**, so there is no need for writing-system-specific code pages.
- **Normalization (NFC, NFD, NFKC, NFKD) can unify potentially different encodings** of the same text.
- **UTF-8 is the most common form of Unicode** in document encoding and NLP tooling.
- **Any string breaking or length computation, byte-level or codepoint-level, must be consistent across documents.**

## 20. Additional reading

Both are blog posts and both are on the slide as recommended reading:

- Joel Spolsky (2003), *The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets*
- Nikita Prokopov (2023), *The Absolute Minimum Every Software Developer Must Know About Unicode in 2023*

## Key Takeaways

> [!tip] Exam Focus
> The assessment rule for this course is unforgiving: you need a mark of **5.5 or higher on both the mini-project and the written exam, independently**. A strong project does not rescue a weak exam and vice versa, so this material has to be actually learned rather than skimmed.
>
> Things to be able to state cold:
>
> 1. **The four English-first assumptions and a counterexample language for each** (section 5). Whitespace tokenization fails for Chinese/Japanese/Thai/Lao/Khmer/Burmese; word-order-encoded grammar fails for Russian/Finnish/Korean; one-word-one-meaning fails for Finnish/Turkish/Hungarian and for German/Dutch/Swedish compounds; left-to-right fails for Arabic/Hebrew/Urdu/Persian.
> 2. **The three types of writing system** with their inventory size and examples: alphabetic (small inventory, Latin/Cyrillic/Hangul), syllabic (Hiragana/Katakana), logographic (Hanzi/Kanji).
> 3. **Script is not language**, in both directions, with the standard examples: Latin and Arabic each serve many languages, Serbian and Uzbek each use two scripts.
> 4. **Codepoint, character, glyph, grapheme**, and the fact that they are four different things.
> 5. **NFC vs NFD vs NFKC vs NFKD**: canonical is reversible, compatibility is lossy; NFD for comparing rendered text, NFKD for search indexing and NLP analysis but not generation.
> 6. **The UTF-8 bit patterns and the three byte rules**, plus the ability to encode a codepoint by hand (section 13.3).
> 7. **Byte inflation**, with the three numbers: 1.0x for "hello", 1.25x for "café", 3.0x for "中文".

> [!warning] The three mistakes this lecture is designed to stop you making
> 1. Assuming a **byte, a codepoint, a character and a grapheme are the same thing**. They coincide only in ASCII, which is where the intuition was formed.
> 2. Skipping **normalization** on crawled data. It does not raise an error, it just makes your deduplication, search and string comparisons quietly wrong.
> 3. Writing **language-independent** in a paper when what you mean is that you tested English and one other language.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L01 - Overview]]
- **Project:** [[MNLP - Mini Project]]
- **Related concept:** [[Tokenization]]
- **Source:** `MNLP_multilingual-nlp.pdf`, 33 slides, Christof Monz, Informatics Institute, University of Amsterdam
</content>
</invoke>
