---
type: book-chapter
course: IR
book: "Search Engines: Information Retrieval in Practice"
chapter: 4
sections: ["4.1", "4.2", "4.3"]
topics:
  - "[[Tokenization]]"
  - "[[Stemming]]"
  - "[[Stop Words]]"
  - "[[Information Retrieval]]"
status: complete
---

# IR Chapter 4: Processing Text

## Overview
Text processing (or text transformation) is the pipeline used to convert raw document text into consistent **index terms**. This process is fundamental to [[Information Retrieval]], moving beyond simple "exact match" find features to handle linguistic variations, noise, and statistical properties of human language.

## 4.1 From Words to Terms
The primary goal of text processing is to normalize the many forms in which words occur into a standardized representation.

> [!definition] Index Terms
> The representation of the content of a document used for searching.

### Key Decisions in Text Processing:
- **Case-sensitivity**: Most search engines utilize case folding (lowercasing) to increase match probability.
- **Punctuation**: Stripping punctuation to simplify queries.
- **[[Tokenization]]**: Splitting text into individual tokens.
- **[[Stop Words]]**: Removing high-frequency, low-content words.
- **[[Stemming]]**: Grouping morphological variants (e.g., "run", "running").

## 4.2 Text Statistics
Language is highly predictable. Understanding the statistical distribution of words is crucial for ranking models and indexing.

### Zipf's Law
Zipf's law describes the skewed distribution of word frequencies: a few words occur very frequently, while many occur only once.

> [!formula] Zipf's Law
> The frequency of the $r$-th most common word ($f$) is inversely proportional to its rank ($r$):
> $$r \cdot f = k$$
> Or in terms of probability $P_r$:
> $$r \cdot P_r = c$$
> For English, $c \approx 0.1$.

**Key Implications:**
- The top 50 words account for ~40% of all text.
- Roughly half of the unique words in a corpus occur only once (**Hapax Legomena**).
- **Mandelbrot's Modification**: $(r + \beta)^\alpha \cdot P_r = \gamma$ (allows for tuning to specific corpora).

### Heaps' Law (Vocabulary Growth)
Predicts how the vocabulary size ($v$) grows relative to the number of tokens in the collection ($n$).

> [!formula] Heaps' Law
> $$v = k \cdot n^\beta$$
> Typical values: $10 \le k \le 100$ and $\beta \approx 0.5$.

### Result Set Size Estimation
Estimating the number of documents containing multiple query terms ($a, b, c$).

1. **Independence Assumption**:
   $$f_{abc} = N \cdot \frac{f_a}{N} \cdot \frac{f_b}{N} \cdot \frac{f_c}{N} = \frac{f_a \cdot f_b \cdot f_c}{N^2}$$
   *Warning*: Often underestimates because words are semantically dependent.
2. **Conditional Probability**:
   $$P(a \cap b \cap c) = P(a \cap b) \cdot P(c | (a \cap b))$$

## 4.3 Document Parsing
Parsing involves recognizing the content and structure of documents.

### 4.3.2 [[Tokenization]]
The process of forming words from a sequence of characters (lexical analysis).

> [!warning] Tokenization Challenges
> - **Small words**: "XP", "II", "J Lo" can be significant.
> - **Hyphens**: "e-bay" vs "ebay", "wal-mart" vs "walmart".
> - **Apostrophes**: Possessives vs. contractions vs. names ("O'Donnell").
> - **Numbers**: Product IDs, dates, patent numbers.

**General Strategy**:
1. Pass 1: Identify markup/tags (HTML/XML).
2. Pass 2: Tokenize text content, typically treating non-alphanumeric characters as word terminators and converting to lowercase.

### 4.3.3 [[Stop Words]] (Stopping)
Function words (determiners, prepositions) that provide structure but little semantic content.

> [!example] Stop Words
> - Determiners: "the", "a", "an", "that"
> - Prepositions: "over", "under", "above"

**Removal Benefits**:
- Decreases [[Inverted Index]] size.
- Increases retrieval efficiency.
- *Risk*: Can break queries like "to be or not to be". Modern systems often index all terms but may remove stop words from queries dynamically.

### 4.3.4 [[Stemming]]
Stemming (conflation) reduces inflected or derived words to a common base form (stem).

> [!intuition] Stemming
> Searching for "swimming" should match documents containing "swam" or "swims" by reducing them to the stem "swim".

**Types of Stemmers**:
1. **Algorithmic**: Rules based on suffixes.
   - **Porter Stemmer**: Most popular for English. Uses 5 steps of suffix stripping.
   - **Porter2**: Improved version with exception handling.
2. **Dictionary-based**: Matches words against a lookup table.
   - **Krovetz Stemmer**: Hybrid approach checking stems against a dictionary to ensure they are valid words.

**Error Types**:
- **False Positive**: Grouping unrelated words (e.g., "policy" / "police").
- **False Negative**: Failing to group related words (e.g., "europe" / "european").

## 4.3.5 Phrases and N-grams
While single words are the default, phrases offer higher precision.

- **POS Tagging**: Identifying simple noun phrases (Adjective + Noun or Noun + Noun).
- **N-grams**: Any sequence of $n$ words.
  - **Unigram**: "tropical"
  - **Bigram**: "tropical fish"
  - **Trigram**: "tropical fish aquarium"

## Summary
Text processing is the foundational stage of the IR pipeline. By understanding the statistical nature of text (Zipf/Heaps) and applying transformations like [[Tokenization]], stopping, and [[Stemming]], search engines create a searchable [[Bag of Words]] or phrase representation that balances efficiency with retrieval effectiveness. Correct normalization ensures that the **[[Term Weighting]]** (e.g., [[BM25]]) can accurately estimate relevance.
