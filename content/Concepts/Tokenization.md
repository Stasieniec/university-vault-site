---
type: concept
aliases: [Tokenization, Tokenizing]
course: [IR]
tags: [foundations]
status: complete
---

# Tokenization

> [!definition] Tokenization
> **Tokenization** is the process of breaking a stream of text into smaller units called **tokens** (e.g., words, phrases, symbols). In IR, tokens are the candidates for becoming terms in the [[Inverted Index]].

## Challenges in Tokenization

Splitting by whitespace is rarely enough. Key issues include:
- **Punctuation**: "O'Neill" → `O'Neill`? `O` and `Neill`? 
- **Hyphenation**: "state-of-the-art" → one token or four?
- **Compounds**: "database" (English) vs "Datimbank" (German) vs "San Francisco" (multi-word expression).
- **Numbers/Dates**: Handling 2024-02-24 or $1,000.50.
- **Case Folding**: Reducing everything to lowercase (e.g., "Apple" vs "apple").

## Normalization Steps

After splitting, tokens often undergo further normalization:
- **Lowercasing**: Standardizing case.
- **Accents**: Stripping diacritics (e.g., `résumé` → `resume`).
- **Standardization**: `U.K.` → `UK`.

## Connections

- Next steps: [[Stemming]] and [[Stop Words]] removal.
- Representation: Part of creating a [[Bag of Words]].
- Indexing: Tokens that survive normalization become **Terms** in the [[Inverted Index]].

## Appears In

- [[IR-L02 - Indexing and Boolean Retrieval]]
