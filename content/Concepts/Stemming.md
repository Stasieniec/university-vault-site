---
type: concept
aliases: [Stemming, Lemmatization, Porter Stemmer]
course: [IR]
tags: [foundations]
status: complete
---

# Stemming

> [!definition] Stemming
> **Stemming** is the process of reducing inflected (or sometimes derived) words to their word stem, base, or root form. It is a heuristic process that "chops off" the ends of words.

## Porter Stemmer
The most common algorithm for English. It applies a series of rules (phases) to iteratively strip suffixes.
- **Example**:
    - `connect`, `connected`, `connecting`, `connections` → `connect`
    - `running` → `run` (Note: some stemmers might result in `run`, others in `runn`)

> [!intuition] Improving Recall
> Stemming helps the retrieval system realize that a user searching for "stems" might also be interested in a document containing "stemming." By mapping different word forms to the same root, we increase the number of matches.

## Stemming vs. Lemmatization

| Feature | Stemming | Lemmatization |
|---------|----------|---------------|
| **Approach** | Heuristic (chopping) | Morphological analysis (lookup) |
| **Output** | May not be a real word (`comput`) | Always a valid word (`compute`) |
| **Speed** | Very Fast | Slower (needs dictionary) |
| **Context** | Ignores context | Uses POS tags (e.g., *saw* as noun vs verb) |

## Trade-offs in IR

- **Increases Recall**: More documents match the query terms.
- **Decreases Precision**: May cause "over-stemming" where unrelated words are conflated (e.g., `organization` and `organ` might both stem to `organ`).

## Connections

- Preprocessing: Happens after [[Tokenization]].
- Related: [[Stop Words]] removal.

## Appears In

- [[IR-L02 - Indexing and Boolean Retrieval]]
