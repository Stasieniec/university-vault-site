---
type: lecture
course: 5204MNLP6Y
week: 3
lecture: 5
date: 2026-09-14
status: complete
topics:
  - Distributional semantics and count-based context vectors
  - Cosine similarity
  - Word2Vec CBOW
  - Word2Vec Skip-gram
  - Negative sampling
  - GloVe
  - Intrinsic and extrinsic evaluation of embeddings
  - Word similarity and analogy benchmarks
  - Visualising embeddings with t-SNE and PCA
  - fastText character n-gram embeddings
  - Cross-lingual embeddings and the shared space hypothesis
  - Seed dictionaries and word alignment
  - Least-squares linear mapping
  - Orthogonal Procrustes mapping
  - Hubness and CSLS
  - Adversarial unsupervised alignment (MUSE)
  - Self-learning refinement
  - Dependent marking and the limits of isomorphism
  - Eigenvector similarity
---

# MNLP-L05: Static Embeddings

> [!abstract] Overview
> A model that sees words as integer IDs knows that `dog` is word 4711 and `cat` is word 812, and nothing else. It has no way to know that whatever it learned about dogs should transfer, partially, to cats, and not at all to carburettors. Everything in this lecture is an answer to one question: how do you give each word a vector such that geometric closeness means linguistic similarity, using only raw text?
>
> The answer the field converged on is Firth's: a word is characterised by the company it keeps. The lecture builds that idea up in three steps. First, literally count the context words (distributional semantics, sparse and interpretable). Then learn dense low-dimensional vectors by training a small network to predict context (Word2Vec CBOW and Skip-gram, made affordable by negative sampling), or by regressing on global co-occurrence counts (GloVe). Then fix the one-vector-per-word-type blindness to morphology with character n-grams (fastText), which matters most for exactly the morphologically rich languages this course cares about.
>
> The second half is the multilingual payoff. If English and Spanish embedding spaces have roughly the same shape, one rotation should map `king` onto `rey`, and you can build a bilingual dictionary from two monolingual corpora plus a small seed dictionary, or, with adversarial training, from no dictionary at all. The lecture ends by taking that claim apart: the "same shape" assumption breaks for typologically distant languages (English and Finnish give 0.0% precision), for mismatched domains and for mismatched embedding algorithms. These vectors are **static**: one vector per word type, regardless of the sentence. The next lecture, [[MNLP-L06 - Contextual Embeddings]], removes that restriction.

The lecture outline (slide 1, repeated as the recap on slide 71):

- Word semantics: distributional semantics
- Embeddings: CBOW and Skip-gram, negative sampling, GloVe
- Evaluation of embeddings
- Morphologically sensitive embeddings: fastText
- Cross-lingual embeddings: supervised alignment, unsupervised alignment, semi-supervised alignment, and a critical evaluation of unsupervised alignment

## 1. What we want from a word representation

We want word representations that let us **compute similarity along multiple dimensions**:

- **meaning**, which itself has several dimensions: synonyms (`car`/`automobile`), topical relatedness (`car`/`road`), and so on
- **morphology** (`run`/`runs`/`running` should be recognisably related)

> [!definition] The distributional hypothesis
> "*You shall know a word by the company it keeps.*" (Firth, 1957)
>
> Produce vector representations of words **based on the contexts in which they occur**. Two words that occur in similar contexts get similar vectors, and are therefore taken to have similar meaning.

There are **two types of approaches**:

| Approach | Also called | How the vector is obtained | Covered in |
|---|---|---|---|
| **count based** | distributional semantics | count context words directly | section 2 |
| **prediction based** | neural embeddings | train a network to predict words from context (or the reverse), keep its weights | sections 3 to 6 |

GloVe (section 7) is the hybrid of the two.

## 2. Distributional semantics: count-based context vectors

### 2.1 The recipe

> [!definition] Count-based approach
> - Define a vocabulary $V_c$ of **context words**.
> - Define a vocabulary $V_t$ of **target words** (can be the same as $V_c$).
> - Define a **window of size $n$** (a sliding window, but the context can also be the whole sentence or document).
> - For each occurrence of a word $w \in V_t$: count how often each word $w' \in V_c$ occurs **within $n$ words to the left or right** of $w$.
> - For each word $w \in V_t$ store the co-occurrence counts in a vector $c_w$.

As pseudocode (assembled from slide 3, which gives the procedure in prose):

```pseudo
Algorithm: Count-based context vectors
──────────────────────────────────────────────────────────
Input:  corpus (sequence of tokens t_1 ... t_N),
        target vocabulary V_t, context vocabulary V_c, window size n
Output: one vector c_w ∈ ℕ^{|V_c|} per target word w

for each w ∈ V_t:
    c_w ← zero vector of length |V_c|
for pos = 1 to N:
    w ← t_pos
    if w ∉ V_t: continue
    for off = -n to n, off ≠ 0:                    // n words left and right
        if 1 ≤ pos+off ≤ N and t_{pos+off} ∈ V_c:
            c_w[index(t_{pos+off})] += 1
return {c_w : w ∈ V_t}
```

### 2.2 Worked example (slides 4 and 5)

The corpus snippets on slide 4. On the slide the target word is in bold and the context words that fall inside the window are underlined; here they are listed after the arrow:

```
...he took the dog for a walk...                 dog  → walk   (3 to the right)
...owner of the dog did not clean...             dog  → owner  (3 to the left)
...people have a dog as a pet...                 dog  → pet    (3 to the right)
...they spotted the car and its owner...         car  → owner  (3 to the right)
...and the cheapest car to run in...             car  → run    (2 to the right)
```

- Context size $n = 3$
- $V_c = \{\textit{leash}, \textit{walk}, \textit{run}, \textit{owner}, \textit{pet}, \textit{bark}\}$

Every underlined word is within 3 tokens of the target, which is why it counts. `took`, `did` or `cheapest` are also within 3 tokens but are not in $V_c$, so they are ignored.

Slide 5 then shows the context vectors for six target words over a larger corpus (the counts are larger than the five snippets alone would give, so the table should be read as illustrative counts over more text):

| word | leash | walk | run | owner | pet | bark |
|---|---|---|---|---|---|---|
| dog | 3 | 5 | 2 | 5 | 3 | 2 |
| cat | 0 | 3 | 3 | 2 | 3 | 0 |
| lion | 0 | 3 | 2 | 0 | 1 | 0 |
| light | 0 | 0 | 0 | 0 | 0 | 0 |
| bark | 1 | 0 | 0 | 2 | 1 | 0 |
| car | 0 | 0 | 1 | 3 | 0 | 0 |

> [!definition] Context vector
> In distributional semantics every word $w$ is represented as a $V$-dimensional context vector $c_w$, where $V = |V_c|$, and
> $$c_w[i] = f$$
> where $f$ is the frequency with which context word $i$ occurs within the (fixed-size) context of $w$.

This is the [[Vector Space Model]] idea from information retrieval applied to words: a document there is a vector of term counts, a word here is a vector of context-word counts.

### 2.3 Similarity as cosine

Word similarity is measured as **cosine similarity in the context vector space**:

> [!formula] Cosine similarity #key-formula
> $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert_2 \, \lVert \mathbf{v} \rVert_2} = \frac{\sum_i u_i v_i}{\sqrt{\sum_i u_i^2}\,\sqrt{\sum_i v_i^2}}$$
>
> where:
> - $\mathbf{u}, \mathbf{v}$ are the two word vectors (here context vectors $c_w$)
> - $\mathbf{u} \cdot \mathbf{v}$ is the dot product
> - $\lVert \mathbf{u} \rVert_2$ is the Euclidean length of $\mathbf{u}$
>
> It is the cosine of the angle between the vectors: 1 for the same direction, 0 for orthogonal. With non-negative counts it lies in $[0,1]$.

Cosine ignores vector length, which is the point: a frequent word has large counts everywhere, a rare word small ones, and only the *proportions* of the contexts should decide similarity.

**The 2D example on slide 6.** Two context words only, `runs` and `legs`:

| word | runs | legs |
|---|---|---|
| dog | 1 | 4 |
| cat | 1 | 5 |
| car | 4 | 1 |

The slide plots these as arrows from the origin (x-axis `runs`, 0 to 6; y-axis `legs`, 0 to 6), with a dotted arc marking the angle between the animal vectors and the car vector:

```
 legs
  5 ┤    ● cat (1,5)
  4 ┤    ● dog (1,4)          both arrows leave the origin almost vertically
  3 ┤
  2 ┤ ·.                      dotted arc: the large angle between
  1 ┤    ·.                   the animal arrows and the car arrow
    ┤       ·           ● car (4,1)
  0 ┼────┬────┬────┬────┬────┬────┬──
    0    1    2    3    4    5    6   runs
```

Working it out:

$$\cos(\text{dog},\text{cat}) = \frac{1\cdot1 + 4\cdot5}{\sqrt{17}\sqrt{26}} = \frac{21}{21.02} = 0.999 \quad (\approx 2.7^\circ)$$

$$\cos(\text{dog},\text{car}) = \frac{1\cdot4 + 4\cdot1}{\sqrt{17}\sqrt{17}} = \frac{8}{17} = 0.471 \quad (\approx 61.9^\circ)$$

$$\cos(\text{cat},\text{car}) = \frac{1\cdot4 + 5\cdot1}{\sqrt{26}\sqrt{17}} = \frac{9}{21.02} = 0.428 \quad (\approx 64.7^\circ)$$

`dog` and `cat` point almost the same way; `car` points elsewhere.

**The same computation on the slide 5 table** (not on the slides, worked here as practice):

| pair | dot product | cosine |
|---|---|---|
| dog, cat | 40 | 0.824 |
| cat, lion | 18 | 0.864 |
| dog, lion | 22 | 0.674 |
| bark, car | 6 | 0.775 |
| dog, bark | 16 | 0.749 |
| dog, car | 17 | 0.617 |
| lion, bark | 1 | 0.109 |
| anything, light | 0 | undefined |

(Norms: $\lVert c_\text{dog} \rVert = 8.718$, $\lVert c_\text{cat} \rVert = 5.568$, $\lVert c_\text{lion} \rVert = 3.742$, $\lVert c_\text{bark} \rVert = 2.449$, $\lVert c_\text{car} \rVert = 3.162$, $\lVert c_\text{light} \rVert = 0$.)

Three things show up. The animals cluster (cat and lion 0.864, dog and cat 0.824). `bark` and `car` come out at 0.775, more similar than `dog` and `car`, purely because both co-occur with `owner`: one shared context word dominates when vectors are this short, which is the "signal versus noise" problem of section 2.4. And `light` has a zero vector because none of its contexts are in $V_c$, so its cosine with anything is $0/0$: a word whose contexts you did not include has no representation at all.

> [!note] Properties of these vectors
> In distributional semantics context vectors are **high-dimensional** (one dimension per context word, so tens or hundreds of thousands), **discrete** (integer counts) and **sparse** (almost all entries are zero, since any one word co-occurs with a tiny fraction of the vocabulary).

### 2.4 Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| **Unsupervised** way to induce word similarity | Context vectors **remain relatively sparse**, despite methods such as stemming to alleviate this |
| **Dimensions are interpretable**: $c_w[i]$ is the frequency of an actual word | **No explicit criterion to discriminate between signal and noise** in the context |

The noise problem in detail:

- some words occur in the context **just due to chance**
- **which context words are the most important discriminators** of a word's meaning? A word like `the` co-occurs with everything and discriminates nothing.

One fix the slide suggests is to weight context words by **inverse document frequency** (the slide writes "ifd", a typo for idf):

> [!formula] Inverse document frequency
> $$\text{idf}(w) = \log \frac{N}{n_w}$$
>
> where:
> - $N$ is the number of contexts
> - $n_w$ is the number of contexts in which $w$ occurs
>
> A context word that appears in nearly every context gets $\text{idf} \approx \log 1 = 0$ and is effectively removed; a word that appears in few contexts gets a large weight.

Example with numbers (not on the slides): with $N = 1000$ contexts, a word in 10 of them gets $\log(1000/10) = 4.61$, a word in 900 of them gets $\log(1000/900) = 0.105$. So the rare, discriminative context word counts about 44 times as much.

## 3. From projections to embeddings

### 3.1 Are projections the same as embeddings?

"Projections" here refers to the projection (lookup) layer of a neural language model such as the probabilistic neural language model (PNLM) referred to on slide 12: the first layer maps each one-hot word to a dense vector $C(w)$, and those vectors are learned along with the rest of the network.

> [!definition] What a good word embedding is
> $C(w) \approx C(w')$ **if and only if**
> - $w$ and $w'$ **mean the same thing**, and
> - $w$ and $w'$ **exhibit the same syntactic behaviour**.
>
> where $C(w)$ is the vector (projection, embedding) assigned to word $w$.

- **For most models, the projections/embeddings are by-products.** The main objective is to optimise a task (next-word prediction, classification, etc.), and the projections are learned only because they help achieve it.
- **Representation learning** is the setting where the *main* objective is to learn good projections/embeddings. Word2Vec is designed this way: the prediction task is a pretext, and the network is thrown away after training except for the embedding matrices.

### 3.2 Word embeddings versus context vectors

> [!definition] Word embeddings
> Same underlying intuition as distributional semantics (a word is characterised by its contexts), but the word vectors are
> - **low dimensional** (e.g. 100, against $|V_c|$ for context vectors)
> - **dense** (no zeros)
> - **continuous** ($c_w \in \mathbb{R}^m$, real-valued, where $m$ is the embedding dimension)
> - **learned by performing a task** (predict)

| Property | Count-based context vectors | Prediction-based [[Word Embeddings]] |
|---|---|---|
| Dimensionality | $\lvert V_c \rvert$ (huge) | $m \approx 100$ to $1000$ |
| Values | discrete counts | real numbers |
| Sparsity | mostly zeros | dense |
| Dimensions interpretable? | yes, each is a context word | no |
| Obtained by | counting | training a predictor |

A popular approach is **Word2Vec** (Mikolov et al.), which consists of two approaches:

- **Continuous Bag of Words (CBOW)**: predict the word from its context
- **Skip-Gram**: predict the context from the word

## 4. Word2Vec: Continuous Bag of Words (CBOW)

### 4.1 The task

> [!definition] CBOW task
> Given a position $t$ in a sentence, the $n$ words to its left $\{w_{t-n}, \ldots, w_{t-1}\}$ and the $m$ words to its right $\{w_{t+1}, \ldots, w_{t+m}\}$, **predict the word in position $t$**.
>
> *the man X the road*, where $X = ?$

> [!warning] Typo on slide 10
> The slide writes the right context as $\{w_{t+1}, \ldots, w_{t+n}\}$ while calling its size $m$. It should end at $w_{t+m}$. In practice the two are equal anyway ($n = m$, see below), which is why every later formula on the slides uses $n$ for both sides.

- It is **seemingly similar to n-gram language modelling**, where $n = \text{LM order} - 1$ and $m = 0$: an n-gram LM predicts a word from the $n$ words to its left only. CBOW also looks to the right, so it is useless as a language model and only good for learning representations.
- It uses a **feed-forward neural network**, with these design choices:
  - **focus on learning the embeddings themselves**
  - a **simpler network** than the PNLM (no hidden non-linear layer)
  - **bring the embedding/projection layer closer to the output**, so the gradient reaching the embeddings is not diluted by layers in between
  - **typically $n = m$, and $n \in \{2, 5, 10\}$**

The name: "bag of words" because the context words are summed, so their order is thrown away, exactly as in a [[Bag of Words]] document representation; "continuous" because the bag is a sum of dense vectors.

### 4.2 Architecture

Slide 11 shows the network for $n = 2$: four input boxes `w(t-2)`, `w(t-1)`, `w(t+1)`, `w(t+2)` in an INPUT column, all with arrows into a single box labelled SUM in the PROJECTION column, which feeds one box `w(t)` in the OUTPUT column.

```mermaid
flowchart LR
    A["w(t-2)"] --> S["SUM<br/>(projection)"]
    B["w(t-1)"] --> S
    C["w(t+1)"] --> S
    D["w(t+2)"] --> S
    S --> O["w(t)<br/>(output)"]
```

### 4.3 The model

> [!formula] CBOW #key-formula
> **No non-linearities. One hidden layer:**
> $$h = \frac{1}{2n} W \, w_C, \qquad w_C = \sum_{\substack{i = t-n \\ i \neq t}}^{t+n} w_i$$
>
> **Output layer:**
> $$\hat{y} = \operatorname{softmax}(W' h)$$
>
> where:
> - $w_i$ is the **1-hot vector** (length $\lvert V \rvert$) for the word in position $i$
> - $w_C$ is the sum of the $2n$ context one-hot vectors (so it is a count vector over the vocabulary)
> - $W$ is a $\lvert h \rvert \times \lvert V \rvert$ matrix, the input (projection) matrix
> - $h$ is the hidden layer, of size $\lvert h \rvert$ (the embedding dimension)
> - $W'$ is a $\lvert V \rvert \times \lvert h \rvert$ matrix, the output matrix
> - $\hat{y} \in \mathbb{R}^{\lvert V \rvert}$ is the predicted distribution over the word at position $t$
> - $W'$ and $W$ are **not (necessarily) shared**, i.e. $W' \neq W^T$
>
> **Loss:** cross entropy. **Training:** SGD.

Reading it:

- Multiplying $W$ by a one-hot vector $w_i$ selects **column** $i$ of $W$. So $W w_C$ is the sum of the context words' columns, and $h$ is their **average**. No matrix multiplication is actually performed in an implementation, only lookups.
- There is no non-linearity between $h$ and the output, so the score of a candidate word $j$ is simply the dot product of row $j$ of $W'$ with the averaged context vector.
- **Cross entropy** with a one-hot target reduces to $-\log \hat{y}[w_t]$, the negative log-probability of the correct word (the same loss as in the PNLM, which is what "see PNLM" refers to).
- The softmax normalises over all $\lvert V \rvert$ words, so every training example touches all of $W'$: cost $O(\lvert V \rvert \cdot \lvert h \rvert)$ per prediction. With $\lvert V \rvert$ in the hundreds of thousands this is the bottleneck that negative sampling (section 6) removes.

### 4.4 Where do the embeddings live?

- **Column $i$ of $W$** ($\lvert h \rvert \times \lvert V \rvert$) is the embedding of word $i$ as an *input* (context) word.
- **Row $i$ of $W'$** ($\lvert V \rvert \times \lvert h \rvert$) is the embedding of word $i$ as an *output* (predicted) word.

Which one is "the" embedding? **Typically $W$**, or

$$W_s = W^T + W'$$

which **combines both into one** $\lvert V \rvert \times \lvert h \rvert$ matrix whose row $i$ is the embedding of word $i$ (the transpose turns $W$'s columns into rows so the shapes match).

## 5. Word2Vec: Skip-gram

### 5.1 The task and architecture

> [!definition] Skip-gram task
> An alternative to CBOW. **Given the word at position $t$, predict the words occurring between positions $t-n$ and $t-1$ and between $t+1$ and $t+n$.**

It is CBOW turned around: one input word, $2n$ predictions. Slide 14 shows one input box `w(t)`, an arrow into a single PROJECTION box, and from there four arrows to OUTPUT boxes `w(t-2)`, `w(t-1)`, `w(t+1)`, `w(t+2)`.

```mermaid
flowchart LR
    I["w(t)<br/>(input)"] --> P["projection"]
    P --> A["w(t-2)"]
    P --> B["w(t-1)"]
    P --> C["w(t+1)"]
    P --> D["w(t+2)"]
```

### 5.2 The model

> [!formula] Skip-gram #key-formula
> **One hidden layer:**
> $$h = W \, w_I$$
> where $w_I$ is the 1-hot vector for the word at position $t$ (the input word).
>
> **$2n$ output layers**, with the context words assumed independent given the input word:
> $$p(w_{t-n} \ldots w_{t-1} w_{t+1} \ldots w_{t+n} \mid w_I) \;\propto\; \prod_{\substack{i=t-n \\ i \neq t}}^{t+n} p(w_i \mid w_I)$$
> $$\hat{y}_i = \operatorname{softmax}(W' h) \qquad (t - n \le i \le t + n \text{ and } i \neq t)$$
>
> where:
> - $W$ ($\lvert h \rvert \times \lvert V \rvert$) and $W'$ ($\lvert V \rvert \times \lvert h \rvert$) are as in CBOW and are not (necessarily) shared, $W' \neq W^T$
> - $\hat{y}_i$ is the predicted distribution for context position $i$
>
> **Loss:** cross entropy. **Training:** SGD.

Two details worth seeing clearly:

- $h$ is just column $w_I$ of $W$, with no averaging: the input word's embedding is used directly.
- The $2n$ "output layers" all use the **same** $W'$ and the **same** $h$, so they produce the **same** distribution $\hat{y}$. What differs is the target each one is scored against. The loss is the sum $-\sum_{i \neq t} \log \hat{y}[w_i]$. Equivalently, Skip-gram is trained on $2n$ independent (input word, context word) pairs per position, which is how negative sampling treats it.

| | CBOW | Skip-gram |
|---|---|---|
| Input | $2n$ context words | 1 centre word |
| Output | the centre word | each of the $2n$ context words |
| Hidden layer | average of context embeddings | embedding of the centre word |
| Predictions per position | 1 | $2n$ |

## 6. Negative sampling

### 6.1 The problem and the idea

- **Both CBOW and Skip-gram benefit from large amounts of data.**
- **Computing activations for the full output layer becomes an issue**: the softmax needs a score for every word in $V$ for every single training example.

> [!definition] Negative sampling
> Replace "predict which word appears in the context" with "**distinguish between words that do and words that do not occur in the context of the input word**":
> - a **classification task** (binary: real context pair or not)
> - **1 positive example** (from the ground truth, the word that actually occurred)
> - **$k$ negative examples** (drawn from a random noise distribution)

Per training example the cost drops from $\lvert V \rvert$ dot products to $k+1$. See [[Negative Sampling]] for the general technique, and [[Contrastive Learning]] for the family it belongs to: pull the true pair together, push sampled pairs apart.

### 6.2 The objective

> [!formula] Negative sampling objective #key-formula
> Given the input word $w$ and a context word $c$ we want
> $$\arg\max_\theta \prod_{(w,c) \in D} p(D = 1 \mid c, w; \theta) \prod_{(w,c) \in D'} p(D = 0 \mid c, w; \theta)$$
>
> where:
> - $D$ is the set of **observed** (word, context) pairs from the data
> - $D'$ is a set of pairs drawn from a **noise distribution**
> - $D = 1$ means "this pair came from the data", $D = 0$ "this pair is noise"
> - $\theta$ are the parameters (the matrices $W$ and $W'$)
>
> The probability of a pair being real is
> $$p(D = 1 \mid c, w; \theta) = \sigma(v_c \cdot v_w), \qquad v_w = W w, \quad v_c = W'^T c$$
> where $\sigma(x) = 1/(1+e^{-x})$ is the logistic sigmoid, $w$ and $c$ are one-hot vectors, so $v_w$ is column $w$ of $W$ (input embedding) and $v_c$ is row $c$ of $W'$ (output embedding).
>
> And $p(D = 0 \mid c, w; \theta) = 1 - p(D = 1 \mid c, w; \theta)$. Since $1 - \sigma(x) = \sigma(-x)$:
> $$\arg\max_\theta \prod_{(w,c) \in D} \sigma(v_c \cdot v_w) \prod_{(w,c) \in D'} \sigma(-v_c \cdot v_w)$$
>
> Taking logs:
> $$\arg\max_\theta \sum_{(w,c) \in D} \log \sigma(v_c \cdot v_w) + \sum_{(w,c) \in D'} \log \sigma(-v_c \cdot v_w)$$

The identity $1 - \sigma(x) = \sigma(-x)$ in one line: $1 - \frac{1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}} = \frac{1}{e^{x}+1} = \sigma(-x)$.

Intuition: the dot product $v_c \cdot v_w$ is a compatibility score. Training pushes it up for pairs seen in the data and down for random pairs. Words that share many contexts get pushed towards the same output vectors, and therefore towards each other.

### 6.3 Practical settings (slide 18)

> [!note] Word2Vec practical considerations
> - **Skip-gram:**
>   - for each observed occurrence $(w, c)$ **add 5 to 20 negative samples** to the data
>   - draw the negative $c$ from the **unigram distribution $P(w)$**
>   - **scale the unigram distribution to $P(w)^{0.75}$ to bias towards rarer words**
> - **Context size typically around 2 to 5.**
> - **The more data, the smaller the context and the negative sample set** can be.

Why the 0.75 power helps (worked here with made-up frequencies, not on the slides). Three words with unigram probabilities 0.9, 0.09 and 0.01:

| word | $P(w)$ | $P(w)^{0.75}$ | renormalised |
|---|---|---|---|
| frequent | 0.90 | 0.924 | 0.825 |
| medium | 0.09 | 0.164 | 0.147 |
| rare | 0.01 | 0.032 | 0.028 |

The raised values sum to 1.12, so they are renormalised. The rare word's sampling probability almost triples (1% to 2.8%) and the frequent word's drops (90% to 82.5%). Without this, nearly every negative would be `the`, `of` or `and`, and the model would learn little about the rest of the vocabulary.

### 6.4 Skip-gram with negative sampling as an algorithm

The slides give the pieces; here they are assembled into one procedure (the update rules are the gradients of the objective above, derived below):

```pseudo
Algorithm: Skip-gram with negative sampling (SGNS)
──────────────────────────────────────────────────────────
Input:  corpus, window n (2 to 5), negatives k (5 to 20),
        embedding dim |h|, learning rate η
Init:   W  (|h| × |V|) small random      // input embeddings, columns v_w
        W' (|V| × |h|) small random      // output embeddings, rows v_c
        P_n(w) ∝ P(w)^0.75               // noise distribution

for each position t in corpus:
    w ← word at t
    for each context position i ∈ [t-n, t+n], i ≠ t:
        c ← word at i                                 // positive pair (w, c)
        draw n_1 .. n_k from P_n                      // k negative pairs (w, n_j)
        g_w ← 0
        // positive: push σ(v_c·v_w) towards 1
        e ← σ(v_c · v_w) − 1
        g_w ← g_w + e · v_c
        v_c ← v_c − η · e · v_w
        // negatives: push σ(v_nj·v_w) towards 0
        for j = 1 to k:
            e ← σ(v_nj · v_w)
            g_w ← g_w + e · v_nj
            v_nj ← v_nj − η · e · v_w
        v_w ← v_w − η · g_w
return W  (or W^T + W')
```

**Where the updates come from** (not on the slides). For one positive pair and $k$ negatives, the loss to minimise is
$$\mathcal{L} = -\log \sigma(v_c \cdot v_w) - \sum_{j=1}^{k} \log \sigma(-v_{n_j} \cdot v_w)$$
Using $\frac{d}{dx} \log \sigma(x) = 1 - \sigma(x)$:
$$\frac{\partial \mathcal{L}}{\partial v_w} = \big(\sigma(v_c \cdot v_w) - 1\big)\, v_c + \sum_{j=1}^{k} \sigma(v_{n_j} \cdot v_w)\, v_{n_j}$$
The coefficient on the positive term is negative (so gradient descent moves $v_w$ towards $v_c$), and the coefficient on each negative is positive (moves $v_w$ away from $v_{n_j}$). Both shrink to zero as the classifier gets the pair right.

> [!example] One step with numbers (illustrative values, not from the slides)
> Suppose $v_c \cdot v_w = 2$ for the true context word, and two negatives score $v_{n_1} \cdot v_w = 0.5$ and $v_{n_2} \cdot v_w = -1$.
>
> | term | score | probability | loss contribution |
> |---|---|---|---|
> | positive $c$ | 2 | $\sigma(2) = 0.881$ | $-\log 0.881 = 0.127$ |
> | negative $n_1$ | 0.5 | $\sigma(-0.5) = 0.378$ | $-\log 0.378 = 0.974$ |
> | negative $n_2$ | $-1$ | $\sigma(1) = 0.731$ | $-\log 0.731 = 0.313$ |
>
> Total loss $\mathcal{L} = 1.414$. The gradient with respect to $v_w$ is
> $$-0.119\, v_c + 0.622\, v_{n_1} + 0.269\, v_{n_2}$$
> so the largest correction comes from $n_1$, the negative the model currently mistakes for a real context word. The positive pair is already mostly right and contributes little.

## 7. GloVe

### 7.1 Motivation: combine counting and predicting

There are **two main strategies** for learning word representations:

| | Count-based | Prediction-based |
|---|---|---|
| How | directly (section 2), or as **global matrix factorisation** (latent semantic analysis: factorise the co-occurrence matrix with SVD) | **local context window** (CBOW or Skip-gram) |
| Strength | **uses global information** (co-occurrence counts) directly | **still tends to outperform** count-based approaches |
| Weakness | **tends to perform worse** than prediction-based approaches | **treats each instance of a context window as an independent event**: it has to repeatedly "rediscover" the association between two words every time they co-occur |

> [!definition] GloVe
> **GloVe** (Pennington et al., 2014) **combines global co-occurrence statistics with local prediction-based updating**: it first counts, once, over the whole corpus, and then trains vectors with gradient updates to fit those counts.

### 7.2 The model

Notation:

- $X_{ij}$ is the **number of times word $j$ occurs in the context of word $i$** (the co-occurrence matrix, built as in section 2)
- $X_i = \sum_k X_{ik}$ is the **number of times any word occurs in the context of word $i$**
- $P_{ij} = P(j \mid i) = X_{ij} / X_i$ is the **probability of word $j$ occurring in the context of word $i$**

> [!formula] GloVe objective #key-formula
> $$J = \sum_{(i,j)} f(X_{ij}) \cdot \left( w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
>
> where:
> - $w_i$ is the learned vector for word $i$ (as a target word)
> - $\tilde{w}_j$ is the learned vector for context word $j$
> - $b_i$ and $\tilde{b}_j$ are scalar bias terms for words $i$ and $j$
> - $\log X_{ij}$ is the log co-occurrence count, the regression target
> - $f(X_{ij})$ is a weighting function (below)

Read it as **weighted least-squares regression**: find vectors whose dot product (plus biases) predicts the log of how often the two words co-occur. Words with similar co-occurrence rows end up with similar vectors.

Not on the slides, but useful to see why logs: if $w_i \cdot \tilde{w}_k \approx \log X_{ik}$, then $(w_i - w_j) \cdot \tilde{w}_k \approx \log \frac{X_{ik}}{X_{jk}}$, a log *ratio* of co-occurrence probabilities. Ratios are what separate meaning: `ice` and `steam` both co-occur with `water`, but the ratio for `solid` is large and for `gas` small. Vector differences then encode such ratios, which is also why analogy arithmetic (section 8.3) works.

> [!formula] GloVe weighting function
> $$f(x) = \begin{cases} (x / x_\text{max})^{\alpha} & \text{if } x < x_\text{max} \\ 1 & \text{otherwise} \end{cases}$$
>
> Typically $x_\text{max} = 100$ and $\alpha = 0.75$.

Its purpose is to **de-emphasise very rare (noisy) and very frequent (uninformative) co-occurrences**. It **increases smoothly from zero for rare pairs** (down-weighting noise) and **caps at 1 for frequent pairs** (preventing uninformative co-occurrences from taking over).

| $X_{ij}$ | 0 | 1 | 5 | 10 | 25 | 50 | $\geq 100$ |
|---|---|---|---|---|---|---|---|
| $f(X_{ij})$ | 0 | 0.032 | 0.106 | 0.178 | 0.354 | 0.595 | 1 |

```
 f(x)
 1.0 ┤                                  ●━━━━━━━━━━━━━  capped at 1
     ┤                         ●·····
 0.5 ┤                 ●····
     ┤          ●···
     ┤     ●··
 0.0 ●···┬────────┬────────┬────────┬────────┬─────
     0   10       25       50       75      100    x = X_ij
```

A detail that makes the objective well defined (not on the slides): $\log X_{ij}$ is $-\infty$ for pairs that never co-occur, but $f(0) = 0$, so those terms vanish. The sum effectively runs only over the non-zero entries of $X$, which is also what makes GloVe cheap: the matrix is sparse.

### 7.3 The algorithm

Reproduced verbatim from the slides:

```pseudo
Build co-occurrence matrix X by a single pass over the corpus:
for each center word i and each context word j within the window:
    X_ij += weight     // often 1/distance, nearer words more important

Initialize w_i, w̃_j, b_i, b̃_j randomly for all words

Repeat (for several epochs):
for each pair (i,j) (sampled/shuffled):
    weight  ← f(X_ij)
    error   ← w_iᵀw̃_j + b_i + b̃_j − log X_ij
    loss_ij ← weight · error²
    update w_i, w̃_j, b_i, b̃_j by AdaGrad on this loss

Final embedding for word i ← w_i + w̃_i   (sum the two vector sets)
```

Notes:

- **One pass over the corpus** builds $X$. After that, training never looks at the corpus again, only at the non-zero cells of $X$. This is the "global" part.
- The count increment is **weighted by distance**: a word 1 position away adds 1, a word 3 positions away adds $1/3$. So $X_{ij}$ is not necessarily an integer.
- **AdaGrad** (not explained on the slides) is SGD with a per-parameter learning rate that shrinks for parameters that have received large gradients, useful here because frequent words get many updates and rare words few.
- **Final embedding $= w_i + \tilde{w}_i$**: the same trick as $W_s = W^T + W'$ in CBOW. With a symmetric window $X$ is symmetric, so the two sets of vectors play interchangeable roles and differ only by random initialisation; summing them averages out that noise.

### 7.4 GloVe versus Word2Vec

| | Word2Vec | GloVe |
|---|---|---|
| **Training procedure** | individual local context windows | global co-occurrence matrix |
| **Training objective** | predict context word(s) | weighted least-squares regression on log-counts |
| **Addresses frequency imbalances by** | subsampling frequent words (can include negative sampling) | explicit weighting function $f(X_{ij})$ |
| **Interpretability** | indirect: the model learns to predict context | directly fits log co-occurrence counts |

"Subsampling frequent words" is not defined on the slides. In Word2Vec it means randomly discarding occurrences of very frequent words before training (each occurrence of $w$ is dropped with a probability that grows with $w$'s frequency), so that `the` does not dominate the training pairs.

## 8. Evaluating word embeddings

### 8.1 Two axes of evaluation

| | Meaning |
|---|---|
| **Intrinsic** | how good are the word embeddings **by themselves**? |
| **Extrinsic** | how useful are they **for downstream tasks**, such as summarisation, MT, IR, etc.? |
| **Qualitative** | **examine the characteristics of selected examples** (nearest neighbours, plots) |
| **Quantitative** | use some **overall (task-dependent) score** |

The two axes are independent: a 2D plot is intrinsic and qualitative, a word similarity correlation is intrinsic and quantitative, a BLEU score after plugging embeddings into MT is extrinsic and quantitative.

### 8.2 Intrinsic evaluation: word similarity

> [!definition] Word similarity task
> - **Rank a list of word pairs**, e.g. (*car*, *bicycle*), **by similarity** (cosine of their embeddings).
> - Compare with human judgements using **Spearman correlation**.
> - Benchmarks: **WS-353** (annotates **relatedness**), **SimLex-999** (annotates **similarity**).
> - These benchmarks **mix all kinds of similarities** (synonyms, topical, unrelated...).

Spearman correlation (definition not on the slides) is the Pearson correlation between the *ranks* of the two lists, so only the ordering of the pairs matters and the scale of the cosine is irrelevant. With no ties, $\rho = 1 - \frac{6 \sum_k d_k^2}{N(N^2-1)}$, where $d_k$ is the difference between the two ranks of pair $k$ and $N$ the number of pairs.

The relatedness/similarity distinction matters (illustration not on the slides): `coffee` and `cup` are highly *related* (topical) but not *similar* (a cup is not a kind of coffee). A relatedness benchmark like WS-353 rewards an embedding for putting them close, a similarity benchmark like SimLex-999 penalises it. Which one an embedding does well on depends on what its training context captures, so the two benchmarks can rank the same embeddings differently.

**WS-353 examples** (slide 25; human scores on a 0 to 10 scale):

| word 1 | word 2 | score |
|---|---|---|
| tiger | cat | 7.35 |
| tiger | tiger | 10.00 |
| plane | car | 5.77 |
| train | car | 6.31 |
| television | radio | 6.77 |
| media | radio | 7.42 |
| bread | butter | 6.19 |
| cucumber | potato | 5.92 |
| doctor | nurse | 7.00 |
| word | similarity | 4.75 |
| peace | plan | 4.75 |
| drink | ear | 1.31 |
| energy | secretary | 1.81 |
| production | hike | 1.75 |

Note `media`/`radio` (7.42) scoring above `television`/`radio` (6.77), and `bread`/`butter` at 6.19: these are relatedness judgements, since bread and butter are not the same kind of thing.

### 8.3 Intrinsic evaluation: analogies

> [!definition] Analogy task
> - *Paris is to France as Berlin is to X*
> - **Evaluated by accuracy** (is the top answer exactly the right word?)
> - **Arithmetic "magic"**: $X = v_\text{king} - v_\text{man} + v_\text{woman}$, and the answer is the vocabulary word whose embedding is closest (by cosine) to $X$; the expected answer is `queen`
> - Also includes **syntactic analogies**: *acquired* is to *acquire* as *tried* is to $X$ (answer *try*)

For *a is to b as c is to X*, the vector is $X = v_b - v_a + v_c$: for the Paris example, $v_\text{France} - v_\text{Paris} + v_\text{Berlin}$, expected nearest word `Germany`. In standard evaluation the three input words are excluded from the candidate answers (not on the slides), since $X$ is usually still closest to one of them.

### 8.4 Visualising embeddings

- Visualising embeddings is **not trivial due to the number of dimensions** (128, 512, 1024, ...).
- **Project the high-dimensional space to 2D** (e.g. using **t-SNE** or **PCA**).

PCA is a linear projection onto the two directions of largest variance. t-SNE is a non-linear method that tries to keep each point's nearest neighbours near it in 2D and gives up on preserving global distances.

**Slide 28** is a 2D projection of the neighbourhood of `power`-related words, shown as labelled light-blue dots. Reading the plot:

- top right: an **electricity cluster**: `voltage`, `battery`, `electric`, `electrical`, `solar`, `electricity`, `generator`
- middle: an **energy/systems cluster**: `cooling`, `motor`, `fuel`, `output`, `supply`, `consumption`, `charging`, `wind`, `switch`, `light`, `efficiency`, `systems`, `speed`, with `energy` and `power` off to the right
- lower middle: a **control/operation cluster**: `generation`, `operate`, `controlled`, `pressure`, `system`, `capacity`, `controlling`, `capability`, `control`, `capable`
- bottom left: an **abstract "ability/authority" cluster**: `turn`, `universal`, `need`, `means`, `state`, `enough`, `thus`, `gain`, `force`, `powerful`, `powers`, `greater`, `strength`, `sense`, `authority`, `ability`
- `Power` (capitalised) sits isolated on the far left, separate from lowercase `power`, a reminder that the vocabulary is case-sensitive and the two forms got different vectors.

The point: different senses of "power" (electrical, mechanical, political/abstract) separate into different regions.

**Slide 29** is a full-vocabulary t-SNE map: thousands of tiny word labels forming dense clumps and islands, too small to read at slide resolution (the slide links to a full-resolution version). It shows that at scale the space has local cluster structure but no readable global layout.

**Slide 30**: "*Select word pairs of interest, compute differences, and compare orientations.*" A 2D projection (x from about $-0.5$ to $0.5$, y from about $-0.5$ to $0.5$) with dashed lines joining male/female word pairs:

| male (lower end) | approx. position | female (upper end) | approx. position |
|---|---|---|---|
| brother | (−0.48, −0.17) | sister | (−0.47, 0.26) |
| nephew | (−0.46, 0.02) | niece | (−0.45, 0.35) |
| uncle | (−0.40, −0.10) | aunt | (−0.41, 0.31) |
| man | (−0.31, −0.49) | woman | (−0.24, −0.06) |
| sir | (0.00, −0.45) | madam | (0.13, 0.11) |
| heir | (0.08, 0.04) | heiress | (0.13, 0.50) |
| king | (0.27, −0.50) | queen | (0.28, −0.12) |
| earl | (0.33, −0.10) | countess | (0.44, 0.33) |
| duke | (0.37, −0.16) | duchess | (0.48, 0.30) |
| emperor | (0.31, −0.27) | empress | (0.49, 0.20) |

All ten lines point roughly the same way (upwards and slightly to the right): there is a consistent "gender direction" in the space. This is the geometric fact behind $v_\text{king} - v_\text{man} + v_\text{woman} \approx v_\text{queen}$, and it is the property cross-lingual alignment relies on in section 10.

> [!warning] Do not over-trust 2D plots
> 2D visualisations are great, but **should not always be taken at face value**:
> - they can use **non-linear projections** that group things that are close in high-dimensional space, so distances in the plot are not distances in the space
> - **t-SNE hyperparameter settings have a substantial impact on results** (Wattenberg 2016): the same data can show different cluster sizes, distances, or even apparent clusters that are not there
>
> They **complement, but do not substitute**, more quantitative and extrinsic evaluations.

## 9. Morphologically sensitive embeddings: fastText

### 9.1 The problem with one vector per word type

- **Word2Vec and GloVe assign one opaque vector per word type.**
  - "run", "runs", "running" and "runner" get **four (potentially) unrelated vectors**
  - there is **no shared structure** between them, despite the obvious morphological relationship
- **Morphologically rich languages suffer most.** Finnish, Turkish, Russian and others have **huge word-form families**, and a whole-word embedding model cannot generalise across them: each inflected form is a separate, rarer word with a worse-estimated vector.
- **No fallback for unseen (OOV) words**: a word not in the training vocabulary has no vector at all.

This is the same problem as in [[MNLP-L04 - Subword Segmentation]], seen from the embedding side instead of the vocabulary side.

How to make word embeddings morphologically sensitive? Two options on the slide:

1. **Sub-segment words** (e.g. with BPE or a morphological analyser) **and learn embeddings separately for each sub-segment.**
2. **Sliding character window: fastText** (introduced by Facebook AI, Bojanowski et al. (2017)). Represent a word by all of its character n-grams.

### 9.2 The fastText algorithm

Reproduced verbatim from the slides:

```pseudo
Input:  corpus, window size m, n-gram range [n_min, n_max] (default 3–6),
        embedding dimension d, negatives K

for each word w in vocabulary:
    G_w ← { all char n-grams of "<w>" for n in [n_min, n_max] } ∪ { "<w>" }
    initialize n-gram vectors z_g (dimension d) and context vectors u_c randomly

for each (center word w, context word c) pair in corpus, with window m:
    sample K negatives n_1..n_K from noise distribution P_n
    v_w ← Σ_{g ∈ G_w} z_g                              // compose center vector
    loss ← −log σ(v_wᵀ u_c) − Σ_k log σ(−v_wᵀ u_{n_k})

    for each g ∈ G_w:
        z_g ← z_g − η · ∂loss/∂v_w          // same gradient to each n-gram
    update u_c and all u_{n_k}

return { z_g }
```

What changes compared with Skip-gram with negative sampling (section 6.4):

- The **centre word's vector is no longer a lookup**. It is **the sum of the vectors of its character n-grams**, $v_w = \sum_{g \in G_w} z_g$.
- `<` and `>` are **boundary markers** added to the word, so that prefixes and suffixes are distinct n-grams: `<ru` can only be word-initial, `ns>` only word-final, and the trigram `her` inside `<where>` differs from the word `<her>`.
- The **whole word** `<w>` is also in $G_w$, so frequent words still get a word-specific vector component.
- The **loss is exactly the negative-sampling loss** of section 6.2, with $u_c$ playing the role of $v_c$.
- Because $v_w$ is a sum, $\partial v_w / \partial z_g = I$ for every $g \in G_w$, so **every n-gram of the word receives the same gradient**.
- Context vectors $u_c$ remain ordinary per-word vectors; only the centre side is decomposed.
- **What is returned is the n-gram table $\{z_g\}$**, and there is no separate word table. The vector of any word, including one never seen in training, is computed on demand by summing its n-grams.

> [!example] The n-grams of `runs` (worked here, not on the slides)
> Wrap the word: `<runs>` (6 characters). With $n \in [3, 6]$:
>
> | $n$ | n-grams |
> |---|---|
> | 3 | `<ru`, `run`, `uns`, `ns>` |
> | 4 | `<run`, `runs`, `uns>` |
> | 5 | `<runs`, `runs>` |
> | 6 | `<runs>` (this is also the whole-word token) |
>
> So $G_\text{runs}$ has 10 elements and $v_\text{runs}$ is the sum of 10 vectors. `<running>` has 23 n-grams, and shares `<ru`, `<run` and `run` with `<runs>`, so the two words' vectors share three summands and are pulled together automatically.
>
> An OOV word such as `runnings` still has n-grams like `<run`, `runn`, `ning`, `ings>`, most of which were seen in training, so it gets a sensible vector where Word2Vec would have nothing.

In practice (not on the slides) the number of distinct n-grams is enormous, so fastText hashes them into a fixed number of buckets (about 2 million) and shares a vector per bucket.

### 9.3 Evaluation

Slide 34 shows two tables from Bojanowski et al. (2017). The slide does not label the columns. From the paper: `sg` and `cbow` are the Word2Vec baselines, `sisg` ("subword information skip-gram") is fastText, and `sisg-` is fastText where words not in the training vocabulary get a null vector instead of being built from n-grams. The left table is word similarity (Spearman correlation × 100 with human judgements), the right is analogy accuracy (%). Bold marks the best score per row as printed on the slide.

**Word similarity** (Spearman × 100):

| Language | Dataset | sg | cbow | sisg- | sisg |
|---|---|---|---|---|---|
| Arabic (AR) | WS353 | 51 | 52 | 54 | **55** |
| German (DE) | GUR350 | 61 | 62 | 64 | **70** |
| German (DE) | GUR65 | 78 | 78 | **81** | **81** |
| German (DE) | ZG222 | 35 | 38 | 41 | **44** |
| English (EN) | RW | 43 | 43 | 46 | **47** |
| English (EN) | WS353 | 72 | **73** | 71 | 71 |
| Spanish (ES) | WS353 | 57 | 58 | 58 | **59** |
| French (FR) | RG65 | 70 | 69 | **75** | **75** |
| Romanian (RO) | WS353 | 48 | 52 | 51 | **54** |
| Russian (RU) | HJ | 59 | 60 | 60 | **66** |

**Word analogy** (accuracy %):

| Language | Type | sg | cbow | sisg |
|---|---|---|---|---|
| Czech (CS) | Semantic | 25.7 | 27.6 | 27.5 |
| Czech (CS) | Syntactic | 52.8 | 55.0 | 77.8 |
| German (DE) | Semantic | 66.5 | 66.8 | 62.3 |
| German (DE) | Syntactic | 44.5 | 45.0 | 56.4 |
| English (EN) | Semantic | 78.5 | 78.2 | 77.8 |
| English (EN) | Syntactic | 70.1 | 69.9 | 74.9 |
| Italian (IT) | Semantic | 52.3 | 54.7 | 52.3 |
| Italian (IT) | Syntactic | 51.5 | 51.8 | 62.7 |

What to read off:

- **Word similarity:** sisg is best or tied-best on 9 of 10 datasets. The only loss is English WS353 (71 against cbow's 73), the dataset of frequent English words where whole-word vectors are already good. The biggest gains are in **morphologically rich languages**: German GUR350 +9 over sg (+8 over cbow), Russian HJ +7. On **English RW** (the Rare Words dataset) it gains 4, because rare words are where n-gram sharing helps.
- **sisg against sisg-:** building OOV vectors from n-grams adds up to 6 points (DE GUR350 64 to 70, RU 60 to 66), which is the direct value of having a fallback for unseen words.
- **Analogies:** the gains are almost entirely **syntactic**: Czech +25.0 (52.8 to 77.8), German +11.9, Italian +11.2, English +4.8 over sg. Syntactic analogies (*acquired : acquire :: tried : try*) are about morphology, which is exactly what the n-grams capture.
- **Semantic analogies do not improve and sometimes get worse** (German 66.5 to 62.3, English 78.5 to 77.8). `France : Paris :: Germany : Berlin` has nothing to do with character overlap, and forcing words with shared n-grams together can add noise there.

## 10. Cross-lingual embeddings

### 10.1 The goal

Embeddings map similar words (e.g. synonyms) to neighbouring points in a high-dimensional space:

- EN↔EN: $E(\text{house}) \approx E(\text{building})$
- NL↔NL: $E(\text{huis}) \approx E(\text{gebouw})$
- What about words **in different languages**? EN↔NL: $E(\text{house}) \approx E(\text{gebouw})$?

(The slide writes the right-hand sides without the $E$, e.g. "$E(\text{house}) \approx (\text{building})$"; the $E$ is clearly intended.)

A cross-lingual embedding space puts translation-equivalent words close together. Its main uses in this course are **bilingual lexicon induction** (building a dictionary by nearest-neighbour search) and transferring models trained on one language to another, which [[MNLP-L07 - Crosslingual NLP]] develops.

**Three basic training set-ups:**

1. **Train embeddings for each language separately, then align the spaces.**
2. **Train embeddings for all languages together** (benefiting from word overlap, e.g. names and numbers shared across languages), **then align the regions**.
3. **Train embeddings on data that is aligned** at the word or sentence level.

**Either way, some form of alignment has to take place:**

- learn a **mapping that brings the two spaces into alignment**
- goal: **translation-equivalent words end up geometrically close together**
- hope: use **large monolingual corpora** and only a **small bilingual signal** for alignment, because monolingual text is plentiful for many languages and parallel text is not

The rest of the lecture is about set-up 1.

### 10.2 The shared space hypothesis

> [!definition] Shared space hypothesis
> - **Different languages, trained separately, still encode similar relational structure.** The geometric relationship between "king" and "queen" in English is structurally similar to that between "rey" and "reina" in Spanish.
> - There is an **approximate isomorphism between embedding spaces in different languages**:
>   - the **overall shape of the point cloud is similar**
>   - the **specific coordinates and orientations can differ** (each training run starts from random initialisation, so the axes mean nothing)
> - **If true (it is a hypothesis)**, a **single geometric transformation can align the spaces**: a general linear mapping applied to one space should bring translations of word pairs close together.

**Slide 37** shows the evidence that motivated this (the slide gives no source). Four 2D projections of embedding spaces, English on the left in red, Spanish on the right in blue, with an arrow between them:

| English word | approx. 2D position | Spanish word | approx. 2D position |
|---|---|---|---|
| one | (0.63, 0.02) | uno (one) | (1.05, 0.10) |
| two | (−0.03, −0.25) | dos (two) | (0.18, −0.56) |
| three | (0.08, −0.13) | tres (three) | (0.27, −0.18) |
| four | (−0.04, 0.14) | cuatro (four) | (−0.05, 0.20) |
| five | (0.01, 0.04) | cinco (five) | (0.08, −0.03) |
| horse | (−0.16, 0.17) | caballo (horse) | (−0.30, 0.41) |
| cow | (−0.11, 0.06) | vaca (cow) | (−0.24, 0.30) |
| pig | (0.00, 0.00) | cerdo (pig) | (0.00, 0.00) |
| dog | (0.14, 0.01) | perro (dog) | (0.43, 0.18) |
| cat | (−0.29, −0.26) | gato (cat) | (−0.43, −0.40) |

The scales differ (the Spanish plots are stretched) but the **arrangement is the same**: in both languages `one` is far to the right of the other numbers, `four` at the top, `two` at the bottom; `cat` is isolated at the bottom left, `horse` and `cow` close together at the top left, `dog` to the right. Two independently trained spaces, one shape.

**The king/queen example (slide 38):**

- EN: king, queen, man, woman. "king" and "queen" are separated by roughly the **same offset** as "man" and "woman" (a consistent gender direction, as in section 8.4).
- ES: rey, reina, hombre, mujer. The **same relational structure holds**, but the whole point cloud may be **rotated and scaled differently** relative to English.

> [!formula] Isomorphism
> A mapping $m$ preserves structure if
> $$m(a \circ b) = m(a) \circ m(b)$$
> where $\circ$ is the operation on the space (here vector addition and subtraction). Applied to the analogy:
> $$m(v_\text{king} - v_\text{man} + v_\text{woman}) \approx m(v_\text{king}) - m(v_\text{man}) + m(v_\text{woman}) \approx m(v_\text{king} - v_\text{man}) + m(v_\text{woman})$$
>
> So if $m$ maps English to Spanish, the English analogy maps to the Spanish one: the gender offset in English becomes the gender offset in Spanish.

For a linear map $m(v) = Wv$ these equalities hold exactly, since $W(a - b + c) = Wa - Wb + Wc$. That is the motivation for using a linear $W$ (section 11.4). The "$\approx$" on the slide reflects that the spaces themselves are only approximately isomorphic. (Strictly, a structure-preserving map is a homomorphism; an isomorphism is additionally invertible. The slide uses the term loosely.)

## 11. Supervised alignment with a seed dictionary

### 11.1 Types of alignment methods

| Type | Bilingual signal | How |
|---|---|---|
| **Supervised** | a **seed bilingual dictionary** as anchors, **typically thousands** of word translations | solve for the mapping directly (sections 11 to 13) |
| **Semi-supervised** | a **very small seed dictionary** (**hundreds** of word translations) | **bootstrap new translations based on statistical confidence**, then re-solve (section 15) |
| **Unsupervised** | **no bilingual signal** | **uses adversarial training** (section 14) (the slide has the typo "sues") |

### 11.2 Where seed dictionaries come from

Slide 40 lists three sources and poses two questions to think about:

1. **Large human-compiled, machine-readable dictionaries.** Ideal, but *"those are often not ideal, why?"* The slide leaves this open. Reasons consistent with the rest of the lecture: they list lemmas (`talk`) while the embedding vocabulary is full of inflected forms (`hablamos`, see slide 48 below), they contain many senses per entry, coverage of low-resource languages is poor, and they do not cover multi-word expressions.
2. **Learn translation dictionaries from actual data.** This **requires human translations at the sentence level** (a parallel corpus), then **word align each sentence pair**.
3. **Exploit the fact that some words occur in multiple languages** (identically spelled words: names, numbers, borrowings). *"Also has some issues, which ones?"* Again left open on the slide. The obvious ones: false friends (Dutch `bad` means bath, English `bad` does not), it only works between languages that share a script, and the shared words are skewed towards names and numbers. Section 16 shows that, despite this, identical-word dictionaries work surprisingly well.

**Word alignment (slide 41)** is shown as an alignment grid between an English sentence (columns) and its Dutch translation (rows); a dot means the two words are aligned:

```
                The  Secretary  of  State  visits  The  Netherlands
De               ●
minister                ●
van                             ●
buitenlandse                         ●
zaken                                ●
brengt                                      ●
een                                         ●
bezoek                                      ●
aan                                         ●
Nederland                                           ●       ●
```

The alignments are not one-to-one. `buitenlandse zaken` (foreign affairs) both align to `State` (as in Secretary of State); the four-word Dutch phrase `brengt een bezoek aan` (literally "brings a visit to") aligns to the single English verb `visits`; and `Nederland` aligns to the two English words `The Netherlands`. A word aligner run over a parallel corpus produces such links, and the most frequent links per word give a dictionary. The non-one-to-one cases show why the resulting dictionary is noisy at the word level.

### 11.3 The general alignment problem

> [!formula] General alignment problem #key-formula
> $$W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2$$
>
> where:
> - $(x_i, y_i)$ are **paired vectors**: a source-language word embedding $x_i \in \mathbb{R}^{d_1}$ and the embedding $y_i \in \mathbb{R}^{d_2}$ of its translation in the target language
> - $W$ is a $d_2 \times d_1$ matrix, the transformation applied to every source vector
> - $\lVert \cdot \rVert$ is the Euclidean norm
>
> Find the transformation $W$ that, applied to every source vector, gives the smallest distance to the corresponding target embedding.

The slide calls this "the **Procrustes** algorithm" (and has the typo "know as").

> [!warning] Terminology: what "Procrustes" means
> As written, with no constraint on $W$, this is ordinary multivariate least-squares regression. The name **Procrustes problem** in the literature, and in the "Procrustes" rows of the results tables on slides 54 and 61, refers to the **orthogonally constrained** version of section 12 (solved with SVD). The slide's naming here is loose. For the exam: unconstrained least squares is Mikolov et al. (2013); orthogonal Procrustes is Xing et al. (2015) onwards.

The **basic questions** the rest of the lecture answers:

- **what should $W$ be?** (linear, section 11.4)
- **what constraints should it satisfy?** (orthogonality, section 12)
- **where do the pairs $(x_i, y_i)$ come from?** (a seed dictionary, section 11.2; or nowhere, section 14)

**Seed dictionaries (slide 43).** A seed dictionary is a **list of (source word, target word) translation pairs**:

- often just the **top few thousand most frequent words**, translated using an existing bilingual dictionary
- **automatic translation** can be used as well (with caution: errors become wrong anchors)
- **limited to words**: there are no embeddings for multi-word expressions (phrases), so they cannot be anchors

Each pair gives **one anchor point across the two spaces**: e.g. $(x_i, y_i)$ = the source embedding of "cat" and the target embedding of "gato". **The goal is a mapping that generalises beyond these anchors**: words never seen in the seed dictionary should also end up correctly aligned. A dictionary of 5000 pairs is useless on its own; a mapping learned from it that translates the other 195 000 words is the point.

### 11.4 The linear transformation hypothesis

> [!definition] Linear transformation hypothesis
> - **Strong assumption:** a **single linear transformation $W$ is sufficient** to map one language's embedding space onto another's.
> - **Why linear (as opposed to non-linear)?** King/queen and man/woman analogies are **already linear regularities within a single space**, and the assumption is that the **same regularities hold across spaces**. A linear map preserves them exactly (section 10.2).
> - It **reduces alignment to a classic regression problem**: given paired points, find the matrix $W$ that best maps one set onto the other.

### 11.5 The least-squares mapping algorithm

Reproduced verbatim from the slides:

```pseudo
Input:  source embeddings, target embeddings, seed dictionary D of n pairs

For each pair (source_word, target_word) ∈ D:
   x_i ← source embedding of source_word
   y_i ← target embedding of target_word
Assemble X (d₁ × n) and Y (d₂ × n)

initialize W randomly

repeat until convergence:
     for each mini-batch of pairs:
          loss ← Σ ‖W x_i − y_i‖²
          W ← W − η · ∂loss/∂W

return W
```

The gradient (not on the slides): $\frac{\partial}{\partial W} \sum_i \lVert W x_i - y_i \rVert^2 = 2 \sum_i (W x_i - y_i)\, x_i^\top$.

> [!formula] Closed-form solution
> $$W^* = Y X^\top (X X^\top)^{-1}$$
>
> where $X$ ($d_1 \times n$) and $Y$ ($d_2 \times n$) hold the paired vectors as columns.

Derivation (not on the slides): the loss is $\lVert WX - Y \rVert_F^2$ (Frobenius norm, the sum of squares of all entries). Setting the gradient $2(WX - Y)X^\top$ to zero gives $W X X^\top = Y X^\top$, so $W = Y X^\top (X X^\top)^{-1}$. This needs $X X^\top$ ($d_1 \times d_1$) to be invertible, which requires at least $d_1$ linearly independent source vectors, i.e. $n \geq d_1$ dictionary pairs.

- **Mikolov et al. (2013) used SGD.**
- **SGD scales better** with large embedding dimensions and dictionaries.
- **The closed form is exact; SGD is approximate but more memory-friendly** (it never forms or inverts $X X^\top$).

At inference, to translate a source word $x$: compute $Wx$ and return the target word whose embedding is nearest (by cosine).

### 11.6 Results of the linear mapping

Slides 46 to 48 carry no citation; they follow the slide on Mikolov et al. (2013) and slide 51 labels the same unconstrained method as Mikolov et al. (2013).

**Slide 46**, word translation precision (P@1: correct translation is the top candidate; P@5: in the top 5):

| Translation | Edit Distance P@1 | P@5 | Word Co-occurrence P@1 | P@5 | Translation Matrix P@1 | P@5 | ED + TM P@1 | P@5 | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| En → Sp | 13% | 24% | 19% | 30% | 33% | 51% | 43% | 60% | 92.9% |
| Sp → En | 18% | 27% | 20% | 30% | 35% | 52% | 44% | 62% | 92.9% |
| En → Cz | 5% | 9% | 9% | 17% | 27% | 47% | 29% | 50% | 90.5% |
| Cz → En | 7% | 11% | 11% | 20% | 23% | 42% | 25% | 45% | 90.5% |

The slide does not explain the columns; reading them from their names:

- **Edit Distance** is a baseline that translates a word to the most similarly spelled target word (it works only through cognates).
- **Word Co-occurrence** is a count-based baseline (section 2 vectors instead of learned embeddings).
- **Translation Matrix** (TM) is the learned linear $W$.
- **ED + TM** combines the two.
- **Coverage** is not defined on the slide; it is presumably the share of test words the method could produce a translation for.

The translation matrix beats both baselines by a wide margin (En→Sp P@1 33% against 13% and 19%), and combining it with edit distance adds another 10 points for Spanish but only 2 for Czech. Spanish and English share far more cognates than Czech and English, so spelling similarity helps more there. Czech, the more distant language, is harder for every method.

**Slide 47**, two plots:

*Left, accuracy against amount of training data.* x-axis: number of monolingual training words, log scale from $10^7$ to $10^{11}$; y-axis: accuracy 0 to 80. Two rising curves:

| training words (approx.) | $2 \cdot 10^7$ | $6 \cdot 10^7$ | $2 \cdot 10^8$ | $6 \cdot 10^8$ | $2 \cdot 10^9$ | $2.5 \cdot 10^{10}$ |
|---|---|---|---|---|---|---|
| Precision@1 (blue) | 9 | 25 | 38 | 44 | 51 | 53 |
| Precision@5 (red) | 19 | 41 | 55 | 61 | 70 | 75 |

Accuracy grows roughly linearly in the log of the data size and flattens above about $2 \cdot 10^9$ words. Better monolingual embeddings make the mapping work better.

*Right, accuracy by word frequency.* A bar chart with no legend and no x-axis title; the bins are labelled 5–7K, 7–9K, ..., 17–19K, which reads as frequency-rank bins of the test words. Blue and red bars, presumably P@1 and P@5 as in the left plot:

| bin | 5–7K | 7–9K | 9–11K | 11–13K | 13–15K | 15–17K | 17–19K |
|---|---|---|---|---|---|---|---|
| blue (P@1?) | 53 | 50 | 45 | 44 | 42 | 44 | 40 |
| red (P@5?) | 73 | 70 | 68 | 65 | 62 | 63 | 60 |

Accuracy drops as words get rarer (P@1 from 53 to 40 between the 5–7K and 17–19K bins): rarer words have worse embeddings, so their mapped positions are less reliable.

**Slide 48**, translation examples (Spanish to English, top 3 computed translations against the dictionary entry):

| Spanish word | Computed English translations | Dictionary entry |
|---|---|---|
| emociones | emotions, emotion, feelings | emotions |
| protegida | wetland, undevelopable, protected | protected |
| imperio | dictatorship, imperialism, tyranny | empire |
| determinante | crucial, key, important | determinant |
| preparada | prepared, ready, prepare | prepared |
| millas | kilometers, kilometres, miles | miles |
| hablamos | talking, talked, talk | talk |
| destacaron | highlighted, emphasized, emphasised | highlighted |

The errors are instructive. `imperio` gives `dictatorship`, `imperialism`, `tyranny`: all topically related, none the translation `empire`. `determinante` gives near-synonyms of the sense used in running text (`crucial`, `key`) rather than the dictionary's `determinant`. `millas` puts `kilometers` above `miles`, since the two units occur in identical contexts. `protegida` (protected, feminine) ranks `wetland` first, presumably from contexts like *zona protegida*. And `hablamos` (we talk) maps to `talking`, `talked`, `talk`: the morphology does not line up one-to-one, so the dictionary entry `talk` is only rank 3. Embedding nearness captures relatedness, and the dictionary demands exact equivalence.

## 12. Orthogonal mappings

### 12.1 What is wrong with an unconstrained $W$

> [!warning] Limitations of unconstrained mappings
> - **An arbitrary matrix $W$ can stretch, shear and rotate the space.** Nothing forces it to preserve the **internal geometry** (distances and angles) of the source language.
> - This can result in **overfitting to the seed dictionary**: words in (or near) the seed pairs are covered well, and the mapping **does not generalise to words not in the dictionary** (the majority).
> - **Mismatch between training and inference objectives:**
>   - training minimises **Euclidean distance** $\lVert W x_i - y_i \rVert^2$
>   - inference retrieves nearest neighbours by **cosine similarity**
>   - the two are **not equivalent unless vectors are unit-normalised**
> - So **additional constraints on the form of $W$ are needed.**

### 12.2 The orthogonality constraint

> [!formula] Orthogonal mapping (Xing et al., 2015) #key-formula
> $$W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2 \quad \text{subject to} \quad W^\top W = I$$
>
> with **normalisation**: all embeddings are of unit length, $\lVert x \rVert^2 = 1$.
>
> where $I$ is the identity matrix. If $W$ is orthogonal it **only allows rotations and reflections**, and **no scaling or shearing**: it **preserves angles and distances**.
>
> **$W^*$ can be solved exactly using SVD.**

Why this fixes both problems:

1. **Geometry is preserved.** $\lVert W x \rVert^2 = x^\top W^\top W x = x^\top x = \lVert x \rVert^2$, and similarly for dot products, so the source space is moved rigidly. It cannot be warped to fit the seed pairs at the expense of everything else, which is the overfitting argument. And a rigid rotation is exactly what the shared space hypothesis says should be enough.
2. **Training and inference agree.** For unit vectors, and with $W$ orthogonal so $\lVert W x \rVert = 1$ (derivation not on the slides):
$$\lVert W x - y \rVert^2 = \lVert W x \rVert^2 + \lVert y \rVert^2 - 2\, (Wx) \cdot y = 2 - 2 \cos(Wx, y)$$
So minimising Euclidean distance **is** maximising cosine similarity.

**Slide 50's figure.** Two 3D scatter plots of the same few thousand points (light blue), with three highlighted red points (a circle, a diamond, a star). Left: unnormalised vectors scattered through a cube with axes from 0 to 100. Right: the same vectors after normalisation, all lying on the surface of the unit sphere (axes $-1$ to $1$), with the three red points at the corresponding positions on the sphere. Normalising throws away length and keeps only direction, which is all cosine looks at; after normalisation an orthogonal map is literally a rotation (or reflection) of that sphere.

**Solving it with SVD** (the slide only states that it can be done; the solution and derivation are not on the slides). With $X$ ($d \times n$) and $Y$ ($d \times n$) holding the dictionary pairs as columns:

> [!formula] Orthogonal Procrustes solution
> $$U \Sigma V^\top = \operatorname{SVD}(Y X^\top), \qquad W^* = U V^\top$$
>
> where $U$, $V$ are orthogonal and $\Sigma$ is the diagonal matrix of singular values of the $d \times d$ matrix $Y X^\top$.

Derivation: $\lVert WX - Y \rVert_F^2 = \lVert WX \rVert_F^2 + \lVert Y \rVert_F^2 - 2\operatorname{tr}(W^\top Y X^\top)$. The first term equals $\lVert X \rVert_F^2$ for orthogonal $W$, so minimising the loss means maximising $\operatorname{tr}(W^\top Y X^\top) = \operatorname{tr}(W^\top U \Sigma V^\top) = \operatorname{tr}(V^\top W^\top U\, \Sigma)$. The matrix $Z = V^\top W^\top U$ is orthogonal, so its diagonal entries are at most 1 and $\operatorname{tr}(Z\Sigma) = \sum_k Z_{kk} \sigma_k \le \sum_k \sigma_k$, with equality at $Z = I$. Hence $W^\top = V U^\top$, i.e. $W = U V^\top$. One SVD of a $d \times d$ matrix, no iteration, no learning rate: this is why the refinement step in section 15 can "re-solve for $W$ exactly".

### 12.3 Results: unconstrained against orthogonal

**Slide 51**, English to Spanish word translation, for different embedding dimensionalities (D-EN for English, D-ES for Spanish):

| D-EN | D-ES | Mikolov et al. (2013) P@1 | P@5 | Xing et al. (2015) P@1 | P@5 |
|---|---|---|---|---|---|
| 300 | 300 | 30.43% | 49.43% | 38.99% | 59.16% |
| 500 | 500 | 25.76% | 44.29% | 39.91% | 59.82% |
| 700 | 700 | 20.69% | 39.12% | 41.04% | 59.38% |
| 800 | 200 | 35.36% | 53.96% | 40.06% | 60.02% |

Left half: unnormalised, unconstrained (Mikolov et al. 2013). Right half: normalised, orthogonal (Xing et al. 2015).

- The **unconstrained mapping gets worse as the dimension grows**: P@1 falls from 30.43% at 300 dimensions to 20.69% at 700. More dimensions means more free parameters in $W$ ($d^2$) for the same seed dictionary, i.e. more room to overfit, which is exactly the slide 49 warning.
- The **orthogonal mapping is better everywhere and improves slightly with dimension** (38.99% to 41.04%), since its number of effective degrees of freedom is constrained.
- The unconstrained method's best setting is the asymmetric 800/200 one (35.36%).

> [!warning] An orthogonality detail the slides skip
> The slide notes "the different embedding dimensionalities for English and Spanish". With $d_\text{EN} = 800$ and $d_\text{ES} = 200$, $W$ is not square, and a non-square $W$ cannot satisfy $W^\top W = I$ in both directions: a $200 \times 800$ matrix has rank at most 200, so $W^\top W$ ($800 \times 800$) cannot be the identity. For that row, the constraint can only hold in the form $W W^\top = I$ (rows orthonormal) or with the roles reversed, depending on the mapping direction. The slide does not say how Xing et al. handled this; the constraint as written applies cleanly only to the square cases.

## 13. Hubness and CSLS

### 13.1 The hubness problem

**Nearest neighbour retrieval** is used **at inference time** to actually return the most similar words, based on the (mapped) embeddings.

> [!definition] Hubs
> In high-dimensional spaces, **some points become "hubs"**: a **small number of vectors end up as the nearest neighbour of many other points, regardless of any actual similarity**.
> - a **known phenomenon in high-dimensional geometry**, not specific to embeddings
> - it **worsens as dimensionality increases**

**Consequence for translation retrieval:** hub words in the target language get nominated as the nearest-neighbour "translation" for many unrelated source words:

$$\text{nn}(\text{cat}) = \text{thing}, \quad \text{nn}(\text{car}) = \text{thing}, \quad \ldots, \quad \text{nn}(\text{house}) = \text{thing}$$

Nearest neighbour is not symmetric: `thing` can be the nearest neighbour of `cat` even though `cat` is far from being the nearest neighbour of `thing`. A hub sits near the "centre" of the cloud and is moderately close to everything. Conneau et al. (2018) propose a modification of cosine similarity to correct for this.

### 13.2 CSLS

> [!formula] Cross-domain Similarity Local Scaling (CSLS) #key-formula
> $$\text{CSLS}(x, y) = 2 \cos(x, y) - r_T(x) - r_S(y)$$
>
> where:
> - $x$ is a mapped source vector ($x = W x_s$) and $y$ a target vector
> - $r_T(x)$ is the **average cosine similarity between $x$ and its $K$ nearest neighbours in the target space**:
> $$r_T(W x_s) = \frac{1}{K} \sum_{y_t \in \mathcal{N}_T(W x_s)} \cos(W x_s, y_t)$$
>   with $\mathcal{N}_T(W x_s)$ the set of the $K$ nearest target vectors to $W x_s$
> - $r_S(y)$ is the **average cosine similarity between $y$ and its $K$ nearest neighbours in the (mapped) source space**
>
> Translation of $x_s$: the target word $y$ that maximises $\text{CSLS}(W x_s, y)$.

- **Subtracting $r_S(y)$ penalises a candidate $y$ if it is, on average, close to lots of things** (a hub): a hub has a high $r_S$, so its score drops.
- $r_T(x)$ is the same for every candidate $y$ when translating a fixed $x$, so it does not change which $y$ wins. It matters when CSLS scores are compared across different source words, e.g. when ranking candidate pairs to build a dictionary (section 15).
- **CSLS significantly increases word translation retrieval accuracy**, while **not requiring any parameter tuning** (only $K$; Conneau et al. use $K = 10$, which is not on the slides).

> [!example] CSLS defusing a hub (toy numbers, not from the slides)
> Translating `cat`. Its mapped vector $x$ has $r_T(x) = 0.50$. Two candidates:
>
> | candidate | $\cos(x, y)$ | $r_S(y)$ | CSLS $= 2\cos - r_T - r_S$ |
> |---|---|---|---|
> | `thing` (a hub) | 0.65 | 0.70 | $1.30 - 0.50 - 0.70 = 0.10$ |
> | `gato` | 0.60 | 0.40 | $1.20 - 0.50 - 0.40 = 0.30$ |
>
> Plain nearest neighbour picks `thing` (0.65 > 0.60). CSLS picks `gato`, because `thing` is similarly close to every source word and gets penalised for it.

### 13.3 Results

**Slide 54**, supervised alignment (seed dictionary) with fastText embeddings, comparing three retrieval criteria on the same Procrustes (orthogonal) mapping. The slide does not name the metric; in Conneau et al. (2018), where the table comes from, it is word translation accuracy P@1 (%):

| Method | en-es | es-en | en-fr | fr-en | en-de | de-en | en-ru | ru-en | en-zh | zh-en | en-eo | eo-en |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Procrustes - NN | 77.4 | 77.3 | 74.9 | 76.1 | 68.4 | 67.7 | 47.0 | 58.2 | 40.6 | 30.2 | 22.1 | 20.4 |
| Procrustes - ISF | 81.1 | 82.6 | 81.1 | 81.3 | 71.1 | 71.5 | 49.5 | 63.8 | 35.7 | **37.5** | 29.0 | 27.9 |
| Procrustes - CSLS | 81.4 | 82.9 | 81.1 | **82.4** | 73.5 | **72.4** | **51.7** | **63.7** | **42.7** | 36.7 | **29.3** | 25.3 |

(Bold as on the slide. The bold marks the best in the full table, which slide 61 shows, so the es and en-fr columns have no bold here: the unsupervised method beats these. The ru-en bold is inconsistent: CSLS 63.7 is bold although ISF shows 63.8 in the same column.)

- **NN** is plain cosine nearest neighbour; **ISF** is "inverted softmax", an alternative hubness correction (not explained on the slides); **CSLS** as above.
- CSLS improves over NN on every pair, by 4.0 points for en-es, 6.3 for fr-en, 4.7 for de-en (5.1 for en-de), 5.5 for ru-en.
- The accuracy order follows language distance: Spanish and French around 80, German low 70s, Russian 50 to 64, Chinese 31 to 43, Esperanto 20 to 29 (Esperanto presumably low because its training corpus is small).

## 14. Unsupervised alignment: adversarial training

### 14.1 Why and how

Supervised methods need a seed dictionary of **thousands of translation pairs**:

- for **many low-resource language pairs**, even a small bilingual dictionary **may not exist**
- dictionaries **may not cover inflected forms**, which are the forms most often observed in actual data

So: **can two independently trained monolingual spaces be aligned from their geometry alone?** No anchors, no seed pairs, just **two point clouds that are hypothesised to be approximately isomorphic**.

> [!warning] Chicken and egg
> We need a **mapping to find translation pairs**, and **translation pairs to learn the mapping**.
>
> The approach of **Conneau et al. (2018)**: **adversarial training bootstraps an initial rough alignment from distributional structure alone**, and the rough alignment then yields translation pairs for the supervised machinery of sections 12 and 13.

### 14.2 The basic idea

Based on the **Generative Adversarial Network (GAN)** framework:

| Role | What it is | Aim |
|---|---|---|
| **Generator** | the mapping $W$ from before, mapping source embeddings into the target space | make mapped source vectors **indistinguishable from real target vectors** |
| **Discriminator** | a classifier, **trained simultaneously** | tell apart **mapped source vectors** and **real target vectors** |

The two are **trained in opposition**: the discriminator gets better at telling real embeddings from mapped ones, and $W$ gets better at fooling the discriminator.

**Data:**

- **real data:** the target-language embedding distribution $y_1, \ldots, y_n$
- **fake data:** the mapped source distribution $W x_1, \ldots, W x_m$

If the discriminator cannot tell $\{W x_i\}$ from $\{y_j\}$, the mapped source cloud has the same shape and position as the target cloud. Under the shared space hypothesis, the only rotation that achieves this is the one that puts translations on top of each other.

### 14.3 Design choices

> [!important] The generator is deliberately not a deep network
> - It is a **single $d \times d$ matrix**: a **linear map with no non-linearity**, and this is deliberate.
> - A **deep generator could learn an arbitrarily complex warping**.
> - It **could make the mapped cloud statistically indistinguishable from the target cloud while completely scrambling which source word is mapped where**. Matching the *distribution* says nothing about matching *individual points*; only a constrained, near-rigid map forces the point-level correspondence to come along with the distribution match.

**Discriminator architecture:**

- **multilayer perceptron, 2 hidden layers of 2048 units each**
- **ReLU** activations
- **dropout on the input layer**
- output: **two-class prediction with label smoothing $s = 0.2$**: instead of target labels $\{0, 1\}$ use $\{s, 1-s\}$, which **prevents the discriminator from becoming pathologically overconfident** (a discriminator that outputs exactly 0 or 1 gives the generator vanishing gradients)

**Sampling:**

- both the source and target vectors fed to the discriminator are drawn from **the most frequent 50k words** of each language
- **rare words have poorly estimated embeddings** (few training occurrences)
- **restricting to frequent words gives a cleaner discriminatory signal**

### 14.4 The algorithm

Reproduced verbatim from the slides:

```pseudo
Input: source embeddings X, target embeddings Y (monolingual, unaligned)
       orthogonalization coefficient β = 0.01

Initialize W ← I  (identity/random orthogonal matrix)
Initialize discriminator parameters θ_D randomly

repeat:
    // Discriminator step (W frozen):
    sample batch of source vectors {x_i}
    sample batch of target vectors {y_j}
    compute mapped vectors {W x_i}

    ℒ_D ← −(1/n) Σ log P_θD(source=1 | W x_i)
          −(1/m) Σ log P_θD(source=0 | y_j)
    θ_D ← θ_D − η · ∂ℒ_D/∂θ_D

    // Generator step (θ_D frozen):
    sample new batches {x_i}, {y_j}
    ℒ_W ← −(1/n) Σ log P_θD(source=0 | W x_i)
          −(1/m) Σ log P_θD(source=1 | y_j)

    // backpropagate ℒ_W through the frozen discriminator into W
    ∇_W ← (1/n) Σ δ_i x_iᵀ // where δ_i = ∂ℒ_W/∂(W x_i)
    W ← W − η · ∇_W

    // Re-Orthogonalization:
    W ← (1 + β) W − β (W Wᵀ) W
```

Line by line:

- $P_{\theta_D}(\text{source} = 1 \mid v)$ is the discriminator's probability that vector $v$ is a **mapped source** vector (fake); $\text{source} = 0$ means real target.
- **Discriminator loss $\mathcal{L}_D$**: ordinary binary cross entropy with the true labels (mapped source = 1, target = 0).
- **Generator loss $\mathcal{L}_W$**: the **same cross entropy with the labels flipped**. $W$ is rewarded when the discriminator calls mapped vectors "target" (source = 0). The second term, over $y_j$, does not depend on $W$ and contributes no gradient; it is written for symmetry.
- **The gradient for $W$**: $W x_i$ is the discriminator's input, so the chain rule gives $\partial \mathcal{L}_W / \partial W = \sum_i \delta_i x_i^\top$ with $\delta_i$ the gradient of the loss with respect to the discriminator's input. (Since $\mathcal{L}_W$ already contains the $1/n$, the extra $1/n$ on the slide only rescales the step and is absorbed into $\eta$.)
- **Re-orthogonalisation** with $\beta = 0.01$ pulls $W$ back towards an orthogonal matrix after each gradient step. If $W$ is already orthogonal, $W W^\top = I$ and the update gives $(1+\beta)W - \beta W = W$: orthogonal matrices are fixed points. Not on the slides: writing $W$ via its SVD, each singular value $s$ is updated as $s \leftarrow (1+\beta)s - \beta s^3$, which pushes $s$ towards 1 (with $\beta = 0.01$: $1.5 \to 1.481$, $0.5 \to 0.504$, $1 \to 1$). Gradient steps alone would let $W$ drift into a general linear map with the overfitting and warping problems of section 12.1.

### 14.5 Model selection without a dictionary

**When should training stop, i.e. which checkpoint should be kept?**

- the **adversarial loss is not a reliable indicator** of alignment quality
- the **discriminator loss can look fine while the mapping is poor**
- in a truly unsupervised setting there is **no validation dictionary** to check against

> [!definition] Unsupervised validation criterion
> 1. Take the **10k most frequent source words**.
> 2. With the current $W$, find each source word's **CSLS nearest neighbour** in the target space.
> 3. Compute the **average cosine similarity** across these induced pairs.
>
> Keep the checkpoint with the highest value.

```pseudo
Algorithm: Unsupervised model selection criterion
─────────────────────────────────────────────────
Input:  mapping W, source embeddings, target embeddings
S ← 10 000 most frequent source words
total ← 0
for each s ∈ S:
    t* ← argmax_t CSLS(W x_s, y_t)        // induced translation
    total ← total + cos(W x_s, y_t*)
return total / |S|
```

**Empirical finding** behind it: a **poor mapping** puts mapped source words **far from any real target words**, so even their best matches have low cosine. A **high average cosine** indicates that the mapping lands on actual (hopefully correct) target words. It is a proxy, and the slide's "(correct?)" is the honest caveat: high cosine does not prove the pairs are translations.

## 15. Semi-supervised refinement and the MUSE pipeline

The mapping $W$ from adversarial training is a **rough alignment**: the **global orientation is approximately right**, but it is **not precise enough** for high-quality translation retrieval. The full **MUSE** pipeline (Conneau et al.'s toolkit) therefore continues with a refinement:

1. **Adversarial training** to obtain a rough $W$.
2. **Build a synthetic dictionary**: use $W$ with **CSLS** to find **mutual nearest-neighbour pairs among frequent words**, and **keep only high-confidence mutual matches**.
3. **Re-solve for $W$ exactly via SVD** on that induced dictionary (orthogonal Procrustes, section 12.2).
4. **Repeat steps 2 and 3** (self-learning).

```mermaid
flowchart TD
    A["Monolingual embeddings<br/>X (source), Y (target)"] --> B["Adversarial training<br/>rough orthogonal W"]
    B --> C["Synthetic dictionary:<br/>mutual CSLS nearest neighbours<br/>among frequent words"]
    C --> D["Procrustes: W = UVᵀ<br/>from SVD of YXᵀ on the dictionary"]
    D --> E{"repeat?"}
    E -- yes --> C
    E -- no --> F["Final W<br/>translate with CSLS"]
```

```pseudo
Algorithm: MUSE refinement (self-learning), assembled from slide 60
────────────────────────────────────────────────────────────────────
Input:  rough W from adversarial training,
        source embeddings, target embeddings, number of iterations R
for r = 1 to R:
    D ← ∅
    for each frequent source word s:
        t ← argmax_t CSLS(W x_s, y_t)                 // best target for s
        if argmax_s' CSLS(W x_s', y_t) = s:           // s is also best source for t
            D ← D ∪ {(s, t)}                          // mutual nearest neighbours
    keep only high-confidence pairs in D
    X_D, Y_D ← embeddings of the pairs in D (as columns)
    U Σ Vᵀ ← SVD(Y_D X_Dᵀ)
    W ← U Vᵀ                                          // exact orthogonal solution
return W
```

Why "semi-supervised": after step 1 there is no human signal at all, but steps 2 to 4 are exactly the supervised method of section 12 run on a dictionary the system made itself. With a small human seed dictionary instead of adversarial training as step 1, the same loop is the semi-supervised method of section 11.1. "Mutual" nearest neighbours matter: a pair is only trusted if each word picks the other, which filters out hub matches.

### 15.1 Results (slide 61)

Word translation accuracy (P@1, %), fastText embeddings, Conneau et al. (2018). Top half with cross-lingual supervision (repeated from slide 54), bottom half without:

| Method | en-es | es-en | en-fr | fr-en | en-de | de-en | en-ru | ru-en | en-zh | zh-en | en-eo | eo-en |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *With supervision* | | | | | | | | | | | | |
| Procrustes - NN | 77.4 | 77.3 | 74.9 | 76.1 | 68.4 | 67.7 | 47.0 | 58.2 | 40.6 | 30.2 | 22.1 | 20.4 |
| Procrustes - ISF | 81.1 | 82.6 | 81.1 | 81.3 | 71.1 | 71.5 | 49.5 | 63.8 | 35.7 | **37.5** | 29.0 | 27.9 |
| Procrustes - CSLS | 81.4 | 82.9 | 81.1 | **82.4** | 73.5 | **72.4** | **51.7** | **63.7** | **42.7** | 36.7 | **29.3** | 25.3 |
| *Without supervision* | | | | | | | | | | | | |
| Adv - NN | 69.8 | 71.3 | 70.4 | 61.9 | 63.1 | 59.6 | 29.1 | 41.5 | 18.5 | 22.3 | 13.5 | 12.1 |
| Adv - CSLS | 75.7 | 79.7 | 77.8 | 71.2 | 70.1 | 66.4 | 37.2 | 48.1 | 23.4 | 28.3 | 18.6 | 16.6 |
| Adv - Refine - NN | 79.1 | 78.1 | 78.1 | 78.2 | 71.3 | 69.6 | 37.3 | 54.3 | 30.9 | 21.9 | 20.7 | 20.6 |
| Adv - Refine - CSLS | **81.7** | **83.3** | **82.3** | 82.1 | **74.0** | 72.2 | 44.0 | 59.1 | 32.5 | 31.4 | 28.2 | **25.6** |

(Bold as printed. As noted for slide 54, the bolding does not always mark the column maximum: ru-en and eo-en have higher ISF values, 63.8 and 27.9.)

What to read off:

- **Each component helps.** For en-es: adversarial alone 69.8, + CSLS 75.7, + refinement 79.1, both 81.7.
- **Fully unsupervised matches or beats supervised** for the close, well-resourced pairs: en-es 81.7 against 81.4, es-en 83.3 against 82.9, en-fr 82.3 against 81.1, en-de 74.0 against 73.5.
- **It falls well short for distant pairs**: en-ru 44.0 against 51.7, en-zh 32.5 against 42.7, zh-en 31.4 against 36.7. The further the language from English, the more the shared space hypothesis is strained and the more a real dictionary helps.

That last point is where the critique starts.

## 16. Critical evaluation of unsupervised alignment

**Søgaard et al. (2018)** criticised some of the assumptions and results of Conneau et al. (2018) from several angles: the choice of languages, the role of morphology, data size, domain, and the embedding algorithm.

### 16.1 The choice of languages

**Slide 62** reproduces Table 1 of Søgaard et al.: the languages used by Conneau et al. (upper half) and the ones Søgaard et al. added (lower half):

| Language | Marking | Type | # Cases |
|---|---|---|---|
| English (EN) | dependent | isolating | None |
| French (FR) | mixed | fusional | None |
| German (DE) | dependent | fusional | 4 |
| Chinese (ZH) | dependent | isolating | None |
| Russian (RU) | dependent | fusional | 6–7 |
| Spanish (ES) | dependent | fusional | None |
| *added by Søgaard et al.:* | | | |
| Estonian (ET) | mixed | agglutinative | 10+ |
| Finnish (FI) | mixed | agglutinative | 10+ |
| Greek (EL) | double | fusional | 3 |
| Hungarian (HU) | dependent | agglutinative | 10+ |
| Polish (PL) | dependent | fusional | 6–7 |
| Turkish (TR) | dependent | agglutinative | 6–7 |

The point of the table: Conneau et al.'s languages are mostly dependent-marking, have few or no cases, and none is agglutinative. Søgaard et al. deliberately added languages with rich case systems, agglutinative morphology, and mixed or double marking. (Isolating, fusional and agglutinative are the morphological types from the morphology lecture, MNLP-L03: roughly, one morpheme per word; several meanings fused into one affix; many affixes chained one per meaning.)

### 16.2 Dependent marking

> [!definition] Head marking and dependent marking
> When two words are in a grammatical relationship, **which one carries the morphological marker** signalling that relationship?
> - **Dependent-marking:** the marker appears on the **dependent**
> - **Head-marking:** the marker appears on the **head**
> - **Double-marking:** **both**
> - **Zero-marking:** **neither** (the relationship is signalled by word order alone)

Example relationships:

- **Possessive phrase:** head = the **possessed** noun, dependent = the **possessor**
- **Clause/sentence:** head = the **verb**, dependents = **subject, object**

German examples (slide 63):

| German | Gloss | Head | Dependent(s) | Where the marking is |
|---|---|---|---|---|
| das Haus des Mannes | the house the.GEN man.GEN | house | man | genitive on the possessor *des Mannes* (dependent) |
| der Mann sieht den Hund | the.NOM man sees the.ACC dog | sees | man, dog | nominative and accusative case on the subject and object (dependents) |

In the second example the **verb (head) also agrees with the subject** (*sieht*, third person singular), so there is **some head-marking as well**, but dependent marking is predominant. English marks the same relations mostly by word order (*the man sees the dog* against *the dog sees the man*) and a few prepositions (*of the man*).

### 16.3 Adversarial against identical-word supervision

**Slide 64**, P@1 results from Søgaard et al.:

| Pair | Unsupervised (Adversarial) | Supervised (Identical) | Similarity (Eigenvectors) |
|---|---|---|---|
| EN-ES | 81.89 | **82.62** | 2.07 |
| EN-ET | 00.00 | **31.45** | 6.61 |
| EN-FI | 00.09 | **28.01** | 7.33 |
| EN-EL | 00.07 | **42.96** | 5.01 |
| EN-HU | 45.06 | **46.56** | 3.27 |
| EN-PL | 46.83 | **52.63** | 2.56 |
| EN-TR | 32.71 | **39.22** | 3.14 |
| ET-FI | **29.62** | 24.35 | 3.98 |

- **Unsupervised (Adversarial)** is the MUSE pipeline of sections 14 and 15.
- **Supervised (Identical)** uses as seed dictionary the **words spelled identically in both languages** (the third source of seed dictionaries on slide 40; the slide does not spell this out, the column name and Søgaard et al. do). It needs no human dictionary either, so it is a fair "free supervision" baseline.
- **Similarity (Eigenvectors)** is the $\Delta$ of section 16.5: **lower means more similar** structure.

> [!note] Observations (slide 64)
> - **Languages with mixed or double dependent marking perform poorly**: EN-ET 0.00, EN-FI 0.09, EN-EL 0.07, a complete failure.
> - **Identity outperforms adversarial training** on every pair involving English, from a small margin (EN-ES, 82.62 against 81.89) to the difference between working and not working (EN-ET, 31.45 against 0.00).
> - **Eigenvector similarity is poor for most languages**: compared with EN-ES (2.07), all other pairs have larger $\Delta$, and the three failing pairs have the largest (EN-ET 6.61, EN-FI 7.33, EN-EL 5.01).

So $\Delta$ tracks adversarial success: the two pairs with the lowest $\Delta$ (EN-ES 2.07, EN-PL 2.56) have the best adversarial scores, and the three with the highest fail completely. **ET-FI**, two structurally similar languages (both mixed-marking, agglutinative), works adversarially (29.62) even though neither works with English, and is the one case where adversarial beats identical.

> [!warning] Marking type is not the whole story
> French is listed as mixed-marking (slide 62), yet English-French unsupervised alignment works well (82.3 on slide 61). The languages that fail are mixed or double marking *and* have case-rich morphology (ET and FI: 10+ cases, agglutinative; EL: 3 cases, fusional). The slides' explanation (16.4) is really about how many distinct surface forms a lemma has, for which marking type is one contributing factor. Hungarian, dependent-marking but agglutinative with 10+ cases, sits in between (45.06).

### 16.4 Why mixed and double marking hurts

Søgaard et al. (2018) show that **MUSE fails dramatically for mixed- and double-marking languages** (Estonian, Finnish, Greek):

- in mixed- or double-marking languages **grammatical information is distributed across more word forms**
- a single **lemma is realised in many more distinct surface forms** (the slide drops the word "forms"), **each with its own embedding, and each in turn of lower frequency**
- this makes the **mapping between a language where marking is expressed by word order and one where it is expressed by morphology more complex (non-isomorphic)**

Concretely: English `house` corresponds to one Finnish lemma `talo`, which appears as `talo`, `talon`, `taloa`, `talossa`, `talosta`, `taloon`, `talolla`, ... (examples not on the slides). There is no rotation that maps one English point onto a dozen Finnish points.

### 16.5 Measuring degrees of isomorphism: eigenvector similarity

**Isomorphism is a strict true/false criterion.** To measure **degrees** of isomorphism, Søgaard et al. use **eigenvector similarity**:

> [!definition] Eigenvector similarity
> 1. Compute the **nearest neighbours of each node (word)** and represent the result as an **adjacency matrix $A$** (a nearest-neighbour graph over the embedding space).
> 2. Compute the **Laplacian matrix $L$** from $A$ and the **diagonal degree matrix $D$** (each diagonal entry is the number of neighbours of that node).
> 3. Obtain the **eigenvalues of $L$** and keep the **largest $n\%$**.
> 4. Given two languages, **sum the squared differences between their Laplacian eigenvalues**:
> $$\Delta = \sum_{i} \left( \lambda^{(1)}_i - \lambda^{(2)}_i \right)^2$$
>    where $\lambda^{(1)}_i$, $\lambda^{(2)}_i$ are the $i$-th largest kept eigenvalues of the two languages' graphs.

```pseudo
Algorithm: Eigenvector similarity Δ, assembled from slide 66
────────────────────────────────────────────────────────────
Input:  embeddings E1 (language 1), E2 (language 2), same number of words N,
        neighbourhood size, fraction n% of eigenvalues to keep
for each language ℓ ∈ {1, 2}:
    A_ℓ ← N × N zero matrix
    for each word u:
        for each nearest neighbour v of u in E_ℓ:
            A_ℓ[u,v] ← 1 ; A_ℓ[v,u] ← 1        // undirected graph
    D_ℓ ← diag(row sums of A_ℓ)                 // number of neighbours
    L_ℓ ← D_ℓ − A_ℓ                              // graph Laplacian (see warning)
    λ_ℓ ← eigenvalues of L_ℓ, sorted descending, keep the top n%
k ← number of kept eigenvalues
return Δ = Σ_{i=1..k} (λ_1[i] − λ_2[i])²
```

> [!warning] Sign of the Laplacian on slide 66
> The slide says to "compute Laplacian matrix $L$ by subtracting diagonal matrix $D$ ... from $A$", i.e. $L = A - D$. The standard graph Laplacian is $L = D - A$, whose eigenvalues are all $\geq 0$. With $A - D$ every eigenvalue flips sign and becomes $\leq 0$, so "keep the largest" would select the values nearest zero, the opposite of what is intended, and "bigger values = dense connectivity" would no longer hold. Read it as $L = D - A$, as in the pseudocode above.

Interpretation:

- **Laplacian eigenvalues indicate how connected the nodes are**: bigger values mean denser connectivity.
- **The bigger $\Delta$, the more structurally different** the two graphs, i.e. the further from isomorphism.
- **It does not require knowing the alignment** between the languages: eigenvalues are invariant to relabelling the nodes, so no dictionary is needed. If the denser portions of the two graphs are similarly dense and the less dense portions similarly less dense, $\Delta$ will be small.

> [!example] $\Delta$ for two tiny graphs (worked here, not on the slides)
> Two graphs on 4 nodes, all eigenvalues kept.
>
> - **Star** (one hub connected to three leaves): $L = D - A$ has eigenvalues $4, 1, 1, 0$.
> - **Path** (a chain of 4 nodes): eigenvalues $2 + \sqrt{2} \approx 3.414$, $2$, $2 - \sqrt{2} \approx 0.586$, $0$.
>
> $$\Delta = (4 - 3.414)^2 + (1 - 2)^2 + (1 - 0.586)^2 + (0 - 0)^2 = 0.343 + 1 + 0.171 = 1.515$$
>
> Two copies of the star (with nodes numbered in any order) would give $\Delta = 0$. The star's large top eigenvalue (4) is its hub: one densely connected node. This is the shape of the Finnish situation below, tight clusters of inflected forms, against the flatter English graph.

**What a high $\Delta$ means (slide 67):**

- **how words cluster together in language A is structurally unlike the way words cluster in language B**
- **no rotation can fix this** (a rotation preserves the neighbour graph exactly, so it cannot change the eigenvalues)
- the **assumption of an isomorphism between A and B does not hold**

**The Finnish case.** A plausible reason relates to morphology:

- a single lemma spreads across **dozens of inflected surface forms**, each its own node (embedding)
- due to their **meaning overlap they cluster tightly with each other**
- **English, with far fewer word forms per lemma, produces a flatter, more evenly connected graph**
- **different connectivity patterns mean a high $\Delta$**

> [!warning] The inference only goes one way
> **A low $\Delta$ does not mean that the spaces are near-isomorphic.** If graphs are (near-)isomorphic they will have a low $\Delta$, **not the other way around**. Not on the slides: there exist pairs of non-isomorphic graphs with identical Laplacian spectra (cospectral graphs), which is a concrete reason the converse fails. So $\Delta$ can rule isomorphism out, never in.

### 16.6 It is not just data size

> [!note] Data size and structural similarity (slide 68)
> These problems are **not only due to smaller data sizes**:
> - Finnish Wikipedia is **12M words**, against **363M** for Spanish
> - retraining on the **Finnish WaC corpus (1.7 billion words)** left **P@1 for English-Finnish at 0.0**
>
> **Structural language similarity matters:**
> - **Estonian and Finnish are both mixed-marking, agglutinative languages**
> - **Estonian-Finnish alignment reaches P@1 of 29.62**
> - **pairing structurally dissimilar languages is the problem**, e.g. English-Finnish

Results also **differ per part of speech** (P@1):

| POS | en-es | en-hu | en-fi |
|---|---|---|---|
| Noun | 80.94 | 26.87 | 00.00 |
| Verb | 66.05 | 25.44 | 00.00 |
| Adjective | 85.53 | 53.28 | 00.00 |
| Adverb | 80.00 | 51.57 | 00.00 |
| Other | 73.00 | 53.40 | 00.00 |

For English-Hungarian, **nouns and verbs are the worst** (26.87, 25.44), and adjectives, adverbs and other words are about twice as good (51.57 to 53.40). Nouns (case) and verbs (person, number, tense) are the parts of speech that inflect most in Hungarian, so this is the per-lemma-form-explosion argument showing up per POS. Adverbs and function words barely inflect. English-Finnish is 0.00 across the board. For English-Spanish, verbs (66.05) are the weakest category: Spanish verbs are the most heavily inflected Spanish words.

### 16.7 Domain sensitivity

**Slide 69**: monolingual corpora from **EuroParl (EP, political discussions)**, **Wikipedia (Wiki, general knowledge)** and **EMEA (medical)**, **1.1M sentences each**. The x-axis of each plot is labelled "Training Corpus (English)"; the bar colour (dark blue EP, red Wiki, green EMEA) is then the corpus used for the Spanish side, which the slide implies but does not state. y-axis: BLI P@1 (bilingual lexicon induction precision at 1), 0 to about 65.

**(d) en-es, identical words** (supervised with identically spelled words):

| English corpus ↓ / Spanish corpus → | EP | Wiki | EMEA |
|---|---|---|---|
| EN:EP | **64.09** | 25.17 | 9.42 |
| EN:Wiki | 25.48 | **46.52** | 9.63 |
| EN:EMEA | 4.84 | 6.63 | **49.24** |

**(g) en-es, fully unsupervised BLI** (adversarial):

| English corpus ↓ / Spanish corpus → | EP | Wiki | EMEA |
|---|---|---|---|
| EN:EP | **61.01** | 0.11 | 0.0 |
| EN:Wiki | 0.13 | **41.38** | 0.08 |
| EN:EMEA | 0.0 | 0.0 | **49.43** |

- **Same domain on both sides** (the diagonal): both methods work, 41 to 64.
- **Different domains**: identical-word supervision degrades but survives (25.17, 25.48 between EP and Wiki; 4.84 to 9.63 involving medical text). **Adversarial induction collapses to essentially zero** (0.0 to 0.13) in every cross-domain cell.

> [!warning] Adversarial induction across domains does not work at all
> This is a problem in practice, because the reason to use an unsupervised method is a low-resource language, and for such a language you rarely get to choose a monolingual corpus from the same domain as your English one. English and Spanish are as close as pairs get in this lecture, and a domain mismatch alone is enough to break it.

### 16.8 Embedding algorithm sensitivity

**What if different embedding models, or variations within a model, are used for the two languages?** Both languages use fastText. English is fixed at skipgram, window 2, character n-grams 3–6. Spanish varies (P@1):

| Spanish setting (relative to English) | Spanish (skipgram) | Spanish (cbow) |
|---|---|---|
| == (same hyperparameters) | 81.89 | 00.00 |
| ≠ win=10 | 81.28 | 00.07 |
| ≠ chn=2-7 | 80.74 | 00.00 |
| ≠ win=10, chn=2-7 | 80.15 | 00.13 |

where `win` is the context window size and `chn` the character n-gram range.

- **Different variations within a model do not matter much**: changing window size and n-gram range for Spanish skipgram costs at most 1.74 points (81.89 to 80.15).
- **Comparing embeddings from two different base models (skipgram vs CBOW) does not work at all**: 0.00 to 0.13 even with otherwise identical hyperparameters.

The two algorithms produce spaces with different geometry even on the same kind of data, so the "approximate isomorphism" that unsupervised alignment depends on is partly an artefact of using the same algorithm on both sides.

> [!tip] What the critique adds up to
> Unsupervised alignment works when the two languages are typologically similar, the corpora come from the same domain, and the embeddings come from the same algorithm. Those are the conditions under which Conneau et al. evaluated it, and they are not the conditions under which you would need it (a low-resource language, whatever text is available). A few identically spelled words, which are free, beat adversarial training in nearly every setting tested.

## Key Takeaways

> [!tip] Exam Focus
> Things to be able to state and derive cold.
>
> 1. **Distributional hypothesis** (Firth 1957) and the **count-based recipe**: $V_t$, $V_c$, window $n$, $c_w[i]$ = count of context word $i$ near $w$. Cosine similarity, and computing it by hand. Sparse, high-dimensional, interpretable; no way to separate signal from noise (idf as a partial fix, $\log N/n_w$).
> 2. **CBOW** $h = \frac{1}{2n} W w_C$, $\hat{y} = \operatorname{softmax}(W'h)$, against **Skip-gram** $h = W w_I$, $2n$ outputs with shared $W'$. Matrix shapes ($W$ is $\lvert h \rvert \times \lvert V \rvert$, $W'$ is $\lvert V \rvert \times \lvert h \rvert$), where the embeddings live (columns of $W$, rows of $W'$, or $W^T + W'$).
> 3. **Negative sampling**: why (the full softmax is $O(\lvert V \rvert)$ per example), the objective $\sum_D \log \sigma(v_c \cdot v_w) + \sum_{D'} \log \sigma(-v_c \cdot v_w)$ and how it follows from $1 - \sigma(x) = \sigma(-x)$, 5 to 20 negatives from $P(w)^{0.75}$.
> 4. **GloVe**: $J = \sum f(X_{ij})(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$, the shape and purpose of $f$ ($x_\text{max} = 100$, $\alpha = 0.75$), final vector $w_i + \tilde{w}_i$, and the four-row comparison with Word2Vec.
> 5. **Evaluation**: intrinsic against extrinsic; word similarity with Spearman (WS-353 relatedness, SimLex-999 similarity); analogies by vector arithmetic; why 2D t-SNE plots are not evidence.
> 6. **fastText**: $v_w = \sum_{g \in G_w} z_g$ over character n-grams of `<w>` (3 to 6) plus the word itself; vectors for OOV words; gains on syntactic analogies and morphologically rich languages, none on semantic analogies.
> 7. **Cross-lingual mapping**: shared space hypothesis, $\min_W \sum \lVert W x_i - y_i \rVert^2$, closed form $Y X^\top (X X^\top)^{-1}$; why unconstrained $W$ overfits and mismatches cosine retrieval; orthogonal constraint $W^\top W = I$ with unit vectors, solved by SVD ($W = UV^\top$).
> 8. **Hubness and CSLS**: $2\cos(x,y) - r_T(x) - r_S(y)$, and which term does the work.
> 9. **MUSE**: adversarial training with a linear generator (and why it must be linear), label smoothing, 50k frequent words, re-orthogonalisation, unsupervised CSLS-based model selection, then Procrustes refinement on mutual nearest neighbours.
> 10. **The critique**: fails for mixed/double-marking, case-rich languages (EN-FI 0.0 even with 1.7B words), across domains, and across embedding algorithms; identical-word supervision beats it; eigenvector similarity $\Delta$ and why a low $\Delta$ proves nothing.
>
> The most likely essay-style question is **"does unsupervised bilingual lexicon induction work?"**. The answer has three layers: the mechanism (shared space hypothesis, linear orthogonal map, adversarial bootstrapping, Procrustes refinement), the headline result (matches supervised on en-es/en-fr/en-de), and the conditions under which it collapses (typology, domain, algorithm), with numbers.

> [!warning] Distinctions to get right
> - **Euclidean training versus cosine retrieval** coincide only for unit-length vectors under an orthogonal map: $\lVert Wx - y \rVert^2 = 2 - 2\cos(Wx, y)$.
> - **"Procrustes"** in the results tables means the **orthogonal** solution via SVD; the unconstrained least-squares map is Mikolov et al.'s translation matrix.
> - **Semi-supervised** in this lecture means two things that share one loop: a small seed dictionary, or a dictionary induced by the adversarial step, followed by iterated Procrustes on mutual nearest neighbours.
> - **Static** means one vector per word type: `bank` has the same vector in *river bank* and *bank account*. That limitation is what [[MNLP-L06 - Contextual Embeddings]] addresses with context models (an [[LSTM]] or, in most current models, a [[Transformers|Transformer]] encoder such as BERT).

## Exam questions

> [!exam]- Does unsupervised bilingual lexicon induction work? Answer with the mechanism, the headline result and the conditions under which it fails, with numbers.
> **Key points:** shared space hypothesis and the MUSE pipeline (adversarial, CSLS selection, Procrustes refinement); matches supervised on close pairs; falls short on distant pairs; collapses for case-rich mixed-marking languages, across domains and across algorithms, not because of data size; identical-word baseline beats it.
>
> Must hit:
> 1. **Mechanism.** The shared space hypothesis says separately trained embedding spaces are approximately isomorphic, so one linear (orthogonal) map should align them. MUSE (Conneau et al., 2018) bootstraps a rough $W$ adversarially (a linear generator against an MLP discriminator, with re-orthogonalisation), picks the checkpoint with the unsupervised CSLS criterion, then refines with Procrustes ($W = UV^\top$) on a synthetic dictionary of mutual CSLS nearest neighbours, repeated.
> 2. **Headline result.** Fully unsupervised matches or beats supervised Procrustes-CSLS on close, well-resourced pairs (P@1): en-es 81.7 against 81.4, es-en 83.3 against 82.9, en-fr 82.3 against 81.1, en-de 74.0 against 73.5.
> 3. **Where it falls short.** Distant pairs: en-ru 44.0 against 51.7, en-zh 32.5 against 42.7.
> 4. **Where it collapses (Søgaard et al., 2018).** Mixed- or double-marking, case-rich languages: EN-ET 0.00, EN-FI 0.09, EN-EL 0.07. EN-FI stays at 0.0 even when retrained on 1.7 billion Finnish words, so data size is not the explanation. Mismatched domains: 0.0 to 0.13 in every cross-domain en-es cell. Mismatched algorithms: Spanish CBOW against English skipgram gives 0.00 to 0.13.
> 5. **A free baseline beats it.** A seed dictionary of identically spelled words beats adversarial training on every pair involving English (EN-ES 82.62 against 81.89, EN-ET 31.45 against 0.00).
> 6. **Verdict.** It works when the languages are typologically similar, the corpora share a domain and both sides use the same embedding algorithm. Those are the conditions it was evaluated under, and they are the opposite of the low-resource setting that motivates it.
>
> Losing marks: giving only the Conneau et al. headline numbers, or blaming the failures on data size alone.

> [!exam]- Derive the negative sampling objective from a binary classification set-up, then derive its gradient with respect to the input vector $v_w$ and interpret it.
> **Key points:** real-or-noise binary classification with $k$ negatives; $p(D=1) = \sigma(v_c \cdot v_w)$ and $1-\sigma(x) = \sigma(-x)$; log objective over $D$ and $D'$; gradient pulls $v_w$ to $v_c$, pushes it from negatives; cost $k+1$ instead of $\lvert V \rvert$.
>
> 1. **Why.** A full softmax needs a score for all $\lvert V \rvert$ words per training example. Replace "which word is the context?" with "is this (word, context) pair real or noise?": 1 positive pair from the data, $k$ negative pairs from a noise distribution.
> 2. **Likelihood.** $\arg\max_\theta \prod_{(w,c) \in D} p(D=1 \mid c,w;\theta) \prod_{(w,c) \in D'} p(D=0 \mid c,w;\theta)$, with $D$ the observed pairs, $D'$ the noise pairs, and $p(D=1 \mid c,w) = \sigma(v_c \cdot v_w)$, where $v_w = Ww$ (column of $W$) and $v_c = W'^\top c$ (row of $W'$).
> 3. **Sigmoid identity.** $1 - \sigma(x) = \frac{e^{-x}}{1+e^{-x}} = \frac{1}{e^x + 1} = \sigma(-x)$, so the objective becomes $\prod_D \sigma(v_c \cdot v_w) \prod_{D'} \sigma(-v_c \cdot v_w)$, and after logs
> $$\sum_{D} \log \sigma(v_c \cdot v_w) + \sum_{D'} \log \sigma(-v_c \cdot v_w)$$
> 4. **Per-example loss.** $\mathcal{L} = -\log \sigma(v_c \cdot v_w) - \sum_{j=1}^{k} \log \sigma(-v_{n_j} \cdot v_w)$.
> 5. **Gradient.** With $\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$:
> $$\frac{\partial \mathcal{L}}{\partial v_w} = (\sigma(v_c \cdot v_w) - 1)\, v_c + \sum_{j} \sigma(v_{n_j} \cdot v_w)\, v_{n_j}$$
> 6. **Interpretation.** The positive coefficient is negative, so a descent step moves $v_w$ towards $v_c$; each negative coefficient is positive, so $v_w$ moves away from $v_{n_j}$. Both shrink to zero once the classifier gets the pair right, so the biggest corrections come from negatives the model mistakes for real contexts. Cost per example drops from $\lvert V \rvert$ dot products to $k+1$.

> [!exam]- Compare Word2Vec (CBOW and Skip-gram) with GloVe: what each optimises, what information each uses, how each handles frequency imbalance, and what the final embedding is.
> **Key points:** CBOW and Skip-gram predict within local windows; GloVe is weighted least squares on global log counts; subsampling against $f(X_{ij})$; final embedding $W$ or $W^\top + W'$ against $w_i + \tilde{w}_i$.
>
> - **CBOW:** predict the centre word from the average of $2n$ context embeddings, $h = \frac{1}{2n} W w_C$, $\hat{y} = \operatorname{softmax}(W'h)$, cross entropy, SGD.
> - **Skip-gram:** predict each of the $2n$ context words from the centre word, $h = W w_I$, context words assumed independent given $w_I$; in practice trained with negative sampling.
> - **GloVe (Pennington et al., 2014):** weighted least-squares regression on log co-occurrence counts, $J = \sum f(X_{ij})(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$.
> - **Information used:** Word2Vec sees individual local context windows, treating each as an independent event, so it keeps "rediscovering" the same association. GloVe counts once over the whole corpus (global statistics) and then fits gradient updates to those counts.
> - **Frequency imbalance:** Word2Vec subsamples frequent words (and draws negatives from $P(w)^{0.75}$); GloVe uses the explicit weighting $f(X_{ij})$ with $x_\text{max} = 100$, $\alpha = 0.75$.
> - **Interpretability:** Word2Vec is indirect (it learns to predict contexts); GloVe directly fits log counts.
> - **Final embedding:** Word2Vec typically $W$, or $W^\top + W'$; GloVe $w_i + \tilde{w}_i$. Both sum the two vector sets.
> - Context: prediction-based methods tend to outperform count-based ones; count-based ones use global information directly. GloVe is the hybrid.

> [!exam]- Why does constraining a cross-lingual mapping to be orthogonal help? Give the objective, the two problems it fixes with a derivation for each, its closed-form solution, and the evidence.
> **Key points:** $W^\top W = I$ with unit vectors; rigid map stops overfitting the seed pairs; $\lVert Wx - y \rVert^2 = 2 - 2\cos$ aligns training with retrieval; $W = UV^\top$ by SVD; unconstrained degrades with dimension, orthogonal better everywhere.
>
> 1. **Objective (Xing et al., 2015).** $W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2$ subject to $W^\top W = I$, with all embeddings normalised to unit length. An orthogonal $W$ only rotates and reflects: no scaling or shearing.
> 2. **Problem 1, overfitting.** An unconstrained $W$ can stretch, shear and rotate, so it can warp the space to fit the seed pairs and fail to generalise to the words outside the dictionary (the majority). Orthogonal $W$ preserves geometry: $\lVert Wx \rVert^2 = x^\top W^\top W x = x^\top x$, so the source space moves rigidly, which is exactly what the shared space hypothesis says should suffice.
> 3. **Problem 2, objective mismatch.** Training minimises Euclidean distance, retrieval uses cosine. For unit vectors and orthogonal $W$: $\lVert Wx - y \rVert^2 = \lVert Wx \rVert^2 + \lVert y \rVert^2 - 2(Wx)\cdot y = 2 - 2\cos(Wx, y)$, so minimising distance is maximising cosine.
> 4. **Solution.** $U\Sigma V^\top = \operatorname{SVD}(YX^\top)$, $W^* = UV^\top$: one SVD of a $d \times d$ matrix, exact, no learning rate.
> 5. **Evidence (en-es, P@1).** Unconstrained (Mikolov et al., 2013) falls from 30.43% at 300 dimensions to 20.69% at 700, because $W$ has $d^2$ free parameters for the same dictionary. Orthogonal is better everywhere and rises slightly, 38.99% to 41.04%.
>
> Losing marks: claiming Euclidean training and cosine retrieval always agree (they coincide only for unit vectors under an orthogonal map), or calling the unconstrained least-squares map "Procrustes" (in the results tables Procrustes means the orthogonal SVD solution).

> [!exam]- Explain the hubness problem in cross-lingual word retrieval and how CSLS corrects it, including which term of the formula does the work.
> **Key points:** hubs are nearest neighbours of many points, worse in high dimensions; $\text{CSLS} = 2\cos - r_T - r_S$; $r_S(y)$ penalises hubs; $r_T(x)$ constant per source word; better retrieval, only $K$ to set.
>
> - **Hubness:** in high-dimensional spaces a few vectors become the nearest neighbour of many points regardless of real similarity. It is a general property of high-dimensional geometry and worsens with dimension. A hub sits near the centre of the cloud and is moderately close to everything.
> - **Effect on translation:** one target hub becomes the nearest-neighbour "translation" of many unrelated source words: nn(cat) = nn(car) = nn(house) = thing. Nearest neighbour is asymmetric: `thing` can be the neighbour of `cat` without `cat` being the neighbour of `thing`.
> - **CSLS (Conneau et al., 2018):** $\text{CSLS}(x, y) = 2\cos(x,y) - r_T(x) - r_S(y)$, with $x = Wx_s$, $r_T(x)$ the mean cosine of $x$ to its $K$ nearest target vectors, $r_S(y)$ the mean cosine of $y$ to its $K$ nearest mapped source vectors.
> - **Which term works:** $r_S(y)$ penalises a candidate that is close to lots of things (a hub has high $r_S$). $r_T(x)$ is constant across candidates for a fixed $x$, so it cannot change which $y$ wins; it matters when scores are compared across source words, e.g. ranking pairs to build a dictionary.
> - **Effect:** significantly better retrieval with no parameter tuning beyond $K$ (Conneau et al. use $K = 10$). Over plain NN on the same Procrustes map: en-es 77.4 to 81.4, fr-en 76.1 to 82.4.

> [!exam]- Describe the full MUSE pipeline of Conneau et al. (2018) for aligning two embedding spaces without any bilingual signal, and justify each design choice.
> **Key points:** linear generator $W$ against MLP discriminator; linear so point correspondence follows the distribution match; label smoothing, 50k frequent words, re-orthogonalisation; CSLS-based unsupervised model selection; Procrustes refinement on mutual nearest neighbours.
>
> 1. **Adversarial step.** Generator = the mapping $W$ (a single $d \times d$ linear matrix); discriminator = an MLP (2 hidden layers of 2048, ReLU, input dropout) that tells mapped source vectors $Wx_i$ from real target vectors $y_j$. They train in alternation: discriminator step with $W$ frozen, generator step with labels flipped and the discriminator frozen.
> 2. **Linear generator, deliberately.** A deep generator could match the target *distribution* while scrambling which source word lands where. Only a constrained, near-rigid map forces point-level correspondence to come with the distribution match.
> 3. **Label smoothing $s = 0.2$.** Targets $\{s, 1-s\}$ stop the discriminator becoming overconfident, which would give the generator vanishing gradients.
> 4. **Only the 50k most frequent words** are fed to the discriminator: rare words have poorly estimated embeddings, so frequent words give a cleaner signal.
> 5. **Re-orthogonalisation** after each step, $W \leftarrow (1+\beta)W - \beta(WW^\top)W$ with $\beta = 0.01$, keeps $W$ near orthogonal so it cannot drift into a warping general linear map.
> 6. **Model selection without a dictionary.** The adversarial loss is unreliable. Instead: for the 10k most frequent source words, find each one's CSLS nearest target, average the cosines, keep the checkpoint with the highest value.
> 7. **Refinement.** Build a synthetic dictionary of high-confidence mutual CSLS nearest neighbours among frequent words, re-solve $W = UV^\top$ by SVD (Procrustes), repeat. Translate with CSLS.
> 8. **Contribution of each part (en-es P@1):** adversarial alone 69.8, + CSLS 75.7, + refinement 79.1, both 81.7.

> [!exam]- What is fastText, how does it differ from Skip-gram with negative sampling, and what does the evidence of Bojanowski et al. (2017) say about where it helps and where it does not?
> **Key points:** sum of character n-grams of `<w>` (3 to 6) plus the word; same SGNS loss, only the centre word decomposed; vectors for OOV words; gains on morphologically rich languages, rare words and syntactic analogies; no gain on semantic analogies.
>
> - **Problem it addresses:** Word2Vec and GloVe give one opaque vector per word type, so `run`, `runs`, `running`, `runner` share no structure; morphologically rich languages (Finnish, Turkish, Russian) split each lemma into many rare forms; OOV words get no vector.
> - **Model:** the centre word vector is the sum of its character n-gram vectors, $v_w = \sum_{g \in G_w} z_g$, where $G_w$ holds all n-grams of `<w>` for $n \in [3, 6]$ plus the whole word `<w>`. `<` and `>` mark word boundaries so prefixes and suffixes are distinct n-grams.
> - **Same as SGNS:** the loss $-\log\sigma(v_w^\top u_c) - \sum_k \log\sigma(-v_w^\top u_{n_k})$, with context vectors $u_c$ still per word.
> - **Different:** every n-gram of $w$ gets the same gradient (since $\partial v_w / \partial z_g = I$); the output is the n-gram table $\{z_g\}$, and any word, seen or not, gets a vector by summing its n-grams.
> - **Word similarity:** sisg (fastText) best or tied-best on 9 of 10 datasets; biggest gains in morphologically rich languages (Russian HJ 59 to 66 over sg); smaller gains on English Rare Words (43 to 47). Building OOV vectors from n-grams (sisg against sisg-) adds up to 6 points. Only loss: English WS353 (71 against cbow's 73), frequent words where whole-word vectors are already good.
> - **Analogies:** gains are syntactic (Czech 52.8 to 77.8, German +11.9, Italian +11.2, English +4.8 over sg). Semantic analogies do not improve and sometimes drop (German 66.5 to 62.3), because capital-country relations have nothing to do with character overlap.

> [!exam]- How can the degree of isomorphism between two embedding spaces be measured without a dictionary, what did it show for English and Finnish, and what can it not show?
> **Key points:** Laplacian eigenvalues of nearest-neighbour graphs; $\Delta$ = sum of squared eigenvalue differences; no dictionary needed; high for EN-FI, EN-ET, EN-EL and tracks adversarial failure; low $\Delta$ does not prove isomorphism.
>
> 1. **Why a measure is needed:** isomorphism is a strict true/false criterion; Søgaard et al. (2018) want a degree.
> 2. **Eigenvector similarity:** build a nearest-neighbour graph per language (adjacency matrix $A$), take the degree matrix $D$, form the Laplacian $L = D - A$, keep the largest $n\%$ of its eigenvalues, and compute $\Delta = \sum_i (\lambda_i^{(1)} - \lambda_i^{(2)})^2$.
> 3. **Reading it:** larger Laplacian eigenvalues mean denser connectivity; larger $\Delta$ means more structurally different graphs. Eigenvalues are invariant to relabelling the nodes, so no alignment is needed.
> 4. **Results:** EN-ES 2.07 is the lowest; the three pairs where adversarial alignment fails completely have the largest values (EN-ET 6.61, EN-FI 7.33, EN-EL 5.01). $\Delta$ tracks adversarial success.
> 5. **Finnish explanation:** a lemma spreads across dozens of inflected forms that cluster tightly by meaning overlap, while English gives a flatter, more evenly connected graph. Different connectivity means high $\Delta$, and no rotation can fix it, since a rotation preserves the neighbour graph and so the eigenvalues.
> 6. **Limit:** near-isomorphic graphs have low $\Delta$, but a low $\Delta$ does not imply near-isomorphism (non-isomorphic cospectral graphs exist). $\Delta$ can rule isomorphism out, never in.
>
> Losing marks: concluding that a low $\Delta$ proves the spaces are isomorphic.

> [!exam]- How should word embeddings be evaluated? Cover the evaluation axes, the word similarity and analogy tasks, and the limits of 2D visualisations.
> **Key points:** intrinsic against extrinsic, qualitative against quantitative; word similarity by Spearman, WS-353 relatedness against SimLex-999 similarity; analogies via $v_b - v_a + v_c$ scored by accuracy; 2D projections distort, t-SNE hyperparameters matter.
>
> - **Two independent axes:** intrinsic (how good the embeddings are by themselves) against extrinsic (usefulness in downstream tasks such as summarisation, MT, IR); qualitative (inspect selected examples, neighbours, plots) against quantitative (an overall task-dependent score). A 2D plot is intrinsic and qualitative; a similarity correlation is intrinsic and quantitative; BLEU after plugging embeddings into MT is extrinsic and quantitative.
> - **Word similarity:** rank word pairs by cosine, compare with human judgements by Spearman correlation. WS-353 annotates relatedness, SimLex-999 similarity; `coffee`/`cup` is related but not similar, so the two benchmarks can rank the same embeddings differently. Both mix kinds of similarity.
> - **Analogies:** *a : b :: c : X*, answer the word closest by cosine to $v_b - v_a + v_c$ (input words excluded), scored by accuracy; semantic and syntactic sets.
> - **Visualisation:** project to 2D with PCA (linear, top-variance directions) or t-SNE (non-linear, keeps local neighbours, gives up global distances). Useful for showing sense clusters or a consistent gender direction.
> - **Caveats:** non-linear projections distort distances; t-SNE hyperparameters change cluster sizes, distances and even apparent clusters (Wattenberg 2016). Plots complement quantitative and extrinsic evaluation and do not substitute for it.

> [!exam]- Derive the closed-form solution of the unconstrained linear mapping $\min_W \sum_i \lVert W x_i - y_i \rVert^2$, state when it exists, and explain why this mapping gets worse as the embedding dimension grows.
> **Key points:** loss $\lVert WX - Y \rVert_F^2$; zero gradient gives $W = YX^\top(XX^\top)^{-1}$; needs $XX^\top$ invertible, $n \geq d_1$; SGD as the scalable alternative; $d^2$ parameters overfit, so accuracy falls with dimension.
>
> 1. **Matrix form.** Stack the $n$ dictionary pairs as columns: $X$ is $d_1 \times n$, $Y$ is $d_2 \times n$. Loss $= \lVert WX - Y \rVert_F^2$.
> 2. **Gradient.** $2(WX - Y)X^\top$ (per pair: $2\sum_i (Wx_i - y_i)x_i^\top$).
> 3. **Set to zero.** $WXX^\top = YX^\top$, so $W^* = YX^\top(XX^\top)^{-1}$.
> 4. **Existence.** $XX^\top$ ($d_1 \times d_1$) must be invertible, which needs at least $d_1$ linearly independent source vectors, i.e. $n \geq d_1$ pairs.
> 5. **Closed form against SGD.** Closed form is exact; SGD (used by Mikolov et al., 2013) is approximate, scales better with large dimensions and dictionaries, and never forms or inverts $XX^\top$.
> 6. **Dimension.** $W$ has $d^2$ free parameters for the same seed dictionary, so more dimensions means more room to overfit: en-es P@1 drops from 30.43% (300 dimensions) to 25.76% (500) to 20.69% (700). The unconstrained map can stretch and shear to fit the anchors and generalises poorly.

## Flashcards

> [!card]- What is the distributional hypothesis behind both count-based and prediction-based word vectors?
> A word is characterised by **the contexts it occurs in** (Firth: "the company it keeps"), so similar contexts mean similar meaning.

> [!card]- When building count-based context vectors, is a word inside the window but outside the context vocabulary $V_c$ counted?
> **No.** A word counts only if it is both inside the window **and in $V_c$**.

> [!card]- Why is word similarity measured with cosine rather than the raw dot product?
> Cosine **ignores vector length**, so a frequent word's large counts do not dominate; only the proportions of contexts matter.

> [!card]- What are the three properties of count-based context vectors?
> **High-dimensional** (one dimension per context word), **discrete** (integer counts), **sparse** (mostly zeros).

> [!card]- What is the main weakness of count-based context vectors besides sparsity?
> **No explicit criterion to separate signal from noise**: chance co-occurrences, and words like `the` that discriminate nothing.

> [!card]- How does idf weighting, $\log(N/n_w)$, help count-based context vectors?
> A context word that occurs in nearly every context gets **idf close to 0**, so non-discriminative words are effectively removed.

> [!card]- What four properties distinguish prediction-based word embeddings from count-based context vectors?
> **Low-dimensional, dense, continuous**, and **learned by performing a prediction task**. Their dimensions are not interpretable.

> [!card]- What is the difference between the CBOW and Skip-gram tasks?
> CBOW predicts **the centre word from its context**. Skip-gram predicts **each context word from the centre word**.

> [!card]- Why is CBOW useless as a language model?
> It also uses the words **to the right** of the target; an n-gram language model uses left context only.

> [!card]- How is the CBOW hidden layer $h$ computed from the context words?
> As the **average** of the $2n$ context words' embeddings: $h = \frac{1}{2n} W w_C$, with no non-linearity.

> [!card]- Why is CBOW called a "bag of words" model?
> The context word vectors are **summed**, so **word order is lost**.

> [!card]- What independence assumption does Skip-gram make?
> The context words are **independent given the input (centre) word**.

> [!card]- Do Skip-gram's $2n$ output layers produce different distributions?
> **No.** They share $W'$ and $h$, so the distribution is identical; only the target each is scored against differs.

> [!card]- What task replaces full-vocabulary prediction in negative sampling?
> **Binary classification**: did this (word, context) pair come from the data or from a noise distribution?

> [!card]- How does negative sampling change the per-example cost of Word2Vec training?
> From **$\lvert V \rvert$ dot products to $k+1$**.

> [!card]- In negative sampling, what is the probability that a (word, context) pair is real?
> $\sigma(v_c \cdot v_w)$: the **sigmoid of the dot product** of the context (output) and word (input) vectors.

> [!card]- Why are Word2Vec negatives drawn from $P(w)^{0.75}$ rather than the raw unigram distribution?
> To **bias sampling towards rarer words**; otherwise nearly every negative would be `the`, `of` or `and`.

> [!card]- What weakness of prediction-based methods like Word2Vec does GloVe address?
> They treat **each context window as an independent event**, repeatedly rediscovering the same association.

> [!card]- Which tends to perform better: count-based or prediction-based word representations?
> **Prediction-based.** Count-based methods use global information directly but tend to perform worse.

> [!card]- What does GloVe combine?
> **Global co-occurrence statistics** (counted once) with **local prediction-based gradient updating**.

> [!card]- In GloVe's weighted least-squares objective, what target is $w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j$ fitted to?
> The **log co-occurrence count** $\log X_{ij}$.

> [!card]- What is the purpose of GloVe's weighting function $f(X_{ij})$?
> To **de-emphasise very rare (noisy) and very frequent (uninformative)** co-occurrences.

> [!card]- What are the typical parameters of GloVe's weighting function?
> **$x_\text{max} = 100$, $\alpha = 0.75$**.

> [!card]- What is the difference between Word2Vec and GloVe in what they train on?
> Word2Vec: **individual local context windows**. GloVe: the **global co-occurrence matrix**, built in one pass over the corpus.

> [!card]- How do Word2Vec and GloVe differ in handling frequency imbalance?
> Word2Vec **subsamples frequent words**. GloVe uses the **explicit weighting function** $f(X_{ij})$.

> [!card]- What is the difference between intrinsic and extrinsic evaluation of word embeddings?
> Intrinsic: how good the embeddings are **by themselves**. Extrinsic: how useful they are **in downstream tasks** (MT, IR, summarisation).

> [!card]- How is word similarity evaluated against human judgements?
> Rank word pairs by **cosine** and compare with the human ratings using **Spearman (rank) correlation**.

> [!card]- What is the difference between the WS-353 and SimLex-999 benchmarks?
> WS-353 annotates **relatedness** (`coffee`/`cup` count as close). SimLex-999 annotates **similarity**.

> [!card]- How is the answer to "a is to b as c is to X" computed with word embeddings?
> The word closest by cosine to **$v_b - v_a + v_c$**, excluding the three input words. Scored by accuracy.

> [!card]- What is the difference between PCA and t-SNE for visualising embeddings?
> PCA: **linear** projection onto the top-variance directions. t-SNE: **non-linear**, keeps local neighbours, gives up global distances.

> [!card]- Why should 2D t-SNE plots of embeddings not be taken at face value?
> Non-linear projections **distort distances**, and t-SNE **hyperparameter settings substantially change** cluster sizes, distances, even apparent clusters.

> [!card]- Why do morphologically rich languages suffer most from one vector per word type?
> Each lemma has **many inflected forms**, each a separate, rarer word with an unrelated, worse-estimated vector.

> [!card]- How does fastText compute a word's vector?
> As the **sum of the vectors of its character n-grams**, plus the whole word.

> [!card]- What character n-gram lengths does fastText use by default?
> **3 to 6**.

> [!card]- Why does fastText wrap each word in `<` and `>`?
> The boundary markers make **prefixes and suffixes distinct n-grams**: `<ru` is only word-initial, `ns>` only word-final.

> [!card]- How does fastText's training differ from Skip-gram with negative sampling?
> Only the **centre word is decomposed** into n-grams. The loss and the per-word context vectors are unchanged.

> [!card]- Where does fastText gain most on word similarity over Word2Vec?
> Most in **morphologically rich languages** (German, Russian); it also helps **rare words** (English Rare Words).

> [!card]- Why do semantic analogies not benefit from fastText's character n-grams?
> Relations like **capital and country have nothing to do with character overlap**; shared n-grams can add noise.

> [!card]- What is building a bilingual dictionary by nearest-neighbour search in a cross-lingual embedding space called?
> **Bilingual lexicon induction** (BLI).

> [!card]- What are the three basic set-ups for training cross-lingual embeddings?
> Train **separately, then align the spaces**; train **together, then align regions**; train on **word- or sentence-aligned data**.

> [!card]- What does the shared space hypothesis claim?
> Separately trained language spaces are **approximately isomorphic**: same overall shape, possibly different coordinates and orientation.

> [!card]- Why is the cross-lingual mapping assumed to be linear?
> Analogies are **already linear regularities** within one space, and a linear map **preserves them exactly** across spaces.

> [!card]- What bilingual signal does each type of cross-lingual alignment use?
> Supervised: **thousands** of seed translations. Semi-supervised: **hundreds**, then bootstrapping. Unsupervised: **none** (adversarial training).

> [!card]- What is the closed-form solution of the unconstrained least-squares cross-lingual mapping $\min_W \sum_i \lVert W x_i - y_i \rVert^2$?
> $W^* = Y X^\top (X X^\top)^{-1}$.

> [!card]- What is the difference between solving the least-squares mapping in closed form and with SGD?
> Closed form is **exact**. SGD is **approximate** but scales better and is more memory-friendly; Mikolov et al. used SGD.

> [!card]- In cross-lingual results tables, what does "Procrustes" refer to?
> The **orthogonally constrained** mapping solved by SVD. The unconstrained map is Mikolov et al.'s translation matrix.

> [!card]- How does linear-mapping translation accuracy change for rarer words?
> It **drops**: rarer words have worse embeddings, so their mapped positions are less reliable.

> [!card]- Why does an unconstrained mapping $W$ overfit the seed dictionary?
> It can **stretch, shear and rotate** the space to fit the seed pairs, so it generalises poorly to other words.

> [!card]- What mismatch between training and inference does the unconstrained mapping have?
> Training minimises **Euclidean distance**, retrieval uses **cosine**. They agree only for unit-normalised vectors.

> [!card]- What constraint and normalisation define Xing et al.'s orthogonal mapping?
> **$W^\top W = I$**, with all embeddings normalised to **unit length**.

> [!card]- What transformations can an orthogonal mapping $W$ perform?
> Only **rotations and reflections**: no scaling or shearing, so angles and distances are preserved.

> [!card]- How is the orthogonal Procrustes mapping solved?
> **Exactly by SVD**: $U \Sigma V^\top = \operatorname{SVD}(Y X^\top)$, then $W = U V^\top$.

> [!card]- Why does the unconstrained mapping get worse as the embedding dimension grows?
> $W$ has **$d^2$ free parameters** for the same seed dictionary, so there is more room to overfit.

> [!card]- How does the orthogonal mapping compare with the unconstrained one across embedding dimensions?
> **Better at every dimension**, and it **improves slightly** as dimension grows.

> [!card]- What is a hub in a high-dimensional embedding space?
> A vector that is **the nearest neighbour of many points regardless of actual similarity**.

> [!card]- What is the CSLS score between a mapped source vector $x$ and a target vector $y$?
> $\text{CSLS}(x, y) = 2\cos(x, y) - r_T(x) - r_S(y)$.

> [!card]- Which CSLS term penalises hubs?
> **$r_S(y)$**: the candidate's average cosine to its $K$ nearest mapped source vectors.

> [!card]- Why does $r_T(x)$ not change which translation CSLS picks for a given source word?
> It is **the same for every candidate** $y$. It matters only when comparing scores across source words.

> [!card]- How does CSLS retrieval compare with plain nearest neighbour on the same Procrustes mapping?
> **CSLS is better on every language pair**, with no tuning beyond $K$.

> [!card]- What is the chicken-and-egg problem of unsupervised cross-lingual alignment?
> You need a **mapping to find translation pairs**, and **translation pairs to learn the mapping**.

> [!card]- In MUSE's adversarial training, what does the discriminator try to distinguish?
> **Mapped source vectors** $Wx$ from **real target vectors** $y$. The generator is the mapping $W$.

> [!card]- Why is MUSE's generator a single linear matrix rather than a deep network?
> A deep generator could **match the target distribution while scrambling which word maps where**.

> [!card]- Why does MUSE use label smoothing ($s = 0.2$) on the discriminator?
> To stop it becoming **overconfident**, which would give the generator **vanishing gradients**.

> [!card]- Why does MUSE feed the discriminator only the 50k most frequent words?
> **Rare words have poorly estimated embeddings**, so frequent words give a cleaner signal.

> [!card]- What does re-orthogonalisation do in MUSE's adversarial training?
> After each gradient step it **pulls $W$ back towards an orthogonal matrix** ($\beta = 0.01$).

> [!card]- What unsupervised criterion does MUSE use to select a checkpoint?
> The **average cosine** between the 10k most frequent source words and their **CSLS nearest** target words.

> [!card]- How does MUSE build its synthetic dictionary for refinement?
> From **mutual CSLS nearest neighbours** among frequent words, keeping only high-confidence pairs.

> [!card]- Why does MUSE's synthetic dictionary keep only mutual nearest neighbours?
> A pair is trusted only if each word picks the other, which **filters out hub matches**.

> [!card]- Why does the MUSE refinement loop count as semi-supervised?
> It runs **supervised Procrustes on a dictionary the system built itself**. Seeded with a small human dictionary, it is the semi-supervised method.

> [!card]- How does fully unsupervised MUSE compare with supervised Procrustes-CSLS for close pairs (en-es, en-fr, en-de)?
> It **matches or slightly beats** supervised alignment.

> [!card]- How does fully unsupervised MUSE compare with supervised alignment for distant pairs (en-ru, en-zh)?
> It **falls well short**.

> [!card]- What is the difference between head-marking and dependent-marking?
> Head-marking puts the grammatical marker **on the head**. Dependent-marking puts it **on the dependent**.

> [!card]- Which language pairs did adversarial alignment fail on completely in Søgaard et al. (2018)?
> **EN-ET, EN-FI, EN-EL**: Estonian, Finnish and Greek, the mixed- or double-marking languages.

> [!card]- What is identical-word supervision for cross-lingual alignment?
> A seed dictionary of **words spelled identically in both languages**. It needs no human dictionary.

> [!card]- What does the Estonian-Finnish result show about unsupervised alignment?
> **Structurally similar** languages align adversarially even when neither aligns with English. It is the only pair in Søgaard et al.'s language comparison where adversarial beats identical.

> [!card]- Why does unsupervised alignment fail for mixed- and double-marking languages, according to Søgaard et al.?
> Each lemma has **many distinct surface forms**, each its own lower-frequency embedding, so the mapping becomes non-isomorphic.

> [!card]- Is the English-Finnish alignment failure caused by too little Finnish training data?
> **No.** Retraining on a far larger Finnish corpus still gave zero precision.

> [!card]- How does a domain mismatch between the two corpora affect identical-word supervision and adversarial alignment?
> Identical-word supervision **degrades but survives**. Adversarial alignment **collapses to about zero**.

> [!card]- How much do hyperparameter changes within one embedding algorithm (skipgram window, n-gram range) hurt unsupervised alignment?
> **Barely**: drops of under two points.

> [!card]- What happens to unsupervised alignment when one language uses skipgram embeddings and the other CBOW?
> It **fails completely** (about zero), even with otherwise identical hyperparameters.

> [!card]- How is eigenvector similarity $\Delta$ computed for two languages?
> **Sum of squared differences** between the largest Laplacian eigenvalues of their nearest-neighbour graphs.

> [!card]- What does a larger eigenvector similarity $\Delta$ mean?
> The two graphs are **more structurally different**: further from isomorphism.

> [!card]- Why does eigenvector similarity not need a bilingual dictionary?
> Laplacian eigenvalues are **invariant to relabelling the nodes**, so no alignment between the languages is needed.

> [!card]- Why can no rotation fix a high eigenvector similarity $\Delta$?
> A rotation **preserves the nearest-neighbour graph**, so it cannot change the Laplacian eigenvalues.

> [!card]- Why does Finnish give a high eigenvector similarity $\Delta$ against English?
> One lemma's inflected forms **cluster tightly**, while English gives a **flatter, more evenly connected** graph.

> [!card]- How does eigenvector similarity $\Delta$ relate to adversarial alignment success?
> It **tracks it**: the lowest-$\Delta$ pairs align best, and the highest-$\Delta$ pairs fail completely.

> [!card]- What does "static" mean for Word2Vec, GloVe and fastText embeddings?
> **One vector per word type**, whatever the sentence: `bank` is the same in *river bank* and *bank account*.

> [!card]- True or false: In Skip-gram, the hidden layer is the average of the context word embeddings.
> **False.** That is CBOW. Skip-gram's hidden layer is the centre word's embedding.

> [!card]- True or false: Word2Vec's context size and number of negative samples can be smaller the more training data there is.
> **True.** More data allows a smaller context and a smaller negative sample set.

> [!card]- True or false: fastText can produce a vector for a word that never occurred in training.
> **True.** It sums the vectors of the word's character n-grams.

> [!card]- True or false: fastText improves over Word2Vec on both semantic and syntactic analogies.
> **False.** The gains are syntactic; semantic analogies do not improve and sometimes get worse.

> [!card]- True or false: For unit-length vectors and an orthogonal $W$, minimising Euclidean distance is equivalent to maximising cosine similarity.
> **True.** $\lVert Wx - y \rVert^2 = 2 - 2\cos(Wx, y)$.

> [!card]- True or false: The unconstrained least-squares mapping continuously improves as the embedding dimension grows.
> **False.** It gets worse, since $d^2$ parameters overfit the same dictionary. The orthogonal map improves slightly.

> [!card]- True or false: In MUSE, the adversarial (discriminator) loss is a reliable criterion for choosing which training checkpoint to keep.
> **False.** It can look fine while the mapping is poor; MUSE selects by average CSLS-neighbour cosine instead.

> [!card]- True or false: Adversarial unsupervised alignment does not suffer from a domain mismatch between the two monolingual corpora.
> **False.** Across domains it collapses to about zero, even for English-Spanish.

> [!card]- True or false: In Søgaard et al.'s language-pair comparison (EN-ES to EN-TR), identical-word supervision beats adversarial alignment on every pair involving English.
> **True.** From a small margin (EN-ES) to working against not working (EN-ET).

> [!card]- True or false: A low eigenvector similarity $\Delta$ proves that two embedding spaces are near-isomorphic.
> **False.** Near-isomorphic spaces give a low $\Delta$, but a low $\Delta$ does not imply near-isomorphism.

> [!card]- Multiple selection: which are true of negative sampling in Word2Vec? (A) It turns prediction into classifying real against noise pairs (B) Each observed pair gets $k$ negatives, typically 5 to 20 (C) It still normalises over the whole vocabulary for each example (D) Negatives are drawn uniformly from the vocabulary
> **A and B.** C: the cost drops to $k+1$ dot products, no full softmax. D: negatives come from $P(w)^{0.75}$.

> [!card]- Multiple selection: which are true of GloVe? (A) It fits dot products plus biases to log co-occurrence counts (B) Its weighting function caps at 1 for frequent pairs (C) It treats each context window as an independent event (D) Its final embedding sums the word and context vectors
> **A, B and D.** C: that is the Word2Vec weakness GloVe addresses by using global counts.

> [!card]- Multiple selection: which are true of hubness and CSLS? (A) Hubness worsens as dimensionality increases (B) CSLS penalises candidates that are close to many mapped source vectors (C) CSLS requires extensive hyperparameter tuning (D) Nearest-neighbour retrieval is symmetric
> **A and B.** C: only the neighbourhood size $K$. D: a hub can be `cat`'s nearest neighbour without `cat` being the hub's.

> [!card]- Multiple selection: which are true of unsupervised (adversarial) alignment, according to Søgaard et al. (2018)? (A) English-Finnish fails even with far more Finnish data (B) It survives a domain mismatch if the languages are close (C) It works when one language uses skipgram and the other CBOW (D) Estonian-Finnish aligns adversarially
> **A and D.** B: cross-domain English-Spanish collapses to about zero. C: mixing skipgram and CBOW fails completely.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L04 - Subword Segmentation]] (subword units; fastText's character n-grams are the embedding-side answer to the same vocabulary problem)
- **Next:** [[MNLP-L06 - Contextual Embeddings]] (vectors that depend on the sentence, built with [[Recurrent Neural Network (RNN)|RNNs]], [[Self-Attention]] and [[Transformers]], the basis of [[Large Language Models (LLM)]])
- **Later:** [[MNLP-L07 - Crosslingual NLP]] (cross-lingual transfer, building on the aligned spaces of sections 10 to 16)
- **Related concepts:** [[Word Embeddings]] · [[Negative Sampling]] · [[Contrastive Learning]] · [[Vector Space Model]] · [[Bag of Words]] · [[Tokenization]]
- **Source:** `embeddings_static.pdf`, Christof Monz, MNLP: Static Embeddings, slides 1 to 71
- **Primary references:** Firth (1957) · Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* and *Distributed Representations of Words and Phrases and their Compositionality* (Word2Vec, negative sampling) · Mikolov, Le and Sutskever (2013), *Exploiting Similarities among Languages for Machine Translation* (linear translation matrix) · Pennington, Socher and Manning (2014), *GloVe: Global Vectors for Word Representation* · Wattenberg, Viégas and Johnson (2016), *How to Use t-SNE Effectively* · Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information* (fastText) · Xing et al. (2015), *Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation* · Conneau et al. (2018), *Word Translation Without Parallel Data* (MUSE, CSLS) · Søgaard, Ruder and Vulić (2018), *On the Limitations of Unsupervised Bilingual Dictionary Induction*

