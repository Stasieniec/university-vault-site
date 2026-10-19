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

> [!abstract] In short
> How do you give every word a vector, so that nearby vectors mean similar words, using only raw text? By the company the word keeps: count its context words, train a small network to predict them (Word2Vec, made affordable by negative sampling), or fit vectors to corpus-wide co-occurrence counts (GloVe). fastText builds a word's vector from its character n-grams, so related forms share parts and unseen words still get a vector.
>
> The second half aligns two languages' spaces with a single linear map, learned from a seed dictionary (least squares, then an orthogonal map solved by SVD) or with no dictionary at all (adversarial training, MUSE), and retrieves translations with CSLS, which discounts "hub" words. Unsupervised alignment breaks down for very different morphology, mismatched domains and mismatched embedding algorithms.
>
> "Static" means one vector per word type, whatever the sentence: `bank` gets the same vector in *river bank* and *bank account*; [[MNLP-L06 - Contextual Embeddings]] removes that limit. The sample exam has no question on this lecture; expect statements about what each method does, which data it uses, what is fixed or constrained, and when it fails.

## The idea: a word is known by its contexts

We want word representations that let us compute similarity along several dimensions: meaning (synonyms such as `car` and `automobile`, topical relatedness such as `car` and `road`) and morphology (`run`, `runs`, `running`).

> [!definition] Distributional hypothesis
> "*You shall know a word by the company it keeps.*" (Firth, 1957)
>
> Produce vector representations of words **based on the contexts in which they occur**: words in similar contexts get similar vectors, and are taken to have similar meanings.

There are two types of approach. **Count-based** methods (distributional semantics) count the context words directly. **Prediction-based** methods (neural embeddings) train a network to predict words from contexts, or the reverse, and keep its weights as the vectors. GloVe combines the two.

### Counting the company a word keeps

Define a vocabulary $V_c$ of **context words**, a vocabulary $V_t$ of **target words** (they can be the same) and a **window of size $n$** (sliding; the context can also be the sentence or document). For each occurrence of $w \in V_t$, count how often each $w' \in V_c$ occurs within $n$ words to its left or right, and store the counts in a vector $c_w$ of length $|V_c|$. With $n = 3$ and $V_c = \{\textit{leash}, \textit{walk}, \textit{owner}, \ldots\}$, "...he took the dog for a walk..." adds 1 to $c_\text{dog}[\textit{walk}]$; `took` is in the window but not in $V_c$, so it is ignored.

Similarity is the **cosine** of the angle between two vectors.

> [!formula] Cosine similarity #key-formula
> $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert} = \frac{\sum_i u_i v_i}{\sqrt{\sum_i u_i^2}\,\sqrt{\sum_i v_i^2}}$$
>
> where $\mathbf{u} \cdot \mathbf{v}$ is the dot product and $\lVert \mathbf{u} \rVert$ the Euclidean length. It is 1 for vectors pointing the same way, 0 for orthogonal ones.

Cosine ignores length on purpose: a frequent word has large counts everywhere, and only the proportions of its contexts should matter. With two context words (`runs`, `legs`), `dog` = (1, 4) and a word seen ten times as often in the same mix, (10, 40), have cosine 1; `car` = (4, 1) gets $8/17 = 0.47$ against `dog`.

Context vectors are **high-dimensional**, **discrete** (counts) and **sparse** (mostly zeros). They are an unsupervised way to induce word similarity, and their dimensions are interpretable ($c_w[i]$ counts an actual word). But they stay sparse even after stemming (cutting words to a common stem, so `walked` counts as `walk`), and there is **no explicit criterion to tell signal from noise**: some words fall in the window by chance, and nothing says which context words discriminate a word's meaning (`the` co-occurs with everything). One fix weights context words by inverse document frequency, $\text{idf}(w) = \log (N / n_w)$, with $N$ the number of contexts and $n_w$ the number containing $w$: a word found in nearly every context gets a weight near 0.

## Learning dense vectors by prediction

The probabilistic neural language model (PNLM) predicts the next word through a **projection layer** that maps each word $w$ to a learned vector $C(w)$, then a non-linear hidden layer and a softmax. A good embedding has $C(w) \approx C(w')$ exactly when $w$ and $w'$ mean the same and behave the same syntactically. In most models such vectors are by-products of the task; in **representation learning** they are the main objective. Word2Vec works this way: prediction is a pretext, and only the embedding matrices are kept.

**Word embeddings** share the intuition of distributional semantics, but they are **low-dimensional** (e.g. 100, against $|V_c|$), **dense**, **continuous** ($c_w \in \mathbb{R}^d$) and **learned by performing a task** (prediction). Their dimensions are not interpretable.

Word2Vec (Mikolov et al., 2013) has two variants: **CBOW** predicts a word from its context, **Skip-gram** predicts the context from the word.

### CBOW: predict the word from its context

**Continuous Bag of Words (CBOW)**: given position $t$, the $n$ words to its left and the $m$ to its right, predict the word at position $t$ ("the man X the road"). An n-gram language model predicts from the left only ($m = 0$); CBOW also looks right, so it is useless as a language model and exists only to learn representations. It is simpler than the PNLM: it drops the non-linear hidden layer, so the embedding layer sits right next to the output. Typically $n = m$, with $n \in \{2, 5, 10\}$.

> [!formula] CBOW #key-formula
> **No non-linearities.** One hidden layer, then the output:
> $$h = \frac{1}{2n} W \, w_C, \quad w_C = \sum_{\substack{i = t-n \\ i \neq t}}^{t+n} w_i, \qquad \hat{y} = \operatorname{softmax}(W' h)$$
>
> where:
> - $w_i$ is the one-hot vector of the word at position $i$
> - $W$ is the $|h| \times |V|$ input (projection) matrix, $|h|$ the embedding size
> - $W'$ is the $|V| \times |h|$ output matrix, $\hat{y}$ the predicted distribution
> - $W$ and $W'$ are **not (necessarily) shared**: $W' \neq W^\top$
>
> Loss: cross entropy. Training: SGD.

Multiplying $W$ by a one-hot vector selects a column of $W$, so $h$ is the **average** of the context embeddings (a sum scaled by $1/2n$). Summing loses word order, hence "bag of words". The softmax normalises over all $|V|$ words, so every example touches all of $W'$: the cost negative sampling removes.

Column $i$ of $W$ is word $i$ as an input word (the context words in CBOW, the centre word in Skip-gram); row $i$ of $W'$ is word $i$ as an output word. Typically $W$ is used as the embedding, or both are combined: $W_s = W^\top + W'$.

### Skip-gram: predict the context from the word

**Skip-gram**: given the word at position $t$, predict the words between $t-n$ and $t-1$ and between $t+1$ and $t+n$. One input word, $2n$ predictions.

> [!formula] Skip-gram #key-formula
> One hidden layer: $h = W \, w_I$, with $w_I$ the one-hot input word at position $t$.
>
> $2n$ output layers, the context words **independent given the input word**:
> $$p(w_{t-n} \ldots w_{t-1} w_{t+1} \ldots w_{t+n} \mid w_I) = \prod_{\substack{i=t-n \\ i \neq t}}^{t+n} p(w_i \mid w_I), \qquad \hat{y}_i = \operatorname{softmax}(W' h)$$
>
> $W$, $W'$ as in CBOW, not necessarily shared. Loss: cross entropy. Training: SGD.

$h$ is the input word's column of $W$, with no averaging. All $2n$ outputs share $W'$ and $h$, so they give the **same** distribution, each scored against a different context word: Skip-gram trains on $2n$ separate (input word, context word) pairs per position.

## Negative sampling: classify pairs instead of predicting words

Both models benefit from large amounts of data, and that is where computing the full output layer hurts: a score for every vocabulary word, for every example. **Negative sampling** instead distinguishes words that do and do not occur in the context of the input word: a **classification task** with **1 positive example** from the ground truth and **$k$ negative examples** from a **random noise distribution**. The cost per example drops from $|V|$ dot products to $k + 1$ (see [[Negative Sampling]], an instance of [[Contrastive Learning]]).

**The objective.** Let $\mathcal{D}$ be the observed (word, context) pairs and $\mathcal{D}'$ pairs from the noise distribution. Choose parameters $\theta$ (the matrices $W$, $W'$) that make observed pairs look real ($D = 1$) and noise pairs look fake ($D = 0$):

$$\arg\max_\theta \prod_{(w,c) \in \mathcal{D}} p(D = 1 \mid c, w; \theta) \prod_{(w,c) \in \mathcal{D}'} p(D = 0 \mid c, w; \theta), \qquad p(D = 1 \mid c, w; \theta) = \sigma(v_c \cdot v_w)$$

where $\sigma(x) = 1/(1+e^{-x})$, $v_w = Ww$ (the input word's column of $W$) and $v_c = W'^\top c$ (the context word's row of $W'$). Since $p(D = 0) = 1 - \sigma(v_c \cdot v_w)$ and $1 - \sigma(x) = \sigma(-x)$, taking logs gives:

> [!formula] Negative sampling objective #key-formula
> $$\arg\max_\theta \sum_{(w,c) \in \mathcal{D}} \log \sigma(v_c \cdot v_w) + \sum_{(w,c) \in \mathcal{D}'} \log \sigma(-v_c \cdot v_w)$$

**One training step of Skip-gram with negative sampling.** Take a real pair $(w, c)$ from the window. Draw $k$ words $n_1, \ldots, n_k$ from the noise distribution and pair each with the **same input word $w$**: a negative keeps $w$ and swaps in a sampled word for $c$. Minimise $-\log \sigma(v_c \cdot v_w) - \sum_{j=1}^{k} \log \sigma(-v_{n_j} \cdot v_w)$ by SGD. That updates $v_w$ (a column of $W$) and the $k + 1$ output vectors $v_c, v_{n_1}, \ldots, v_{n_k}$ (rows of $W'$), and nothing else.

The dot product is a compatibility score, pushed up for real pairs and down for random ones, so words that share many contexts are pulled towards the same output vectors, and so towards each other. (Outside the lecture: the gradient for $v_w$ is $(\sigma(v_c \cdot v_w) - 1)\, v_c + \sum_j \sigma(v_{n_j} \cdot v_w)\, v_{n_j}$, so a step moves $v_w$ towards $v_c$ and away from the negatives.)

> [!definition] Word2Vec practical settings
> - Skip-gram: for each observed pair $(w, c)$, add **5 to 20 negative samples**.
> - Draw negatives from the **unigram distribution** $P(w)$ (each word's share of all tokens), **raised to the power 0.75** and renormalised, to bias sampling **towards rarer words**.
> - Context size typically around **2 to 5** (a practical recommendation; the CBOW description allows $n$ up to 10).
> - **The more data, the smaller the context and the negative sample set** can be.

Without the 0.75 power nearly every negative would be `the`, `of` or `and`.

## GloVe: fit vectors to global counts

Two strategies, each strong where the other is weak. **Count-based** methods, used directly or as global matrix factorisation (latent semantic analysis: factorise the co-occurrence matrix with SVD), use global information directly but tend to perform worse. **Prediction-based** methods with a local window (CBOW, Skip-gram) tend to outperform them, but **treat each context window as an independent event**, rediscovering the association between two words every time they co-occur.

**GloVe** (Global Vectors; Pennington et al., 2014) **combines global co-occurrence statistics with local prediction-based updating**: count once over the whole corpus, then fit vectors to those counts with gradient updates. $X_{ij}$ is the number of times word $j$ occurs in the context of word $i$.

> [!formula] GloVe objective #key-formula
> $$J = \sum_{(i,j)} f(X_{ij}) \cdot \left( w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
>
> where $w_i$ is the vector for word $i$, $\tilde{w}_j$ the vector for context word $j$, $b_i$ and $\tilde{b}_j$ scalar biases, $\log X_{ij}$ the target, and $f$ a weighting function.

So GloVe is **weighted least-squares regression**: vectors whose dot product (plus biases) predicts the log of how often two words co-occur. Words with similar co-occurrence rows get similar vectors.

> [!formula] GloVe weighting function
> $$f(x) = \begin{cases} (x / x_\text{max})^{\alpha} & \text{if } x < x_\text{max} \\ 1 & \text{otherwise} \end{cases} \qquad \text{typically } x_\text{max} = 100,\ \alpha = 0.75$$

It de-emphasises very rare (noisy) and very frequent (uninformative) co-occurrences: it rises smoothly from zero for rare pairs and **caps at 1** for frequent ones. Frequent pairs still get the highest weight, 1; the cap only stops their weight growing with the count. Since $f(0) = 0$, pairs that never co-occur (where $\log X_{ij}$ is undefined) drop out, and training runs only over the non-zero cells of $X$.

**Training.** A **single pass over the corpus** builds $X$, each co-occurrence often weighted by 1/distance so nearer words count more. Training then sees only $X$, never the corpus: that is what "global" means. Each epoch visits the pairs in shuffled order and updates $w_i$, $\tilde{w}_j$, $b_i$, $\tilde{b}_j$ with AdaGrad (outside the lecture: SGD with a per-parameter learning rate that shrinks for parameters that have had large gradients). The **final embedding is $w_i + \tilde{w}_i$**, the same trick as $W^\top + W'$ in Word2Vec.

Word2Vec instead handles frequent words by **subsampling** (defined here, not in the lecture): randomly discarding occurrences of very frequent words before training, more often the more frequent the word.

| | Word2Vec | GloVe |
|---|---|---|
| Trains on | local context windows | global co-occurrence matrix |
| Objective | predict context words | weighted least squares on log counts |
| Frequency imbalance | subsampling (and negative sampling) | weighting function $f(X_{ij})$ |
| Interpretability | indirect: learns to predict context | directly fits log counts |

Neither has interpretable dimensions; the last row is about what the objective fits.

## Judging whether embeddings are any good

Evaluation has two independent axes. **Intrinsic** evaluation asks how good the embeddings are by themselves, **extrinsic** how useful they are for downstream tasks (summarisation, MT, IR). **Qualitative** evaluation examines selected examples, **quantitative** uses an overall, task-dependent score. A 2D plot is intrinsic and qualitative, a similarity benchmark intrinsic and quantitative.

**Word similarity.** Rank word pairs such as (`car`, `bicycle`) by cosine and compare with human judgements by **Spearman correlation**, a correlation between the two rankings, so only the order matters. **WS-353 annotates relatedness**: `bread` and `butter` score fairly high although neither is a kind of the other. **SimLex-999 annotates similarity** in the strict sense, so related but unlike pairs such as `coffee` and `cup` (an example from outside the lecture) score low. The same embeddings can therefore rank differently on the two. The pair lists mix all kinds of pairs: synonyms, topically related and unrelated ones (WS-353 has both `tiger`/`cat` and `drink`/`ear`).

**Analogies.** *Paris is to France as Berlin is to X*, scored by **accuracy**. The "arithmetic magic": $X = v_\text{king} - v_\text{man} + v_\text{woman}$, answered by the word nearest to $X$ by cosine (`queen`), excluding the input words. There are syntactic analogies too: *acquired* is to *acquire* as *tried* is to X.

**Visualisation.** Project to 2D with **PCA** (linear, onto the directions of largest variance) or **t-SNE** (non-linear, keeps neighbours close, gives up global distances). Plots show senses separating, and relations as directions: `brother` to `sister` and `king` to `queen` point the same way, a consistent gender direction. But plots should not be taken at face value: non-linear projections distort distances, and **t-SNE hyperparameters have a substantial impact** (Wattenberg 2016). They complement, but do not substitute, quantitative and extrinsic evaluation.

## fastText: words built from character n-grams

Word2Vec and GloVe give **one opaque vector per word type**: `run`, `runs`, `running`, `runner` get four potentially unrelated vectors. Morphologically rich languages suffer most: Finnish, Turkish and Russian have huge families of word forms, each a separate, rarer word with a worse vector. And there is **no fallback for unseen (out-of-vocabulary, OOV) words**. This is the problem of [[MNLP-L04 - Subword Segmentation]] from the embedding side.

There are two fixes. One splits words into subword segments (e.g. with BPE) and learns an embedding per segment. The other slides a character window over the word: **fastText** (Bojanowski et al., 2017). fastText is Skip-gram with negative sampling, with these changes:

- **The centre word's vector is the sum of its character n-gram vectors**, $v_w = \sum_{g \in G_w} z_g$. $G_w$ holds the n-grams of `<w>` of **3 to 6 characters** by default, plus the **whole word** `<w>`.
- `<` and `>` are boundary markers, so prefixes and suffixes are distinct n-grams (`<ru` only starts a word, `ns>` only ends one).
- The loss is the negative-sampling loss. **Only the centre word is decomposed; context vectors stay per word.**
- Every n-gram of the word gets **the same gradient**, since $v_w$ is a sum.
- Training returns **the n-gram table** $\{z_g\}$. Any word's vector, seen in training or not, is the sum of its n-grams.

> [!example] The n-grams of `runs`
> `<runs>` gives `<ru`, `run`, `uns`, `ns>`, `<run`, `runs`, `uns>`, `<runs`, `runs>`, `<runs>`. `<running>` shares `<ru`, `<run` and `run`, so the two vectors are pulled together. The unseen `runnings` still has known n-grams such as `<run` and `ings>`, so it gets a sensible vector where Word2Vec has none.

**The evidence** (Bojanowski et al.), against Word2Vec Skip-gram and CBOW:

- **Word similarity:** fastText is best or tied best on almost every dataset, with the **largest gains in morphologically rich languages** (German, Russian) and a gain on **English rare words**. Its one loss is English WS-353 (a likely reason, outside the lecture: its words are frequent, so their whole-word vectors are already good). Building unseen words from n-grams, rather than giving them a null vector, adds further.
- **Analogies:** the gains are on **syntactic** analogies (largest for Czech), which are about morphology. **Semantic analogies barely change, and for German get worse.** A likely reason (outside the lecture): a country and its capital share nothing in spelling, and tying words with shared n-grams together adds noise.

## Crosslingual embeddings: one space for two languages

Embeddings give $E(\text{house}) \approx E(\text{building})$ in English and $E(\text{huis}) \approx E(\text{gebouw})$ (*gebouw*, building) in Dutch. Can we get $E(\text{house}) \approx E(\text{gebouw})$? Such a space gives **bilingual lexicon induction** (BLI: building a dictionary by nearest-neighbour search) and transfer between languages ([[MNLP-L07 - Crosslingual NLP]]).

There are three basic set-ups: train each language separately, then **align the spaces**; train all languages together on one mixed corpus (shared names and numbers tie them together), then align the parts of the space each language occupies; or train on data aligned at the word or sentence level. In all three some alignment is needed, so that translation-equivalent words end up close together. The hope is to use **large monolingual corpora** and only a **small bilingual signal**. Everything below concerns the first set-up.

### The shared space hypothesis

Plot English number words and animals in 2D next to their Spanish translations, and the arrangement is the same (`one`/`uno` apart from the other numbers, `horse`/`caballo` near `cow`/`vaca`), only turned and stretched.

> [!definition] Shared space hypothesis
> - **Different languages, trained separately, still encode similar relational structure**: `king` and `queen` relate in English as `rey` and `reina` do in Spanish.
> - There is an **approximate isomorphism** between the spaces: the overall shape of the point cloud is similar, but the specific coordinates and orientations can differ.
> - **If true (it is a hypothesis)**, a single geometric transformation can align the spaces: **a general linear mapping** applied to one space brings translations close together.

In English, `king` and `queen` are separated by roughly the same offset as `man` and `woman`; in Spanish the same holds for `rey`, `reina`, `hombre`, `mujer`, but the whole cloud may be rotated and scaled differently. A structure-preserving map $m$ (an isomorphism) keeps the operations between vectors, $m(a \circ b) = m(a) \circ m(b)$ with $\circ$ an operation such as $+$ or $-$, so $m(v_\text{king} - v_\text{man} + v_\text{woman}) \approx m(v_\text{king}) - m(v_\text{man}) + m(v_\text{woman})$: the English gender offset becomes the Spanish one. A linear map $m(v) = Wv$ satisfies this exactly; the "$\approx$" is there because the spaces are only approximately isomorphic.

### How much bilingual signal: seed dictionaries

> [!definition] Types of alignment methods
> - **Supervised:** a seed bilingual dictionary as anchors, **typically thousands** of translations.
> - **Semi-supervised:** a very small seed dictionary (**hundreds** of translations), then bootstrapping new translations based on statistical confidence.
> - **Unsupervised:** **no bilingual signal**; uses adversarial training.

A **seed dictionary** is a list of (source word, target word) translation pairs, often the top few thousand most frequent words translated with an existing dictionary; automatic translation can be used too, with caution. It is **limited to words**, because there are no embeddings for multi-word expressions. Each pair is one anchor across the spaces (the source vector of `cat`, the target vector of `gato`), and the goal is a mapping that **generalises beyond the anchors**.

Dictionaries come from three sources, each with problems the lecture leaves as questions (likely answers in brackets): **human-compiled dictionaries** (lemmas such as `hablar`, while embeddings hold forms such as `hablamos`); **dictionaries learned from data** by word-aligning human sentence translations (alignments are not one-to-one, so noisy); and **words spelled identically in both languages** (false friends; a shared script is needed). The last works surprisingly well, as the critique at the end shows.

## Learning a translation matrix from a seed dictionary

The **linear transformation hypothesis** is a strong assumption: **a single linear transformation $W$ is sufficient** to map one language's space onto another's. Why linear rather than non-linear? King/queen and man/woman analogies are **already linear regularities within one space**, assumed to hold across spaces too. It reduces alignment to a classic regression problem.

> [!formula] General alignment problem #key-formula
> $$W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2$$
>
> where $(x_i, y_i)$ are paired vectors, a source embedding $x_i \in \mathbb{R}^{d_1}$ and the target embedding $y_i \in \mathbb{R}^{d_2}$ of its translation, and $W$ ($d_2 \times d_1$) is applied to every source vector.

Stack the dictionary pairs as columns of $X$ ($d_1 \times n$) and $Y$ ($d_2 \times n$). Either run gradient descent on the squared loss, or use the closed form (derivation outside the lecture: set the gradient $2(WX - Y)X^\top$ to zero):

> [!formula] Closed-form solution
> $$W^* = Y X^\top (X X^\top)^{-1}$$
> It needs $XX^\top$ invertible: at least $d_1$ dictionary pairs whose source vectors are linearly independent.

**Mikolov et al. (2013) used SGD**, which scales better with large embedding dimensions and dictionaries; the closed form is exact, SGD approximate but more memory-friendly. To translate $x$, compute $Wx$ and return the nearest target word by cosine.

**How well it worked** (Mikolov et al., English with Spanish and Czech): the translation matrix **beats both baselines by a wide margin**, edit distance (pick the most similarly spelled word) and word co-occurrence (count-based vectors). Accuracy **rises with monolingual data size** and **drops for rarer words**, whose embeddings are worse. The errors are related words (`imperio`, empire, gives `dictatorship`; `millas` ranks `kilometers` above `miles`): nearness captures relatedness, while a dictionary demands equivalence.

## Keeping the map rigid: orthogonal mappings

> [!warning] Limitations of unconstrained mappings
> - An arbitrary $W$ can **stretch, shear and rotate**; nothing preserves the source space's internal geometry (distances, angles).
> - So it can **overfit the seed dictionary**: pairs in or near the dictionary fit well, and the mapping does not generalise to the words outside it (the majority).
> - Training minimises Euclidean distance, retrieval uses cosine, and the two are **not equivalent unless vectors are unit-normalised**.

> [!formula] Orthogonal mapping (Xing et al., 2015) #key-formula
> $$W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2 \quad \text{subject to} \quad W^\top W = I$$
> with all embeddings **normalised to unit length**.
>
> A strictly orthogonal $W$ is square, which needs both spaces to have the same dimension $d$; Xing et al. also report a run with 800 English and 200 Spanish dimensions, where $W$ is non-square and can be orthogonal in one direction only. It **only allows rotations and reflections, no scaling or shearing**, so it preserves angles and distances. $W^*$ can be **solved exactly using SVD**.

Orthogonality is an extra constraint added by Xing et al.; the shared space hypothesis itself asks only for a linear map. The constraint fixes both problems:

1. **Geometry is preserved**: $\lVert Wx \rVert^2 = x^\top W^\top W x = \lVert x \rVert^2$, so the source space moves rigidly and cannot be warped to fit the seed pairs. With every vector at unit length, a uniform difference in scale between the two clouds disappears, so a rotation (or reflection) can carry the shared shape across.
2. **Training and retrieval agree**: for unit vectors and orthogonal $W$, $\lVert W x - y \rVert^2 = 1 + 1 - 2\,(Wx) \cdot y = 2 - 2 \cos(Wx, y)$, so minimising distance is maximising cosine.

> [!formula] Orthogonal Procrustes solution
> $$U \Sigma V^\top = \operatorname{SVD}(Y X^\top), \qquad W^* = U V^\top$$
> One SVD of a $d \times d$ matrix: exact, no iterations, no learning rate.

**The name "Procrustes" is used two ways.** Monz uses it for the general least-squares problem above, so a statement such as "the Procrustes algorithm finds the $W$ minimising $\sum_i \lVert Wx_i - y_i \rVert^2$", with no constraint on $W$, is true in the lecture's sense. In the literature and in Conneau et al.'s results tables, "Procrustes" means the orthogonal version solved by SVD.

**The evidence** (English to Spanish): the **unconstrained map gets worse as the embedding dimension grows** (a likely reason, not given in the lecture: more free parameters, $d^2$, to overfit the same dictionary). The **orthogonal map is better at every dimension and improves slightly** with it; its runs also normalise the vectors, so both changes contribute. One setting gives the two languages different dimensions (800 English, 200 Spanish), where a non-square $W$ can be orthogonal in one direction only.

## Hubs, and how CSLS discounts them

Retrieval is nearest-neighbour search at inference time, and in high-dimensional spaces some points become **hubs**: a small number of vectors are **the nearest neighbour of many other points, regardless of actual similarity**. This is a known phenomenon of high-dimensional geometry, not specific to embeddings, and it **worsens as dimensionality increases**. A target hub becomes the "translation" of many unrelated source words: nn(`cat`) = nn(`car`) = nn(`house`) = `thing`. Nearest neighbour is not symmetric: `thing` can be `cat`'s nearest neighbour while `cat` is far down `thing`'s list. Conneau et al. (2018) modify cosine to correct for this.

> [!formula] Cross-domain Similarity Local Scaling (CSLS) #key-formula
> $$\text{CSLS}(x, y) = 2 \cos(x, y) - r_T(x) - r_S(y)$$
>
> where $x = Wx_s$ is a mapped source vector and $y$ a target vector,
> - $r_T(x)$ is the average cosine between $x$ and its $K$ nearest neighbours in the target space
> - $r_S(y)$ is the average cosine between $y$ and its $K$ nearest neighbours in the (mapped) source space

Subtracting $r_S(y)$ **penalises a candidate that is close to lots of things**: a hub has a high $r_S$. $r_T(x)$ is the same for every candidate when translating a fixed $x$, so it cannot change the winner; it matters when scores are compared across source words, as when building a dictionary. CSLS **significantly increases translation accuracy without parameter tuning**; only $K$ is set ($K = 10$ in Conneau et al.).

> [!example] CSLS defusing a hub (toy numbers)
> Translating `cat`, with $r_T(x) = 0.5$. The hub `thing`: $\cos = 0.65$, $r_S = 0.70$, CSLS $= 1.30 - 0.5 - 0.70 = 0.10$. `gato`: $\cos = 0.60$, $r_S = 0.40$, CSLS $= 1.20 - 0.5 - 0.40 = 0.30$. Nearest neighbour picks `thing`; CSLS picks `gato`.

**The evidence** (supervised orthogonal mapping on fastText embeddings): CSLS **beats plain nearest neighbour on every language pair**, and is about level with inverted softmax, another hubness correction. Accuracy falls from Spanish and French through German and Russian to Chinese, which fits distance from English. Esperanto is lowest, which distance does not explain (likely a small corpus; outside the lecture).

## No dictionary at all: adversarial training (MUSE)

For many low-resource pairs even a small dictionary may not exist, and dictionaries may not cover the inflected forms common in real text. Can two spaces be aligned from their geometry alone? The catch is a **chicken-and-egg problem**: you need a mapping to find translation pairs, and translation pairs to learn the mapping. Conneau et al. (2018) break it with **adversarial training, which bootstraps a rough alignment from distributional structure alone**. Their toolkit is called MUSE.

### The adversarial game

The set-up is borrowed from Generative Adversarial Networks (GANs). The **generator** is the mapping $W$: it maps source embeddings into the target space, aiming to make them indistinguishable from real target vectors. The **discriminator**, a classifier **trained simultaneously**, tells mapped source vectors $Wx_1, \ldots, Wx_m$ (fake data) from real target vectors $y_1, \ldots, y_n$ (real data). They are **trained in opposition**: the discriminator gets better at telling them apart, $W$ at fooling it. If the discriminator cannot tell the clouds apart, they have the same shape, and under the shared space hypothesis that puts translations on top of each other.

> [!important] The generator is deliberately not a deep network
> Unlike a standard GAN, the generator is a **single $d \times d$ matrix, a linear map with no non-linearity**. A deep generator could learn an arbitrarily complex warping that makes the mapped cloud statistically indistinguishable from the target cloud **while scrambling which source word lands where**. A near-rigid map cannot reshuffle words, so matching the two clouds also lines up the translations.

The discriminator is a multilayer perceptron with **2 hidden layers of 2048 units**, ReLU and dropout on the input layer. Its two-class output uses **label smoothing $s = 0.2$**: targets $\{s, 1-s\}$ instead of $\{0, 1\}$, which keeps it from becoming pathologically overconfident (outside the lecture: an overconfident discriminator gives the generator almost no gradient). Source and target vectors come only from **the 50k most frequent words** of each language, because rare words have poorly estimated embeddings and frequent ones give a cleaner signal.

```pseudo
Algorithm: Adversarial training (MUSE)
──────────────────────────────────────────────
Input: source embeddings X, target embeddings Y
       (monolingual, unaligned), β = 0.01
Initialise W ← I (identity/random orthogonal)
Initialise discriminator parameters θ_D

repeat:
  // Discriminator step (W frozen):
  sample batches {x_i} (source), {y_j} (target)
  ℒ_D ← −(1/n) Σ log P_θD(source=1 | W x_i)
        −(1/m) Σ log P_θD(source=0 | y_j)
  θ_D ← θ_D − η · ∂ℒ_D/∂θ_D

  // Generator step (θ_D frozen):
  sample new batches {x_i}, {y_j}
  ℒ_W ← −(1/n) Σ log P_θD(source=0 | W x_i)
        −(1/m) Σ log P_θD(source=1 | y_j)
  W ← W − η · ∂ℒ_W/∂W

  // Re-orthogonalisation:
  W ← (1 + β) W − β (W Wᵀ) W
```

$P_{\theta_D}(\text{source} = 1 \mid v)$ is the discriminator's probability that $v$ is a mapped source vector. **The two steps alternate, each with the other side frozen.** The discriminator trains on the true labels, the generator on **flipped labels**, so $W$ is rewarded when mapped vectors are judged "target". **Re-orthogonalisation** pulls $W$ back towards an orthogonal matrix after each step (an orthogonal $W$ has $WW^\top = I$ and is left unchanged); without it, $W$ would drift into a general linear map with the warping problems of the unconstrained mapping.

**Choosing a checkpoint without a dictionary.** The **adversarial loss is not a reliable indicator** of alignment quality: the discriminator loss can look fine while the mapping is poor, and there is no validation dictionary. Instead, take the **10k most frequent source words**, find each one's **CSLS nearest neighbour** in the target space with the current $W$, compute the **average cosine** over these pairs, and keep the checkpoint where it is highest. A poor mapping puts mapped words far from any real target word, so a high average means the mapping lands on actual target words; whether they are the correct ones it cannot tell.

### Refinement: from a rough map to a precise one

The adversarial $W$ is rough: the global orientation is about right, but it is not precise enough for good retrieval. The full MUSE pipeline continues:

1. Adversarial training for a rough $W$.
2. Build a **synthetic dictionary**: use $W$ with CSLS to find **mutual nearest-neighbour pairs among frequent words**, keeping only high-confidence matches. "Mutual" means $t$ is $s$'s best match and $s$ is $t$'s, which filters out hubs.
3. **Re-solve for $W$ exactly via SVD** on that dictionary (orthogonal Procrustes).
4. Repeat steps 2 and 3 (self-learning).

MUSE as a whole is **unsupervised**: it uses no bilingual data. Conneau et al. call the refinement step **semi-supervised**, because steps 2 to 4 are the supervised orthogonal method run on a dictionary the system built itself. The semi-supervised alignment type from earlier is the same loop started from a small human seed dictionary instead of from adversarial training.

**How well it works.** Each component helps: adversarial alone is weakest, CSLS retrieval and refinement each add, and both together are best. Fully unsupervised MUSE **matches or slightly beats supervised Procrustes with CSLS** on close, well-resourced pairs (English with Spanish, French, German), and **falls well short on distant pairs** (English with Russian, Chinese). The critique starts there.

## When unsupervised mapping fails

Søgaard, Ruder and Vulić (2018) questioned Conneau et al.'s assumptions and results from several angles. They classify languages by where grammar is marked and by morphological type.

> [!definition] Head marking and dependent marking
> When two words are in a grammatical relationship, which one carries the morphological marker?
> - **Dependent-marking:** the dependent. **Head-marking:** the head. **Double-marking:** both. **Zero-marking:** neither.
> - **Mixed marking** (not defined in the lecture; the usual meaning in typology): some constructions mark the head, others the dependent.
>
> In a possessive phrase the head is the possessed noun and the dependent the possessor; in a clause the head is the verb, the dependents subject and object.

German *der Mann sieht den Hund* (the man sees the dog) puts case (marking of a noun's role in the sentence, here on the articles: *der* nominative, *den* accusative) on subject and object, the dependents; the verb also agrees with its subject, so there is some head-marking, but dependent marking predominates. English uses word order instead. The morphological types come from [[MNLP-L03 - Morphology and Word Formation]]: **fusional** languages pack several meanings into one affix, **agglutinative** ones string together separable affixes, one meaning each.

**The choice of languages.** Conneau et al.'s languages (Spanish, French, German, Russian, Chinese) are mostly dependent-marking and none is agglutinative. Spanish, French and Chinese have no cases, German has 4 and Russian 6 to 7, but none has 10 or more. Søgaard et al. added **Estonian and Finnish** (mixed marking, agglutinative, 10+ cases), **Greek** (double marking, fusional, 3 cases), and Hungarian, Polish and Turkish (dependent marking, 6 or more cases).

**A free baseline: identical-word supervision.** Use the words spelled identically in both languages as the seed dictionary, then run ordinary supervised alignment. It needs no human dictionary either. The results:

- Adversarial alignment **fails completely** (essentially zero precision) for **English-Estonian, English-Finnish and English-Greek**.
- **Identical-word supervision beats adversarial training on every pair involving English**, from a small margin (English-Spanish) to working against not working (English-Estonian).
- **Estonian-Finnish**, two structurally similar languages, **aligns adversarially** though neither aligns with English: the one pair where adversarial beats identical.

The lecture sums this up as "languages with mixed or double dependent marking perform poorly", but **marking type alone does not decide it**: French is mixed-marking and aligns well with English. The failing languages combine mixed or double marking with rich morphology (Estonian and Finnish are agglutinative with 10+ cases; Greek is fusional with 3), and Hungarian (dependent-marking, agglutinative, 10+ cases) lands in between.

**Why it hurts.** In mixed- or double-marking languages grammatical information is **distributed across more word forms**: a lemma is realised in many more distinct surface forms, **each with its own embedding and each of lower frequency**. That makes the mapping between a language that marks relations by word order and one that marks them by morphology more complex: **non-isomorphic**. English `house` faces Finnish `talo`, `talon`, `taloa`, `talossa`, ... (an illustration from outside the lecture), and no rotation maps one point onto a dozen.

### Measuring how isomorphic two spaces are

Isomorphism is a strict true/false criterion. To measure degrees of it, Søgaard et al. compare nearest-neighbour graphs through their eigenvalues.

> [!definition] Eigenvector similarity
> 1. Compute each word's **nearest neighbours** and represent them as an **adjacency matrix $A$**.
> 2. Form the **Laplacian $L = D - A$**, with $D$ the diagonal matrix of neighbour counts.
> 3. Take the **eigenvalues of $L$** and keep the **largest $n\%$**.
> 4. For two languages, **sum the squared differences of their Laplacian eigenvalues**:
> $$\Delta = \sum_{i} \left( \lambda^{(1)}_i - \lambda^{(2)}_i \right)^2$$
> with $\lambda^{(1)}_i, \lambda^{(2)}_i$ the $i$-th largest kept eigenvalues of each graph.

(The lecture words this step the other way round: "subtracting diagonal matrix $D$ ... from $A$", which is $A - D$. A statement that the Laplacian is obtained by subtracting $D$ from $A$ follows the lecture. $A - D$ only flips the sign of every eigenvalue; the standard form is $D - A$, with eigenvalues $\geq 0$.)

Laplacian eigenvalues indicate **how connected the nodes are**: bigger values, denser connectivity. **The bigger $\Delta$, the more structurally different** the graphs, the further from isomorphism. It **does not require knowing the alignment**, since eigenvalues do not change when nodes are relabelled: if dense parts are similarly dense and sparse parts similarly sparse, $\Delta$ is small.

A high $\Delta$ means that words cluster in language A unlike in language B, that **no rotation can fix it** (a rotation keeps the neighbour graph, so the eigenvalues), and that the isomorphism assumption does not hold. For Finnish the plausible reason is morphology: one lemma spreads across dozens of inflected forms that cluster tightly through overlapping meaning, while English, with far fewer forms per lemma, gives a flatter, more evenly connected graph. In the results the three pairs where adversarial alignment fails have the **largest** $\Delta$, but the fit is loose (Estonian-Finnish aligns despite a larger $\Delta$ than English-Hungarian), and eigenvector similarity is poor for most pairs: every pair has a larger $\Delta$ than English-Spanish, and the three failing pairs the largest.

> [!warning] The inference only goes one way
> **A low $\Delta$ does not mean the spaces are near-isomorphic.** Near-isomorphic graphs have a low $\Delta$, not the other way round (outside the lecture: different graphs can share the same eigenvalues). $\Delta$ can rule isomorphism out, never in.

### Data size, domain and embedding algorithm

**Not only data size.** Finnish Wikipedia is far smaller than Spanish, so too little data was a suspect. Retraining on a much larger Finnish corpus (WaC, 1.7 billion words) still left English-Finnish at zero, while Estonian-Finnish aligns. **Structural similarity matters: pairing structurally dissimilar languages is the problem.** Results also differ by part of speech: for English-Hungarian, nouns and verbs, the most inflected, do worst.

**Domain.** With equal-sized corpora from EuroParl (political debates), Wikipedia or EMEA (medical texts), English aligned with Spanish, both methods work when both sides share a domain. Across domains, identical-word supervision **degrades but survives**, while adversarial induction **collapses to essentially zero in every combination**, even for English and Spanish.

**Embedding algorithm.** Both sides use fastText. English is fixed at fastText Skip-gram (window 2, n-grams 3 to 6) and the Spanish settings vary. **Variations within one model barely matter**: another window size or n-gram range for Spanish Skip-gram costs very little. **Two different base models do not work at all**: Spanish fastText CBOW against English fastText Skip-gram gives essentially zero, even with otherwise identical settings. So the approximate isomorphism depends partly on using the same algorithm on both sides.

> [!tip] What the critique adds up to
> Unsupervised alignment works when the languages are typologically similar, the corpora share a domain, and both sides use the same embedding algorithm: the conditions it was evaluated under, and the opposite of the low-resource setting that motivates it. A free dictionary of identically spelled words beats it on every English pair tested.

## The alignment methods side by side

| Method | Signal | Constraint on $W$ | Solved by |
|---|---|---|---|
| Translation matrix (Mikolov) | seed dictionary | none | SGD or closed form |
| Orthogonal (Xing) | seed dictionary | orthogonal, unit vectors | SVD |
| Procrustes + CSLS (Conneau) | seed dictionary | orthogonal | SVD, CSLS retrieval |
| MUSE (Conneau) | none | near-orthogonal | adversarial, SVD refinement |
| Identical words (Søgaard) | identical spellings | as supervised | as supervised |

## What to be able to answer

- CBOW against Skip-gram; how negative sampling forms negatives and what one step updates.
- GloVe against Word2Vec; WS-353 against SimLex-999.
- fastText: what is decomposed, where it gains and where it does not.
- The shared space hypothesis; the three kinds of bilingual signal.
- Unconstrained against orthogonal mapping; the two senses of "Procrustes"; CSLS.
- MUSE's design choices, and when unsupervised alignment fails.

Model answers are in the exam questions below.

## Exam questions

> [!exam]- Does unsupervised bilingual lexicon induction work? Answer with the mechanism, how it compares with supervised alignment, and the conditions under which it fails.
> **Key points:** shared space hypothesis and the MUSE pipeline (adversarial, CSLS checkpoint selection, Procrustes refinement); matches supervised on close pairs, falls short on distant ones; collapses for mixed- and double-marking languages (Estonian, Finnish, Greek), across domains and across base models, and data size does not explain it; the free identical-word baseline beats it.
>
> 1. **Mechanism.** The shared space hypothesis says separately trained spaces are approximately isomorphic, so a single linear map should align them. MUSE (Conneau et al., 2018) bootstraps a rough $W$ adversarially (a linear generator against an MLP discriminator, kept near-orthogonal), picks the checkpoint by the unsupervised CSLS criterion, then refines with Procrustes ($W = UV^\top$) on a synthetic dictionary of mutual CSLS nearest neighbours, repeated.
> 2. **Against supervised alignment.** Fully unsupervised MUSE matches or slightly beats supervised Procrustes with CSLS on close, well-resourced pairs (English with Spanish, French, German), and falls well short on distant ones (English with Russian, Chinese).
> 3. **Where it collapses (Søgaard et al., 2018).** English-Estonian, English-Finnish and English-Greek give essentially zero: the mixed- and double-marking languages. Marking type alone does not decide it, since French is mixed-marking and aligns well. English-Finnish stays at zero with a much larger Finnish corpus (1.7 billion words), so data size is not the whole story; Estonian-Finnish, structurally similar, does align. Mismatched domains give essentially zero even for English-Spanish, and so does Spanish fastText CBOW against English fastText Skip-gram.
> 4. **A free baseline beats it.** A seed dictionary of identically spelled words beats adversarial training on every pair involving English, from a small margin (English-Spanish) to working against not working (English-Estonian).
> 5. **Verdict.** It works when the languages are typologically similar, the corpora share a domain and both sides use the same base model: the conditions it was evaluated under, and the opposite of the low-resource setting that motivates it.

> [!exam]- Explain Skip-gram with negative sampling: the classification set-up, how negative pairs are formed, the objective, and what one training step updates.
> **Key points:** real-or-noise classification with 1 positive and $k$ negatives; a negative keeps the input word and swaps in a sampled word; $p(D=1) = \sigma(v_c \cdot v_w)$ and $1-\sigma(x) = \sigma(-x)$ give the log objective; one step updates $v_w$ and $k+1$ output vectors; cost $k+1$ instead of $|V|$.
>
> 1. **Why.** A full softmax needs a score for all $|V|$ words per example. Instead ask "is this (word, context) pair real or noise?".
> 2. **Pairs.** For a real pair $(w, c)$, draw $k$ (5 to 20) words from the unigram distribution raised to 0.75, which favours rarer words, and pair each with the same input word $w$.
> 3. **Likelihood.** $\arg\max_\theta \prod_{\mathcal{D}} p(D=1 \mid c,w;\theta) \prod_{\mathcal{D}'} p(D=0 \mid c,w;\theta)$, with $p(D=1 \mid c,w) = \sigma(v_c \cdot v_w)$, $v_w = Ww$, $v_c = W'^\top c$. Since $1 - \sigma(x) = \sigma(-x)$, after logs: $\sum_{\mathcal{D}} \log \sigma(v_c \cdot v_w) + \sum_{\mathcal{D}'} \log \sigma(-v_c \cdot v_w)$.
> 4. **One step.** Minimise $-\log \sigma(v_c \cdot v_w) - \sum_j \log \sigma(-v_{n_j} \cdot v_w)$ by SGD. It updates $v_w$ (a column of $W$) and $v_c, v_{n_1}, \ldots, v_{n_k}$ (rows of $W'$), nothing else.
> 5. **Effect.** Real pairs' dot products go up, random pairs' go down, so words sharing contexts end up close. (Outside the lecture: the gradient for $v_w$ is $(\sigma(v_c \cdot v_w) - 1)\, v_c + \sum_j \sigma(v_{n_j} \cdot v_w)\, v_{n_j}$.)

> [!exam]- Compare Word2Vec (CBOW and Skip-gram) with GloVe: what each optimises, what information each uses, how each handles frequency imbalance, and what the final embedding is.
> **Key points:** CBOW and Skip-gram predict within local windows; GloVe is weighted least squares on global log counts; subsampling against $f(X_{ij})$; final embedding $W$ or $W^\top + W'$ against $w_i + \tilde{w}_i$.
>
> - **CBOW:** predict the centre word from the average of $2n$ context embeddings, $h = \frac{1}{2n} W w_C$, $\hat{y} = \operatorname{softmax}(W'h)$, no non-linearity, cross entropy, SGD.
> - **Skip-gram:** predict each of the $2n$ context words from the centre word, $h = W w_I$, context words independent given $w_I$; in practice trained with negative sampling.
> - **GloVe (Pennington et al., 2014):** weighted least squares on log co-occurrence counts, $J = \sum f(X_{ij})(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$.
> - **Information used:** Word2Vec sees individual local windows, each an independent event, so it keeps rediscovering the same association. GloVe counts once over the whole corpus and fits those global counts.
> - **Frequency imbalance:** Word2Vec subsamples frequent words (and draws negatives from $P(w)^{0.75}$); GloVe uses $f(X_{ij})$ ($x_\text{max} = 100$, $\alpha = 0.75$), rising from zero and capped at 1.
> - **Final embedding:** Word2Vec typically $W$, or $W^\top + W'$; GloVe $w_i + \tilde{w}_i$.

> [!exam]- Why does constraining a crosslingual mapping to be orthogonal help? Give the objective, the two problems it fixes with a derivation for each, its solution, and the evidence.
> **Key points:** $W^\top W = I$ with unit vectors, an added constraint beyond the shared space hypothesis; a rigid map cannot overfit the seed pairs; $\lVert Wx - y \rVert^2 = 2 - 2\cos$ aligns training with retrieval; $W = UV^\top$ by SVD; unconstrained degrades with dimension, orthogonal better everywhere.
>
> 1. **Objective (Xing et al., 2015).** $\arg\min_W \sum_i \lVert W x_i - y_i \rVert^2$ subject to $W^\top W = I$, all embeddings unit length. A strictly orthogonal $W$ is square and only rotates and reflects. The shared space hypothesis itself asks only for a linear map.
> 2. **Overfitting.** An unconstrained $W$ can stretch and shear to fit the seed pairs and fail on the words outside the dictionary. Orthogonal $W$ preserves geometry, $\lVert Wx \rVert^2 = x^\top W^\top W x = \lVert x \rVert^2$, so the space moves rigidly.
> 3. **Objective mismatch.** Training uses Euclidean distance, retrieval cosine. For unit vectors and orthogonal $W$: $\lVert Wx - y \rVert^2 = 2 - 2\cos(Wx, y)$, so they coincide.
> 4. **Solution.** $U\Sigma V^\top = \operatorname{SVD}(YX^\top)$, $W^* = UV^\top$: exact, one SVD.
> 5. **Evidence (English to Spanish).** The unconstrained map gets worse as the dimension grows (likely overfitting, a reason not given in the lecture); the orthogonal map, which also normalises the vectors, is better at every dimension and improves slightly.
>
> Losing marks: claiming Euclidean training and cosine retrieval always agree, or saying the shared space hypothesis requires an orthogonal map.

> [!exam]- Explain the hubness problem in crosslingual word retrieval and how CSLS corrects it, including which term of the formula does the work.
> **Key points:** hubs are nearest neighbours of many points, worse in high dimensions; $\text{CSLS} = 2\cos - r_T - r_S$; $r_S(y)$ penalises hubs; $r_T(x)$ constant per source word; better retrieval with only $K$ to set.
>
> - **Hubness:** in high-dimensional spaces a few vectors become the nearest neighbour of many points regardless of real similarity; a general property of high-dimensional geometry that worsens with dimension. One target hub becomes the "translation" of many unrelated source words. Nearest neighbour is asymmetric.
> - **CSLS (Conneau et al., 2018):** $2\cos(x,y) - r_T(x) - r_S(y)$, with $r_T(x)$ the mean cosine of $x$ to its $K$ nearest target vectors and $r_S(y)$ the mean cosine of $y$ to its $K$ nearest mapped source vectors.
> - **Which term works:** $r_S(y)$ penalises a candidate close to many things. $r_T(x)$ is the same for every candidate of a fixed $x$, so it matters only when comparing scores across source words.
> - **Effect:** better than plain nearest neighbour on every language pair, with no tuning beyond $K$.

> [!exam]- Describe the full MUSE pipeline of Conneau et al. (2018) for aligning two embedding spaces without any bilingual signal, and justify each design choice.
> **Key points:** linear generator $W$ against MLP discriminator; linear so word correspondence follows the distribution match; label smoothing, 50k frequent words, re-orthogonalisation; CSLS-based checkpoint selection; Procrustes refinement on mutual nearest neighbours.
>
> 1. **Adversarial step.** Generator = the mapping $W$, a single $d \times d$ linear matrix initialised to the identity (or a random orthogonal matrix); discriminator = an MLP (2 hidden layers of 2048, ReLU, input dropout) telling $Wx_i$ from real $y_j$. Steps alternate with the other side frozen; the generator uses flipped labels.
> 2. **Linear generator.** A deep generator could match the target distribution while scrambling which word lands where.
> 3. **Label smoothing $s = 0.2$** keeps the discriminator from becoming pathologically overconfident.
> 4. **Only the 50k most frequent words**: rare words have poorly estimated embeddings.
> 5. **Re-orthogonalisation** $W \leftarrow (1+\beta)W - \beta(WW^\top)W$, $\beta = 0.01$, keeps $W$ near orthogonal.
> 6. **Checkpoint selection.** The adversarial loss is unreliable; instead average the cosine between the 10k most frequent source words and their CSLS nearest targets, and keep the best checkpoint.
> 7. **Refinement.** Synthetic dictionary of high-confidence mutual CSLS nearest neighbours among frequent words, $W = UV^\top$ by SVD, repeat. This step is called semi-supervised, but MUSE as a whole uses no bilingual data. Each component adds accuracy.

> [!exam]- What is fastText, how does it differ from Skip-gram with negative sampling, and where does it help and not help?
> **Key points:** sum of character n-grams of `<w>` (3 to 6) plus the word; same loss, only the centre word decomposed; vectors for unseen words; gains on word similarity (most for rich morphology and rare words) and on syntactic analogies; semantic analogies barely change.
>
> - **Problem:** one opaque vector per word type, so `run` and `runs` share nothing, morphologically rich languages split lemmas into many rare forms, and unseen words get no vector.
> - **Model:** $v_w = \sum_{g \in G_w} z_g$, with $G_w$ the n-grams of `<w>` for $n$ from 3 to 6 plus `<w>` itself; `<` and `>` make prefixes and suffixes distinct.
> - **Same as Skip-gram with negative sampling:** the loss; context vectors stay per word.
> - **Different:** every n-gram of $w$ gets the same gradient; the output is the n-gram table, so any word gets a vector.
> - **Evidence (Bojanowski et al., 2017):** best or tied on almost all word-similarity sets, most in morphologically rich languages and on English rare words; the only loss is English WS-353. Building unseen words from n-grams helps further. Analogy gains are syntactic; semantic analogies barely change, and for German get worse.

> [!exam]- How can the degree of isomorphism between two embedding spaces be measured without a dictionary, what did it show for English and Finnish, and what can it not show?
> **Key points:** Laplacian eigenvalues of nearest-neighbour graphs; $\Delta$ = sum of squared eigenvalue differences; no dictionary needed; highest for the failing pairs; low $\Delta$ does not prove isomorphism.
>
> 1. **Why:** isomorphism is true/false; Søgaard et al. want a degree.
> 2. **Eigenvector similarity:** nearest-neighbour graph per language (adjacency $A$), degree matrix $D$, Laplacian $L = D - A$, keep the largest $n\%$ of eigenvalues, $\Delta = \sum_i (\lambda_i^{(1)} - \lambda_i^{(2)})^2$.
> 3. **Reading it:** larger eigenvalues mean denser connectivity; larger $\Delta$ means more structurally different graphs. Eigenvalues ignore node labels, so no alignment is needed.
> 4. **Results:** English-Spanish has the lowest $\Delta$; English-Estonian, English-Finnish and English-Greek, where adversarial alignment fails, the highest.
> 5. **Finnish:** one lemma's many inflected forms cluster tightly, while English gives a flatter graph; no rotation can fix that, since a rotation keeps the neighbour graph.
> 6. **Limit:** near-isomorphic graphs have low $\Delta$, but low $\Delta$ does not imply near-isomorphism. $\Delta$ rules isomorphism out, never in.

> [!exam]- How should word embeddings be evaluated? Cover the evaluation axes, the word similarity and analogy tasks, and the limits of 2D visualisations.
> **Key points:** intrinsic against extrinsic, qualitative against quantitative; word similarity by Spearman, WS-353 relatedness against SimLex-999 similarity; analogies via $v_b - v_a + v_c$ scored by accuracy; 2D projections distort, t-SNE hyperparameters matter.
>
> - **Axes:** intrinsic (the embeddings by themselves) against extrinsic (downstream tasks such as MT, IR, summarisation); qualitative (selected examples, plots) against quantitative (an overall score). Independent of each other.
> - **Word similarity:** rank pairs by cosine, compare with human ratings by Spearman correlation. WS-353 annotates relatedness (`bread`/`butter` high), SimLex-999 similarity in the strict sense, so the same embeddings can rank differently on the two.
> - **Analogies:** *a : b :: c : X*, answer the word nearest to $v_b - v_a + v_c$, scored by accuracy; semantic and syntactic sets.
> - **Visualisation:** PCA (linear) or t-SNE (non-linear, local). Shows sense clusters and a consistent gender direction, but distorts distances, and t-SNE hyperparameters change the picture. Plots complement quantitative and extrinsic evaluation.

> [!exam]- Derive the closed-form solution of the unconstrained linear mapping $\min_W \sum_i \lVert W x_i - y_i \rVert^2$, state when it exists, and say how this mapping behaves as the embedding dimension grows.
> **Key points:** loss $\lVert WX - Y \rVert_F^2$; zero gradient gives $W = YX^\top(XX^\top)^{-1}$; needs $XX^\top$ invertible; SGD as the scalable alternative; accuracy falls as dimension grows.
>
> 1. **Matrix form.** Dictionary pairs as columns: $X$ is $d_1 \times n$, $Y$ is $d_2 \times n$. Loss $\lVert WX - Y \rVert_F^2$.
> 2. **Gradient** $2(WX - Y)X^\top$; set to zero: $WXX^\top = YX^\top$, so $W^* = YX^\top(XX^\top)^{-1}$ (the derivation is outside the lecture; the formula is in it).
> 3. **Existence.** $XX^\top$ must be invertible: at least $d_1$ linearly independent source vectors.
> 4. **Closed form against SGD.** Closed form is exact; SGD (Mikolov et al., 2013) is approximate, scales better and is more memory-friendly.
> 5. **Dimension.** Accuracy falls as $d$ grows. A likely reason, not given in the lecture: $W$ has $d^2$ free parameters for the same dictionary, so it stretches and shears to fit the anchors and generalises poorly.

## Flashcards

> [!card]- What does "static" mean for Word2Vec, GloVe and fastText embeddings?
> **One vector per word type**, whatever the sentence: `bank` is the same in *river bank* and *bank account*.

> [!card]- What is the distributional hypothesis behind both count-based and prediction-based word vectors?
> A word is characterised by **the contexts it occurs in** (Firth: "the company it keeps").

> [!card]- What are the three properties of count-based context vectors?
> **High-dimensional**, **discrete** (counts), **sparse** (mostly zeros).

> [!card]- What is the main weakness of count-based context vectors besides sparsity?
> **No explicit criterion to separate signal from noise**: chance co-occurrences, and words like `the` that discriminate nothing.

> [!card]- What four properties distinguish prediction-based word embeddings from count-based context vectors?
> **Low-dimensional, dense, continuous**, and **learned by performing a prediction task**.

> [!card]- What is representation learning?
> Training where **learning good embeddings is the main objective**, the task being a pretext. In most models embeddings are by-products.

> [!card]- What is the difference between the CBOW and Skip-gram tasks?
> CBOW predicts **the centre word from its context**. Skip-gram predicts **each context word from the centre word**.

> [!card]- Are Word2Vec's input matrix $W$ and output matrix $W'$ tied together?
> **No, not necessarily**: $W' \neq W^\top$.

> [!card]- Which matrix is typically used as the Word2Vec embedding?
> **$W$** (its columns), or the combination $W^\top + W'$.

> [!card]- What independence assumption does Skip-gram make?
> The context words are **independent given the input (centre) word**.

> [!card]- In negative sampling, what is the probability that a (word, context) pair is real?
> $\sigma(v_c \cdot v_w)$: the **sigmoid of the dot product** of the context and word vectors.

> [!card]- Which tends to perform better: count-based or prediction-based word representations?
> **Prediction-based.** Count-based methods use global information directly but tend to perform worse.

> [!card]- What is the purpose of GloVe's weighting function $f(X_{ij})$?
> To **de-emphasise very rare (noisy) and very frequent (uninformative)** co-occurrences.

> [!card]- What is the difference between Word2Vec and GloVe in what they train on?
> Word2Vec: **individual local context windows**. GloVe: the **global co-occurrence matrix**.

> [!card]- How do Word2Vec and GloVe differ in handling frequency imbalance?
> Word2Vec **subsamples frequent words**. GloVe uses the **explicit weighting function** $f(X_{ij})$.

> [!card]- What is the difference between intrinsic and extrinsic evaluation of word embeddings?
> Intrinsic: the embeddings **by themselves**. Extrinsic: their usefulness **in downstream tasks**.

> [!card]- How is word similarity evaluated against human judgements?
> Rank word pairs by **cosine** and compare with human ratings by **Spearman (rank) correlation**.

> [!card]- What is the difference between the WS-353 and SimLex-999 benchmarks?
> WS-353 annotates **relatedness** (`bread`/`butter` high). SimLex-999 annotates **similarity** in the strict sense.

> [!card]- How is the answer to "a is to b as c is to X" computed with word embeddings?
> The word nearest by cosine to **$v_b - v_a + v_c$**, excluding the input words. Scored by accuracy.

> [!card]- Why should 2D t-SNE plots of embeddings not be taken at face value?
> Non-linear projections **distort distances**, and t-SNE **hyperparameters substantially change** the picture.

> [!card]- Why do morphologically rich languages suffer most from one vector per word type?
> Each lemma has **many inflected forms**, each a separate, rarer word with a worse-estimated vector.

> [!card]- Where does fastText gain most on word similarity over Word2Vec?
> In **morphologically rich languages**; it also helps **rare words**.

> [!card]- What are the three basic set-ups for training crosslingual embeddings?
> **Separately, then align spaces**; **together, then align regions**; on **word- or sentence-aligned data**.

> [!card]- Why is the crosslingual mapping assumed to be linear?
> Analogies are **already linear regularities** within one space, assumed to hold across spaces.

> [!card]- What bilingual signal does each type of crosslingual alignment use?
> Supervised: **thousands** of seed translations. Semi-supervised: **hundreds**, then bootstrapping. Unsupervised: **none**.

> [!card]- Why can seed dictionaries contain only single words?
> There are **no embeddings for multi-word expressions**, so phrases cannot be anchors.

> [!card]- What is the difference between solving the least-squares mapping in closed form and with SGD?
> Closed form is **exact**. SGD is **approximate** but scales better and saves memory; Mikolov et al. used SGD.

> [!card]- Which mapping does the name "Procrustes" refer to?
> Monz: the **general least-squares** problem. Literature and Conneau et al.'s tables: the **orthogonal** map solved by SVD.

> [!card]- How does linear-mapping translation accuracy change for rarer words?
> It **drops**: rarer words have worse embeddings.

> [!card]- Why does an unconstrained mapping $W$ overfit the seed dictionary?
> It can **stretch, shear and rotate** the space to fit the seed pairs, and generalises poorly.

> [!card]- What mismatch between training and inference does the unconstrained mapping have?
> Training minimises **Euclidean distance**, retrieval uses **cosine**. They agree only for unit-normalised vectors.

> [!card]- What constraint and normalisation define Xing et al.'s orthogonal mapping?
> **$W^\top W = I$**, with all embeddings normalised to **unit length**.

> [!card]- How is the orthogonal Procrustes mapping solved?
> **Exactly by SVD**: $U \Sigma V^\top = \operatorname{SVD}(Y X^\top)$, $W = U V^\top$.

> [!card]- How does the orthogonal mapping compare with the unconstrained one across embedding dimensions?
> **Better at every dimension**, and it **improves slightly** as dimension grows.

> [!card]- What is a hub in a high-dimensional embedding space?
> A vector that is **the nearest neighbour of many points regardless of actual similarity**.

> [!card]- How does CSLS retrieval compare with plain nearest neighbour on the same supervised mapping?
> **CSLS is better on every language pair**, with no tuning beyond $K$.

> [!card]- What is the chicken-and-egg problem of unsupervised crosslingual alignment?
> You need a **mapping to find translation pairs**, and **translation pairs to learn the mapping**.

> [!card]- In MUSE's adversarial training, what does the discriminator try to distinguish?
> **Mapped source vectors** $Wx$ from **real target vectors** $y$. The generator is $W$.

> [!card]- Why is MUSE's generator a single linear matrix rather than a deep network?
> A deep generator could **match the target distribution while scrambling which word maps where**.

> [!card]- What unsupervised criterion does MUSE use to select a checkpoint?
> The **average cosine** between the 10k most frequent source words and their **CSLS nearest** targets.

> [!card]- Is MUSE a supervised, semi-supervised or unsupervised method?
> **Unsupervised** as a whole; its refinement step is called semi-supervised.

> [!card]- How does fully unsupervised MUSE compare with supervised alignment for distant pairs (English with Russian, Chinese)?
> It **falls well short**.

> [!card]- What is the difference between head-marking and dependent-marking?
> Head-marking puts the grammatical marker **on the head**. Dependent-marking puts it **on the dependent**.

> [!card]- What is identical-word supervision for crosslingual alignment?
> A seed dictionary of **words spelled identically in both languages**. No human dictionary needed.

> [!card]- Why does unsupervised alignment fail for mixed- and double-marking languages, according to Søgaard et al.?
> Each lemma has **many distinct surface forms**, each its own lower-frequency embedding, so the spaces are non-isomorphic.

> [!card]- How does a domain mismatch between the two corpora affect identical-word supervision and adversarial alignment?
> Identical-word supervision **degrades but survives**. Adversarial alignment **collapses to about zero**.

> [!card]- How much do hyperparameter changes within one embedding algorithm (window, n-gram range) hurt unsupervised alignment?
> **Barely.**

> [!card]- How is eigenvector similarity $\Delta$ computed for two languages?
> **Sum of squared differences** between the largest Laplacian eigenvalues of their nearest-neighbour graphs.

> [!card]- What does a larger eigenvector similarity $\Delta$ mean?
> The graphs are **more structurally different**: further from isomorphism.

> [!card]- Why does Finnish give a high eigenvector similarity $\Delta$ against English?
> One lemma's inflected forms **cluster tightly**; English gives a **flatter, more evenly connected** graph.

> [!card]- True or false: In Skip-gram, the hidden layer is the average of the context word embeddings.
> **False.** That is CBOW. Skip-gram's hidden layer is the centre word's embedding.

> [!card]- True or false: CBOW passes its averaged context vector through a non-linear activation before the output layer.
> **False.** CBOW has no non-linearities.

> [!card]- True or false: In Skip-gram with negative sampling, a negative example keeps the input word and replaces the context word with a sampled word.
> **True.** Each of the $k$ negatives pairs the same input word with a word from the noise distribution.

> [!card]- True or false: Word2Vec's context size and number of negative samples can be smaller the more training data there is.
> **True.** More data allows a smaller context and a smaller negative set.

> [!card]- True or false: Raising the unigram distribution to the power 0.75 biases negative sampling towards frequent words.
> **False.** It biases sampling towards **rarer** words.

> [!card]- True or false: GloVe builds its co-occurrence matrix in a single pass over the corpus.
> **True.** Training then uses only the matrix.

> [!card]- True or false: GloVe's weighting function gives very frequent pairs a lower weight than medium-frequency pairs.
> **False.** It never decreases; frequent pairs get the cap, 1.

> [!card]- True or false: In fastText, only the centre word is split into character n-grams.
> **True.** Context vectors stay per word.

> [!card]- True or false: fastText can produce a vector for a word that never occurred in training.
> **True.** It sums the word's character n-gram vectors.

> [!card]- True or false: The shared space hypothesis says the two embedding spaces differ only by a rotation.
> **False.** It says a general linear map aligns them; orthogonality is Xing et al.'s added constraint.

> [!card]- True or false: Semi-supervised alignment starts from a seed dictionary of thousands of translations.
> **False.** It starts from **hundreds**; thousands is the supervised setting.

> [!card]- True or false: For unit-length vectors and an orthogonal $W$, minimising Euclidean distance is equivalent to maximising cosine similarity.
> **True.** $\lVert Wx - y \rVert^2 = 2 - 2\cos(Wx, y)$.

> [!card]- True or false: An orthogonal mapping can rotate, reflect and scale the source space.
> **False.** No scaling: only rotations and reflections.

> [!card]- True or false: The unconstrained least-squares mapping continuously improves as the embedding dimension grows.
> **False.** It gets worse as the dimension grows.

> [!card]- True or false: MUSE's generator is a deep non-linear network, as in a standard GAN.
> **False.** It is a single linear $d \times d$ matrix, deliberately.

> [!card]- True or false: In MUSE, the discriminator only sees vectors of the 50k most frequent words of each language.
> **True.** Rare words' embeddings are poorly estimated.

> [!card]- True or false: Fully unsupervised MUSE can match supervised alignment for English-Spanish.
> **True.** It matches or slightly beats supervised Procrustes with CSLS on close pairs.

> [!card]- True or false: English-Finnish unsupervised alignment fails because the Finnish corpus is too small.
> **False.** It stays at zero with a much larger Finnish corpus.

> [!card]- True or false: Estonian-Finnish aligns adversarially although neither language aligns with English.
> **True.** Structurally similar languages can align.

> [!card]- True or false: Søgaard et al. found that adversarial alignment fails for mixed- and double-marking languages (Estonian, Finnish, Greek).
> **True.** English-Estonian, -Finnish and -Greek give essentially zero.

> [!card]- True or false: Unsupervised alignment fails when Spanish uses fastText CBOW and English fastText Skip-gram, even with otherwise identical settings.
> **True.** Different base models give about zero.

> [!card]- True or false: In Søgaard et al.'s language comparison, identical-word supervision beats adversarial alignment on every pair involving English.
> **True.** From a small margin (English-Spanish) to working against not working (English-Estonian).

> [!card]- True or false: A rotation of one embedding space cannot change the eigenvector similarity $\Delta$.
> **True.** A rotation keeps the nearest-neighbour graph, so the eigenvalues.

> [!card]- True or false: A low eigenvector similarity $\Delta$ proves that two embedding spaces are near-isomorphic.
> **False.** Near-isomorphic spaces give a low $\Delta$, but not the other way round.

> [!card]- Multiple selection: which are true of negative sampling in Word2Vec? (A) It turns prediction into classifying real against noise pairs (B) Each observed pair gets $k$ negatives, typically 5 to 20 (C) It still normalises over the whole vocabulary for each example (D) Negatives are drawn uniformly from the vocabulary
> **A and B.** C: the cost drops to $k+1$ dot products. D: negatives come from $P(w)^{0.75}$.

> [!card]- Multiple selection: which are true of GloVe? (A) It fits dot products plus biases to log co-occurrence counts (B) Its weighting function caps at 1 for frequent pairs (C) It treats each context window as an independent event (D) Its final embedding sums the word and context vectors
> **A, B and D.** C: that is the Word2Vec weakness GloVe addresses.

> [!card]- Multiple selection: which are true of fastText? (A) A word's vector is the sum of its character n-gram vectors and the word itself (B) Context vectors are also built from n-grams (C) Its analogy gains over Word2Vec are on syntactic analogies (D) It returns a table of n-gram vectors
> **A, C and D.** B: only the centre word is decomposed.

> [!card]- Multiple selection: which are true of hubness and CSLS? (A) Hubness worsens as dimensionality increases (B) CSLS penalises candidates that are close to many mapped source vectors (C) CSLS requires extensive hyperparameter tuning (D) Nearest-neighbour retrieval is symmetric
> **A and B.** C: only $K$. D: a hub can be `cat`'s nearest neighbour without `cat` being the hub's.

> [!card]- Multiple selection: which are true of MUSE? (A) The discriminator and the mapping are trained in alternation, each with the other frozen (B) The mapping is kept near-orthogonal after each step (C) The adversarial loss is used to pick the final checkpoint (D) Refinement re-solves the mapping by SVD on mutual nearest neighbours
> **A, B and D.** C: the checkpoint is picked by average CSLS-neighbour cosine.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L04 - Subword Segmentation]] (subword units; fastText's character n-grams are the embedding-side answer to the same vocabulary problem)
- **Next:** [[MNLP-L06 - Contextual Embeddings]] (vectors that depend on the sentence, built with [[Recurrent Neural Network (RNN)|RNNs]], [[Self-Attention]] and [[Transformers]], the basis of [[Large Language Models (LLM)]])
- **Later:** [[MNLP-L07 - Crosslingual NLP]] (crosslingual transfer, building on the aligned spaces of this lecture)
- **Related concepts:** [[Word Embeddings]] · [[Negative Sampling]] · [[Contrastive Learning]] · [[Vector Space Model]] · [[Bag of Words]] · [[Tokenization]]
- **Source:** `embeddings_static.pdf`, Christof Monz, MNLP: Static Embeddings, slides 1 to 71
- **Primary references:** Firth (1957) · Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* and *Distributed Representations of Words and Phrases and their Compositionality* (Word2Vec, negative sampling) · Mikolov, Le and Sutskever (2013), *Exploiting Similarities among Languages for Machine Translation* (linear translation matrix) · Pennington, Socher and Manning (2014), *GloVe: Global Vectors for Word Representation* · Wattenberg, Viégas and Johnson (2016), *How to Use t-SNE Effectively* · Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information* (fastText) · Xing et al. (2015), *Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation* · Conneau et al. (2018), *Word Translation Without Parallel Data* (MUSE, CSLS) · Søgaard, Ruder and Vulić (2018), *On the Limitations of Unsupervised Bilingual Dictionary Induction*
