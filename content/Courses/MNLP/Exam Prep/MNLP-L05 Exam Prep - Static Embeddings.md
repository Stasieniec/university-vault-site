---
type: exam-prep
course: 5204MNLP6Y
lecture: 5
status: complete
---

# MNLP-L05 Exam Prep - Static Embeddings

> [!abstract] What the exam can ask here
> Statements turn on what each method predicts or fits, what is shared or constrained, and which way a result goes. Traps: swapped pairs (CBOW and Skip-gram, WS-353 and SimLex-999, supervised and semi-supervised, $r_T$ and $r_S$), absolute words ("every", "always", "continuously") on findings with an exception, and the conditions under which unsupervised alignment fails (language structure, domain, embedding model; data size alone does not explain it).
> Explanations in the [[MNLP-L05 - Static Embeddings|lecture note]], numbers on the [[MNLP - Cheat Sheet|cheat sheet]].

## Count vectors and prediction-based embeddings

- Two types of approach: **count-based** (distributional semantics) and **prediction-based** (neural embeddings).
- Count model: context vocabulary $V_c$, target vocabulary $V_t$ (**can be the same** as $V_c$), counts within $n$ words to the **left or right**; the window is sliding but **can also be a sentence or document**.
- Count vectors are **high-dimensional, discrete and sparse**. Similarity is **cosine** in the context vector space.
- Advantages: an **unsupervised** way to induce similarity; dimensions are **interpretable** ($c_w[i]$ counts an actual word).
- Disadvantages: vectors stay relatively sparse **despite** stemming; **no explicit criterion** to tell signal from noise (chance co-occurrences, which context words discriminate meaning). idf, $\log \frac{N}{n_w}$, is offered as one way to weight context words.
- Embeddings are **low-dimensional** (e.g. 100 against $|V_c|$), **dense**, **continuous** and **learned by performing a task** (prediction).
- In most models embeddings are **by-products** of the task objective. **Representation learning** is when learning good embeddings is the main objective.

> [!warning] Traps
> - *"Stemming makes count-based context vectors dense."* False: they remain relatively sparse despite stemming.
> - *"Interpretable dimensions are a disadvantage of count-based vectors."* False: listed as an advantage.
> - *"The context vocabulary and the target vocabulary of a count model must be disjoint."* False: they can be the same.
> - *"Word embeddings are high-dimensional, discrete and sparse."* False: that describes count vectors.

## CBOW and Skip-gram

- **CBOW** predicts the word at position $t$ **from** the $n$ words to its left and $m$ to its right. Typically $n = m$, $n \in \{2, 5, 10\}$. With $m = 0$ it would resemble n-gram language modelling.
- CBOW is a **feed-forward** network, **simpler than the PNLM**, focused on learning the embeddings, with the embedding layer brought **closer to the output**.
- CBOW has **no non-linearities** and **one hidden layer**: $h = \frac{1}{2n} W w_C$, where $w_C$ sums the context words' one-hot vectors. So $h$ is the **average** of the context embeddings (the architecture figure labels it "SUM") and word order is lost.
- Output $\hat{y} = \operatorname{softmax}(W'h)$. Loss **cross entropy**, training **SGD**, for both CBOW and Skip-gram.
- $W$ is $|h| \times |V|$, $W'$ is $|V| \times |h|$, and they are **not (necessarily) shared**: $W' \neq W^\top$.
- Word $i$'s embedding is column $i$ of $W$ or row $i$ of $W'$. **Typically $W$**, or both combined: $W_s = W^\top + W'$.
- **Skip-gram** takes the word at $t$ and predicts the words from $t-n$ to $t+n$. Hidden layer $h = W w_I$, the **input word's** embedding, with **$2n$ output layers**, each $\operatorname{softmax}(W'h)$.
- Skip-gram treats the context words as **independent given the input word**: $p(w_{t-n} \ldots w_{t+n} \mid w_I) \propto \prod_{i \neq t} p(w_i \mid w_I)$.

| | CBOW | Skip-gram |
|---|---|---|
| Input | the context words | the centre word |
| Predicts | the centre word | each context word |
| Hidden layer | average of context embeddings | centre word's embedding |
| Output layers | one | $2n$ |

> [!warning] Traps
> - *"Skip-gram predicts the centre word from the average of its context embeddings."* False: that is CBOW.
> - *"Like the PNLM, CBOW passes the hidden layer through a non-linearity."* False: no non-linearities; simpler than the PNLM.
> - *"Word2Vec ties the output matrix to the input matrix, $W' = W^\top$."* False: not necessarily shared.
> - *"CBOW is sensitive to the order of the context words."* False: the context vectors are summed.
> - *"Word2Vec embeddings can only be read from the output matrix $W'$."* False: typically $W$, or $W^\top + W'$.

## Negative sampling

- Both CBOW and Skip-gram **benefit from large amounts of data**, and computing the **full output layer** for every example becomes the problem.
- Negative sampling makes it a **classification task**: tell words that **do** occur in the input word's context from words that **do not**.
- Per observed pair: **1 positive** example (ground truth) and **$k$ negatives** from a **random noise distribution**.
- $p(D=1 \mid c, w) = \sigma(v_c \cdot v_w)$, a **sigmoid of a dot product** ($v_w = Ww$, $v_c = W'^\top c$), and $p(D=0) = \sigma(-v_c \cdot v_w)$. Maximise $\sum \log \sigma(v_c \cdot v_w)$ over observed pairs plus $\sum \log \sigma(-v_c \cdot v_w)$ over noise pairs.
- Skip-gram in practice: **5 to 20** negatives per observed pair, drawn from the **unigram** distribution raised to **0.75**, which biases sampling **towards rarer words**.
- Context size typically **2 to 5**.
- **The more data, the smaller** the context and the negative sample set.

> [!warning] Traps
> - *"Negative samples are drawn uniformly from the vocabulary."* False: from the unigram distribution raised to 0.75.
> - *"The 0.75 exponent makes frequent words more likely as negatives."* False: it biases towards rarer words.
> - *"The more training data, the more negative samples per pair are needed."* False: more data, fewer negatives and a smaller context.
> - *"Negative sampling scores a pair with a softmax over the vocabulary."* False: a sigmoid of the dot product; avoiding the full output layer is the point.

## GloVe and Word2Vec

- **Count-based** methods (directly, or global matrix factorisation such as LSA) use **global** co-occurrence counts directly but **tend to perform worse**.
- **Prediction-based** methods (local windows) treat **each context window as an independent event**, rediscovering associations repeatedly, but **still tend to outperform** count-based ones.
- **GloVe** (Pennington et al., 2014) **combines** global co-occurrence statistics with local prediction-based updating.
- $J = \sum_{(i,j)} f(X_{ij}) \cdot (w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$: **weighted least-squares regression on log counts**, with a word vector $w_i$, a separate **context** vector $\tilde{w}_j$, and **bias terms**. $X_{ij}$ counts word $j$ in the context of word $i$.
- $f(x) = (x/x_{max})^\alpha$ if $x < x_{max}$, else 1; typically $x_{max} = 100$, $\alpha = 0.75$.
- $f$ **de-emphasises very rare (noisy) and very frequent (uninformative)** co-occurrences: it rises smoothly **from zero** and **caps at 1**, so frequent pairs cannot take over.
- Co-occurrence matrix built in **a single pass**, counts often weighted by **1/distance**; vectors start **random** and are updated with **AdaGrad**.
- Final embedding: **$w_i + \tilde{w}_i$** (sum of the two vector sets).

| | Word2Vec | GloVe |
|---|---|---|
| Trains on | local context windows | global co-occurrence matrix |
| Objective | predict context words | weighted least squares on log counts |
| Frequency imbalance | subsampling frequent words | weighting function $f(X_{ij})$ |
| Interpretability | indirect | directly fits log counts |

> [!warning] Traps
> - *"GloVe treats each context window as an independent event."* False: that is the prediction-based weakness GloVe addresses.
> - *"Count-based methods tend to outperform prediction-based ones."* False: the reverse.
> - *"GloVe's weighting gives the most frequent pairs the lowest weight."* False: the weight never falls; it caps at 1.
> - *"GloVe handles frequency imbalance by subsampling frequent words."* False: that is Word2Vec; GloVe uses $f(X_{ij})$.
> - *"GloVe's final embedding is the word vector $w_i$ alone."* False: $w_i + \tilde{w}_i$.

## Evaluating embeddings

- **Intrinsic**: how good the embeddings are by themselves. **Extrinsic**: how useful they are downstream (summarisation, MT, IR).
- **Qualitative**: examine selected examples. **Quantitative**: an overall, task-dependent score.
- Word similarity: rank word pairs by similarity and compare with human judgements by **Spearman** correlation.
- **WS-353 annotates relatedness**; **SimLex-999 annotates similarity**. The pair lists mix all kinds of similarity (synonyms, topical, unrelated).
- Analogies (Paris : France :: Berlin : X) are scored by **accuracy**; $X = v_{king} - v_{man} + v_{woman}$; the set also includes **syntactic** analogies (acquired : acquire :: tried : X).
- Visualising needs a 2D projection, **t-SNE or PCA**.
- 2D plots should not be taken at face value: projections can be **non-linear**, and **t-SNE hyperparameters have a substantial impact**. They **complement, but do not substitute**, quantitative and extrinsic evaluation.

> [!warning] Traps
> - *"SimLex-999 annotates relatedness and WS-353 strict similarity."* False: swapped.
> - *"Word similarity is scored by accuracy."* False: Spearman correlation; accuracy is for analogies.
> - *"Analogy benchmarks test only semantic relations."* False: they include syntactic analogies.
> - *"Measuring how much embeddings improve MT is an intrinsic evaluation."* False: extrinsic.
> - *"A t-SNE plot can replace quantitative evaluation."* False: it complements and does not substitute.

## fastText

- Word2Vec and GloVe assign **one opaque vector per word type**: `run`, `runs`, `running`, `runner` get (potentially) unrelated vectors.
- **Morphologically rich languages suffer most** (Finnish, Turkish, Russian), and there is **no fallback for unseen (OOV) words**.
- Remedies: segment words and embed each segment, or a **sliding character window**: fastText (Bojanowski et al., 2017).
- fastText is Skip-gram with negative sampling where the **centre word's vector is the sum of its character n-gram vectors**: n-grams of `<w>` with $n$ from **3 to 6** by default, **plus the whole word** `<w>`.
- **Context vectors stay one per word**; only the centre word is decomposed.
- Every n-gram of the word gets **the same gradient**.
- fastText **returns the n-gram vectors**, so any word, **seen or unseen**, gets a vector as a sum of n-grams.
- Word similarity against Skip-gram and CBOW: best or tied on every set except **English WS-353** (CBOW wins there); gains in rich-morphology languages (German, Russian) and on English **rare words**.
- Analogies: clear gains on **syntactic** analogies; **semantic** analogies barely change and **drop for German**.

> [!warning] Traps
> - *"fastText also builds the context word vectors from character n-grams."* False: only the centre word.
> - *"fastText cannot represent words unseen in training."* False: it sums their n-gram vectors.
> - *"fastText's largest analogy gains are on semantic analogies."* False: syntactic; semantic barely move, German drops.
> - *"fastText beats Word2Vec on every word similarity dataset."* False: not on English WS-353.
> - *"fastText represents a word only by its n-grams, without a vector for the word itself."* False: `<w>` is in the set.

## The shared space hypothesis and bilingual signal

- Goal: translation-equivalent words end up geometrically close. Hope: **large monolingual corpora** and only a **small bilingual signal**.
- Three set-ups: train **separately, then align the spaces**; train **together** (benefiting from word overlap), then align the **regions**; train on data **aligned at word or sentence level**. **Either way, some alignment has to take place.**
- **Shared space hypothesis**: separately trained languages still encode **similar relational structure**. There is an **approximate isomorphism**: the overall shape of the point cloud is similar, the **coordinates and orientations can differ** (the cloud may be rotated and **scaled**).
- It **is a hypothesis**. If true, **a single** transformation, **a general linear mapping**, aligns the spaces.
- **Linear transformation hypothesis**: a **strong assumption**, motivated by analogies being **already linear regularities within one space**. It reduces alignment to **regression**.
- **Supervised**: seed dictionary of **thousands** of pairs. **Semi-supervised**: **hundreds**, then **bootstraps** new translations by statistical confidence. **Unsupervised**: **no bilingual signal**, adversarial training.
- Seed dictionaries are often the **top few thousand most frequent** words; automatic translation can be used **with caution**; they are **limited to words** (no embeddings for multi-word expressions). Each pair is one anchor; the mapping must **generalise beyond the anchors**.
- Dictionary sources: human-compiled dictionaries, **word-aligned** human sentence translations, and words that **occur in several languages**.

> [!warning] Traps
> - *"The shared space hypothesis says the spaces differ only by a rotation."* False: a general linear map; orientation and scale can differ.
> - *"The shared space hypothesis is an established property of embedding spaces."* False: the slides stress it is a hypothesis.
> - *"Semi-supervised alignment starts from thousands of seed translations."* False: hundreds; thousands is supervised.
> - *"Training all languages together removes the need for alignment."* False: either way some alignment has to take place.
> - *"Seed dictionaries can anchor multi-word phrases."* False: limited to words.

## Supervised mapping: unconstrained against orthogonal

- $W^* = \arg\min_W \sum_i \lVert Wx_i - y_i \rVert^2$ over dictionary pairs. The slides call this **the Procrustes algorithm** (Conneau et al.'s paper reserves "Procrustes" for the orthogonal map solved by SVD).
- Closed form: $W^* = YX^\top(XX^\top)^{-1}$, **exact**. **Mikolov et al. (2013) used SGD**: scales better with large dimensions and dictionaries, approximate but **memory-friendlier**.
- Mikolov's translation matrix **beats edit distance and word co-occurrence** baselines; **combined with edit distance** it does better still. Accuracy **rises with more training words** (levelling off) and **falls for less frequent words**.
- An unconstrained $W$ can **stretch, shear and rotate**; nothing preserves distances and angles, so it **overfits the seed dictionary** and does not generalise to the words outside it (the majority).
- **Mismatch**: training minimises **Euclidean** distance, retrieval uses **cosine**; **not equivalent unless vectors are unit-normalised**.
- **Xing et al. (2015)**: add $W^\top W = I$ (**orthogonal**) and normalise embeddings to **unit length**. Only **rotations and reflections**, **no scaling or shearing**: angles and distances are preserved. Solved **exactly with SVD**.
- English-Spanish results: as the dimension grows (300, 500, 700), the unconstrained map **gets worse** and the orthogonal map's P@1 **improves slightly**; the orthogonal map is **better in every setting**, including different English and Spanish dimensionalities.

| | Mikolov et al. (2013) | Xing et al. (2015) |
|---|---|---|
| Constraint | none | orthogonal |
| Vectors | unnormalised | unit length |
| Solved by | SGD (closed form exists) | SVD, exact |
| Larger dimension | worse | slightly better |

> [!warning] Traps
> - *"An orthogonal mapping can rescale the source space to match the target."* False: rotations and reflections only.
> - *"The unconstrained mapping continuously improves as the dimension grows."* False: it gets worse.
> - *"Euclidean training and cosine retrieval always agree."* False: only for unit-normalised vectors.
> - *"Mikolov et al. used the closed form because SGD does not scale."* False: they used SGD because it scales better.
> - *"The orthogonal mapping must be approximated by gradient descent."* False: exact via SVD.

## Hubs and CSLS

- **Hubs**: in nearest-neighbour retrieval at inference, a small number of vectors become the nearest neighbour of **many** points **regardless of actual similarity**, so a target hub is returned as the "translation" of many unrelated source words.
- Hubness is a known phenomenon of **high-dimensional geometry**, **not specific to embeddings**, and it **worsens as dimensionality increases**.
- **CSLS** (Conneau et al., 2018): $2\cos(x,y) - r_T(x) - r_S(y)$. $r_T(x)$ is the average cosine between $x$ and its $K$ nearest neighbours **in the target space**; $r_S(y)$ is the average cosine between $y$ and its $K$ nearest neighbours **in the source space**.
- Subtracting them **penalises a candidate that is close to lots of things** (a hub).
- CSLS **significantly increases** translation retrieval accuracy **without parameter tuning**.
- Results (supervised Procrustes, fastText embeddings): CSLS beats plain nearest neighbour on **every** pair; ISF is about level with CSLS and wins two pairs (Chinese-English, Esperanto-English).

> [!warning] Traps
> - *"Hubness is specific to crosslingual embeddings."* False: a general high-dimensional phenomenon.
> - *"Hubness decreases as dimensionality increases."* False: it worsens.
> - *"CSLS boosts candidates that are close to many points."* False: it penalises them.
> - *"$r_T(x)$ averages $x$'s similarity to its neighbours in the source space."* False: target space; $r_S(y)$ uses the source space.
> - *"CSLS needs extensive hyperparameter tuning."* False: no parameter tuning.
> - *"CSLS beats ISF on every language pair."* False: it beats plain nearest neighbour on every pair; ISF wins two.

## Adversarial alignment (MUSE)

- Why unsupervised: many low-resource pairs lack even a small dictionary, and dictionaries may not cover **inflected forms**.
- **Chicken and egg**: a mapping is needed to find translation pairs, and pairs to learn the mapping. Conneau et al. (2018) bootstrap a **rough** alignment adversarially from **distributional structure alone**.
- GAN framework: the **generator is the mapping $W$**, making mapped source vectors indistinguishable from real target vectors. The **discriminator**, trained **simultaneously**, tells mapped source vectors (fake) from real target vectors (real). They are trained **in opposition**.
- Unlike a standard GAN, the generator is **not a deep network**: a single $d \times d$ matrix, **linear, no non-linearity, deliberately**. A deep generator could make the clouds indistinguishable **while scrambling** which source word maps where.
- Discriminator: MLP with **2 hidden layers of 2048 units**, **ReLU**, **dropout on the input layer**, two-class output with **label smoothing $s = 0.2$** (targets $\{s, 1-s\}$ against overconfidence).
- Vectors fed to the discriminator come from the **50k most frequent words** of each language: rare words have poorly estimated embeddings.
- Algorithm: $W$ starts as the **identity** (or a random orthogonal matrix). A discriminator step with **$W$ frozen** alternates with a generator step with the **discriminator frozen**, which uses **flipped labels**. After each update, **re-orthogonalise**: $W \leftarrow (1+\beta)W - \beta(WW^\top)W$, $\beta = 0.01$.
- Model selection: the **adversarial loss is not a reliable indicator**; the discriminator loss can look fine while the mapping is poor, and there is no validation dictionary. Proxy: take the **10k most frequent** source words, find each one's **CSLS nearest neighbour** with the current $W$, and use the **average cosine**.
- **Refinement** (titled "semi-supervised" on the slides): the adversarial $W$ is rough (global orientation roughly right). Then build a **synthetic dictionary** of high-confidence **mutual** CSLS nearest neighbours among **frequent** words, **re-solve $W$ exactly by SVD**, and **repeat** (self-learning). No human dictionary is involved.
- Results: CSLS and refinement each help; adversarial with refinement and CSLS is the best unsupervised setting. It is **on par with or better than** supervised Procrustes-CSLS for English with Spanish, French and German, and **falls well short** for English with Russian and Chinese.

> [!warning] Traps
> - *"MUSE's generator is a deep non-linear network, as in standard GANs."* False: a single linear matrix, deliberately.
> - *"The discriminator is trained only after the mapping has converged."* False: simultaneously, alternating steps.
> - *"The adversarial loss is a reliable criterion for picking the checkpoint."* False: the average CSLS-neighbour cosine is used.
> - *"Label smoothing makes the discriminator more confident."* False: it prevents overconfidence.
> - *"The discriminator sees the whole vocabulary, rare words included."* False: the 50k most frequent words.
> - *"The refinement step needs a small human seed dictionary."* False: it induces its own from mutual nearest neighbours.
> - *"Unsupervised MUSE is always worse than supervised alignment."* False: on par or better on close pairs.

## Where unsupervised alignment breaks

- Marking: **dependent-marking** puts the marker on the dependent, **head-marking** on the head, **double-marking** on both, **zero-marking** on neither. Possessive: head = possessed noun, dependent = possessor. Clause: head = verb, dependents = subject, object. German is mainly dependent-marking (verb agreement adds some head-marking).
- Søgaard et al. (2018) note that Conneau et al.'s languages are mostly dependent-marking and none is agglutinative, and add Estonian and Finnish (mixed-marking, agglutinative), Greek (double-marking), Hungarian, Polish and Turkish.
- **Identity** baseline: supervised alignment from a seed dictionary of identically spelled words.
- MUSE **fails dramatically** (P@1 about zero) for **English-Estonian, English-Finnish, English-Greek**: the mixed- and double-marking languages.
- Identity **outperforms** adversarial on **every pair with English**. **Estonian-Finnish** is the exception: adversarial aligns them and beats identity.
- Why: in mixed- or double-marking languages grammatical information is spread over **more word forms**; a lemma has many more surface forms, each with its own embedding and **lower frequency**, so mapping to a language that marks by word order becomes **non-isomorphic**.
- **Not only data size**: retraining Finnish on a 1.7-billion-word corpus left English-Finnish at **0.0**. Estonian and Finnish (both mixed-marking, agglutinative) align: **pairing structurally dissimilar languages** is the problem.
- Results differ **per part of speech** (English-Hungarian: nouns and verbs worst).
- **Domain** (English-Spanish; EuroParl, Wikipedia, EMEA, 1.1M sentences each): within one domain both methods work. **Across domains adversarial induction does not work at all**; identity drops sharply but stays above zero.
- **Embedding model** (both fastText): **variations within a model** (window, n-gram range) **do not matter much**; **different base models** (Skip-gram against CBOW) **do not work at all**.

> [!warning] Traps
> - *"English-Finnish fails because Finnish Wikipedia is small; a much larger Finnish corpus fixes it."* False: still 0.0 at 1.7 billion words.
> - *"Adversarial alignment fails for every pair involving Finnish."* False: Estonian-Finnish aligns.
> - *"Identity supervision beats adversarial training on every language pair tested."* False: on every pair with English; Estonian-Finnish is the exception.
> - *"Adversarial alignment across domains still works for similar languages such as English and Spanish."* False: it fails completely.
> - *"Skip-gram for one language and CBOW for the other is fine as long as both come from fastText."* False: different base models do not work at all.
> - *"Changing the window size for one language breaks unsupervised alignment."* False: variations within a model barely matter.
> - *"In a dependent-marking possessive, the possessed noun carries the marker."* False: the possessor (the dependent) does.

## Eigenvector similarity

- Isomorphism is a **strict true/false** criterion; eigenvector similarity measures **degrees** of it.
- Build each language's **nearest-neighbour graph** (adjacency matrix $A$), form the **Laplacian** $L$ by subtracting the diagonal neighbour-count matrix $D$ from $A$ (slide wording; Søgaard et al. write $L = D - A$), and keep the **largest $n\%$** of its eigenvalues.
- Laplacian eigenvalues show **how connected** nodes are: bigger values, denser connectivity.
- $\Delta$ = **sum of squared differences** between the two languages' Laplacian eigenvalues. **Bigger $\Delta$**, more structurally different, further from isomorphism.
- It **does not require knowing the alignment** between the languages.
- High $\Delta$: words cluster differently in the two languages, **no rotation can fix this**, and the isomorphism assumption does not hold (Finnish: a lemma's many inflected forms cluster tightly; English gives a flatter graph).
- **One direction only**: (near-)isomorphic graphs have a low $\Delta$, but a low $\Delta$ does **not** mean near-isomorphic.
- "Eigenvector similarity is **poor for most languages**"; English-Spanish has the lowest $\Delta$, the three failing English pairs the highest.

> [!warning] Traps
> - *"A low $\Delta$ shows that two spaces are near-isomorphic."* False: the inference only works the other way.
> - *"Eigenvector similarity needs a seed dictionary to match the nodes."* False: no alignment needed.
> - *"A high $\Delta$ can be removed by finding a better rotation."* False: no rotation can fix it.
> - *"A larger $\Delta$ means the languages are structurally more similar."* False: more different.

## Practice questions

**Q1.** Which of the following statements about Word2Vec are true? (multiple answers possible)
- [A] In CBOW, the hidden layer combines the embeddings of the context words and applies no non-linearity.
- [B] Skip-gram has one output layer per context position, all using the same output matrix $W'$.
- [C] Word2Vec requires $W' = W^\top$, so that each word has exactly one embedding.
- [D] Skip-gram treats the context words as independent given the input word.

> [!example]- Answer
> **A, B and D.** C: $W$ and $W'$ are not necessarily shared; the embedding is typically $W$, or $W^\top + W'$.

**Q2.** Negative sampling is used to train Skip-gram efficiently. Which of the following statements are true? (multiple answers possible)
- [A] It replaces predicting a word over the whole vocabulary with classifying whether a word occurs in the input word's context.
- [B] Negatives are drawn from the unigram distribution raised to the power 0.75.
- [C] The exponent 0.75 increases the share of the most frequent words among the negatives.
- [D] The more training data is available, the more negative samples per observed pair are needed.

> [!example]- Answer
> **A and B.** C: the exponent biases sampling towards rarer words. D: the more data, the smaller the negative sample set (and the context).

**Q3.** GloVe combines count-based and prediction-based ideas. Which of the following statements are true? (multiple answers possible)
- [A] Like Word2Vec, GloVe treats each context window as an independent training event.
- [B] GloVe's weighting function increases with the co-occurrence count up to $x_{max}$ and is 1 from there on.
- [C] GloVe fits the dot product of a word vector and a context vector, plus biases, to the log co-occurrence count.
- [D] GloVe handles frequency imbalance by subsampling frequent words.

> [!example]- Answer
> **B and C.** A: GloVe trains on the global co-occurrence matrix; independent windows are Word2Vec's weakness. D: subsampling is Word2Vec; GloVe uses the weighting function $f(X_{ij})$.

**Q4.** fastText makes word embeddings morphologically sensitive. Which of the following statements are true? (multiple answers possible)
- [A] Both the centre word and the context words are represented as sums of character n-gram vectors.
- [B] fastText can produce a vector for a word that never occurred in training.
- [C] Compared with Skip-gram and CBOW, fastText's analogy gains are mainly on semantic analogies.
- [D] Word similarity benchmarks such as WS-353 are scored by Spearman correlation with human judgements.

> [!example]- Answer
> **B and D.** A: only the centre word is decomposed; context vectors stay per word. C: the gains are on syntactic analogies; semantic ones barely change and drop for German.

**Q5.** Supervised crosslingual alignment maps one language's embedding space onto another's using a seed dictionary. Which of the following statements are true? (multiple answers possible)
- [A] The shared space hypothesis assumes that the two spaces differ only by a rotation.
- [B] Constraining the mapping to be orthogonal preserves distances and angles of the source space.
- [C] The unconstrained least-squares mapping continuously improves as the embedding dimensionality grows.
- [D] CSLS lowers the score of a candidate target word that is, on average, close to many source vectors.

> [!example]- Answer
> **B and D.** A: the hypothesis asks for a general linear mapping; orientation and scale may differ, and orthogonality is Xing et al.'s added constraint. C: its accuracy drops as the dimensionality grows.

**Q6.** Conneau et al. (2018) align embedding spaces without any bilingual signal, and Søgaard et al. (2018) test where this works. Which of the following statements are true? (multiple answers possible)
- [A] The generator is a single linear matrix, so that matching the two distributions cannot scramble which source word maps where.
- [B] Because no dictionary is available, the checkpoint is chosen by the lowest discriminator loss.
- [C] English-Finnish alignment succeeds once the Finnish embeddings are trained on a much larger corpus.
- [D] Alignment fails when one language's embeddings come from fastText CBOW and the other's from fastText Skip-gram.

> [!example]- Answer
> **A and D.** B: the adversarial loss is not a reliable indicator; the average cosine to CSLS nearest neighbours of the 10k most frequent source words is used. C: English-Finnish stays at 0.0 with a 1.7-billion-word Finnish corpus; structural dissimilarity is the problem.
