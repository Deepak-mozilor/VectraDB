# Information Retrieval Theory — Mathematical Foundations

This document covers the mathematical foundations behind VectraDB's retrieval algorithms, evaluation metrics, and design decisions.

---

## 1. Vector Space Model

In the Vector Space Model (VSM), documents and queries are represented as vectors in a high-dimensional space where each dimension corresponds to a feature (term, embedding dimension, etc.).

**Document-Term Matrix:**

Given a corpus of `N` documents and a vocabulary of `V` terms, the document-term matrix `A` is:

```
A ∈ ℝ^(N × V)
A[i][j] = weight of term j in document i
```

**Cosine Similarity Derivation:**

For two vectors **a** and **b**, cosine similarity measures the angle between them:

```
cos(θ) = (a · b) / (‖a‖ × ‖b‖)

where:
  a · b = Σᵢ aᵢbᵢ           (dot product)
  ‖a‖  = √(Σᵢ aᵢ²)          (L2 norm)
```

**Properties:**
- Range: [-1, 1] (for non-negative vectors: [0, 1])
- cos(θ) = 1: identical direction (maximum similarity)
- cos(θ) = 0: orthogonal (no similarity)
- cos(θ) = -1: opposite direction

**Why cosine over Euclidean?** Cosine similarity is invariant to vector magnitude. Two documents about the same topic but of different lengths will have similar cosine scores but different Euclidean distances. This makes cosine preferred for text retrieval where document length varies.

---

## 2. TF-IDF Formulation

### Term Frequency (TF)

Measures how often a term appears in a document. Multiple variants exist:

| Variant | Formula | Use Case |
|---------|---------|----------|
| Raw count | `tf(t,d) = f(t,d)` | Simple, frequency-proportional |
| Sublinear | `tf(t,d) = 1 + log(f(t,d))` if f > 0, else 0 | Diminishes impact of high-frequency terms |
| Boolean | `tf(t,d) = 1` if t ∈ d, else 0 | Presence-only |

VectraDB uses **sublinear TF** by default to prevent common terms from dominating.

### Inverse Document Frequency (IDF)

Measures how rare a term is across the corpus. Rare terms are more discriminative.

```
idf(t) = log(N / df(t))

where:
  N     = total number of documents
  df(t) = number of documents containing term t
```

**Smoothed IDF** (used in VectraDB):

```
idf(t) = log(1 + N / df(t))
```

Smoothing prevents division by zero and reduces the impact of extremely rare terms.

### TF-IDF Weight

```
w(t, d) = tf(t, d) × idf(t)
```

High weight when a term is frequent in the document (high TF) but rare globally (high IDF).

### BM25 (Robertson-Sparck Jones)

BM25 generalizes TF-IDF with saturation and document length normalization:

```
BM25(t, d) = idf(t) × (f(t,d) × (k₁ + 1)) / (f(t,d) + k₁ × (1 - b + b × |d|/avgdl))

where:
  k₁    = term frequency saturation (typically 1.2–2.0)
  b     = length normalization (typically 0.75)
  |d|   = document length
  avgdl = average document length
```

BM25 is the de facto standard for sparse retrieval. VectraDB currently implements classic TF-IDF; BM25 can be added by extending the scoring module.

---

## 3. Approximate Nearest Neighbor (ANN) Complexity

### Brute Force

```
Time:  O(N × d)      — compare query against all N vectors of dimension d
Space: O(N × d)      — store all vectors
```

Guarantees 100% recall but is impractical for large datasets.

### HNSW (Hierarchical Navigable Small World)

HNSW builds a multi-layer navigable small-world graph:

**Construction:**

```
Time:  O(N × log(N) × M × ef_construction)
Space: O(N × M × L)

where:
  M              = max connections per node (default 16)
  ef_construction = beam width during construction (default 200)
  L              = expected number of layers ≈ log(N) / log(M)
```

**Search:**

```
Time:  O(ef × log(N) × M)
Space: O(ef)  (search state)

where:
  ef = beam width during search (default 50)
```

**Recall-Speed Tradeoff:**
- Increasing `ef` improves recall but slows search
- Increasing `M` improves recall and slows both construction and search
- The `ef_construction` parameter affects index quality but not search speed

### Product Quantization (PQ)

Reduces memory by compressing vectors:

```
Original: N × d × 4 bytes (float32)
Compressed: N × m bytes (m subspaces, 1 byte each)

Compression ratio: 4d/m (e.g., d=768, m=8 → 384× compression)
```

**Asymmetric Distance Computation (ADC):**

```
d(q, x̃) ≈ Σⱼ ‖qⱼ - cⱼ(xⱼ)‖²

where:
  qⱼ    = j-th subvector of query
  cⱼ(·) = codebook entry for j-th subspace
```

Pre-computing `‖qⱼ - cⱼ‖²` lookup tables makes distance computation O(m) instead of O(d).

### ES4D (Dimension-Level Early Termination)

VectraDB's ES4D optimizes HNSW by computing distance in shards:

```
L2²(a, b) = Σₛ (Σᵢ∈shard_s (aᵢ - bᵢ)²)
```

After computing each shard `s`, if the partial sum already exceeds the current k-th best distance, the remaining shards are skipped. Combined with dimension reordering (high-variance dimensions first), this can skip 30-60% of computation.

---

## 4. Evaluation Metrics

### Recall@k

Fraction of relevant documents found in the top-k results:

```
Recall@k = |{relevant} ∩ {retrieved@k}| / |{relevant}|
```

- Range: [0, 1]
- Measures: completeness (did we find what we should have?)

### Precision@k

Fraction of top-k results that are relevant:

```
Precision@k = |{relevant} ∩ {retrieved@k}| / k
```

- Range: [0, 1]
- Measures: accuracy (are results useful?)

### F1 Score

Harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Mean Reciprocal Rank (MRR)

Average of reciprocal ranks across queries:

```
MRR = (1/Q) × Σᵢ (1 / rankᵢ)

where rankᵢ = position of first relevant result for query i
```

- Range: (0, 1]
- Emphasizes: how quickly the first relevant result appears

### Normalized Discounted Cumulative Gain (NDCG)

Handles graded relevance (not just binary relevant/irrelevant):

```
DCG@k  = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)
IDCG@k = DCG@k with ideal ordering (sort by relevance descending)
NDCG@k = DCG@k / IDCG@k
```

- Range: [0, 1]
- NDCG = 1.0 means perfect ranking order

### Mean Average Precision (MAP)

Average precision across all relevant items:

```
AP = (1/|relevant|) × Σₖ (Precision@k × rel(k))

where rel(k) = 1 if item at position k is relevant, else 0

MAP = (1/Q) × Σᵢ APᵢ
```

- Range: [0, 1]
- Considers the entire ranking, not just top-k

---

## 5. Curse of Dimensionality

### Distance Concentration

As dimensionality `d` increases, the ratio of nearest to farthest distances converges to 1:

```
lim(d→∞) [max_dist(q, X) - min_dist(q, X)] / min_dist(q, X) → 0
```

This means all points become nearly equidistant, making similarity search less meaningful.

### Johnson-Lindenstrauss Lemma

For any set of `N` points in ℝᵈ and any ε ∈ (0, 1), there exists a linear map f: ℝᵈ → ℝᵏ where:

```
k = O(log(N) / ε²)
```

such that for all pairs:

```
(1 - ε) ‖u - v‖² ≤ ‖f(u) - f(v)‖² ≤ (1 + ε) ‖u - v‖²
```

**Implication:** Distances can be approximately preserved in much lower dimensions. This justifies dimensionality reduction (PCA, random projection) and validates that ANN algorithms can work in moderate dimensions.

### Practical Impact on VectraDB

| Dimension | Behavior | Recommendation |
|-----------|----------|----------------|
| d < 100 | Distance separation is strong | Brute force or HNSW with low `ef` |
| 100 < d < 1000 | ANN algorithms work well | HNSW with tuned `ef`, consider PQ |
| d > 1000 | Distance concentration begins | Use ES4D (early termination helps), increase `ef` |
| d > 2000 | Diminishing returns | Consider dimensionality reduction before indexing |

---

## 6. Hybrid Retrieval Theory

### Why Combine Dense and Sparse?

Dense retrieval (embedding-based) excels at semantic similarity but can miss exact keyword matches. Sparse retrieval (TF-IDF/BM25) excels at keyword matching but misses semantic relationships.

```
Dense:  "automobile" ≈ "car"     ✓  (semantic)
        "ML" ≈ "machine learning" ✓  (learned association)
        "error code 42" ≈ ???    ✗  (no semantic equivalent)

Sparse: "error code 42" = exact  ✓  (keyword match)
        "automobile" ≈ "car"     ✗  (different tokens)
```

### Reciprocal Rank Fusion (RRF)

Combines ranked lists without requiring score calibration:

```
RRF_score(d) = Σᵣ 1 / (k + rankᵣ(d))

where:
  k  = smoothing constant (default 60)
  rankᵣ(d) = rank of document d in ranker r
```

**Properties:**
- Robust to score distribution differences between rankers
- Does not require score normalization
- k=60 (from original Cormack et al. 2009) works well empirically

### Weighted Sum Fusion

When scores are calibrated (same scale):

```
score(d) = α × dense_score(d) + (1-α) × sparse_score(d)
```

Requires careful α tuning (typically α ∈ [0.5, 0.8] for dense-heavy blends).

---

## 7. Problematic Models and Failure Modes

### Embedding Collapse

When an embedding model maps semantically different texts to nearly identical vectors:

```
embed("The cat sat on the mat") ≈ embed("A dog ran through the park")
```

**Causes:** Under-trained models, domain mismatch, excessive dimensionality reduction.
**Detection:** Measure average pairwise cosine similarity across a diverse corpus. If mean > 0.8, collapse is likely.
**Mitigation:** Use domain-specific fine-tuned models; increase embedding dimension.

### Isotropy Issues

Embeddings from transformer models often occupy a narrow cone in vector space rather than being uniformly distributed:

```
Ideal:   vectors spread across unit sphere (isotropic)
Reality: vectors clustered in a small region (anisotropic)
```

**Impact:** Cosine similarity becomes less discriminative when all vectors are similar.
**Mitigation:** Apply whitening transformation: `z = W(x - μ)` where W decorrelates and normalizes.

### Domain Shift

A model trained on general text may perform poorly on domain-specific data (medical, legal, code):

```
General model: "patent claims" ↔ "IP rights"  (weak association)
Legal model:   "patent claims" ↔ "IP rights"  (strong association)
```

**Detection:** Compare recall@10 on in-domain vs general queries.
**Mitigation:** Use domain-adapted embedding models or fine-tune on domain data.

### Out-of-Distribution Queries

Queries containing terms or concepts not seen during model training produce unreliable embeddings. This is particularly problematic for:
- Newly coined terms or acronyms
- Highly technical jargon
- Multilingual queries on monolingual models

**Mitigation:** Hybrid search (TF-IDF catches exact matches that embeddings miss).

---

## References

1. Salton, G., Wong, A., Yang, C.S. (1975). "A Vector Space Model for Automatic Indexing." *Communications of the ACM*.
2. Robertson, S., Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in IR*.
3. Malkov, Y., Yashunin, D. (2020). "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs." *IEEE TPAMI*.
4. Jégou, H., Douze, M., Schmid, C. (2011). "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI*.
5. Johnson, W., Lindenstrauss, J. (1984). "Extensions of Lipschitz mappings into a Hilbert space." *Contemp. Math.*
6. Cormack, G.V., Clarke, C.L.A., Büttcher, S. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." *SIGIR*.
