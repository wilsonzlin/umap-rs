# How UMAP works

UMAP (Uniform Manifold Approximation and Projection) compresses high-dimensional data into 2D/3D while preserving local neighborhoods. Think: "keep nearby points nearby, don't worry much about far points."

## Core concept

**Input:** 1000 samples × 784 features (e.g., MNIST digits)
**Output:** 1000 samples × 2 coordinates (plottable!)

**Core insight:** High-dimensional data often lies on a lower-dimensional manifold (like a curved surface embedded in high-D space). UMAP learns this manifold structure and flattens it.

## Algorithm steps

### 1. Find k-nearest neighbors

For each point, find its 15 nearest neighbors (default k=15) in high-D space using Euclidean distance.

```
Point A: [neighbors: B, C, D, E, ...]
Point B: [neighbors: A, F, G, H, ...]
```

This is just brute-force distance calculation or a fast ANN index.

### 2. Build fuzzy graph (membership strengths)

**Problem:** KNN is too rigid - either you're a neighbor or not.

**Solution:** Assign fuzzy membership weights to edges based on distances.

For each point `i` and its neighbor `j`:
- Compute local distance scale `σᵢ` (how spread out i's neighbors are)
- Compute offset `ρᵢ` (distance to nearest neighbor)
- Weight: `wᵢⱼ = exp(-(dist(i,j) - ρᵢ) / σᵢ)`

**Result:** Directed weighted graph where `wᵢⱼ` = "strength of i→j connection"

**Intuition:** Nearby points get high weights (~1.0), far points get low weights (~0.0). The exponential decay is calibrated so each point sees roughly k neighbors with significant weight.

### 3. Symmetrize graph (fuzzy set union)

The graph from step 2 is asymmetric: `wᵢⱼ ≠ wⱼᵢ` (i might think j is close, but j disagrees).

**Symmetrize using fuzzy set union:**
```
w_final[i,j] = w[i,j] + w[j,i] - w[i,j] * w[j,i]
```

This is probabilistic OR: "i→j OR j→i is an edge."

**Result:** Undirected weighted graph representing the high-D manifold.

### 4. Initialize low-dimensional embedding

Need starting positions for 1000 points in 2D space.

**Method:** Spectral embedding (Laplacian eigenvectors)
- Compute graph Laplacian matrix
- Find top eigenvectors (like PCA for graphs)
- Use eigenvector components as initial 2D coordinates

**Why spectral?** Gives decent initial layout that roughly respects graph structure. Better than random → faster convergence.

### 5. Optimize layout via SGD

**Goal:** Move points in 2D to match the high-D graph structure.

**Force model:**
- **Attractive force:** Points connected in high-D graph should be close in 2D
- **Repulsive force:** Points NOT connected in high-D should spread out in 2D

**SGD Loop (200-500 epochs):**

```python
for epoch in range(n_epochs):
    # Sample edges based on their weights (high weight = sample more often)
    for edge (i, j) with weight w:
        dist_2d = distance(embedding[i], embedding[j])

        # Attractive: pull i and j together
        attractive_force = 2 * b * (w - 1) / (1 + a * dist_2d^(2b))
        embedding[i] += learning_rate * attractive_force * (j - i)
        embedding[j] -= learning_rate * attractive_force * (j - i)

        # Negative sampling: push i away from random points
        for k in random_sample(n_negative):
            dist_2d = distance(embedding[i], embedding[k])
            repulsive_force = 2 * b / (1 + a * dist_2d^(2b))
            embedding[i] -= learning_rate * repulsive_force * (k - i)

    learning_rate *= decay  # Reduce learning rate each epoch
```

**Key parameters:**
- `a, b`: Control attractive vs repulsive force balance (learned from `min_dist`, `spread`)
- `negative_sample_rate`: How many random points to repel (default 5)
- `learning_rate`: Step size (starts at 1.0, decays linearly to 0)

**Why negative sampling?** Can't compute repulsion for all pairs (O(n²)). Random sampling gives unbiased gradient estimate (like word2vec skip-gram).

### 6. Parallelization (Hogwild! SGD)

**Key insight:** Most edges don't share vertices in the same batch → races are rare.

**Parallel approach:**
- Split edges across threads
- Each thread updates embeddings WITHOUT locks
- Occasional race conditions (two threads update same point) → acceptable
- **Why it works:** SGD is already stochastic; lost updates don't break convergence

**Implementation:**
```rust
edges.par_iter().for_each(|edge| {
    // Multiple threads write to embedding array concurrently
    // Races on embedding[i] if two edges share vertex i
    // But this is rare and doesn't hurt the optimization
});
```

**Result:** Multi-core speedup with negligible accuracy loss.

## Intuitive analogy

**Physical simulation:**
- Start with points randomly scattered in 2D
- Connect nearby high-D neighbors with springs (attractive force)
- All points repel each other like charged particles (repulsive force)
- Run physics simulation → points settle into configuration that respects high-D structure

## Comparison to t-SNE

| Feature | t-SNE | UMAP |
|---------|-------|------|
| **Speed** | Slow (O(n² log n)) | Fast (O(n^1.14)) + parallelizable |
| **Scalability** | Struggles >10k points | Handles millions |
| **Global structure** | Poor | Better preserves global distances |
| **Theory** | Ad-hoc | Grounded in manifold theory |
| **Tuning** | Sensitive to perplexity | More robust defaults |

## Code flow in this repo

```
umap.fit(X)
├─ 1. KNN search (external, precomputed)
├─ 2. fuzzy_simplicial_set [PARALLEL]
│   ├─ smooth_knn_dist (compute σ, ρ for each point) [PARALLEL - per sample]
│   ├─ compute_membership_strengths (compute weights wᵢⱼ) [PARALLEL]
│   └─ set_operations (symmetrize graph) [PARALLEL - fused formula]
├─ 3. simplicial_set_embedding
│   ├─ init_graph_transform (spectral initialization)
│   └─ optimize_layout_euclidean (SGD with parallelism) [PARALLEL - Hogwild!]
│       ├─ Attractive updates (connected edges)
│       └─ Repulsive updates (negative sampling)
└─ Return 2D embedding
```

## Practical tips

**When to use UMAP:**
- Visualization of high-D data (the primary use case)
- Clustering (use UMAP → run k-means on 2D output)
- Preprocessing for ML (compress 784D → 50D, then train classifier)

**Key parameters to tune:**
- `n_neighbors` (5-50): Smaller = focus on local structure, larger = preserve global
- `min_dist` (0.0-0.99): Smaller = tighter clusters, larger = more dispersed
- `n_components` (2-100): Usually 2 for viz, 10-50 for preprocessing

**Common gotchas:**
- Output is NOT deterministic (random init, SGD)
- Different runs give different layouts (but similar structure)
- Distances in 2D are NOT metric (close = similar, but far ≠ dissimilar)
- Only fit() is implemented here, no transform() for new points

## Mathematical footnote

The fuzzy set operations and membership functions come from category theory / algebraic topology. The "manifold" interpretation: locally, high-D data looks Euclidean, globally it's curved. UMAP learns this curvature via the graph, then flattens it. The `a, b` parameters are derived from a specific choice of distance function on the manifold. You don't need to understand this to use UMAP effectively.
