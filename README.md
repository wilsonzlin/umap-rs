# umap-rs

Fast, parallel Rust implementation of the core UMAP algorithm. Stripped-down port focused on performance.

## What this is

- Core UMAP dimensionality reduction algorithm (~1300 lines)
- Parallel SGD optimization (4-8x speedup via Rayon)
- Euclidean distance + generic metric trait
- Dense arrays only (no sparse matrix support)
- Fit only (no transform for new points)

## What this is NOT

- Drop-in replacement for Python umap-learn
- General-purpose ML library
- Production-ready with validation/error handling

See [DIVERGENCES.md](DIVERGENCES.md) for detailed comparison to Python umap-learn.

## Usage

```rust
use ndarray::Array2;
use umap::Umap;

// Precompute KNN (use your favorite ANN library)
let knn_indices: Array2<u32> = /* shape (n_samples, n_neighbors) */;
let knn_dists: Array2<f32> = /* shape (n_samples, n_neighbors) */;

// Optional: precompute initialization (PCA, spectral, etc.)
let init: Array2<f32> = /* shape (n_samples, n_components) */;

let mut umap = UmapBuilder::default()
    .n_neighbors(15)
    .n_components(2)
    .init(init.view())
    .knn_indices(knn_indices.view())
    .knn_dists(knn_dists.view())
    .metric(&EuclideanMetric)
    .output_metric(&EuclideanMetric)
    .build();

let embedding = umap.fit(X.view());  // Returns Array2<f32>
```

## Key parameters

- `n_neighbors` (default 15): Smaller = local structure, larger = global structure
- `n_components` (default 2): Output dimensionality
- `min_dist` (default 0.1): Minimum distance between points in embedding
- `init`: Custom initialization (you provide the array)

See [UMAP.md](UMAP.md) for algorithm explanation.

## Build

```bash
cargo build --release
```

## Performance

Parallel optimization enabled by default. To use:
- SGD runs in parallel across edges (Hogwild! algorithm)
- 4-8x speedup on multi-core machines
- Requires Rayon (already in dependencies)

## Custom distance metrics

Implement the `Metric` trait:

```rust
pub trait Metric: Send + Sync {
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>)
        -> (f32, Array1<f32>);
    fn is_euclidean(&self) -> bool { false }
}
```

Return `(distance, gradient)`. See `src/distances.rs` for Euclidean example.

## Documentation

- [UMAP.md](UMAP.md) - How UMAP works (algorithm explanation)
- [DIVERGENCES.md](DIVERGENCES.md) - Differences from Python umap-learn
- [AGENTS.md](AGENTS.md) - Notes for developers/future agents

## Design principles

1. **Minimal** - Core algorithm only, no feature creep
2. **Fast** - Parallel by default, zero-copy where possible
3. **Explicit** - Caller provides KNN, initialization, etc.
4. **Auditable** - All code fits in your head (~2000 lines total)

## Limitations

- No input validation (assumes clean data)
- No transform() for new points
- Dense arrays only
- Euclidean + generic trait (not 30+ metrics like Python)
- Panics on invalid input (not graceful errors)

This is a specialized tool for the core algorithm. Wrap it in validation/error handling if needed.

## License

[Your license here]

## References

- Original paper: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426
- Hogwild! SGD: Recht, B., et al. (2011). Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. NIPS 2011
- Python umap-learn: https://github.com/lmcinnes/umap
