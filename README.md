# umap-rs

Fast, parallel Rust implementation of the core UMAP algorithm. Clean, modern Rust implementation focused on performance and correctness.

## What this is

- Core UMAP dimensionality reduction algorithm (~1800 lines)
- Parallel SGD optimization (4-8x speedup via Rayon)
- Euclidean distance + extensible metric trait
- Dense arrays only (no sparse matrix support)
- Fit only (transform for new points not yet implemented)

## What this is NOT

- Drop-in replacement for Python umap-learn
- General-purpose ML library
- Production-ready with validation/error handling

See [DIVERGENCES.md](DIVERGENCES.md) for detailed comparison to Python umap-learn.

## Usage

```rust
use umap::{Umap, UmapConfig};
use ndarray::Array2;

// Configure UMAP parameters
let config = UmapConfig {
    n_components: 2,
    graph: GraphParams {
        n_neighbors: 15,
        ..Default::default()
    },
    ..Default::default()
};

// Create UMAP instance
let umap = Umap::new(config);

// Precompute KNN (use your favorite ANN library: pynndescent, hnswlib, etc.)
let knn_indices: Array2<u32> = /* shape (n_samples, n_neighbors) */;
let knn_dists: Array2<f32> = /* shape (n_samples, n_neighbors) */;

// Provide initialization (see Initialization section below)
// Common choices: random, PCA, or your own custom embedding
let init: Array2<f32> = /* shape (n_samples, n_components) */;

// Fit UMAP to data
let model = umap.fit(
    data.view(),
    knn_indices.view(),
    knn_dists.view(),
    init.view(),
);

// Get the embedding
let embedding = model.embedding();  // Returns ArrayView2<f32>

// Or take ownership of the embedding
let embedding = model.into_embedding();  // Returns Array2<f32>
```

## Initialization

**You must provide your own initialization.** This library is designed to be minimal and focused on the core UMAP optimization - initialization is left to the caller.

### Recommended Approaches

**Random initialization** (simplest):
```rust
use rand::Rng;
use ndarray::Array2;

fn random_init(n_samples: usize, n_components: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((n_samples, n_components), |_| {
        rng.gen_range(-10.0..10.0)
    })
}
```

**PCA initialization** (recommended for better convergence):
```rust
// Use any PCA library (e.g., linfa-reduction, ndarray-stats, etc.)
// Project data to first n_components principal components
// Scale to roughly [-10, 10] range
```

**Custom initialization**:
- Spectral embedding (use sparse eigensolvers like arpack-ng for large datasets)
- t-SNE initialization
- Pre-trained neural network embeddings
- Domain-specific embeddings

**Note:** The MNIST example includes a toy spectral implementation, but it's **not suitable for production** use (O(n³) complexity, only works on small datasets). See the example code documentation for details.

## Configuration

UMAP parameters are grouped logically:

### Basic

```rust
use umap::UmapConfig;

let config = UmapConfig {
    n_components: 2,  // Output dimensions
    ..Default::default()
};
```

### Manifold parameters

```rust
use umap::config::ManifoldParams;

let manifold = ManifoldParams {
    min_dist: 0.1,   // Minimum distance in embedding
    spread: 1.0,     // Scale of embedding
    a: None,         // Auto-computed from min_dist/spread
    b: None,         // Auto-computed from min_dist/spread
};
```

### Graph construction

```rust
use umap::config::GraphParams;

let graph = GraphParams {
    n_neighbors: 15,              // Number of nearest neighbors
    local_connectivity: 1.0,      // Local neighborhood connectivity
    set_op_mix_ratio: 1.0,        // Fuzzy union (1.0) vs intersection (0.0)
    disconnection_distance: None, // Auto-computed from metric
};
```

### Optimization

```rust
use umap::config::OptimizationParams;

let optimization = OptimizationParams {
    n_epochs: None,           // Auto-determined from dataset size
    learning_rate: 1.0,       // SGD learning rate
    negative_sample_rate: 5,  // Negative samples per positive
    repulsion_strength: 1.0,  // Weight for negative samples
};
```

### Complete example

```rust
let config = UmapConfig {
    n_components: 3,
    manifold: ManifoldParams {
        min_dist: 0.05,
        ..Default::default()
    },
    graph: GraphParams {
        n_neighbors: 30,
        ..Default::default()
    },
    optimization: OptimizationParams {
        n_epochs: Some(500),
        ..Default::default()
    },
};
```

## Custom distance metrics

Implement the `Metric` trait:

```rust
use umap::Metric;
use ndarray::{Array1, ArrayView1};

#[derive(Debug)]
struct MyMetric;

impl Metric for MyMetric {
    fn distance(&self, a: ArrayView1<f32>, b: ArrayView1<f32>) -> (f32, Array1<f32>) {
        // Return (distance, gradient)
        // gradient = ∂distance/∂a
        todo!()
    }

    // Optional: provide fast squared distance for optimization
    fn squared_distance(&self, a: ArrayView1<f32>, b: ArrayView1<f32>) -> Option<f32> {
        None  // Return Some(dist_sq) if available
    }
}

// Use custom metric
let umap = Umap::with_metrics(
    config,
    Box::new(MyMetric),       // Input space metric
    Box::new(EuclideanMetric), // Output space metric
);
```

See `src/distances.rs` for the Euclidean implementation example.

## Build

```bash
cargo build --release
```

## Performance

- **Parallel SGD**: Enabled by default via Rayon
- **4-8x speedup** on multi-core machines
- **Hogwild! algorithm**: Lock-free parallel gradient descent
- **Euclidean fast path**: Specialized optimization using squared distances

## Documentation

- [UMAP.md](UMAP.md) - How UMAP works (algorithm explanation)
- [DIVERGENCES.md](DIVERGENCES.md) - Differences from Python umap-learn
- [AGENTS.md](AGENTS.md) - Developer notes

Run `cargo doc --open` to browse the API documentation.

## Design principles

1. **Minimal** - Core algorithm only, no feature creep
2. **Fast** - Parallel by default, zero-copy where possible
3. **Explicit** - Caller provides KNN, initialization, etc.
4. **Rust-native** - Idiomatic patterns, not Python translations

## Limitations

- No input validation (assumes clean data)
- Transform not yet implemented
- Dense arrays only
- Panics on invalid input (not Result-based errors)
- Requires external KNN computation and initialization

This is a specialized tool for the core algorithm. Wrap it in validation/error handling for production use.

## License

BSD-3-Clause (see LICENSE file)

## References

- Original paper: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426
- Hogwild! SGD: Recht, B., et al. (2011). Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. NIPS 2011
- Python umap-learn: https://github.com/lmcinnes/umap
