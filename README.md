# umap-rs

Fast, parallel Rust implementation of the core UMAP algorithm. Clean, modern Rust implementation focused on performance and correctness.

Embeds 264 million samples to 2D in 2 hours (500 epochs) on a 126-core machine (see [Performance](#performance)).

## What this is

- Core UMAP dimensionality reduction algorithm
- Fully parallelized via Rayon with memory-efficient sparse matrix construction
- Extensible metric system (Euclidean + custom metrics)
- Checkpointing and fault-tolerant training
- Dense arrays only (no sparse matrix support)
- Fit only (transform for new points not yet implemented)

See [DIVERGENCES.md](DIVERGENCES.md) for detailed comparison to Python umap-learn.

## Usage

```rust
use ndarray::Array2;
use umap::{GraphParams, Umap, UmapConfig};

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

### Checkpointing

UMAP training has two phases: **learning the manifold** (building the graph) and **optimizing the embedding** (running gradient descent). The first is deterministic and expensive, the second is iterative and can be interrupted.

For long training runs, you can checkpoint the optimization and resume if interrupted:

```rust
use umap_rs::{Metric, Optimizer};

// Phase 1: Learn the manifold structure from your data
// This builds the fuzzy topological graph. It's slow but deterministic - 
// same inputs always give same manifold.
let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());

// Phase 2: Optimize the embedding via gradient descent
// Create an optimizer that will run 500 epochs of SGD
let metric = umap_rs::EuclideanMetric;
let mut opt = Optimizer::new(manifold, init, 500, &config, metric.metric_type());

// Train in chunks of 10 epochs at a time
while opt.remaining_epochs() > 0 {
  opt.step_epochs(10, &metric);  // Run 10 more epochs
  
  // Periodically save a checkpoint (embedding + all optimization state)
  if opt.current_epoch() % 50 == 0 {
    std::fs::write(
      format!("checkpoint_{}.bin", opt.current_epoch()),
      bincode::serialize(&opt)?
    )?;
  }
}

// Training done - convert to final lightweight model
let fitted = opt.into_fitted(config);
```

If your process is interrupted, load the checkpoint and continue:

```rust
// Deserialize the optimizer state from disk
let mut opt: Optimizer = bincode::deserialize(&std::fs::read("checkpoint_250.bin")?)?;

// Continue from epoch 250 to 500
while opt.remaining_epochs() > 0 {
  opt.step_epochs(10, &metric);
}

let fitted = opt.into_fitted(config);
```

The checkpoint contains everything: current embedding, epoch counters, and the manifold. When training completes, convert to a final model:

```rust
// Training done - drop the heavy optimization state
let fitted = opt.into_fitted(config);

// Access the embedding
let embedding = fitted.embedding();  // Zero-copy view

// Or take ownership
let embedding = fitted.into_embedding();

// Save the final model (much smaller than checkpoints)
std::fs::write("model.bin", bincode::serialize(&fitted)?)?;
```

The serialized `FittedUmap` contains just the manifold and embedding, not the optimization state, making it lightweight for long-term storage.

## Initialization

**You must provide your own initialization.** This library is designed to be minimal and focused on the core UMAP optimization - initialization is left to the caller.

### Recommended Approaches

**Random initialization** (simplest):
```rust
use ndarray::Array2;
use rand::Rng;

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
    symmetrize: true,             // Symmetrize graph (set false to save memory)
};
```

The `symmetrize` option controls whether the fuzzy graph is symmetrized via fuzzy set union. For very large datasets, setting `symmetrize: false` roughly halves memory usage with minimal impact on 2D visualization quality.

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

This implementation is designed for large-scale datasets (100M+ samples) on high-core-count machines.

### Real-world benchmark

264 million text embeddings from a [search engine](https://blog.wilsonl.in/search-engine) embedded to 2D in 2 hours on a 126-core AMD EPYC 9J45 with 1.4 TB RAM:
- Precomputed KNN (n_neighbors=100)
- Precomputed PCA initialization
- Symmetrization disabled
- 500 epochs
- ~1 TB peak memory

### Parallelization

Every phase is fully parallelized via Rayon:

- **Graph construction**: Parallel smooth KNN distance, parallel CSR matrix construction
- **Set operations**: Parallel CSC structure building, parallel symmetrization
- **Optimizer initialization**: Parallel edge filtering, parallel epoch scheduling
- **SGD optimization**: Lock-free Hogwild! algorithm for parallel gradient descent

### Memory Efficiency

Optimized for minimal memory footprint at scale:

- **Direct CSR construction**: Builds sparse matrices in-place without intermediate COO/triplet format. Avoids O(nnz) temporary allocations and O(nnz log nnz) global sorting.
- **u32 indices**: Uses 4-byte indices instead of 8-byte, halving index memory for datasets up to ~4B samples.
- **CSC structure-only**: Transpose operations store only structure (indptr + indices), looking up values in original CSR via O(log k) binary search.
- **Sequential array allocation**: Large arrays are allocated one at a time to avoid memory spikes.
- **No cloning**: Avoids sequential `.clone()` on large arrays; uses parallel copies when needed.

### Scaling Guidelines

Memory scales with `n_samples × n_neighbors`:

| n_samples | n_neighbors | Approx. Memory |
|-----------|-------------|----------------|
| 10M       | 30          | ~10 GB         |
| 100M      | 30          | ~100 GB        |
| 250M      | 30          | ~250 GB        |
| 250M      | 256         | ~2 TB          |

To reduce memory:
- Use smaller `n_neighbors` (15-50 is typical for visualization)
- Disable symmetrization: `config.graph.symmetrize = false`
- Slice KNN arrays to use fewer neighbors than computed

### Configuration for Large Datasets

```rust
let config = UmapConfig {
    graph: GraphParams {
        n_neighbors: 30,      // Lower = less memory
        symmetrize: false,    // Skip symmetrization to save memory
        ..Default::default()
    },
    ..Default::default()
};
```

### Timing Logs

The library emits structured logs via the `tracing` crate. Enable a subscriber to see timing for each phase:

```rust
tracing_subscriber::fmt::init();  // or your preferred subscriber
```

Example output:
```
INFO umap_rs::umap::fuzzy_simplicial_set: smooth_knn_dist complete duration_ms=52033
INFO umap_rs::umap::fuzzy_simplicial_set: csr row_counts complete duration_ms=48495
INFO umap_rs::umap::fuzzy_simplicial_set: csr indptr complete duration_ms=560 nnz=62586367074
INFO umap_rs::optimizer: optimizer edge filtering complete duration_ms=725 total_edges=23276942679
```

### Advanced: Accessing the Graph

The fuzzy simplicial set graph is exposed as `SparseMat` (a `CsMatI<f32, u32, usize>`):

```rust
use umap_rs::SparseMat;

let manifold = umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view());
let graph: &SparseMat = manifold.graph();
// graph uses u32 column indices (memory efficient) and usize row pointers (handles large nnz)
```

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

- **Maximum ~4 billion samples**: Uses `u32` indices internally for memory efficiency
- No input validation (assumes clean data)
- Transform not yet implemented
- Dense arrays only
- Panics on invalid input (not Result-based errors)
- Requires external KNN computation and initialization

### KNN Sentinel Values

If your KNN search couldn't find `k` neighbors for some points (e.g., isolated points), use `u32::MAX` as a sentinel index and any distance value (commonly `f32::INFINITY`). These entries are automatically skipped during graph construction:

```rust
// Point 5 only has 2 real neighbors, rest are sentinels
knn_indices[[5, 0]] = 10;           // real neighbor
knn_indices[[5, 1]] = 23;           // real neighbor  
knn_indices[[5, 2]] = u32::MAX;     // sentinel - skipped
knn_indices[[5, 3]] = u32::MAX;     // sentinel - skipped
```

This is a specialized tool for the core algorithm. Wrap it in validation/error handling for production use.

## License

BSD-3-Clause (see LICENSE file)

## References

- Original paper: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426
- Hogwild! SGD: Recht, B., et al. (2011). Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. NIPS 2011
- Python umap-learn: https://github.com/lmcinnes/umap
