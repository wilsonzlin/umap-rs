# Semantic Differences from Python UMAP

This document describes intentional differences between this Rust port and the original Python umap-learn implementation. All differences maintain algorithmic correctness while optimizing for Rust's performance and safety model.

## Parallel Execution Model

**Python UMAP:** Uses Numba's `@njit(parallel=True)` which allows data races on shared arrays during SGD optimization. These races are rare (most edges don't share vertices) and don't affect convergence due to the stochastic nature of SGD.

**Rust Port:** Uses `UnsafeSyncCell<T>` to explicitly allow the same data race pattern via Rayon's parallel iterators. This matches Python's behavior exactly and implements the same lock-free parallel SGD pattern described in the Hogwild! algorithm (Recht et al., 2011).

**Impact:** Identical parallelism semantics. Both allow rare lost updates that don't affect convergence. Output may differ in exact values due to different race outcomes, but converges to equivalent solutions.

## Random Number Generation

**Python UMAP:** Uses NumPy's MT19937-based RNG.

**Rust Port:** Uses Rust's `rand` crate (Xoshiro256++ by default).

**Impact:** Different RNG means different random samples for negative edges. Results are statistically equivalent but not bit-identical.

## Floating Point Operations

**Python UMAP:** NumPy operations with specific BLAS backend optimizations.

**Rust Port:** Direct ndarray operations, compiler may reorder FP arithmetic differently.

**Impact:** Minor numerical differences due to different rounding/optimization. All differences within expected FP variance. No algorithmic impact.

## Data Structure Changes

**Python UMAP:** Uses `@dataclass` with mutable fields, passes arrays by reference.

**Rust Port:** Uses `TypedBuilder` pattern, passes lightweight views (`ArrayView2<'a, T>`) by value.

**Impact:** Zero semantic difference. `ArrayView2` is a thin wrapper (pointer + metadata) that's cheap to copy. Equivalent to passing pointers in Python.

## Removed Features

The following Python UMAP features were intentionally removed per project requirements:

- Sparse matrix input support (dense arrays only)
- Transform operations (fit-only)
- Supervised/semi-supervised learning
- Custom initializations beyond spectral/random
- Validation and edge case handling
- Multiple distance metrics (Euclidean + generic via trait only)

## Simplifications

**Set operations in FuzzySimplicialSet:** Python uses sparse matrix scalar multiplication then element-wise ops (transpose, hadamard, add). Rust fuses all operations into a single parallel pass: for each unique position, the formula `mix*(A + A^T) + (1-2*mix)*(A ⊙ A^T)` is computed directly. Mathematically equivalent, but may differ by floating-point epsilon (~1e-7) due to different operation ordering. This is acceptable since UMAP's SGD is inherently stochastic.

**Direct CSR construction:** Python UMAP builds sparse matrices via COO/triplet format then converts to CSR (O(nnz log nnz) sorting). Rust builds CSR directly by: (1) counting entries per row in parallel, (2) computing prefix sums for row pointers, (3) filling data in parallel per-row, (4) sorting within each row in parallel. This is O(nnz) with only O(k log k) sorting per row where k is row length. This avoids allocating O(nnz) intermediate triplet arrays.

**u32 sparse matrix indices:** Rust uses `CsMatI<f32, u32>` instead of `CsMat<f32>` (which uses `usize`). This halves index memory (4 bytes vs 8 bytes per entry). Valid for datasets up to ~4 billion samples.

**CSC structure-only for transpose:** During set operations, Python UMAP fully converts the matrix to CSC format (duplicating data). Rust builds only the CSC structure (indptr + indices) and looks up values in the original CSR via binary search O(log k). Since k ≈ 256, this is fast and avoids duplicating the data array.

**Parallel graph construction:** Python UMAP's `smooth_knn_dist` runs sequentially (Numba JIT but single-threaded for this phase). Rust parallelizes the per-sample binary search via Rayon. Results are identical.

**Parallel optimizer initialization:** Python UMAP's optimizer setup (filtering edges, computing epoch schedules) is sequential. Rust parallelizes all phases:
- Parallel max value computation
- Parallel row counting and edge filtering
- Parallel edge extraction (head/tail/weights)
- Parallel epochs_per_sample computation
- Parallel embedding normalization
- Parallel epoch scheduling array creation

**Sequential array allocation:** Python allocates multiple large arrays simultaneously. Rust allocates them one at a time to avoid memory spikes. No performance loss since each allocation is still parallel internally.

**Epoch tracking:** Python modifies arrays in-place within numba kernels. Rust uses the same pattern but with explicit `UnsafeSyncCell` wrapper for clarity.

**Optional symmetrization:** Python UMAP always symmetrizes the fuzzy graph (A + A^T fuzzy union). Rust exposes `config.graph.symmetrize` to optionally skip this, halving memory for large datasets. For 2D visualization, skipping symmetrization typically has minimal impact on output quality.

**Structured timing logs:** Rust emits timing information via the `tracing` crate (structured logging). Python uses verbose print statements. Enable a tracing subscriber to see phase-by-phase timing.

**Sentinel value handling:** Rust gracefully handles `u32::MAX` in KNN indices as a sentinel for missing neighbors (when KNN search couldn't find k neighbors for a point). These entries are skipped during graph construction.

## Semantic Equivalences Maintained

- Graph construction (fuzzy simplicial set) is identical
- Spectral initialization uses same eigensolver
- SGD update rules are mathematically identical
- Negative sampling rate and probabilities match exactly
- Learning rate schedule (linear decay) is identical
- Membership strength computation uses same formulas
- Set operation mixing (union/intersection) uses same math

## Verification Approach

The port maintains semantic equivalence by:

1. **Same mathematical formulas**: All SGD updates, distance computations, and probability calculations use identical math
2. **Same algorithmic flow**: Graph construction → initialization → optimization follows Python exactly
3. **Comparable parallelism**: Lock-free parallel SGD matches Numba's behavior
4. **Documented divergences**: All differences are intentional, documented, and maintain correctness

**Expected behavior:** Given the same input data and RNG seed, outputs will be similar but not bit-identical due to RNG and FP differences. The learned manifold structure and relative distances should be equivalent.