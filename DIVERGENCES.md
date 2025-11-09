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

**Set operations in FuzzySimplicialSet:** Python uses sparse matrix scalar multiplication then element-wise ops. Rust uses direct TriMat construction with scaled values. Mathematically identical, fewer intermediate allocations.

**Epoch tracking:** Python modifies arrays in-place within numba kernels. Rust uses the same pattern but with explicit `UnsafeSyncCell` wrapper for clarity.

## Semantic Equivalences Maintained

✓ Graph construction (fuzzy simplicial set) is identical
✓ Spectral initialization uses same eigensolver
✓ SGD update rules are mathematically identical
✓ Negative sampling rate and probabilities match exactly
✓ Learning rate schedule (linear decay) is identical
✓ Membership strength computation uses same formulas
✓ Set operation mixing (union/intersection) uses same math

## Verification Approach

The port maintains semantic equivalence by:

1. **Same mathematical formulas**: All SGD updates, distance computations, and probability calculations use identical math
2. **Same algorithmic flow**: Graph construction → initialization → optimization follows Python exactly
3. **Comparable parallelism**: Lock-free parallel SGD matches Numba's behavior
4. **Documented divergences**: All differences are intentional, documented, and maintain correctness

**Expected behavior:** Given the same input data and RNG seed, outputs will be similar but not bit-identical due to RNG and FP differences. The learned manifold structure and relative distances should be equivalent.
