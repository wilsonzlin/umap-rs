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

## Why Python umap-learn Has 10x More Code

**The core algorithm is ~1000 lines.** Python umap-learn is ~15,000+ lines. Why?

### Production Library Infrastructure (~70% of code)

**Input validation & error handling:**
- Check array shapes, dtypes, NaN/inf values
- Meaningful error messages for 50+ edge cases
- Automatic type coercion (lists → arrays, int → float)
- Warnings for suboptimal parameters

**Multiple distance metrics (30+ supported):**
- Euclidean, cosine, manhattan, chebyshev, minkowski
- Haversine (for lat/lon), correlation, canberra
- Hamming, jaccard (for categorical data)
- Custom metric support via numba JIT compilation

**Sparse matrix support:**
- CSR/CSC/COO input handling
- Sparse-specific optimizations for KNN
- Memory-efficient operations (no densification)

**Transform operations:**
- Embed new points into existing embedding
- Requires storing training graph, parameters
- Separate optimization path for new points

**Supervised/semi-supervised learning:**
- Use class labels to guide embedding
- Weighted graph edges based on label similarity
- Metric learning integration

### Scikit-learn Integration (~10% of code)

- BaseEstimator/TransformerMixin inheritance
- fit/transform/fit_transform API
- Pipeline compatibility
- Cross-validation support
- Parameter validation via `_validate_params`
- get_params/set_params for grid search

### User Experience Features (~10% of code)

**Initialization methods:**
- Spectral (Laplacian eigenvectors)
- Random
- PCA-based
- User-provided coordinates
- tSNE initialization

**Progress tracking:**
- Verbose output with timing info
- Progress bars for long operations
- Callback hooks for custom monitoring

**Plotting utilities:**
- Built-in scatter plot with labels
- Interactive plots (Plotly/Bokeh)
- Connectivity diagram visualization

### Backward Compatibility (~5% of code)

- Deprecated parameter aliases
- Legacy API support
- Old serialization format readers
- Warnings for parameter changes

### Documentation & Examples (~5% of code)

- Extensive docstrings (NumPy style)
- Example code in docstrings
- Parameter descriptions
- References to papers

### What This Port Eliminates

**This Rust port is ~2000 lines** (core algorithm + minimal infrastructure):

```
✓ Core algorithm:          ~800 lines  (identical to Python)
✓ Parallel optimization:   ~400 lines  (UnsafeSyncCell wrapper)
✓ Type definitions:        ~300 lines  (builders, structs)
✓ Utilities:               ~200 lines  (clip, constants)
✓ Distance metrics:        ~100 lines  (Euclidean + trait)
✓ Documentation:           ~200 lines  (this + HOW_UMAP_WORKS.md)

✗ Validation:                 0 lines  (assumes caller provides clean data)
✗ Transform:                  0 lines  (fit only)
✗ Multiple metrics:           0 lines  (Euclidean + generic trait)
✗ Sparse support:             0 lines  (dense only)
✗ Supervised learning:        0 lines  (unsupervised only)
✗ Scikit-learn compat:        0 lines  (standalone)
✗ Error handling:             0 lines  (panic on bad input)
✗ Plotting:                   0 lines  (output raw coordinates)
```

### The 80/20 Rule

**80% of use cases need 20% of the code.** Most users:
1. Have clean dense data (no need for validation/sparse support)
2. Want basic visualization (don't need transform/supervised)
3. Use Euclidean distance (don't need 30+ metrics)

This port focuses on the 20% that delivers 80% of the value: **fast, parallel, core UMAP algorithm.**

### Code Comparison

**Python umap-learn structure:**
```
umap/
├── umap_.py           (~3000 lines: main UMAP class, all features)
├── distances.py       (~1500 lines: 30+ metric implementations)
├── spectral.py        (~400 lines: initialization methods)
├── utils.py           (~800 lines: validation, conversion)
├── sparse.py          (~1200 lines: sparse-specific code)
├── plot/              (~2000 lines: plotting utilities)
├── aligned_umap.py    (~1000 lines: multi-dataset alignment)
├── parametric_umap.py (~2000 lines: neural network variant)
└── ...                (~3000+ lines: misc features)
```

**This Rust port structure:**
```
src/
├── umap/
│   ├── umap.rs                    (~150 lines: main API)
│   ├── fuzzy_simplicial_set.rs   (~150 lines: graph construction)
│   ├── simplicial_set_embedding.rs (~100 lines: init + optimize)
│   └── ...                        (~400 lines: supporting functions)
├── layout/
│   ├── optimize_layout_euclidean.rs (~250 lines: parallel SGD)
│   └── optimize_layout_generic.rs   (~150 lines: metric-agnostic SGD)
├── distances.rs                   (~100 lines: Euclidean + trait)
└── utils/                         (~100 lines: clip, constants)

Total: ~1300 lines of implementation + ~700 lines docs/tests
```

### Why The Simplicity Matters

**Advantages of minimal port:**
- **Easier to audit:** All code fits in your head
- **Faster compilation:** Less code = faster builds
- **Clearer intent:** No legacy baggage or edge case sprawl
- **Better performance:** No abstraction overhead from generalization
- **Maintainable:** Core algorithm changes don't break unrelated features

**Tradeoff:** Not a drop-in replacement for umap-learn. This is a **specialized tool** for the core algorithm, not a general-purpose library.

