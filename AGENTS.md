# Notes for future agents/developers

## Critical things that look wrong but are correct

### UnsafeSyncCell and data races
`src/layout/optimize_layout_euclidean.rs:30-57` implements `UnsafeSyncCell<T>` which **intentionally allows data races**. This is correct for parallel SGD:

- Multiple threads write to the same embedding array without locks
- Races occur when two edges share a vertex (rare)
- Lost updates are acceptable because SGD is stochastic
- This matches Python UMAP's Numba `parallel=True` behavior
- Academic justification: Hogwild! algorithm (Recht et al., 2011)

**Do NOT "fix" this by adding locks/atomics.** Performance would tank and it's unnecessary.

### ParallelVec and UnsafeCell usage
`src/umap/fuzzy_simplicial_set.rs` uses `ParallelVec<T>` for parallel CSR construction. This uses `std::cell::UnsafeCell<Vec<T>>` internally, which is **required** for correct unsafe code:

- `UnsafeCell<T>` is Rust's primitive for interior mutability
- It tells the compiler "the T inside may be mutated through shared references"
- Without it, the compiler assumes `&Vec<T>` means immutable contents and may optimize incorrectly (UB)
- **Correct pattern**: `UnsafeCell<Vec<T>>` - the Vec's contents may be mutated
- **Wrong pattern**: `UnsafeCell<*mut T>` - only wraps the pointer, not the data

```rust
struct ParallelVec<T> {
  data: UnsafeCell<Vec<T>>,  // Vec contents may be mutated through &self
}
unsafe impl<T: Send> Sync for ParallelVec<T> {}  // Safe if disjoint writes

impl<T> ParallelVec<T> {
  unsafe fn write(&self, index: usize, value: T) {
    let vec = &mut *self.data.get();  // UnsafeCell::get() → *mut Vec<T>
    *vec.get_unchecked_mut(index) = value;
  }
  fn into_inner(self) -> Vec<T> {
    self.data.into_inner()  // Extract Vec after parallel work
  }
}
```

**Key invariant**: Each row i writes only to `[indptr[i]..indptr[i+1]]`, which are disjoint by construction of the prefix sum.

### Lifetime patterns in fuzzy_simplicial_set.rs
`FuzzySimplicialSet<'a, 'd>` has two lifetimes but no bound between them. This is intentional:

- `'a` is for KNN data (indices/distances)
- `'d` is for disconnections set (DashSet)
- They're independent - no relationship needed
- Tried `'d: 'a` - causes borrow checker errors in caller

**Do NOT add lifetime bounds without testing the full call chain.**

### ArrayView2 passed by value
Throughout the codebase, `ArrayView2<'a, T>` is passed by value, not reference:

```rust
fn foo(distances: ArrayView2<'a, f32>) { ... }  // Correct
// NOT: fn foo(distances: &ArrayView2<'a, f32>) { ... }
```

`ArrayView2` is a thin wrapper (pointer + shape metadata), cheap to copy. Passing by value is idiomatic for ndarray views.

### No TypedBuilder struct in optimize_layout_generic.rs
Earlier version tried using a `TypedBuilder` struct for single-epoch optimization. This failed due to lifetime issues (can't borrow embeddings mutably multiple times through struct fields).

Solution: standalone function with many parameters. Yes, it triggers `clippy::too_many_arguments`. That's fine - the alternative doesn't compile.

**Do NOT try to refactor into a builder struct without solving the lifetime puzzle.**

## Performance-critical sections

### Parallel loop in optimize_layout_euclidean.rs:338-436
This is the hot path. ~80% of runtime is here. Changes here affect performance directly:

- Keep allocations outside the parallel loop if possible
- `Vec::with_capacity(dim)` is acceptable inside (stack-like, tiny)
- Do NOT add synchronization primitives
- Do NOT call external functions that aren't inline

### Direct CSR construction in fuzzy_simplicial_set.rs
Graph construction uses direct CSR building instead of TriMat/COO to avoid OOM on large datasets:

1. **Count phase**: Parallel count of entries per row
2. **Indptr phase**: Sequential prefix sum (fast, O(n))
3. **Fill phase**: Parallel fill, each row writes to `[indptr[i]..indptr[i+1]]`
4. **Sort phase**: Parallel per-row sort (insertion sort, O(k²) for k~256)

This avoids:
- Allocating O(nnz) intermediate triplet arrays
- O(nnz log nnz) global sort (replaced by O(n × k log k) local sorts)
- Multiple copies during format conversion

**Memory**: Only stores the final CSR arrays (indptr, indices, data) plus temporary row counts. No intermediate triplet/COO storage.

### Distance calculations
Both Euclidean implementations use squared distance to avoid sqrt:

```rust
let dist_squared = rdist(&current, &other);  // No sqrt
// Then use dist_squared directly in formulas
```

**Do NOT add sqrt calls.** The formulas are designed for squared distance.

## Common mistakes to avoid

### Don't store ArrayView2 in long-lived structs
Bad:
```rust
struct Foo<'a> {
    data: ArrayView2<'a, f32>,  // Ties struct lifetime to data
}
```

Good:
```rust
fn foo(data: ArrayView2<f32>) { ... }  // Pass as parameter
```

Views tie lifetimes in complex ways. Use them as function parameters, not struct fields (unless you really understand the lifetime implications).

### Don't remove the `unsafe` blocks thinking they're bugs
Every `unsafe` block has a safety comment explaining WHY it's needed. If you see unsafe code:

1. Read the safety comment
2. Check if it references Hogwild!/parallel SGD
3. If yes, it's intentional
4. If no safety comment, THEN investigate

### Don't add `Send`/`Sync` bounds everywhere
Rayon requires `Send + Sync` for parallel iterators. But this is satisfied by `UnsafeSyncCell`. Adding manual bounds elsewhere usually means you're fighting the borrow checker instead of solving the real problem.

## Architecture decisions

### Why no error types?
This is a library component, not a user-facing API. Invalid input (wrong shapes, NaN, etc.) is a programming error, not a runtime condition. Panics are appropriate.

If you need error handling, wrap this in a higher-level API that validates before calling.

### Why ArrayView instead of owned arrays?
Caller owns the data, we just borrow it. This allows:
- Zero-copy when data is already in memory
- Caller controls allocation strategy
- No hidden clones

Tradeoff: Lifetime annotations everywhere. Accept this.

### Why no trait objects for metrics in hot paths?
`&dyn Metric` is used, which is a trait object (dynamic dispatch). This costs ~2-5ns per call. For Euclidean distance, there's a specialized version that avoids this.

Generic optimization uses trait objects because it's not the hot path (most users use Euclidean). If you want to eliminate this, you'd need to make the entire call chain generic over `M: Metric`, which explodes compile times.

## What NOT to change

1. **Don't add validation** - this is a stripped-down port, validation is caller's responsibility
2. **Don't add logging/progress bars** - adds dependencies, bloat
3. **Don't generalize distance metrics further** - Euclidean + generic trait is enough
4. **Don't add transform()** - out of scope for this port
5. **Don't "clean up" the unsafe code** - it's correct as-is

## What TO change

1. **Do add more distance metric implementations** via the `Metric` trait
2. **Do optimize the SGD formulas** if you find mathematically equivalent but faster versions
3. **Do improve the spectral initialization** if you have a better eigensolver
4. **Do add examples/benchmarks** in a separate directory

## Testing strategy

There are no unit tests because:
1. Most functions are thin wrappers around math
2. Integration test would require real KNN data
3. Correctness is verified by comparing output to Python UMAP

If you add tests, test at the integration level: full UMAP fit on small dataset (100 points), compare structure to known-good output.

## When to ask for help

If you're considering changes that involve:
- Adding lifetimes or lifetime bounds
- Modifying the parallel optimization
- Changing ArrayView to owned arrays
- Adding locks/atomics

Stop and document WHY you think this is needed. The current design is the result of fighting the borrow checker for hours. Your "obvious fix" probably won't work.
