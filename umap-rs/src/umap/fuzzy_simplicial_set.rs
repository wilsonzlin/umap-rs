use crate::umap::smooth_knn_dist::SmoothKnnDist;
use dashmap::DashSet;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use rayon::prelude::*;
use sprs::CsMatI;
use std::cell::UnsafeCell;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::info;
use typed_builder::TypedBuilder;

/// Sparse matrix with u32 indices to save memory (4 bytes vs 8 bytes per index).
/// Valid for n_samples < 2^32 (~4 billion).
pub type SparseMat = CsMatI<f32, u32>;

/// CSC structure without data - only stores indptr and indices for transpose traversal.
/// Values are looked up in the original CSR via binary search O(log k).
struct CscStructure {
  indptr: Vec<u32>,  // Column pointers
  indices: Vec<u32>, // Row indices (which rows have entries in each column)
}

impl CscStructure {
  /// Iterate over (row_index) for entries in the given column
  #[inline]
  fn col_row_indices(&self, col: usize) -> &[u32] {
    let start = self.indptr[col] as usize;
    let end = self.indptr[col + 1] as usize;
    &self.indices[start..end]
  }
}

/// A Vec wrapped in UnsafeCell for parallel write access to disjoint regions.
///
/// `UnsafeCell<Vec<T>>` tells the compiler that the Vec's contents may be mutated
/// even through shared references (`&self`). This is required for soundness -
/// without UnsafeCell, the compiler assumes `&Vec<T>` means immutable contents.
///
/// # Safety
///
/// The caller must ensure that concurrent writes are to disjoint indices.
/// This is used for parallel CSR construction where each row writes to its own
/// section: `[indptr[i]..indptr[i+1]]`.
struct ParallelVec<T> {
  /// The Vec is inside UnsafeCell because we mutate it through &self.
  /// This is the correct pattern - UnsafeCell<Vec<T>> not UnsafeCell<*mut T>.
  data: UnsafeCell<Vec<T>>,
}

// SAFETY: Access is only safe when writes are to disjoint indices.
// The algorithm guarantees this: each row i writes only to [indptr[i]..indptr[i+1]].
unsafe impl<T: Send> Send for ParallelVec<T> {}
unsafe impl<T: Send> Sync for ParallelVec<T> {}

impl<T> ParallelVec<T> {
  /// Create from an owned Vec.
  fn new(vec: Vec<T>) -> Self {
    Self {
      data: UnsafeCell::new(vec),
    }
  }

  /// Write a value at the given index.
  ///
  /// # Safety
  ///
  /// - Index must be in bounds
  /// - No other thread may be accessing the same index concurrently
  #[inline]
  unsafe fn write(&self, index: usize, value: T) {
    // SAFETY: Caller ensures index is valid and no concurrent access to this index.
    // UnsafeCell::get() returns *mut Vec<T>, we deref to get &mut Vec<T>.
    unsafe {
      let vec = &mut *self.data.get();
      debug_assert!(index < vec.len());
      *vec.get_unchecked_mut(index) = value;
    }
  }

  /// Get a mutable slice for a range. Used for sorting.
  ///
  /// # Safety
  ///
  /// - Range must be in bounds
  /// - No other thread may be accessing the same range concurrently
  #[inline]
  unsafe fn get_mut_slice(&self, start: usize, len: usize) -> &mut [T] {
    // SAFETY: Caller ensures range is valid and no concurrent access.
    unsafe {
      let vec = &mut *self.data.get();
      debug_assert!(start + len <= vec.len());
      &mut vec[start..start + len]
    }
  }

  /// Consume and return the inner Vec.
  fn into_inner(self) -> Vec<T> {
    self.data.into_inner()
  }
}

/*
  Given a set of data X, a neighborhood size, and a measure of distance
  compute the fuzzy simplicial set (here represented as a fuzzy graph in
  the form of a sparse matrix) associated to the data. This is done by
  locally approximating geodesic distance at each point, creating a fuzzy
  simplicial set for each such point, and then combining all the local
  fuzzy simplicial sets into a global one via a fuzzy union.

  Parameters
  ----------
  X: array of shape (n_samples, n_features)
      The data to be modelled as a fuzzy simplicial set.

  n_neighbors: int
      The number of neighbors to use to approximate geodesic distance.
      Larger numbers induce more global estimates of the manifold that can
      miss finer detail, while smaller values will focus on fine manifold
      structure to the detriment of the larger picture.

  knn_indices: array of shape (n_samples, n_neighbors) (optional)
      If the k-nearest neighbors of each point has already been calculated
      you can pass them in here to save computation time. This should be
      an array with the indices of the k-nearest neighbors as a row for
      each data point.

  knn_dists: array of shape (n_samples, n_neighbors) (optional)
      If the k-nearest neighbors of each point has already been calculated
      you can pass them in here to save computation time. This should be
      an array with the distances of the k-nearest neighbors as a row for
      each data point.

  set_op_mix_ratio: float (optional, default 1.0)
      Interpolate between (fuzzy) union and intersection as the set operation
      used to combine local fuzzy simplicial sets to obtain a global fuzzy
      simplicial sets. Both fuzzy set operations use the product t-norm.
      The value of this parameter should be between 0.0 and 1.0; a value of
      1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
      intersection.

  local_connectivity: int (optional, default 1)
      The local connectivity required -- i.e. the number of nearest
      neighbors that should be assumed to be connected at a local level.
      The higher this value the more connected the manifold becomes
      locally. In practice this should be not more than the local intrinsic
      dimension of the manifold.

  verbose: bool (optional, default False)
      Whether to report information on the current progress of the algorithm.

  return_dists: bool or None (optional, default None)
      Whether to return the pairwise distance associated with each edge.

  Returns
  -------
  fuzzy_simplicial_set: coo_matrix
      A fuzzy simplicial set represented as a sparse matrix. The (i,
      j) entry of the matrix represents the membership strength of the
      1-simplex between the ith and jth sample points.
*/
#[derive(TypedBuilder, Debug)]
pub struct FuzzySimplicialSet<'a, 'd> {
  n_samples: usize,
  n_neighbors: usize,
  knn_indices: ArrayView2<'a, u32>,
  knn_dists: ArrayView2<'a, f32>,
  knn_disconnections: &'d DashSet<(usize, usize)>,
  #[builder(default = 1.0)]
  set_op_mix_ratio: f32,
  #[builder(default = 1.0)]
  local_connectivity: f32,
  #[builder(default = true)]
  apply_set_operations: bool,
}

impl<'a, 'd> FuzzySimplicialSet<'a, 'd> {
  pub fn exec(self) -> (SparseMat, Array1<f32>, Array1<f32>) {
    assert!(
      self.n_samples < u32::MAX as usize,
      "n_samples must be < 2^32 for u32 indices"
    );

    // Extract the fields we need
    let knn_dists = self.knn_dists;
    let knn_indices = self.knn_indices;
    let knn_disconnections = self.knn_disconnections;
    let n_neighbors = self.n_neighbors;
    let n_samples = self.n_samples;
    let local_connectivity = self.local_connectivity;
    let set_op_mix_ratio = self.set_op_mix_ratio;
    let apply_set_operations = self.apply_set_operations;

    let started = Instant::now();
    let (sigmas, rhos) = SmoothKnnDist::builder()
      .distances(knn_dists)
      .k(n_neighbors)
      .local_connectivity(local_connectivity)
      .build()
      .exec();
    info!(
      duration_ms = started.elapsed().as_millis(),
      "smooth_knn_dist complete"
    );

    // Build CSR directly - no intermediate allocations
    // Uses u32 indices to halve index memory
    let started = Instant::now();
    let mut result = build_membership_csr(
      n_samples,
      n_neighbors,
      knn_indices,
      knn_dists,
      knn_disconnections,
      &sigmas.view(),
      &rhos.view(),
    );
    info!(
      duration_ms = started.elapsed().as_millis(),
      nnz = result.nnz(),
      "build_membership_csr complete"
    );

    if apply_set_operations {
      let started = Instant::now();
      result = apply_set_operations_parallel(&result, set_op_mix_ratio);
      info!(
        duration_ms = started.elapsed().as_millis(),
        "set_operations complete"
      );
    }

    (result, sigmas, rhos)
  }
}

/// Build CSR matrix directly from KNN data without intermediate allocations.
/// Uses u32 indices to halve index memory (4 bytes vs 8 bytes per index).
fn build_membership_csr(
  n_samples: usize,
  n_neighbors: usize,
  knn_indices: ArrayView2<u32>,
  knn_dists: ArrayView2<f32>,
  knn_disconnections: &DashSet<(usize, usize)>,
  sigmas: &ArrayView1<f32>,
  rhos: &ArrayView1<f32>,
) -> SparseMat {
  // Step 1: Count valid (non-zero) entries per row in parallel
  let started = Instant::now();
  let row_counts: Vec<u32> = (0..n_samples)
    .into_par_iter()
    .map(|i| {
      let mut count = 0u32;
      for j in 0..n_neighbors {
        if knn_disconnections.contains(&(i, j)) {
          continue;
        }
        let knn_idx = knn_indices[(i, j)] as usize;
        // Skip self-loops and sentinel values (e.g. u32::MAX used when KNN couldn't find k neighbors)
        if knn_idx == i || knn_idx >= n_samples {
          continue;
        }
        let val = compute_membership_strength(i, j, knn_dists, rhos, sigmas);
        if val != 0.0 {
          count += 1;
        }
      }
      count
    })
    .collect();
  info!(
    duration_ms = started.elapsed().as_millis(),
    "csr row_counts complete"
  );

  // Step 2: Build indptr from prefix sum (u32 for memory efficiency)
  let started = Instant::now();
  let mut indptr: Vec<u32> = Vec::with_capacity(n_samples + 1);
  indptr.push(0);
  let mut total = 0u32;
  for &count in &row_counts {
    total = total.checked_add(count).expect("nnz overflows u32");
    indptr.push(total);
  }
  let nnz = total as usize;
  info!(
    duration_ms = started.elapsed().as_millis(),
    nnz, "csr indptr complete"
  );

  // Step 3: Pre-allocate indices and data, wrap in UnsafeCell for parallel access
  // SAFETY: Each row i writes only to [indptr[i]..indptr[i+1]], which are disjoint
  let indices_vec = ParallelVec::new(vec![0u32; nnz]);
  let data_vec = ParallelVec::new(vec![0.0f32; nnz]);

  // Step 4: Fill indices and data in parallel (each row writes to its own section)
  // No false sharing: each row is ~256 elements (~1KB with u32), writes are sequential within row.
  // Threads work on different rows, not adjacent elements.
  let started = Instant::now();
  (0..n_samples).into_par_iter().for_each(|i| {
    let row_start = indptr[i] as usize;
    let mut offset = 0;

    for j in 0..n_neighbors {
      if knn_disconnections.contains(&(i, j)) {
        continue;
      }
      let knn_idx = knn_indices[(i, j)];
      // Skip self-loops and sentinel values (must match count phase exactly)
      if knn_idx as usize == i || knn_idx as usize >= n_samples {
        continue;
      }
      let val = compute_membership_strength(i, j, knn_dists, rhos, sigmas);
      if val != 0.0 {
        // SAFETY: Each row writes to disjoint section [indptr[i]..indptr[i+1]]
        unsafe {
          indices_vec.write(row_start + offset, knn_idx);
          data_vec.write(row_start + offset, val);
        }
        offset += 1;
      }
    }
  });
  info!(
    duration_ms = started.elapsed().as_millis(),
    "csr fill complete"
  );

  // Step 5: Sort column indices within each row (required for valid CSR)
  // Each row can be sorted independently in parallel
  let started = Instant::now();
  (0..n_samples).into_par_iter().for_each(|i| {
    let row_start = indptr[i] as usize;
    let row_len = (indptr[i + 1] - indptr[i]) as usize;
    if row_len > 0 {
      // SAFETY: Each row accesses disjoint section [indptr[i]..indptr[i+1]]
      let row_indices = unsafe { indices_vec.get_mut_slice(row_start, row_len) };
      let row_data = unsafe { data_vec.get_mut_slice(row_start, row_len) };

      // Sort by column index (insertion sort is fast for small rows, ~256 elements)
      for k in 1..row_len {
        let mut m = k;
        while m > 0 && row_indices[m - 1] > row_indices[m] {
          row_indices.swap(m - 1, m);
          row_data.swap(m - 1, m);
          m -= 1;
        }
      }
    }
  });
  info!(
    duration_ms = started.elapsed().as_millis(),
    "csr row_sort complete"
  );

  // Extract Vecs from UnsafeCell wrappers and build CSR
  let indices = indices_vec.into_inner();
  let data = data_vec.into_inner();
  CsMatI::new((n_samples, n_samples), indptr, indices, data)
}

#[inline]
fn compute_membership_strength(
  i: usize,
  j: usize,
  knn_dists: ArrayView2<f32>,
  rhos: &ArrayView1<f32>,
  sigmas: &ArrayView1<f32>,
) -> f32 {
  if knn_dists[(i, j)] - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
    1.0
  } else {
    f32::exp(-(knn_dists[(i, j)] - rhos[i]) / sigmas[i])
  }
}

/// Build CSC structure (indptr + indices only, no data) for transpose traversal.
/// Values are looked up in original CSR when needed via binary search O(log k).
fn build_csc_structure(csr: &SparseMat) -> CscStructure {
  let n_rows = csr.shape().0;
  let n_cols = csr.shape().1;
  let nnz = csr.nnz();

  // Step 1: Count entries per column (parallel with atomics)
  let started = Instant::now();
  let col_counts: Vec<AtomicU32> = (0..n_cols).map(|_| AtomicU32::new(0)).collect();

  (0..n_rows).into_par_iter().for_each(|row| {
    let row_start = csr.indptr().index(row) as usize;
    let row_end = csr.indptr().index(row + 1) as usize;
    for &col in &csr.indices()[row_start..row_end] {
      col_counts[col as usize].fetch_add(1, Ordering::Relaxed);
    }
  });
  info!(
    duration_ms = started.elapsed().as_millis(),
    "csc col_counts complete"
  );

  // Step 2: Build column pointers (prefix sum)
  let started = Instant::now();
  let mut indptr: Vec<u32> = Vec::with_capacity(n_cols + 1);
  indptr.push(0);
  let mut total = 0u32;
  for count in &col_counts {
    total = total
      .checked_add(count.load(Ordering::Relaxed))
      .expect("nnz overflows u32");
    indptr.push(total);
  }
  assert_eq!(total as usize, nnz);
  info!(
    duration_ms = started.elapsed().as_millis(),
    "csc indptr complete"
  );

  // Step 3: Fill indices only (sequential to avoid atomic contention)
  // No data array - values will be looked up in CSR when needed.
  let started = Instant::now();
  let mut indices: Vec<u32> = vec![0; nnz];
  let mut col_offsets: Vec<u32> = vec![0; n_cols];

  for row in 0..n_rows {
    let row_start = csr.indptr().index(row) as usize;
    let row_end = csr.indptr().index(row + 1) as usize;
    let row_indices = &csr.indices()[row_start..row_end];

    for &col in row_indices {
      let write_pos = (indptr[col as usize] + col_offsets[col as usize]) as usize;
      indices[write_pos] = row as u32;
      col_offsets[col as usize] += 1;
    }
  }
  info!(
    duration_ms = started.elapsed().as_millis(),
    "csc fill complete"
  );
  // No sorting needed: iterating rows in order guarantees sorted row indices per column

  CscStructure { indptr, indices }
}

/// Binary search for value A[row, col] in CSR matrix. Returns 0.0 if not found.
#[inline]
fn csr_get(csr: &SparseMat, row: usize, col: u32) -> f32 {
  let row_start = csr.indptr().index(row) as usize;
  let row_end = csr.indptr().index(row + 1) as usize;
  let row_indices = &csr.indices()[row_start..row_end];
  let row_data = &csr.data()[row_start..row_end];

  match row_indices.binary_search(&col) {
    Ok(idx) => row_data[idx],
    Err(_) => 0.0,
  }
}

/// Apply fuzzy set union/intersection operations, building CSR directly.
///
/// Computes: set_op_mix_ratio * (A + A^T) + (1 - 2*set_op_mix_ratio) * (A ⊙ A^T)
/// where ⊙ is the Hadamard (elementwise) product.
///
/// The result is symmetric: for each pair (i,j) where A[i,j] OR A[j,i] is non-zero,
/// both output[i,j] and output[j,i] are set to the same computed value.
fn apply_set_operations_parallel(input: &SparseMat, set_op_mix_ratio: f32) -> SparseMat {
  let n_samples = input.shape().0;
  let prod_coeff = 1.0 - 2.0 * set_op_mix_ratio;

  // Build CSC structure (no data) for efficient transpose traversal
  // Values are looked up in original CSR via binary search (avoids duplicating data array)
  let started = Instant::now();
  let csc = build_csc_structure(input);
  info!(
    duration_ms = started.elapsed().as_millis(),
    "set_operations csc_structure complete"
  );

  // Step 1: Count output entries per row
  // For row r, entries come from:
  //   - A's row r (direct entries)
  //   - A's column r where A[r,c] doesn't exist (transpose entries without direct counterpart)
  let started = Instant::now();
  let row_counts: Vec<u32> = (0..n_samples)
    .into_par_iter()
    .map(|row| {
      // Count from A's row (direct entries)
      let row_start = input.indptr().index(row) as usize;
      let row_end = input.indptr().index(row + 1) as usize;
      let row_indices = &input.indices()[row_start..row_end];
      let row_data = &input.data()[row_start..row_end];

      let mut count = 0u32;
      for (&col, &val_rc) in row_indices.iter().zip(row_data) {
        let val_cr = csr_get(input, col as usize, row as u32);
        let final_val =
          set_op_mix_ratio * val_rc + set_op_mix_ratio * val_cr + prod_coeff * val_rc * val_cr;
        if final_val != 0.0 {
          count += 1;
        }
      }

      // Count from A's column (transpose entries without direct counterpart)
      // CSC tells us which rows c have entries in column `row` (i.e., A[c, row] exists)
      for &c in csc.col_row_indices(row) {
        // Skip if direct entry A[row, c] exists (already counted above)
        if csr_get(input, row, c) != 0.0 {
          continue;
        }
        // val_rc = 0 since no direct entry, val_cr = A[c, row]
        let val_cr = csr_get(input, c as usize, row as u32);
        let final_val = set_op_mix_ratio * val_cr; // Simplified: 0 + mix*val_cr + 0
        if final_val != 0.0 {
          count += 1;
        }
      }

      count
    })
    .collect();
  info!(
    duration_ms = started.elapsed().as_millis(),
    "set_operations row_counts complete"
  );

  // Step 2: Build indptr
  let started = Instant::now();
  let mut indptr: Vec<u32> = Vec::with_capacity(n_samples + 1);
  indptr.push(0);
  let mut total = 0u32;
  for &count in &row_counts {
    total = total.checked_add(count).expect("nnz overflows u32");
    indptr.push(total);
  }
  let nnz = total as usize;
  info!(
    duration_ms = started.elapsed().as_millis(),
    nnz, "set_operations indptr complete"
  );

  // Step 3: Pre-allocate and wrap in UnsafeCell for parallel access
  // SAFETY: Each row writes only to [indptr[row]..indptr[row+1]], which are disjoint
  // No false sharing: each row section is ~512 elements (~2KB after symmetrization with u32),
  // writes are sequential within row. Threads work on different rows.
  let indices_vec = ParallelVec::new(vec![0u32; nnz]);
  let data_vec = ParallelVec::new(vec![0.0f32; nnz]);

  let started = Instant::now();
  (0..n_samples).into_par_iter().for_each(|row| {
    let out_start = indptr[row] as usize;
    let mut offset = 0;

    // Fill from A's row (direct entries)
    let row_start = input.indptr().index(row) as usize;
    let row_end = input.indptr().index(row + 1) as usize;
    let row_indices = &input.indices()[row_start..row_end];
    let row_data = &input.data()[row_start..row_end];

    for (&col, &val_rc) in row_indices.iter().zip(row_data) {
      let val_cr = csr_get(input, col as usize, row as u32);
      let final_val =
        set_op_mix_ratio * val_rc + set_op_mix_ratio * val_cr + prod_coeff * val_rc * val_cr;
      if final_val != 0.0 {
        // SAFETY: Each row writes to disjoint section [indptr[row]..indptr[row+1]]
        unsafe {
          indices_vec.write(out_start + offset, col);
          data_vec.write(out_start + offset, final_val);
        }
        offset += 1;
      }
    }

    // Fill from A's column (transpose entries without direct counterpart)
    for &c in csc.col_row_indices(row) {
      // Skip if direct entry exists (already filled above)
      if csr_get(input, row, c) != 0.0 {
        continue;
      }
      let val_cr = csr_get(input, c as usize, row as u32);
      let final_val = set_op_mix_ratio * val_cr;
      if final_val != 0.0 {
        // SAFETY: Each row writes to disjoint section
        unsafe {
          indices_vec.write(out_start + offset, c);
          data_vec.write(out_start + offset, final_val);
        }
        offset += 1;
      }
    }
  });
  info!(
    duration_ms = started.elapsed().as_millis(),
    "set_operations fill complete"
  );

  // Step 4: Sort columns within each row (entries may be unsorted after combining)
  let started = Instant::now();
  (0..n_samples).into_par_iter().for_each(|row| {
    let row_start = indptr[row] as usize;
    let row_len = (indptr[row + 1] - indptr[row]) as usize;
    if row_len > 1 {
      // SAFETY: Each row accesses disjoint section
      let row_indices = unsafe { indices_vec.get_mut_slice(row_start, row_len) };
      let row_data = unsafe { data_vec.get_mut_slice(row_start, row_len) };

      // Insertion sort (rows are small after set operations)
      for k in 1..row_len {
        let mut m = k;
        while m > 0 && row_indices[m - 1] > row_indices[m] {
          row_indices.swap(m - 1, m);
          row_data.swap(m - 1, m);
          m -= 1;
        }
      }
    }
  });
  info!(
    duration_ms = started.elapsed().as_millis(),
    "set_operations row_sort complete"
  );

  // Extract Vecs from UnsafeCell wrappers and build CSR
  let indices = indices_vec.into_inner();
  let data = data_vec.into_inner();
  CsMatI::new((n_samples, n_samples), indptr, indices, data)
}
