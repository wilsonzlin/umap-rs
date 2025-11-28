use crate::utils::clip::clip;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut2;
use rand::Rng;
use rayon::prelude::*;

/// Wrapper to allow concurrent mutable access to embedding arrays in parallel SGD.
///
/// # Safety
///
/// This type explicitly allows data races on the underlying f32 values. This is acceptable
/// for stochastic gradient descent because:
///
/// 1. **The algorithm is inherently stochastic** - SGD already has randomness, and occasional
///    lost updates don't affect convergence.
///
/// 2. **Races are rare** - In typical graphs, most edges don't share vertices in the same
///    parallel batch, so conflicts are infrequent.
///
/// 3. **Performance is critical** - The speedup from parallelism vastly outweighs
///    the negligible impact of rare lost updates.
///
/// 4. **Matches reference implementation** - Python UMAP with Numba uses `parallel=True`
///    which has the same race behavior.
///
/// This is a well-known pattern in parallel SGD implementations. See:
/// - Hogwild! algorithm (Recht et al., 2011)
/// - Numba's parallel prange with relaxed memory ordering
struct UnsafeSyncCell<T> {
  ptr: *mut T,
}

unsafe impl<T> Send for UnsafeSyncCell<T> {}
unsafe impl<T> Sync for UnsafeSyncCell<T> {}

impl<T> UnsafeSyncCell<T> {
  /// Creates a new UnsafeSyncCell from a mutable pointer.
  ///
  /// # Safety
  ///
  /// The caller must ensure that:
  /// - The pointer remains valid for the lifetime of this cell
  /// - Concurrent unsynchronized access is acceptable for the use case
  unsafe fn new(ptr: *mut T) -> Self {
    Self { ptr }
  }

  /// Returns the underlying mutable pointer.
  ///
  /// # Safety
  ///
  /// The caller must ensure proper synchronization if required by the use case.
  #[inline(always)]
  fn get(&self) -> *mut T {
    self.ptr
  }
}

/// Execute a single epoch of Euclidean UMAP optimization.
///
/// Performs one epoch of stochastic gradient descent to optimize the embedding,
/// using squared Euclidean distance for better performance.
///
/// The epoch state arrays are updated in-place, allowing for stateful optimization
/// with checkpointing support.
///
/// # Arguments
///
/// * `head_embedding` - The embedding being optimized (mutated in-place)
/// * `tail_embedding` - Reference embedding (usually same as head for standard UMAP)
/// * `head` - Head vertex indices for each edge
/// * `tail` - Tail vertex indices for each edge
/// * `n_vertices` - Total number of vertices in the dataset
/// * `epochs_per_sample` - Sampling frequency for each edge
/// * `a`, `b` - Curve parameters for distance-to-probability transformation
/// * `gamma` - Repulsion strength for negative samples
/// * `alpha` - Current learning rate for this epoch
/// * `epochs_per_negative_sample` - Negative sampling frequency per edge
/// * `epoch_of_next_sample` - Next epoch to sample each positive edge (mutated)
/// * `epoch_of_next_negative_sample` - Next epoch for negative sampling (mutated)
/// * `current_epoch` - Current epoch number
/// * `parallel` - Whether to use parallel execution
/// * `move_other` - Whether to update tail_embedding as well
#[allow(clippy::too_many_arguments)]
pub fn optimize_layout_euclidean_single_epoch_stateful(
  head_embedding: &mut ArrayViewMut2<f32>,
  tail_embedding: &mut ArrayViewMut2<f32>,
  head: &ArrayView1<u32>,
  tail: &ArrayView1<u32>,
  n_vertices: usize,
  epochs_per_sample: &ArrayView1<f64>,
  a: f32,
  b: f32,
  gamma: f32,
  alpha: f32,
  epochs_per_negative_sample: &mut Array1<f64>,
  epoch_of_next_sample: &mut Array1<f64>,
  epoch_of_next_negative_sample: &mut Array1<f64>,
  current_epoch: usize,
  parallel: bool,
  move_other: bool,
) {
  let dim = head_embedding.shape()[1];

  if parallel {
    optimize_layout_euclidean_single_epoch_parallel(
      head_embedding,
      tail_embedding,
      head,
      tail,
      n_vertices,
      epochs_per_sample,
      a,
      b,
      gamma,
      dim,
      move_other,
      alpha,
      epochs_per_negative_sample,
      epoch_of_next_negative_sample,
      epoch_of_next_sample,
      current_epoch,
    );
  } else {
    optimize_layout_euclidean_single_epoch(
      head_embedding,
      tail_embedding,
      head,
      tail,
      n_vertices,
      epochs_per_sample,
      a,
      b,
      gamma,
      dim,
      move_other,
      alpha,
      epochs_per_negative_sample,
      epoch_of_next_negative_sample,
      epoch_of_next_sample,
      current_epoch,
    );
  }
}

#[allow(clippy::too_many_arguments)]
fn optimize_layout_euclidean_single_epoch(
  head_embedding: &mut ArrayViewMut2<f32>,
  tail_embedding: &mut ArrayViewMut2<f32>,
  head: &ArrayView1<u32>,
  tail: &ArrayView1<u32>,
  n_vertices: usize,
  epochs_per_sample: &ArrayView1<f64>,
  a: f32,
  b: f32,
  gamma: f32,
  dim: usize,
  move_other: bool,
  alpha: f32,
  epochs_per_negative_sample: &mut Array1<f64>,
  epoch_of_next_negative_sample: &mut Array1<f64>,
  epoch_of_next_sample: &mut Array1<f64>,
  n: usize,
) {
  let mut rng = rand::rng();

  for i in 0..epochs_per_sample.len() {
    if epoch_of_next_sample[i] <= n as f64 {
      let j = head[i] as usize;
      let k = tail[i] as usize;

      let mut dist_squared = 0.0_f32;
      for d in 0..dim {
        let diff = head_embedding[(j, d)] - tail_embedding[(k, d)];
        dist_squared += diff * diff;
      }

      let grad_coeff = if dist_squared > 0.0 {
        let dist_pow_b = f32::powf(dist_squared, b);
        let mut gc = -2.0 * a * b * dist_pow_b / dist_squared;
        gc /= a * dist_pow_b * dist_squared + 1.0;
        gc
      } else {
        0.0
      };

      for d in 0..dim {
        let diff = head_embedding[(j, d)] - tail_embedding[(k, d)];
        let grad_d = clip(grad_coeff * diff);
        head_embedding[(j, d)] += grad_d * alpha;
        if move_other {
          tail_embedding[(k, d)] -= grad_d * alpha;
        }
      }

      epoch_of_next_sample[i] += epochs_per_sample[i];

      let n_neg_samples =
        ((n as f64 - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]) as usize;

      for _p in 0..n_neg_samples {
        let k = rng.random_range(0..n_vertices);
        if j == k {
          continue;
        }

        let mut dist_squared = 0.0_f32;
        for d in 0..dim {
          let diff = head_embedding[(j, d)] - tail_embedding[(k, d)];
          dist_squared += diff * diff;
        }

        let grad_coeff = if dist_squared > 0.0 {
          let dist_pow_b = f32::powf(dist_squared, b);
          2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_pow_b * dist_squared + 1.0))
        } else {
          0.0
        };

        if grad_coeff > 0.0 {
          for d in 0..dim {
            let diff = head_embedding[(j, d)] - tail_embedding[(k, d)];
            let grad_d = clip(grad_coeff * diff);
            head_embedding[(j, d)] += grad_d * alpha;
          }
        }
      }

      epoch_of_next_negative_sample[i] += n_neg_samples as f64 * epochs_per_negative_sample[i];
    }
  }
}

#[allow(clippy::too_many_arguments)]
fn optimize_layout_euclidean_single_epoch_parallel(
  head_embedding: &mut ArrayViewMut2<f32>,
  tail_embedding: &mut ArrayViewMut2<f32>,
  head: &ArrayView1<u32>,
  tail: &ArrayView1<u32>,
  n_vertices: usize,
  epochs_per_sample: &ArrayView1<f64>,
  a: f32,
  b: f32,
  gamma: f32,
  dim: usize,
  move_other: bool,
  alpha: f32,
  epochs_per_negative_sample: &mut Array1<f64>,
  epoch_of_next_negative_sample: &mut Array1<f64>,
  epoch_of_next_sample: &mut Array1<f64>,
  n: usize,
) {
  // SAFETY: We allow concurrent mutable access to the embeddings because:
  // 1. SGD is inherently stochastic - races don't affect convergence
  // 2. Most edges don't share vertices, so conflicts are rare
  // 3. This matches the Python/Numba implementation's behavior
  let head_cell = unsafe { UnsafeSyncCell::new(head_embedding.as_mut_ptr()) };
  let tail_cell = unsafe { UnsafeSyncCell::new(tail_embedding.as_mut_ptr()) };
  let head_stride = head_embedding.shape()[1];
  let tail_stride = tail_embedding.shape()[1];

  // SAFETY: Each parallel iteration i accesses only epoch_of_next_sample[i] and
  // epoch_of_next_negative_sample[i], so there are no data races between threads
  let epoch_sample_cell = unsafe { UnsafeSyncCell::new(epoch_of_next_sample.as_mut_ptr()) };
  let epoch_neg_cell = unsafe { UnsafeSyncCell::new(epoch_of_next_negative_sample.as_mut_ptr()) };

  (0..epochs_per_sample.len()).into_par_iter().for_each(|i| {
    let mut rng = rand::rng();

    unsafe {
      let head_ptr = head_cell.get();
      let tail_ptr = tail_cell.get();
      let epoch_sample_ptr = epoch_sample_cell.get();
      let epoch_neg_ptr = epoch_neg_cell.get();

      if *epoch_sample_ptr.add(i) <= n as f64 {
        let j = head[i] as usize;
        let k = tail[i] as usize;

        let current_base = head_ptr.add(j * head_stride);
        let other_base = tail_ptr.add(k * tail_stride);

        let mut dist_squared = 0.0_f32;
        for d in 0..dim {
          let diff = *current_base.add(d) - *other_base.add(d);
          dist_squared += diff * diff;
        }

        let grad_coeff = if dist_squared > 0.0 {
          let dist_pow_b = f32::powf(dist_squared, b);
          let mut gc = -2.0 * a * b * dist_pow_b / dist_squared;
          gc /= a * dist_pow_b * dist_squared + 1.0;
          gc
        } else {
          0.0
        };

        for d in 0..dim {
          let diff = *current_base.add(d) - *other_base.add(d);
          let grad_d = clip(grad_coeff * diff);
          *current_base.add(d) += grad_d * alpha;
          if move_other {
            *other_base.add(d) -= grad_d * alpha;
          }
        }

        *epoch_sample_ptr.add(i) += epochs_per_sample[i];

        let n_neg_samples =
          ((n as f64 - *epoch_neg_ptr.add(i)) / epochs_per_negative_sample[i]) as usize;

        for _p in 0..n_neg_samples {
          let k = rng.random_range(0..n_vertices);
          if j == k {
            continue;
          }

          let other_base = tail_ptr.add(k * tail_stride);

          let mut dist_squared = 0.0_f32;
          for d in 0..dim {
            let diff = *current_base.add(d) - *other_base.add(d);
            dist_squared += diff * diff;
          }

          let grad_coeff = if dist_squared > 0.0 {
            let dist_pow_b = f32::powf(dist_squared, b);
            2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_pow_b * dist_squared + 1.0))
          } else {
            0.0
          };

          if grad_coeff > 0.0 {
            for d in 0..dim {
              let diff = *current_base.add(d) - *other_base.add(d);
              let grad_d = clip(grad_coeff * diff);
              *current_base.add(d) += grad_d * alpha;
            }
          }
        }

        *epoch_neg_ptr.add(i) += n_neg_samples as f64 * epochs_per_negative_sample[i];
      }
    }
  });
}
