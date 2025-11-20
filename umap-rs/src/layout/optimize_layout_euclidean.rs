use crate::utils::clip::clip;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut2;
use rand::Rng;
use rayon::prelude::*;
use typed_builder::TypedBuilder;

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
/// 3. **Performance is critical** - The speedup from parallelism (4-8x) vastly outweighs
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

/*
  Improve an embedding using stochastic gradient descent to minimize the
  fuzzy set cross entropy between the 1-skeletons of the high dimensional
  and low dimensional fuzzy simplicial sets. In practice this is done by
  sampling edges based on their membership strength (with the (1-p) terms
  coming from negative sampling similar to word2vec).

  This is the Euclidean specialization of the optimization, using squared
  Euclidean distance (rdist) for better performance.

  Parameters
  ----------
  head_embedding: array of shape (n_samples, n_components)
      The initial embedding to be improved by SGD.

  tail_embedding: array of shape (source_samples, n_components)
      The reference embedding of embedded points. If not embedding new
      previously unseen points with respect to an existing embedding this
      is simply the head_embedding (again); otherwise it provides the
      existing embedding to embed with respect to.

  head: array of shape (n_1_simplices)
      The indices of the heads of 1-simplices with non-zero membership.

  tail: array of shape (n_1_simplices)
      The indices of the tails of 1-simplices with non-zero membership.

  n_epochs: int
      The number of training epochs to use in optimization.

  n_vertices: int
      The number of vertices (0-simplices) in the dataset.

  epochs_per_sample: array of shape (n_1_simplices)
      A float value of the number of epochs per 1-simplex. 1-simplices with
      weaker membership strength will have more epochs between being sampled.

  a: float
      Parameter of differentiable approximation of right adjoint functor

  b: float
      Parameter of differentiable approximation of right adjoint functor

  gamma: float (optional, default 1.0)
      Weight to apply to negative samples.

  initial_alpha: float (optional, default 1.0)
      Initial learning rate for the SGD.

  negative_sample_rate: float (optional, default 5.0)
      Number of negative samples to use per positive sample.

  parallel: bool (optional, default true)
      Whether to run the computation in parallel using rayon.

  move_other: bool (optional, default false)
      Whether to adjust tail_embedding alongside head_embedding

  Returns
  -------
  head_embedding: array of shape (n_samples, n_components)
      The optimized embedding.
*/
#[derive(TypedBuilder)]
pub struct OptimizeLayoutEuclidean<'a> {
  head_embedding: &'a mut ArrayViewMut2<'a, f32>,
  tail_embedding: &'a mut ArrayViewMut2<'a, f32>,
  head: &'a ArrayView1<'a, u32>,
  tail: &'a ArrayView1<'a, u32>,
  n_epochs: usize,
  n_vertices: usize,
  epochs_per_sample: &'a ArrayView1<'a, f64>,
  a: f32,
  b: f32,
  #[builder(default = 1.0)]
  gamma: f32,
  #[builder(default = 1.0)]
  initial_alpha: f32,
  #[builder(default = 5.0)]
  negative_sample_rate: f64,
  #[builder(default = true)]
  parallel: bool,
  #[builder(default = false)]
  move_other: bool,
}

impl<'a> OptimizeLayoutEuclidean<'a> {
  pub fn exec(self) {
    let Self {
      head_embedding,
      tail_embedding,
      head,
      tail,
      n_epochs,
      n_vertices,
      epochs_per_sample,
      a,
      b,
      gamma,
      initial_alpha,
      negative_sample_rate,
      parallel,
      move_other,
    } = self;

    let dim = head_embedding.shape()[1];

    // Calculate epochs per negative sample
    let mut epochs_per_negative_sample = Array1::<f64>::zeros(epochs_per_sample.len());
    for i in 0..epochs_per_sample.len() {
      epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate;
    }
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let mut epoch_of_next_sample = epochs_per_sample.to_owned();

    let mut alpha = initial_alpha;

    for n in 0..n_epochs {

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
          &mut epochs_per_negative_sample,
          &mut epoch_of_next_negative_sample,
          &mut epoch_of_next_sample,
          n,
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
          &mut epochs_per_negative_sample,
          &mut epoch_of_next_negative_sample,
          &mut epoch_of_next_sample,
          n,
        );
      }

      alpha = initial_alpha * (1.0 - (n as f32 / n_epochs as f32));
    }
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
