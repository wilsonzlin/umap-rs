use ndarray::{Array1, ArrayView1, ArrayViewMut2};
use rand::Rng;
use rayon::prelude::*;
use typed_builder::TypedBuilder;

use crate::{distances::rdist, utils::clip::clip};

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
  pub fn exec(self) -> &'a mut ArrayViewMut2<'a, f32> {
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

    for n in 0..n_epochs {
      let alpha = initial_alpha * (1.0 - (n as f32 / n_epochs as f32));

      if parallel {
        // Parallel version using rayon - iterate over edges
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
        // Sequential version
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
    }

    head_embedding
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

      let current = head_embedding.row(j).to_owned();
      let other = tail_embedding.row(k).to_owned();

      let dist_squared = rdist(&current.view(), &other.view());

      let grad_coeff = if dist_squared > 0.0 {
        let mut gc = -2.0 * a * b * f32::powf(dist_squared, b - 1.0);
        gc /= a * f32::powf(dist_squared, b) + 1.0;
        gc
      } else {
        0.0
      };

      for d in 0..dim {
        let grad_d = clip(grad_coeff * (current[d] - other[d]));

        head_embedding[(j, d)] += grad_d * alpha;
        if move_other {
          tail_embedding[(k, d)] += -grad_d * alpha;
        }
      }

      epoch_of_next_sample[i] += epochs_per_sample[i];

      let n_neg_samples = ((n as f64 - epoch_of_next_negative_sample[i])
        / epochs_per_negative_sample[i]) as usize;

      for _p in 0..n_neg_samples {
        let k = rng.random_range(0..n_vertices);

        let current = head_embedding.row(j).to_owned();
        let other = tail_embedding.row(k).to_owned();

        let dist_squared = rdist(&current.view(), &other.view());

        let grad_coeff = if dist_squared > 0.0 {
          let mut gc = 2.0 * gamma * b;
          gc /= (0.001 + dist_squared) * (a * f32::powf(dist_squared, b) + 1.0);
          gc
        } else if j == k {
          continue;
        } else {
          0.0
        };

        for d in 0..dim {
          if grad_coeff > 0.0 {
            let grad_d = clip(grad_coeff * (current[d] - other[d]));
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
  // For parallel execution, we need to be careful about concurrent writes.
  // Since each edge updates different vertices (with possible conflicts),
  // we use unsafe code with raw pointers to allow parallel mutation.

  let head_ptr = head_embedding.as_mut_ptr();
  let tail_ptr = tail_embedding.as_mut_ptr();
  let head_stride = head_embedding.shape()[1];
  let tail_stride = tail_embedding.shape()[1];

  (0..epochs_per_sample.len()).into_par_iter().for_each(|i| {
    let mut rng = rand::rng();

    if epoch_of_next_sample[i] <= n as f64 {
      let j = head[i] as usize;
      let k = tail[i] as usize;

      unsafe {
        // Read current and other
        let current_base = head_ptr.add(j * head_stride);
        let other_base = tail_ptr.add(k * tail_stride);

        let mut current = Vec::with_capacity(dim);
        let mut other = Vec::with_capacity(dim);
        for d in 0..dim {
          current.push(*current_base.add(d));
          other.push(*other_base.add(d));
        }

        let dist_squared = {
          let mut sum = 0.0;
          for d in 0..dim {
            let diff = current[d] - other[d];
            sum += diff * diff;
          }
          sum
        };

        let grad_coeff = if dist_squared > 0.0 {
          let mut gc = -2.0 * a * b * f32::powf(dist_squared, b - 1.0);
          gc /= a * f32::powf(dist_squared, b) + 1.0;
          gc
        } else {
          0.0
        };

        for d in 0..dim {
          let grad_d = clip(grad_coeff * (current[d] - other[d]));
          *current_base.add(d) += grad_d * alpha;
          if move_other {
            *other_base.add(d) += -grad_d * alpha;
          }
        }
      }

      epoch_of_next_sample[i] += epochs_per_sample[i];

      let n_neg_samples = ((n as f64 - epoch_of_next_negative_sample[i])
        / epochs_per_negative_sample[i]) as usize;

      for _p in 0..n_neg_samples {
        let k = rng.random_range(0..n_vertices);
        if j == k {
          continue;
        }

        unsafe {
          let current_base = head_ptr.add(j * head_stride);
          let other_base = tail_ptr.add(k * tail_stride);

          let mut current = Vec::with_capacity(dim);
          let mut other = Vec::with_capacity(dim);
          for d in 0..dim {
            current.push(*current_base.add(d));
            other.push(*other_base.add(d));
          }

          let dist_squared = {
            let mut sum = 0.0;
            for d in 0..dim {
              let diff = current[d] - other[d];
              sum += diff * diff;
            }
            sum
          };

          let grad_coeff = if dist_squared > 0.0 {
            let mut gc = 2.0 * gamma * b;
            gc /= (0.001 + dist_squared) * (a * f32::powf(dist_squared, b) + 1.0);
            gc
          } else {
            0.0
          };

          for d in 0..dim {
            if grad_coeff > 0.0 {
              let grad_d = clip(grad_coeff * (current[d] - other[d]));
              *current_base.add(d) += grad_d * alpha;
            }
          }
        }
      }

      epoch_of_next_negative_sample[i] += n_neg_samples as f64 * epochs_per_negative_sample[i];
    }
  });
}
