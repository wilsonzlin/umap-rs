use super::constants::MIN_K_DIST_SCALE;
use super::constants::SMOOTH_K_TOLERANCE;
use ndarray::Array1;
use ndarray::ArrayView2;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Debug)]
pub struct SmoothKnnDist<'a> {
  distances: ArrayView2<'a, f32>,
  k: usize,
  #[builder(default = 64)]
  n_iter: usize,
  #[builder(default = 1.0)]
  local_connectivity: f32,
  #[builder(default = 1.0)]
  bandwidth: f32,
}

impl<'a> SmoothKnnDist<'a> {
  /*
    Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each sample. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
  */
  pub fn exec(self) -> (Array1<f32>, Array1<f32>) {
    let SmoothKnnDist {
      distances,
      k,
      n_iter,
      local_connectivity,
      bandwidth,
    } = self;

    let target = (k as f32).log2() * bandwidth;
    let mut rho = Array1::<f32>::zeros(distances.shape()[0]);
    let mut result = Array1::<f32>::zeros(distances.shape()[0]);

    let mean_distances = distances.mean().unwrap();

    for i in 0..distances.shape()[0] {
      let mut lo = 0.0;
      let mut hi = f32::INFINITY;
      let mut mid = 1.0;

      let ith_distances = distances.row(i);
      let non_zero_dists: Vec<f32> = ith_distances
        .iter()
        .filter(|&&d| d > 0.0)
        .copied()
        .collect();

      if non_zero_dists.len() >= local_connectivity as usize {
        let index = local_connectivity.floor() as usize;
        let interpolation = local_connectivity - local_connectivity.floor();
        if index > 0 {
          rho[i] = non_zero_dists[index - 1];
          if interpolation > SMOOTH_K_TOLERANCE {
            rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1]);
          }
        } else {
          rho[i] = interpolation * non_zero_dists[0];
        }
      } else if !non_zero_dists.is_empty() {
        rho[i] = *non_zero_dists
          .iter()
          .max_by(|a, b| a.partial_cmp(b).unwrap())
          .unwrap();
      }

      for _n in 0..n_iter {
        let mut psum = 0.0;
        for j in 1..distances.shape()[1] {
          let d = distances[(i, j)] - rho[i];
          if d > 0.0 {
            psum += f32::exp(-(d / mid));
          } else {
            psum += 1.0;
          }
        }

        if f32::abs(psum - target) < SMOOTH_K_TOLERANCE {
          break;
        }

        if psum > target {
          hi = mid;
          mid = (lo + hi) / 2.0;
        } else {
          lo = mid;
          if hi == f32::INFINITY {
            mid *= 2.0
          } else {
            mid = (lo + hi) / 2.0;
          }
        }
      }

      result[i] = mid;

      if rho[i] > 0.0 {
        let mean_ith_distances = ith_distances.mean().unwrap();
        if result[i] < MIN_K_DIST_SCALE * mean_ith_distances {
          result[i] = MIN_K_DIST_SCALE * mean_ith_distances;
        }
      } else if result[i] < MIN_K_DIST_SCALE * mean_distances {
        result[i] = MIN_K_DIST_SCALE * mean_distances;
      }
    }

    (result, rho)
  }
}
