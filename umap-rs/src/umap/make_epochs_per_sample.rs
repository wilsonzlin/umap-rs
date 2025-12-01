use ndarray::Array1;
use ndarray::ArrayView1;
use rayon::prelude::*;

/*
  Given a set of weights and number of epochs generate the number of
  epochs per sample for each weight.

  Parameters
  ----------
  weights: array of shape (n_1_simplices)
      The weights of how much we wish to sample each 1-simplex.

  n_epochs: int
      The total number of epochs we want to train for.

  Returns
  -------
  An array of number of epochs per sample, one for each 1-simplex.
*/
pub fn make_epochs_per_sample(weights: &ArrayView1<f32>, n_epochs: usize) -> Array1<f64> {
  // Find max weight (parallel)
  let max_weight = weights
    .as_slice()
    .unwrap()
    .par_iter()
    .copied()
    .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));
  let max_weight = if max_weight <= 0.0 { 1.0 } else { max_weight };

  // Calculate epochs per sample (parallel)
  // n_samples = n_epochs * (weight / max_weight)
  // result = n_epochs / n_samples = max_weight / weight
  let n_epochs_f64 = n_epochs as f64;
  let max_weight_f64 = max_weight as f64;

  let result_vec: Vec<f64> = weights
    .as_slice()
    .unwrap()
    .par_iter()
    .map(|&w| {
      let n_samples = n_epochs_f64 * (w as f64 / max_weight_f64);
      if n_samples > 0.0 {
        n_epochs_f64 / n_samples
      } else {
        -1.0
      }
    })
    .collect();

  Array1::from(result_vec)
}
