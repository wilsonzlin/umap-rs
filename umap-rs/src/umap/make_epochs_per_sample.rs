use ndarray::Array1;
use ndarray::ArrayView1;

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
  let mut result = Array1::<f64>::from_elem(weights.len(), -1.0);

  // Find max weight
  let max_weight = weights
    .iter()
    .copied()
    .max_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap_or(1.0);

  // Calculate n_samples = n_epochs * (weights / max_weight)
  // Then result[n_samples > 0] = n_epochs / n_samples[n_samples > 0]
  for i in 0..weights.len() {
    let n_samples = n_epochs as f64 * (weights[i] as f64 / max_weight as f64);
    if n_samples > 0.0 {
      result[i] = n_epochs as f64 / n_samples;
    }
  }

  result
}
