use ndarray::{Array1, ArrayView1};


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
pub fn make_epochs_per_sample(weights: &ArrayView1<f64>, n_epochs: usize) -> Array1<f64> {
  let result = -1.0 * Array1::<f64>::ones(weights.shape()[0]);
  let n_samples = n_epochs * (weights / weights.max());
  result[n_samples > 0] = n_epochs as f64 / n_samples[n_samples > 0];
  result
}
