use ndarray::ArrayView1;

/*
  Standard clamping of a value into a fixed range (in this case -4.0 to 4.0)

  Parameters
  ----------
  val: float
      The value to be clamped.

  Returns
  -------
  The clamped value, now fixed to be in the range -4.0 to 4.0.
*/
#[inline]
pub fn clip(val: f32) -> f32 {
    if val > 4.0 {
        return 4.0;
    } else if val < -4.0 {
        return -4.0;
    } else {
        return val;
    }
}

/*
  Reduced Euclidean distance (squared Euclidean).

  Parameters
  ----------
  x: array of shape (embedding_dim,)
  y: array of shape (embedding_dim,)

  Returns
  -------
  The squared euclidean distance between x and y
*/
#[inline]
pub fn rdist(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    let mut result = 0.0;
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        result += diff * diff;
    }
    result
}
