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
#[inline(always)]
pub fn clip(val: f32) -> f32 {
  // OPTIMIZATION: Manual clamp with fast path for common case (no clipping needed)
  // Most values in UMAP are within range, so this helps branch prediction
  if val > -4.0 && val < 4.0 {
    val
  } else if val <= -4.0 {
    -4.0
  } else {
    4.0
  }
}
