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
  val.clamp(-4.0, 4.0)
}
