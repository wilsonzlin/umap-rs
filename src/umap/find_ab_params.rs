use ndarray::linspace;

/*
  Fit a, b params for the differentiable curve used in lower
  dimensional fuzzy simplicial complex construction. We want the
  smooth curve (from a pre-defined family with simple gradient) that
  best matches an offset exponential decay.
*/
pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    fn curve(x: f32, a: f32, b: f32) {
        1.0 / (1.0 + a * x ** (2 * b))
    }

    let xv = linspace(0, spread * 3, 300);
    let yv = np.zeros(xv.shape);
    yv[xv < min_dist] = 1.0;
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread);
    let (params, covar) = curve_fit(curve, xv, yv);
    (params[0], params[1])
}
