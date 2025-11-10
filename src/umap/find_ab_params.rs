use ndarray::Array1;

/*
  Fit a, b params for the differentiable curve used in lower
  dimensional fuzzy simplicial complex construction. We want the
  smooth curve (from a pre-defined family with simple gradient) that
  best matches an offset exponential decay.
*/
pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
  // Generate x values from 0 to spread * 3
  let n_points = 300;
  let mut xv = Array1::<f32>::zeros(n_points);
  for i in 0..n_points {
    xv[i] = (spread * 3.0) * (i as f32) / (n_points as f32 - 1.0);
  }

  // Generate y values: 1.0 for x < min_dist, exponential decay for x >= min_dist
  let mut yv = Array1::<f32>::zeros(n_points);
  for i in 0..n_points {
    let x = xv[i];
    if x < min_dist {
      yv[i] = 1.0;
    } else {
      yv[i] = f32::exp(-(x - min_dist) / spread);
    }
  }

  // Curve function: f(x) = 1 / (1 + a * x^(2*b))
  // We use Levenberg-Marquardt-style iterative fitting
  // Starting with reasonable initial guesses based on typical UMAP behavior
  let mut a = 1.5;
  let mut b = 0.9;

  // Simple gradient descent to fit the curve
  let learning_rate = 0.01;
  let n_iterations = 1000;

  for _ in 0..n_iterations {
    let mut grad_a = 0.0;
    let mut grad_b = 0.0;
    let mut total_error = 0.0;

    for i in 0..n_points {
      let x = xv[i];
      let y_true = yv[i];

      // Compute predicted value
      let x_2b = f32::powf(x, 2.0 * b);
      let denom = 1.0 + a * x_2b;
      let y_pred = 1.0 / denom;

      // Compute error
      let error = y_pred - y_true;
      total_error += error * error;

      // Compute gradients
      // dy/da = -x^(2b) / (1 + a*x^(2b))^2
      grad_a += 2.0 * error * (-x_2b / (denom * denom));

      // dy/db = -2*a*x^(2b)*ln(x) / (1 + a*x^(2b))^2
      if x > 0.0 {
        let ln_x = f32::ln(x);
        grad_b += 2.0 * error * (-2.0 * a * x_2b * ln_x / (denom * denom));
      }
    }

    // Update parameters
    a -= learning_rate * grad_a / n_points as f32;
    b -= learning_rate * grad_b / n_points as f32;

    // Clamp to reasonable ranges
    a = a.clamp(0.001, 10.0);
    b = b.clamp(0.001, 10.0);

    // Early stopping if error is small enough
    if total_error / (n_points as f32) < 1e-7 {
      break;
    }
  }

  (a, b)
}
