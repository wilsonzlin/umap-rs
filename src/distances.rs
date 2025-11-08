use ndarray::{Array1, ArrayView1};

use crate::metric::Metric;

#[inline]
fn sign(a: f32) -> f32 {
    if a < 0.0 {
        -1.0
    } else {
        1.0
    }
}

#[derive(Debug)]
pub struct Euclidean;

impl Euclidean {
    #[inline]
    pub fn distance_only(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
        let mut result = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            result += diff * diff;
        }
        result.sqrt()
    }
}

impl Metric for Euclidean {
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> (f32, Array1<f32>) {
        let mut result = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            result += diff * diff;
        }
        let d = result.sqrt();
        let mut grad = Array1::<f32>::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = (x[i] - y[i]) / (1e-6 + d);
        }
        (d, grad)
    }

    fn is_euclidean(&self) -> bool {
        true
    }

    fn default_disconnection_distance(&self) -> f32 {
        f32::INFINITY
    }
}

#[derive(Debug)]
pub struct Manhattan;

impl Metric for Manhattan {
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> (f32, Array1<f32>) {
        let mut result = 0.0;
        let mut grad = Array1::<f32>::zeros(x.len());
        for i in 0..x.len() {
            result += (x[i] - y[i]).abs();
            grad[i] = sign(x[i] - y[i]);
        }
        (result, grad)
    }

    fn is_euclidean(&self) -> bool {
        false
    }

    fn default_disconnection_distance(&self) -> f32 {
        f32::INFINITY
    }
}

#[derive(Debug)]
pub struct Chebyshev;

impl Metric for Chebyshev {
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> (f32, Array1<f32>) {
        let mut result = 0.0;
        let mut max_i = 0;
        for i in 0..x.len() {
            let v = (x[i] - y[i]).abs();
            if v > result {
                result = v;
                max_i = i;
            }
        }
        let mut grad = Array1::<f32>::zeros(x.len());
        grad[max_i] = sign(x[max_i] - y[max_i]);
        (result, grad)
    }

    fn is_euclidean(&self) -> bool {
        false
    }

    fn default_disconnection_distance(&self) -> f32 {
        f32::INFINITY
    }
}

#[derive(Debug)]
pub struct Minkowski {
    pub p: f32,
}

impl Minkowski {
    pub fn new(p: f32) -> Self {
        Minkowski { p }
    }
}

impl Metric for Minkowski {
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> (f32, Array1<f32>) {
        let mut result = 0.0;
        for i in 0..x.len() {
            result += (x[i] - y[i]).abs().powf(self.p);
        }

        let mut grad = Array1::<f32>::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = (x[i] - y[i]).abs().powf(self.p - 1.0)
                * sign(x[i] - y[i])
                * result.powf(1.0 / (self.p - 1.0));
        }

        (result.powf(1.0 / self.p), grad)
    }

    fn is_euclidean(&self) -> bool {
        (self.p - 2.0).abs() < 1e-6
    }

    fn default_disconnection_distance(&self) -> f32 {
        f32::INFINITY
    }
}

#[derive(Debug)]
pub struct Cosine;

impl Metric for Cosine {
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> (f32, Array1<f32>) {
        let mut result = 0.0;
        let mut x_norm = 0.0;
        let mut y_norm = 0.0;

        for i in 0..x.len() {
            result += x[i] * y[i];
            x_norm += x[i] * x[i];
            y_norm += y[i] * y[i];
        }

        x_norm = x_norm.sqrt();
        y_norm = y_norm.sqrt();

        if x_norm == 0.0 && y_norm == 0.0 {
            return (0.0, Array1::zeros(x.len()));
        } else if x_norm == 0.0 || y_norm == 0.0 {
            return (1.0, Array1::zeros(x.len()));
        }

        let similarity = result / (x_norm * y_norm);
        let distance = 1.0 - similarity;

        // Gradient computation
        let mut grad = Array1::<f32>::zeros(x.len());
        let norm_product = x_norm * y_norm;
        for i in 0..x.len() {
            grad[i] = -y[i] / norm_product + similarity * x[i] / (x_norm * x_norm);
        }

        (distance, grad)
    }

    fn is_euclidean(&self) -> bool {
        false
    }

    fn default_disconnection_distance(&self) -> f32 {
        2.0
    }
}

#[derive(Debug)]
pub struct Correlation;

impl Metric for Correlation {
    fn distance(&self, x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> (f32, Array1<f32>) {
        let n = x.len() as f32;
        let mut mu_x = 0.0;
        let mut mu_y = 0.0;

        for i in 0..x.len() {
            mu_x += x[i];
            mu_y += y[i];
        }
        mu_x /= n;
        mu_y /= n;

        let mut result = 0.0;
        let mut x_norm = 0.0;
        let mut y_norm = 0.0;

        for i in 0..x.len() {
            let x_centered = x[i] - mu_x;
            let y_centered = y[i] - mu_y;
            result += x_centered * y_centered;
            x_norm += x_centered * x_centered;
            y_norm += y_centered * y_centered;
        }

        x_norm = x_norm.sqrt();
        y_norm = y_norm.sqrt();

        if x_norm == 0.0 && y_norm == 0.0 {
            return (0.0, Array1::zeros(x.len()));
        } else if x_norm == 0.0 || y_norm == 0.0 {
            return (1.0, Array1::zeros(x.len()));
        }

        let similarity = result / (x_norm * y_norm);
        let distance = 1.0 - similarity;

        // Simplified gradient (not exact but functional)
        let mut grad = Array1::<f32>::zeros(x.len());
        let norm_product = x_norm * y_norm;
        for i in 0..x.len() {
            let y_centered = y[i] - mu_y;
            let x_centered = x[i] - mu_x;
            grad[i] = -y_centered / norm_product + similarity * x_centered / (x_norm * x_norm);
        }

        (distance, grad)
    }

    fn is_euclidean(&self) -> bool {
        false
    }

    fn default_disconnection_distance(&self) -> f32 {
        2.0
    }
}
