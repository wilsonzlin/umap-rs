use dashmap::DashSet;
use ndarray::{Array1, Array2, ArrayView2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sprs::{CsMat, TriMat};
use typed_builder::TypedBuilder;

use crate::{
    metric::Metric,
    umap::{
        find_ab_params::find_ab_params,
        fuzzy_simplicial_set::FuzzySimplicialSet,
        raise_disconnected_warning::raise_disconnected_warning,
        simplicial_set_embedding::SimplicialSetEmbedding,
    },
};


#[derive(TypedBuilder, Debug)]
#[allow(dead_code)]
pub struct Umap<'a> {
  #[builder(default = 15)]
  n_neighbors: usize,
  #[builder(default = 2)]
  n_components: usize,
  metric: &'a dyn Metric,
  output_metric: &'a dyn Metric,
  n_epochs: Option<usize>,
  #[builder(default = 1.0)]
  learning_rate: f32,
  init: ArrayView2<'a, f32>,
  #[builder(default = 0.1)]
  min_dist: f32,
  #[builder(default = 1.0)]
  spread: f32,
  #[builder(default = 1.0)]
  set_op_mix_ratio: f32,
  #[builder(default = 1.0)]
  local_connectivity: f32,
  #[builder(default = 1.0)]
  repulsion_strength: f32,
  #[builder(default = 5)]
  negative_sample_rate: usize,
  #[builder(default = 4.0)]
  transform_queue_size: f32,
  /// Provide NaN to autoconfigure.
  #[builder(default = f32::NAN)]
  a: f32,
  /// Provide NaN to autoconfigure.
  #[builder(default = f32::NAN)]
  b: f32,
  #[builder(default = 42)]
  transform_seed: usize,
  /// Provide NaN to autoconfigure.
  #[builder(default = f32::NAN)]
  disconnection_distance: f32,
  knn_dists: ArrayView2<'a, f32>,
  knn_indices: ArrayView2<'a, u32>,

  // [DIVERGE] How many rows are in the input dataset.
  // We store this to avoid needing to store the entire dataset, for cheaper serialization.
  #[builder(setter(skip), default = usize::MAX)]
  n: usize,
  #[builder(setter(skip), default = f32::NAN)]
  _initial_alpha: f32,
  #[builder(setter(skip), default = TriMat::new((0, 0)).to_csr())]
  _graph: CsMat<f32>,
  #[builder(setter(skip), default = Default::default())]
  _sigmas: Array1<f32>,
  #[builder(setter(skip), default = Default::default())]
  _rhos: Array1<f32>,
}

impl<'a> Umap<'a> {
    fn _validate_parameters(&mut self) {
        if self.set_op_mix_ratio < 0.0 || self.set_op_mix_ratio > 1.0 {
            panic!("set_op_mix_ratio must be between 0.0 and 1.0");
        }
        if self.repulsion_strength < 0.0 {
            panic!("repulsion_strength cannot be negative");
        }
        if self.min_dist > self.spread {
            panic!("min_dist must be less than or equal to spread");
        }
        if self.min_dist < 0.0 {
            panic!("min_dist cannot be negative");
        };
        if self.init.shape()[1] != self.n_components {
            panic!("init ndarray must match n_components value");
        }
        // negative_sample_rate is usize, so it's always >= 0
        if self._initial_alpha < 0.0 && !self._initial_alpha.is_nan() {
            panic!("learning_rate must be positive");
        }
        if self.n_neighbors < 2 {
            panic!("n_neighbors must be greater than 1")
        }
        if self.n_components < 1 {
            panic!("n_components must be greater than 0");
        }

        // This will be used to prune all edges of greater than a fixed value from our knn graph.
        // We have preset defaults described in DISCONNECTION_DISTANCES for our bounded measures.
        // Otherwise a user can pass in their own value.
        if self.disconnection_distance.is_nan() {
          self.disconnection_distance = self.metric.default_disconnection_distance();
        };

        if self.knn_dists.shape() != self.knn_indices.shape() {
            panic!(
                "knn_dists and knn_indices must be numpy arrays of the same size"
            );
        }
        if self.knn_dists.shape()[1] != self.n_neighbors {
            panic!("knn_dists has a number of neighbors not equal to n_neighbors parameter");
        }
        if self.knn_dists.shape()[0] != self.n {
            panic!(
                "knn_dists has a different number of samples than the data you are fitting"
            );
        }
    }

    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: ArrayView2<f32>) -> Array2<f32> {
        self.n = X.shape()[0];

        // Handle all the optional arguments, setting default
        if self.a.is_nan() || self.b.is_nan() {
            let (a, b) = find_ab_params(self.spread, self.min_dist);
            self.a = a;
            self.b = b;
        }

        self._initial_alpha = self.learning_rate;

        self._validate_parameters();

        // Error check n_neighbors based on data size
        assert!(self.n > self.n_neighbors);

        // Disconnect any vertices farther apart than _disconnection_distance
        let knn_disconnections = DashSet::new();
        (0..self.n).into_par_iter().for_each(|row_no| {
          let row = self.knn_dists.row(row_no);
          for (col_no, &dist) in row.iter().enumerate() {
            if dist >= self.disconnection_distance {
              knn_disconnections.insert((row_no, col_no));
            }
          }
        });
        let edges_removed = knn_disconnections.len();

        let (graph, sigmas, rhos) = FuzzySimplicialSet::builder()
          .n_samples(self.n)
          .n_neighbors(self.n_neighbors)
          .knn_indices(self.knn_indices)
          .knn_dists(self.knn_dists)
          .knn_disconnections(&knn_disconnections)
          .local_connectivity(self.local_connectivity)
          .set_op_mix_ratio(self.set_op_mix_ratio)
          .apply_set_operations(true)
          .build()
          .exec();
        self._graph = graph;
        self._sigmas = sigmas;
        self._rhos = rhos;

        // Report the number of vertices with degree 0 in our graph
        // This ensures that they were properly disconnected.
        let vertices_disconnected = self._graph.outer_iterator()
            .filter(|row| {
                let sum: f32 = row.data().iter().sum();
                sum == 0.0
            })
            .count();
        raise_disconnected_warning(
            edges_removed,
            vertices_disconnected,
            self.disconnection_distance,
            self.n,
            0.1,
        );

        let mut embedding = self._fit_embed_data(
            self.n_epochs,
            self.init,
        );

        // Assign any points that are fully disconnected from our manifold(s) to have embedding
        // coordinates of NaN. These will be filtered by plotting functions automatically.
        for (i, row) in self._graph.outer_iterator().enumerate() {
            let sum: f32 = row.data().iter().sum();
            if sum == 0.0 {
                for j in 0..embedding.shape()[1] {
                    embedding[(i, j)] = f32::NAN;
                }
            }
        }

        embedding
    }

    /*
      A method wrapper for simplicial_set_embedding that can be
      replaced by subclasses.
    */
    fn _fit_embed_data(&self, n_epochs: Option<usize>, init: ArrayView2<f32>) -> Array2<f32> {
      SimplicialSetEmbedding::builder()
        .graph(self._graph.view())
        .initial_alpha(self._initial_alpha)
        .a(self.a)
        .b(self.b)
        .gamma(self.repulsion_strength)
        .negative_sample_rate(self.negative_sample_rate)
        .n_epochs(n_epochs)
        .init(init)
        .output_metric(self.output_metric)
        .build()
        .exec()
    }

    #[allow(non_snake_case)]
    pub fn fit_transform(&mut self, X: ArrayView2<f32>) -> Array2<f32> {
        self.fit(X)
    }
}
