use dashmap::DashSet;
use derive_builder::Builder;
use ndarray::ArrayView2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{metric::Metric, umap::{find_ab_params::find_ab_params, fuzzy_simplicial_set::FuzzySimplicialSet, raise_disconnected_warning::raise_disconnected_warning}};



#[derive(Debug, Default)]
struct State {
  // [DIVERGE] How many rows are in the input dataset.
  // We store this to avoid needing to store the entire dataset, for cheaper serialization.
  n: usize,
  a: f32,
  b: f32,
  initial_alpha: f32,
  disconnection_distance: f32,
}

#[derive(Builder, Debug)]
#[builder(setter(into))]
pub struct Umap<'a> {
  #[builder(default = "15")]
  n_neighbors: usize,
  #[builder(default = "2")]
  n_components: usize,
  metric: &'a dyn Metric,
  output_metric: &'a dyn Metric,
  n_epochs: Option<usize>,
  #[builder(default = "1.0")]
  learning_rate: f32,
  init: &'a ArrayView2<'a, f32>,
  #[builder(default = "0.1")]
  min_dist: f32,
  #[builder(default = "1.0")]
  spread: f32,
  #[builder(default = "1.0")]
  set_op_mix_ratio: f32,
  #[builder(default = "1.0")]
  local_connectivity: f32,
  #[builder(default = "1.0")]
  repulsion_strength: f32,
  #[builder(default = "5")]
  negative_sample_rate: usize,
  #[builder(default = "4.0")]
  transform_queue_size: f32,
  a: Option<f32>,
  b: Option<f32>,
  #[builder(default = "42")]
  transform_seed: usize,
  disconnection_distance: Option<f32>,
  knn_dists: &'a ArrayView2<'a, f32>,
  knn_indices: &'a ArrayView2<'a, u32>,

  #[builder(setter(skip))]
  _s: State,
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
        if self.negative_sample_rate < 0 {
            panic!("negative sample rate must be positive");
        }
        if self._s.initial_alpha < 0.0 {
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
        self._s.disconnection_distance = self.disconnection_distance.unwrap_or_else(|| self.metric.default_disconnection_distance());

        if self.knn_dists.shape() != self.knn_indices.shape() {
            panic!(
                "knn_dists and knn_indices must be numpy arrays of the same size"
            );
        }
        if self.knn_dists.shape()[1] != self.n_neighbors {
            panic!("knn_dists has a number of neighbors not equal to n_neighbors parameter");
        }
        if self.knn_dists.shape()[0] != self._s.n {
            panic!(
                "knn_dists has a different number of samples than the data you are fitting"
            );
        }
    }

    fn fit(&mut self, X: &ArrayView2<f32>) {
        self._s.n = X.shape()[0];

        // Handle all the optional arguments, setting default
        if self.a.is_none() || self.b.is_none() {
            let (a, b) = find_ab_params(self.spread, self.min_dist);
            self._s.a = a;
            self._s.b = b;
        } else {
            self._s.a = self.a.unwrap();
            self._s.b = self.b.unwrap();
        }

        let init = self.init;

        self._s.initial_alpha = self.learning_rate;

        self._validate_parameters();

        // Error check n_neighbors based on data size
        assert!(X.shape()[0] > self.n_neighbors);

        let random_state = check_random_state(self.random_state);

        // Disconnect any vertices farther apart than _disconnection_distance
        let knn_disconnections = DashSet::new();
        (0..self._s.n).into_par_iter().for_each(|row_no| {
          let row = self.knn_dists.row(row_no);
          for (col_no, &dist) in row.iter().enumerate() {
            if dist >= self._s.disconnection_distance {
              knn_disconnections.insert((row_no, col_no));
            }
          }
        });
        let edges_removed = knn_disconnections.len();

        (
            self.graph_,
            self._sigmas,
            self._rhos,
            self.graph_dists_,
        ) = FuzzySimplicialSet::default()(
            X,
            self.n_neighbors,
            random_state,
            nn_metric,
            self._metric_kwds,
            self._knn_indices,
            self._knn_dists,
            self.angular_rp_forest,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
        );
        // Report the number of vertices with degree 0 in our umap.graph_
        // This ensures that they were properly disconnected.
        let vertices_disconnected = np.sum(
            np.array(self.graph_.sum(axis=1)).flatten() == 0
        );
        raise_disconnected_warning(
            edges_removed,
            vertices_disconnected,
            self._disconnection_distance,
            self._raw_data.shape[0],
            verbose=self.verbose,
        );

        let epochs = (
            self.n_epochs_list if self.n_epochs_list is not None else self.n_epochs
        );
        self.embedding_, aux_data = self._fit_embed_data(
            X,
            epochs,
            init,
            random_state,  // JH why raw data?
        );

        // Assign any points that are fully disconnected from our manifold(s) to have embedding
        // coordinates of np.nan.  These will be filtered by our plotting functions automatically.
        // They also prevent users from being deceived a distance query to one of these points.
        // Might be worth moving this into simplicial_set_embedding or _fit_embed_data
        let disconnected_vertices = np.array(self.graph_.sum(axis=1)).flatten() == 0;
        if len(disconnected_vertices) > 0 {
            self.embedding_[disconnected_vertices] = np.full(
                self.n_components, np.nan
            );
        }

        self
    }

    fn _fit_embed_data(&self, X, n_epochs, init, random_state) {
        """A method wrapper for simplicial_set_embedding that can be
        replaced by subclasses. Arbitrary keyword arguments can be passed
        through .fit() and .fit_transform().
        """
        return simplicial_set_embedding(
            self.graph_,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._output_distance_func,
            self._output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
        )
    }

    fn fit_transform(self, X) {
        self.fit(X, **kwargs)
        return self.embedding_
    }

    fn transform(self, X) {
        // If we fit just a single instance then error
        if self._raw_data.shape[0] == 1 {
            panic!(
                "Transform unavailable when model was fit with only a single data sample."
            );
        }

        // #848: knn_search_index is allowed to be None if not transforming new data,
        // so now we must validate that if it exists it is not None
        if hasattr(self, "_knn_search_index") and self._knn_search_index is None:
            raise NotImplementedError(
                "No search index available: transforming data"
                " into an existing embedding is not supported"
            )

        // X = check_array(X, dtype=np.float32, order="C", accept_sparse="csr")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        epsilon = 0.24 if self._knn_search_index._angular_trees else 0.12
        indices, dists = self._knn_search_index.query(
            X, self.n_neighbors, epsilon=epsilon
        )

        dists = dists.astype(np.float32, order="C")
        // Remove any nearest neighbours who's distances are greater than our disconnection_distance
        indices[dists >= self._disconnection_distance] = -1
        adjusted_local_connectivity = max(0.0, self.local_connectivity - 1.0)
        sigmas, rhos = smooth_knn_dist(
            dists,
            float(self._n_neighbors),
            local_connectivity=float(adjusted_local_connectivity),
        )

        let (rows, cols, vals) = compute_membership_strengths(
            indices, dists, sigmas, rhos, bipartite=True
        );

        let graph = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
        );

        // This was a very specially constructed graph with constant degree.
        // That lets us do fancy unpacking by reshaping the csr matrix indices
        // and data. Doing so relies on the constant degree assumption!
        // csr_graph = normalize(graph.tocsr(), norm="l1")
        // inds = csr_graph.indices.reshape(X.shape[0], self._n_neighbors)
        // weights = csr_graph.data.reshape(X.shape[0], self._n_neighbors)
        // embedding = init_transform(inds, weights, self.embedding_)
        // This is less fast code than the above numba.jit'd code.
        // It handles the fact that our nearest neighbour graph can now contain variable numbers of vertices.
        csr_graph = graph.tocsr()
        csr_graph.eliminate_zeros()
        embedding = init_graph_transform(csr_graph, self.embedding_)

        let n_epochs = if self.n_epochs.is_none() {
            // For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000 {
                100
            } else {
                30
            }
        } else {
            self.n_epochs.unwrap() / 3
        };

        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0;
        graph.eliminate_zeros();

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs);

        head = graph.row;
        tail = graph.col;
        weight = graph.data;

        if self.output_metric.is_euclidean() {
            embedding = optimize_layout_euclidean(
                embedding,
                self.embedding_.astype(np.float32, copy=True),  // Fixes #179 & #217,
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self.random_state is None,
            )
        } else {
            embedding = optimize_layout_generic(
                embedding,
                self.embedding_.astype(np.float32, copy=True),  // Fixes #179 & #217
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self._output_distance_func,
                tuple(self._output_metric_kwds.values()),
                verbose=self.verbose,
                tqdm_kwds=self.tqdm_kwds,
            )
        };

        embedding
    }
}
