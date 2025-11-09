/// Configuration for manifold shape and embedding space properties.
///
/// These parameters control the geometric properties of the low-dimensional
/// embedding space and how the manifold is shaped.
#[derive(Debug, Clone)]
pub struct ManifoldParams {
    /// Minimum distance between points in the embedding space.
    ///
    /// Controls how tightly points can be packed together. Smaller values
    /// create more clustered embeddings, larger values spread points out more.
    ///
    /// Default: 0.1
    pub min_dist: f32,

    /// The effective scale of embedded points.
    ///
    /// Together with `min_dist`, this determines the embedding's overall spread.
    /// The curve used in optimization is calibrated using these parameters.
    ///
    /// Default: 1.0
    pub spread: f32,

    /// Parameter 'a' of the distance-to-probability curve: 1 / (1 + a*x^(2b))
    ///
    /// If `None`, will be automatically computed from `min_dist` and `spread`.
    /// Manually setting this overrides automatic calibration.
    ///
    /// Default: None (auto-compute)
    pub a: Option<f32>,

    /// Parameter 'b' of the distance-to-probability curve: 1 / (1 + a*x^(2b))
    ///
    /// If `None`, will be automatically computed from `min_dist` and `spread`.
    /// Manually setting this overrides automatic calibration.
    ///
    /// Default: None (auto-compute)
    pub b: Option<f32>,
}

impl Default for ManifoldParams {
    fn default() -> Self {
        Self {
            min_dist: 0.1,
            spread: 1.0,
            a: None,
            b: None,
        }
    }
}

/// Configuration for k-nearest neighbor graph construction.
///
/// These parameters control how the high-dimensional manifold structure
/// is captured via a fuzzy topological representation.
#[derive(Debug, Clone)]
pub struct GraphParams {
    /// Number of nearest neighbors to use for manifold approximation.
    ///
    /// Larger values capture more global structure but may miss fine details.
    /// Smaller values focus on local structure but may fragment the manifold.
    ///
    /// Must be >= 2.
    ///
    /// Default: 15
    pub n_neighbors: usize,

    /// Local connectivity requirement (number of nearest neighbors assumed connected).
    ///
    /// Higher values make the manifold more locally connected, which can help
    /// with datasets that have variable density. Should generally not exceed
    /// the local intrinsic dimensionality.
    ///
    /// Default: 1.0
    pub local_connectivity: f32,

    /// Interpolation between fuzzy union (1.0) and fuzzy intersection (0.0).
    ///
    /// Controls how local fuzzy simplicial sets are combined. Pure union (1.0)
    /// gives equal weight to all edges, pure intersection (0.0) only keeps
    /// mutually nearest neighbors.
    ///
    /// Must be in range [0.0, 1.0].
    ///
    /// Default: 1.0
    pub set_op_mix_ratio: f32,

    /// Distance threshold beyond which edges are disconnected.
    ///
    /// If `None`, uses the metric's default (typically infinity for unbounded metrics).
    /// Useful for bounded metrics or to explicitly remove long-range connections.
    ///
    /// Default: None (use metric default)
    pub disconnection_distance: Option<f32>,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            disconnection_distance: None,
        }
    }
}

/// Configuration for stochastic gradient descent optimization.
///
/// These parameters control the embedding optimization process via SGD
/// on the fuzzy set cross-entropy.
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    /// Number of optimization epochs.
    ///
    /// If `None`, will be automatically determined based on dataset size:
    /// - <= 10,000 samples: 500 epochs
    /// - > 10,000 samples: 200 epochs
    ///
    /// More epochs improve convergence but increase runtime.
    ///
    /// Default: None (auto-determine)
    pub n_epochs: Option<usize>,

    /// Initial learning rate for SGD.
    ///
    /// The learning rate decays linearly to 0 over the course of optimization.
    /// Higher values converge faster but may overshoot; lower values are more stable.
    ///
    /// Default: 1.0
    pub learning_rate: f32,

    /// Number of negative samples per positive sample.
    ///
    /// Negative sampling is used to push apart non-neighboring points.
    /// Higher values improve separation but increase computation.
    ///
    /// Default: 5
    pub negative_sample_rate: usize,

    /// Weight applied to negative samples (repulsion strength).
    ///
    /// Higher values push non-neighbors apart more strongly.
    /// Lower values focus more on pulling neighbors together.
    ///
    /// Default: 1.0
    pub repulsion_strength: f32,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
        }
    }
}

/// Complete UMAP configuration.
///
/// Groups all parameters for dimensionality reduction into a coherent structure.
/// All parameter groups have sensible defaults and can be customized individually.
///
/// # Example
///
/// ```ignore
/// use umap::config::{UmapConfig, GraphParams};
///
/// // Use all defaults
/// let config = UmapConfig::default();
///
/// // Customize specific groups
/// let config = UmapConfig {
///     n_components: 3,
///     graph: GraphParams {
///         n_neighbors: 30,
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct UmapConfig {
    /// Number of dimensions in the output embedding.
    ///
    /// Typically 2 for visualization or 3-50 for downstream ML tasks.
    ///
    /// Must be >= 1.
    ///
    /// Default: 2
    pub n_components: usize,

    /// Manifold shape configuration.
    pub manifold: ManifoldParams,

    /// Graph construction configuration.
    pub graph: GraphParams,

    /// Optimization configuration.
    pub optimization: OptimizationParams,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            manifold: ManifoldParams::default(),
            graph: GraphParams::default(),
            optimization: OptimizationParams::default(),
        }
    }
}
