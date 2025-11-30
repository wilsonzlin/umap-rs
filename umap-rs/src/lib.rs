//! Fast, parallel Rust implementation of the UMAP dimensionality reduction algorithm.
//!
//! This library provides a streamlined implementation of UMAP (Uniform Manifold
//! Approximation and Projection) focused on performance and correctness. It requires
//! precomputed k-nearest neighbors and initialization, allowing you to use your
//! preferred libraries for these steps.
//!
//! # Example
//!
//! ```ignore
//! use umap::{Umap, UmapConfig};
//! use umap::EuclideanMetric;
//!
//! // Configure UMAP
//! let config = UmapConfig::default();
//! let umap = Umap::new(config);
//!
//! // Fit to data (requires precomputed KNN and initialization)
//! let model = umap.fit(
//!     data.view(),
//!     knn_indices.view(),
//!     knn_dists.view(),
//!     init.view(),
//! );
//!
//! // Get the embedding
//! let embedding = model.embedding();
//! ```
//!
//! # Features
//!
//! - **Parallel optimization**: Rayon's parallel SGD (Hogwild! algorithm)
//! - **Extensible metrics**: Custom distance functions via the `Metric` trait
//! - **Zero-copy views**: Efficient array handling with `ndarray`
//!
//! # Limitations
//!
//! - Dense arrays only (no sparse matrix support)
//! - Fit only (transform for new points not yet implemented)
//! - Panics on invalid input (no Result-based error handling)
//! - Requires external KNN computation and initialization
//!
//! # Public API
//!
//! The library exposes a minimal, well-defined API:
//!
//! * [`Umap`] - Main algorithm struct
//! * [`FittedUmap`] - Fitted model with embeddings
//! * [`UmapConfig`] - Configuration parameters
//! * [`Metric`] - Distance metric trait
//! * [`EuclideanMetric`] - Euclidean distance implementation

// Public modules
pub mod config;
pub mod metric;

// Public re-exports (primary API)
pub use config::GraphParams;
pub use config::ManifoldParams;
pub use config::OptimizationParams;
pub use config::UmapConfig;
pub use embedding::FittedUmap;
pub use embedding::Umap;
pub use manifold::LearnedManifold;
pub use metric::Metric;
pub use metric::MetricType;
pub use optimizer::Optimizer;
pub use umap::SparseMat;

// Internal modules (not exposed)
mod distances;
mod embedding;
mod layout;
mod umap;
mod utils;

// Public modules (for advanced users)
pub mod manifold;
pub mod optimizer;

// Re-export distances for convenience
pub use distances::EuclideanMetric;

// Tests
#[cfg(test)]
mod tests;
