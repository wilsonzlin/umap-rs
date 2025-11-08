use std::iter::zip;

use ndarray::{Array2, ArrayView2};
use sprs::CsMatView;

/*
  Given a bipartite graph representing the 1-simplices and strengths between the
  new points and the original data set along with an embedding of the original points
  initialize the positions of new points relative to the strengths (of their neighbors in the source data).

  If a point is in our original data set it embeds at the original points coordinates.
  If a point has no neighbours in our original dataset it embeds as the np.nan vector.
  Otherwise a point is the weighted average of it's neighbours embedding locations.

  Parameters
  ----------
  graph: csr_matrix (n_new_samples, n_samples)
      A matrix indicating the 1-simplices and their associated strengths.  These strengths should
      be values between zero and one and not normalized.  One indicating that the new point was identical
      to one of our original points.

  embedding: array of shape (n_samples, dim)
      The original embedding of the source data.

  Returns
  -------
  new_embedding: array of shape (n_new_samples, dim)
      An initial embedding of the new sample points.
*/
pub fn init_graph_transform(
  graph: &CsMatView<f32>,
  embedding: &ArrayView2<f32>,
) -> Array2<f32> {
  let mut result = Array2::<f32>::zeros((graph.shape().0, embedding.shape()[1]));

  for row_index in 0..graph.shape().0 {
    let graph_row = graph.outer_view(row_index).unwrap();
    if graph_row.nnz() == 0 {
      result.row_mut(row_index).fill(f32::NAN);
      continue;
    }
    let row_sum: f32 = graph_row.data().iter().sum();
    for (&graph_value, &col_index) in zip(graph_row.data(), graph_row.indices()) {
      if graph_value == 1.0 {
        result.row_mut(row_index).assign(&embedding.row(col_index));
        break;
      }
      let weighted_embedding = &embedding.row(col_index) * (graph_value / row_sum);
      let mut row = result.row_mut(row_index);
      row += &weighted_embedding;
    }
  }

  result
}
