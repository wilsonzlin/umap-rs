use tracing::debug;
use tracing::warn;

pub fn raise_disconnected_warning(
  edges_removed: usize,
  vertices_disconnected: usize,
  disconnection_distance: f32,
  total_rows: usize,
  threshold: f64, // Default 0.1
) {
  if vertices_disconnected == 0 && edges_removed > 0 {
    debug!(
      "Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges. This is not a problem as no vertices were disconnected."
    );
  } else if vertices_disconnected > 0
    && vertices_disconnected <= (threshold * total_rows as f64) as usize
  {
    warn!(
      "A few of your vertices were disconnected from the manifold.  This shouldn't cause problems.\nDisconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\nIt has only fully disconnected {vertices_disconnected} vertices.\nUse umap.utils.disconnected_vertices() to identify them."
    );
  } else if vertices_disconnected > (threshold * total_rows as f64) as usize {
    warn!(
      "A large number of your vertices were disconnected from the manifold.\nDisconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\nIt has fully disconnected {vertices_disconnected} vertices.\nYou might consider using find_disconnected_points() to find and remove these points from your data.\nUse umap.utils.disconnected_vertices() to identify them."
    )
  }
}
