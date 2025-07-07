def raise_disconnected_warning(
  edges_removed,
  vertices_disconnected,
  disconnection_distance,
  total_rows,
  threshold=0.1,
  verbose=False,
):
  """A simple wrapper function to avoid large amounts of code repetition."""
  if verbose & (vertices_disconnected == 0) & (edges_removed > 0):
      print(
          f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.  "
          f"This is not a problem as no vertices were disconnected."
      )
  elif (vertices_disconnected > 0) & (
      vertices_disconnected <= threshold * total_rows
  ):
      warn(
          f"A few of your vertices were disconnected from the manifold.  This shouldn't cause problems.\n"
          f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\n"
          f"It has only fully disconnected {vertices_disconnected} vertices.\n"
          f"Use umap.utils.disconnected_vertices() to identify them.",
      )
  elif vertices_disconnected > threshold * total_rows:
      warn(
          f"A large number of your vertices were disconnected from the manifold.\n"
          f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\n"
          f"It has fully disconnected {vertices_disconnected} vertices.\n"
          f"You might consider using find_disconnected_points() to find and remove these points from your data.\n"
          f"Use umap.utils.disconnected_vertices() to identify them.",
      )
