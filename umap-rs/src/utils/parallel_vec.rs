use std::cell::UnsafeCell;

/// A Vec wrapped in UnsafeCell for parallel write access to disjoint regions.
///
/// `UnsafeCell<Vec<T>>` tells the compiler that the Vec's contents may be mutated
/// even through shared references (`&self`). This is required for soundness -
/// without UnsafeCell, the compiler assumes `&Vec<T>` means immutable contents.
///
/// # Safety
///
/// The caller must ensure that concurrent writes are to disjoint indices.
/// This is used for parallel construction where each worker writes to its own
/// section (e.g., `[offsets[i]..offsets[i+1]]`).
pub struct ParallelVec<T> {
  data: UnsafeCell<Vec<T>>,
}

// SAFETY: Access is only safe when writes are to disjoint indices.
// The caller must guarantee this via the algorithm structure.
unsafe impl<T: Send> Send for ParallelVec<T> {}
unsafe impl<T: Send> Sync for ParallelVec<T> {}

impl<T> ParallelVec<T> {
  /// Create from an owned Vec.
  pub fn new(vec: Vec<T>) -> Self {
    Self {
      data: UnsafeCell::new(vec),
    }
  }

  /// Write a value at the given index.
  ///
  /// # Safety
  ///
  /// - Index must be in bounds
  /// - No other thread may be accessing the same index concurrently
  pub unsafe fn write(&self, index: usize, value: T) {
    unsafe {
      let vec = &mut *self.data.get();
      debug_assert!(index < vec.len());
      *vec.get_unchecked_mut(index) = value;
    }
  }

  /// Get a mutable slice for a range. Used for sorting.
  ///
  /// # Safety
  ///
  /// - Range must be in bounds
  /// - No other thread may be accessing the same range concurrently
  pub unsafe fn get_mut_slice(&self, start: usize, len: usize) -> &mut [T] {
    unsafe {
      let vec = &mut *self.data.get();
      debug_assert!(start + len <= vec.len());
      &mut vec[start..start + len]
    }
  }

  /// Consume and return the inner Vec.
  pub fn into_inner(self) -> Vec<T> {
    self.data.into_inner()
  }
}
