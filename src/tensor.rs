/// An N-dimensional array.
///
/// `Tensor<T>` is the central data structure of the library, providing a contiguous,
/// row-major memory layout for elements of type `T`. It supports an arbitrary
/// number of dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    /// A flat vector holding the array's data in a contiguous block.
    pub(crate) data: Vec<T>,
    /// The shape of the array (e.g. `vec![2, 3]` for a 2x3 matrix).
    pub(crate) shape: Vec<usize>,
    /// Strides determine the number of elements to skip in `data` to move
    /// one step along each dimension. Crucial for efficient views and broadcasting.
    pub(crate) strides: Vec<usize>,
}
