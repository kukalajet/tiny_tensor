use std::fmt;
use std::fmt::{Debug, Display, Formatter};

use crate::error::TensorError;

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

impl<T: Copy + Clone> Tensor<T> {
    /// Creates a new `Tensor` from a flat data buffer and a shape.
    ///
    /// The constructor validates that the number of elements in `data` matches
    /// the total number of elements specified by the `shape`. It also calculates
    /// the initial row-major strides.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ShapeError` if `data.len()` does not equal the product
    /// of the dimensions in `shape`.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let num_elements: usize = shape.iter().product();
        if data.len() != num_elements {
            return Err(TensorError::ShapeError(format!(
                "Data size ({}) does not match shape product ({})",
                data.len(),
                num_elements
            )));
        }

        let strides = Self::calculate_strides(&shape);

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Calculates row-major strides for a given shape.
    pub(crate) fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }
}

/// Helper function for pretty-printing tensors.
fn format_recursive<T: Debug>(
    f: &mut Formatter<'_>,
    data: &[T],
    shape: &[usize],
    strides: &[usize],
    level: usize,
) -> fmt::Result {
    if shape.is_empty() {
        return write!(f, "{:?}", data[0]);
    }

    let indent = " ".repeat(level * 2);
    writeln!(f, "[")?;

    let elements_in_dim = shape[0];
    for i in 0..elements_in_dim {
        write!(f, "{}  ", indent)?;
        let offset = i * strides[0];
        if shape.len() > 1 {
            format_recursive(f, &data[offset..], &shape[1..], &strides[1..], level + 1)?;
        } else {
            write!(f, "{:?}", data[offset])?;
        }
        if i < elements_in_dim - 1 {
            writeln!(f, ",")?;
        } else {
            writeln!(f)?;
        }
    }

    write!(f, "{}]", indent)
}

impl<T: Debug> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.shape.is_empty() {
            return writeln!(f, "[]");
        }

        if self.shape.iter().any(|&dim| dim == 0) {
            return writeln!(f, "[]");
        }

        format_recursive(f, &self.data, &self.shape, &self.strides, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor_success() {
        let result = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

        assert_eq!(result.shape, &[2, 3]);
        assert_eq!(result.strides, &[3, 1]);
        assert_eq!(result.data, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_new_tensor_shape_error() {
        let result = Tensor::new(vec![1, 2, 3], vec![2, 3]);

        assert!(matches!(result, Err(TensorError::ShapeError(_))))
    }
}
