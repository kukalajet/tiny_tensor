use std::error::Error;
use std::fmt::{Display, Formatter, Result};

/// A custom error enum for all fallible operations within the `tiny_tensor` library.
#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Error indicating a mismatch in shapes for an operation.
    ShapeError(String),
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            TensorError::ShapeError(msg) => write!(f, "ShapeError: {}", msg),
        }
    }
}

impl Error for TensorError {}
