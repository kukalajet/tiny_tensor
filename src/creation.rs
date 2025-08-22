use crate::tensor::Tensor;

/// Creates a `Tensor` of a given shape filled with zeros.
///
/// # Panics
///
/// Panics if the total number of elements overflows `usize`.
pub fn zeros<T: Default + Copy>(shape: &[usize]) -> Tensor<T> {
    let num_elements = shape.iter().product();
    let data = vec![T::default(); num_elements];

    Tensor::new(data, shape.to_vec()).unwrap()
}

/// A macro for creating `Tensor`s with a convenient, literal-like syntax.
///
/// # Examples
///
/// ```
/// use tiny_tensor::tensor;
///
/// // 1D vector
/// let v = tensor![1, 2, 3];
///
/// // 2D matrix
/// let m = tensor![[1, 2], [3, 4]];
///
/// // 3D tensor
/// let t = tensor![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
/// ```
#[macro_export]
macro_rules! tensor {
    // 3D: tensor![[[a, b], [c, d]], [[e, f], [g, h]]]
    (
        [
            $(
                [
                    $(
                        [ $( $val:expr ),+ $(,)? ]
                    ),+ $(,)?
                ]
            ),+ $(,)?
        ]
    ) => {{
        let layers = vec![
            $(
                vec![
                    $(
                        vec![ $( $val ),+ ]
                    ),+
                ]
            ),+
        ];
        let d1 = layers.len();
        let d2 = layers[0].len();
        let d3 = layers[0][0].len();
        if !layers.iter().all(|m| m.len() == d2 && m.iter().all(|r| r.len() == d3)) {
            panic!("tensor!: all inner 2D matrices must have equal sizes");
        }
        let data: Vec<_> = layers.into_iter().flatten().flatten().collect();
        $crate::tensor::Tensor::new(data, vec![d1, d2, d3]).unwrap()
    }};
    // 2D: tensor![[a, b], [c, d]]
    (
        [
            $(
                [ $( $val:expr ),+ $(,)? ]
            ),+ $(,)?
        ]
    ) => {{
        let rows = vec![
            $(
                vec![ $( $val ),+ ]
            ),+
        ];
        let d1 = rows.len();
        let d2 = rows[0].len();
        if !rows.iter().all(|r| r.len() == d2) {
            panic!("tensor!: all rows must have the same length");
        }
        let data: Vec<_> = rows.into_iter().flatten().collect();
        $crate::tensor::Tensor::new(data, vec![d1, d2]).unwrap()
    }};
    // 1D: tensor![a, b, c]
    ( $( $val:expr ),+ $(,)? ) => {{
        let data = vec![ $( $val ),+ ];
        $crate::tensor::Tensor::new(data, vec![data.len()]).unwrap()
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let result: Tensor<i32> = zeros(&[2, 3]);

        assert_eq!(result.shape, &[2, 3]);
        assert_eq!(result.data, vec![0, 0, 0, 0, 0, 0])
    }
}
