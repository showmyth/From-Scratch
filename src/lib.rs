pub mod layers;
pub mod tensor;
pub mod helpers;

#[cfg(test)]
mod tests {
    use crate::tensor::array::Array;
    use crate::tensor::matrix::Matrix;

    #[test]
    fn test_softmax_output_sums_to_one() {
        let arr = Array { data: [1.0, 2.0, 3.0] };
        let result = arr.softmax();
        let sum: f64 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum was {}", sum);
    }

    #[test]
    fn test_softmax_preserves_order() {
        let arr = Array { data: [1.0, 2.0, 3.0] };
        let result = arr.softmax();
        assert!(result.data[0] < result.data[1]);
        assert!(result.data[1] < result.data[2]);
    }

    #[test]
    fn test_softmax_uniform_input() {
        let arr = Array { data: [2.0, 2.0, 2.0] };
        let result = arr.softmax();
        for val in result.data.iter() {
            assert!((val - 1.0 / 3.0).abs() < 1e-10, "Expected ~0.333, got {}", val);
        }
    }

    #[test]
    fn test_softmax_large_values_no_overflow() {
        let arr = Array { data: [1000.0, 1001.0, 1002.0] };
        let result = arr.softmax();
        let sum: f64 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for val in result.data.iter() {
            assert!(val.is_finite(), "Got non-finite value: {}", val);
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let arr = Array { data: [-1.0, -2.0, -3.0] };
        let result = arr.softmax();
        let sum: f64 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(result.data[0] > result.data[1]);
        assert!(result.data[1] > result.data[2]);
    }

    #[test]
    fn test_softmax_single_element() {
        let arr = Array { data: [42.0] };
        let result = arr.softmax();
        assert!((result.data[0] - 1.0).abs() < 1e-10);
    }
}
