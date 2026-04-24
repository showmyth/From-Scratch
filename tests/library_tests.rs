use from_scratch::{
    helpers::helper_math::randn,
    layers::{attention::multi_head_attn::MultiHeadAttention, attention::self_attn::SelfAttention, linear::Linear},
    tensor::{array::Array, matrix::{InitStrategy, Matrix}},
};

#[test]
fn matrix_transpose_and_matmul_produces_expected_values() {
    let a = Matrix::from_arr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let at = a.transpose();
    assert_eq!(at.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

    let product = a.matmul(&at);
    assert_eq!(product.data, [[14.0, 32.0], [32.0, 77.0]]);
}

#[test]
fn matrix_scale_and_softmax_rows_produce_normalized_rows() {
    let m = Matrix::from_arr([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
    let scaled = m.scale(0.5);
    assert_eq!(scaled.data, [[0.5, 1.0, 1.5], [0.0, 0.0, 0.0]]);

    let softmaxed = m.softmax_rows();
    for row in softmaxed.data.iter() {
        let row_sum: f64 = row.iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }
}

#[test]
fn linear_forward_and_forward_seq_match_manual_computation() {
    let linear = Linear {
        weights: Matrix::from_arr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        bias: Array::new([0.5, -1.0]),
        grad_weights: None,
        grad_bias: None,
    };

    let input = Array::new([1.0, 2.0, 3.0]);
    let output = linear.forward(&input);
    assert_eq!(output.data, [22.5, 27.0]);

    let sequence = Matrix::from_arr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let seq_output = linear.forward_seq(&sequence);
    assert_eq!(seq_output.data, [[22.5, 27.0], [49.5, 63.0]]);
}

#[test]
fn helper_randn_is_reproducible_and_finite() {
    let mut seed_a = 123;
    let mut seed_b = 123;
    let a = randn(&mut seed_a);
    let b = randn(&mut seed_b);
    assert_eq!(a, b);
    assert!(a.is_finite());
    assert_ne!(a, randn(&mut seed_a));
}

#[test]
fn self_attention_forward_returns_finite_output() {
    let attention = SelfAttention::<2>::new(42, Some(InitStrategy::Naive));
    let x = Matrix::from_arr([[1.0, 0.0], [0.0, 1.0]]);
    let out = attention.forward(&x);
    for row in out.data.iter() {
        for value in row.iter() {
            assert!(value.is_finite());
        }
    }
}

#[test]
fn multi_head_attention_forward_preserves_shape_and_returns_finite_values() {
    let mha = MultiHeadAttention::<4, 2>::new(2, 42, Some(InitStrategy::Naive));
    let x = Matrix::from_arr([[1.0, 0.0, 0.0, 1.0]]);
    let out = mha.forward(&x);
    assert_eq!(out.data.len(), 1);
    assert_eq!(out.data[0].len(), 4);
    for value in out.data[0].iter() {
        assert!(value.is_finite());
    }
}
