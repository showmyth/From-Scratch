use from_scratch::{
    helpers::helper_math::randn,
    layers::{
        self,
        attention::{multi_head_attn::MultiHeadAttention, self_attn::Attention},
        decoder::Decoder,
        linear::Linear,
    },
    tensor::{
        array::Array,
        matrix::{InitStrategy, Matrix},
    },
};

#[test]
fn matrix_transpose_and_matmul_produces_expected_values() {
    let a = Matrix::from_arr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let at = a.transpose();
    assert_eq!(at.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let product = a.matmul(&at);
    assert_eq!(product.data, vec![14.0, 32.0, 32.0, 77.0]);
}

#[test]
fn matrix_scale_and_softmax_rows_produce_normalized_rows() {
    let m = Matrix::from_arr([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
    let scaled = m.scale(0.5);
    assert_eq!(scaled.data, vec![0.5, 1.0, 1.5, 0.0, 0.0, 0.0]);

    let softmaxed = layers::activation::softmax_rows(&m);
    for i in 0..2 {
        let row = softmaxed.get_row(i);
        let row_sum: f32 = row.iter().sum();
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
    assert_eq!(output.data, vec![22.5, 27.0]);

    let sequence = Matrix::from_arr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let seq_output = linear.forward_seq(&sequence);
    assert_eq!(seq_output.data, vec![22.5, 27.0, 49.5, 63.0]);
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
    let attention = Attention::<2>::new(42, Some(InitStrategy::Naive));
    let x = Matrix::from_arr([[1.0, 0.0], [0.0, 1.0]]);
    let out = attention.forward(&x);
    for value in out.data.iter() {
        assert!(value.is_finite());
    }
}

#[test]
fn multi_head_attention_forward_preserves_shape_and_returns_finite_values() {
    let mha = MultiHeadAttention::<4, 2>::new(2, 42, Some(InitStrategy::Naive));
    let x = Matrix::from_arr([[1.0, 0.0, 0.0, 1.0]]);
    let out = mha.forward(&x);
    // Matrix is now 1D. Shape should logically be (1, 4), which means data.len() == 4
    assert_eq!(out.data.len(), 4);
    for value in out.data.iter() {
        assert!(value.is_finite());
    }
}

#[test]
fn test_layer_norm_shape_and_forward() {
    let mut rng = 67;
    let x = [
        Matrix::<5, 8>::randn(&mut rng, None),
        Matrix::<5, 8>::randn(&mut rng, None),
    ];

    let layer_norm = layers::norm::Norm::<8>::new();
    let mut out = [Matrix::<5, 8>::zeros(), Matrix::<5, 8>::zeros()];

    for i in 0..2 {
        out[i] = layer_norm.forward_seq(&x[i]);
    }

    assert_eq!(out.len(), x.len());
    assert_eq!(out[0].data.len(), x[0].data.len()); // Should be 5 * 8 = 40
}

#[test]
fn test_cross_attention_forward() {
    let cross_attn = layers::attention::cross_attention::CrossAttention::<8>::new(42, None);

    let mut rng = 123;
    let x_q = Matrix::<10, 8>::randn(&mut rng, None);
    let x_kv = Matrix::<50, 8>::randn(&mut rng, None);

    let output = cross_attn.forward_cross(&x_q, &x_kv);

    // Shape is logically 10x8. Under the hood, Vec should be 80 elements long.
    assert_eq!(output.data.len(), 80);

    for value in output.data.iter() {
        assert!(value.is_finite());
    }
}

#[test]
fn test_decoder_forward_preserves_shape_and_is_finite() {
    let num_heads = 2;
    let decoder = Decoder::<8, 4, 32>::new(num_heads, 42, Some(InitStrategy::Xavier));

    let mut rng = 99;
    let x_decoder = Matrix::<10, 8>::randn(&mut rng, None);
    let x_encoder = Matrix::<20, 8>::randn(&mut rng, None);

    let output = decoder.forward(&x_decoder, &x_encoder);

    // Output must logically be (10, 8), making data.len() 80
    assert_eq!(output.data.len(), 80);

    for value in output.data.iter() {
        assert!(
            value.is_finite(),
            "output contains non-finite value: {value}"
        );
    }
}
