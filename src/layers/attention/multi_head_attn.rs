use crate::{
    layers::{attention::self_attn::Attention, linear::Linear},
    tensor::matrix::{InitStrategy, Matrix},
};

pub struct MultiHeadAttention<const Dim: usize, const HeadDim: usize> {
    heads: Vec<Attention<HeadDim>>,
    w_o: Linear<Dim, Dim>,
}

impl<const Dim: usize, const HeadDim: usize> MultiHeadAttention<Dim, HeadDim> {
    pub fn new(num_heads: usize, seed: u64, strategy: Option<InitStrategy>) -> Self {
        assert_eq!(
            Dim,
            HeadDim * num_heads,
            "Dim must equal HeadDim * num_heads"
        );
        let heads = (0..num_heads)
            .map(|i| Attention::new(seed + i as u64, strategy))
            .collect();
        Self {
            heads,
            w_o: Linear::new(seed + num_heads as u64, strategy),
        }
    }

    pub fn forward<const Seq: usize>(&self, x: &Matrix<Seq, Dim>) -> Matrix<Seq, Dim> {
        // first split the batches
        let head_outputs: Vec<Matrix<Seq, HeadDim>> = self
            .heads
            .iter()
            .enumerate()
            .map(|(h, head)| {
                let mut slice = [[0.0; HeadDim]; Seq];
                for s in 0..Seq {
                    for d in 0..HeadDim {
                        slice[s][d] = x.get(s, h * HeadDim + d);
                    }
                }
                head.forward(&Matrix::from_arr(slice))
            })
            .collect();

        let mut concatenated = [[0.0; Dim]; Seq];

        for (h, head_out) in head_outputs.iter().enumerate() {
            for s in 0..Seq {
                for d in 0..HeadDim {
                    concatenated[s][h * HeadDim + d] = head_out.get(s, d);
                }
            }
        }

        self.w_o.forward_seq(&Matrix::from_arr(concatenated))
    }
}
