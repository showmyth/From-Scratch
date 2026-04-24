use core::num;

use crate::{layers::{attention::self_attn::SelfAttention, linear::Linear}, tensor::matrix::{InitStrategy, Matrix}};

pub struct MultiHeadAttention<const Dim: usize, const HeadDim: usize> {
    heads: Vec<SelfAttention<HeadDim>>,
    w_o: Linear<Dim, Dim>,
} 

impl<const Dim: usize, const HeadDim: usize> MultiHeadAttention<Dim, HeadDim> {
    pub fn new(num_heads: usize, seed: u64, strategy: Option<InitStrategy>) -> Self {
        assert_eq!(Dim, HeadDim * num_heads, "Dim must equal HeadDim * num_heads");
        let heads = (0..num_heads)
            .map(|i| SelfAttention::new(seed + i as u64, strategy))
            .collect();
        Self {
            heads,
            w_o: Linear::new(seed + num_heads as u64, strategy),
        }
    }

    pub fn forward<const Seq: usize>(&self, x: &Matrix<Seq, Dim>) -> Matrix<Seq, Dim> {
        let num_heads = self.heads.len();

        // first split the batches
        let mut head_outputs: Vec<Matrix<Seq, HeadDim>> = self.heads.iter().enumerate().map(|(h, head)| {
            let mut slice = [[0.0; HeadDim]; Seq];
            for s in 0..Seq {
                for d in 0..HeadDim {
                    slice[s][h] = x.data[s][h * HeadDim + d];
                }
            }
            head.forward( &Matrix { data: slice} )
        }).collect();

        let mut concatenated = [[0.0; Dim]; Seq];

        for (h, head_out) in head_outputs.iter().enumerate() {
            for s in 0..Seq {
                for d in 0..HeadDim {
                    concatenated[s][h * HeadDim + d] = head_out.data[s][d];
                }
            }
        }

        self.w_o.forward_seq( &Matrix { data: concatenated })
    }
}