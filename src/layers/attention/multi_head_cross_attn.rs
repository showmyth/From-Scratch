use crate::{
    layers::{attention::cross_attention::CrossAttention, linear::Linear},
    tensor::matrix::{InitStrategy, Matrix},
};

pub struct MultiHeadCrossAttention<const Dim: usize, const HeadDim: usize> {
    heads: Vec<CrossAttention<HeadDim>>,
    w_o: Linear<Dim, Dim>,
}

impl<const Dim: usize, const HeadDim: usize> MultiHeadCrossAttention<Dim, HeadDim> {
    pub fn new(num_heads: usize, seed: u64, strategy: Option<InitStrategy>) -> Self {
        assert_eq!(
            Dim,
            HeadDim * num_heads,
            "Dim must equal HeadDim * num_heads"
        );
        let heads = (0..num_heads)
            .map(|i| CrossAttention::new(seed + i as u64, strategy))
            .collect();
        Self {
            heads,
            w_o: Linear::new(seed + num_heads as u64, strategy),
        }
    }

    /// Forward cross-attention pass.
    ///
    /// Both `x_q` and `x_kv` are split into `num_heads` slices along the Dim axis.
    /// Each head operates independently on its slice, then outputs are concatenated
    /// and projected back through `w_o`.
    ///
    /// - `x_q`  : Decoder input  — shape (SeqQ,  Dim)
    /// - `x_kv` : Encoder output — shape (SeqKV, Dim)
    pub fn forward<const SeqQ: usize, const SeqKV: usize>(
        &self,
        x_q: &Matrix<SeqQ, Dim>,
        x_kv: &Matrix<SeqKV, Dim>,
    ) -> Matrix<SeqQ, Dim> {
        let head_outputs: Vec<Matrix<SeqQ, HeadDim>> = self
            .heads
            .iter()
            .enumerate()
            .map(|(h, head)| {
                // slice the q heads
                let mut slice_q = [[0.0; HeadDim]; SeqQ];
                for s in 0..SeqQ {
                    for d in 0..HeadDim {
                        slice_q[s][d] = x_q.data[s][h * HeadDim + d];
                    }
                }

                // slice out the kv heads
                let mut slice_kv = [[0.0; HeadDim]; SeqKV];
                for s in 0..SeqKV {
                    for d in 0..HeadDim {
                        slice_kv[s][d] = x_kv.data[s][h * HeadDim + d];
                    }
                }

                head.forward_cross(&Matrix { data: slice_q }, &Matrix { data: slice_kv })
            })
            .collect();

        // Concatenate all head outputs back into (SeqQ, Dim)
        let mut concatenated = [[0.0; Dim]; SeqQ];
        for (h, head_out) in head_outputs.iter().enumerate() {
            for s in 0..SeqQ {
                for d in 0..HeadDim {
                    concatenated[s][h * HeadDim + d] = head_out.data[s][d];
                }
            }
        }

        self.w_o.forward_seq(&Matrix { data: concatenated })
    }
}
