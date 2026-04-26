use crate::{
    layers::{
        attention::multi_head_attn::MultiHeadAttention, feed_forward::FeedForward, norm::Norm,
    },
    tensor::matrix::{InitStrategy, Matrix},
};

pub struct Encoder<const Dim: usize, const HeadDim: usize, const Hidden: usize> {
    self_attn: MultiHeadAttention<Dim, HeadDim>,
    ff: FeedForward<Dim, Hidden>,
    norm1: Norm<Dim>,
    norm2: Norm<Dim>,
}

impl<const Dim: usize, const HeadDim: usize, const Hidden: usize> Encoder<Dim, HeadDim, Hidden> {
    pub fn new(num_heads: usize, seed: u64, strat: Option<InitStrategy>) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(num_heads, seed, strat),
            ff: FeedForward::new(seed + 1, strat),
            norm1: Norm::new(),
            norm2: Norm::new(),
        }
    }

    pub fn forward<const Seq: usize>(&self, x: &Matrix<Seq, Dim>) -> Matrix<Seq, Dim> {
        // self attention + add + norm
        let attn = self.self_attn.forward(x);
        let x = self.norm1.forward_seq(&x.add(&attn));

        // feed forward + add + norm
        let ff_out = self.ff.forward_seq(&x);
        self.norm2.forward_seq(&x.add(&ff_out))
    }
}
