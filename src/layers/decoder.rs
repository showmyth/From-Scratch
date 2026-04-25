use crate::{
    layers::{
        attention::{
            multi_head_attn::MultiHeadAttention, multi_head_cross_attn::MultiHeadCrossAttention,
        },
        feed_forward::FeedForward,
        norm::Norm,
    },
    tensor::matrix::{InitStrategy, Matrix},
};

/// A single Transformer Decoder block (faithful to the original paper).
///
/// Structure:
///   x -> [Multi-Head Masked Self-Attention]  -> Add & Norm
///     -> [Multi-Head Cross-Attention]        -> Add & Norm
///     -> [Feed Forward]                      -> Add & Norm
///
/// `Dim`    : model embedding dimension (must equal HeadDim * num_heads)
/// `HeadDim`: dimension per attention head
/// `Hidden` : inner dimension of the FFN (typically 4 * Dim)
pub struct Decoder<const Dim: usize, const HeadDim: usize, const Hidden: usize> {
    self_attn: MultiHeadAttention<Dim, HeadDim>,
    cross_attn: MultiHeadCrossAttention<Dim, HeadDim>,
    ff: FeedForward<Dim, Hidden>,
    norm1: Norm<Dim>,
    norm2: Norm<Dim>,
    norm3: Norm<Dim>,
}

impl<const Dim: usize, const HeadDim: usize, const Hidden: usize> Decoder<Dim, HeadDim, Hidden> {
    pub fn new(num_heads: usize, seed: u64, strat: Option<InitStrategy>) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(num_heads, seed, strat),
            cross_attn: MultiHeadCrossAttention::new(num_heads, seed + 100, strat),
            ff: FeedForward::new(seed + 200, strat),
            norm1: Norm::new(),
            norm2: Norm::new(),
            norm3: Norm::new(),
        }
    }

    /// Forward pass for a single decoder block.
    ///
    /// - `x`      : Decoder input, shape  (SeqQ,  Dim)
    /// - `enc_out`: Encoder output, shape (SeqKV, Dim)
    pub fn forward<const SeqQ: usize, const SeqKV: usize>(
        &self,
        x: &Matrix<SeqQ, Dim>,
        enc_out: &Matrix<SeqKV, Dim>,
    ) -> Matrix<SeqQ, Dim> {
        // 1. Masked Multi-Head Self-Attention + residual + norm
        let attn1 = self.self_attn.forward(x);
        let x = self.norm1.forward_seq(&x.add(&attn1));

        // 2. Multi-Head Cross-Attention (Q from decoder, K/V from encoder) + residual + norm
        let attn2 = self.cross_attn.forward(&x, enc_out);
        let x = self.norm2.forward_seq(&x.add(&attn2));

        // 3. Feed-Forward + residual + norm
        let ff_out = self.ff.forward_seq(&x);
        self.norm3.forward_seq(&x.add(&ff_out))
    }
}
