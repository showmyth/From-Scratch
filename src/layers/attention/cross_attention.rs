use crate::{
    layers::linear::Linear,
    tensor::matrix::{InitStrategy, Matrix},
};

pub struct CrossAttention<const Dim: usize> {
    w_q: Linear<Dim, Dim>,
    w_k: Linear<Dim, Dim>,
    w_v: Linear<Dim, Dim>,
    w_o: Linear<Dim, Dim>,
}

impl<const Dim: usize> CrossAttention<Dim> {
    pub fn new(seed: u64, strat: Option<InitStrategy>) -> Self {
        Self {
            w_q: Linear::new(seed, strat),
            w_k: Linear::new(seed + 1, strat),
            w_v: Linear::new(seed + 2, strat),
            w_o: Linear::new(seed + 3, strat),
        }
    }

    pub fn forward_cross<const SEQ_Q: usize, const SEQ_KV: usize>(
        &self,
        x_q: &Matrix<SEQ_Q, Dim>,
        x_kv: &Matrix<SEQ_KV, Dim>,
    ) -> Matrix<SEQ_Q, Dim> {
        let q = self.w_q.forward_seq(x_q);

        let k = self.w_k.forward_seq(x_kv);
        let v = self.w_v.forward_seq(x_kv);

        let scale = (Dim as f32).sqrt();
        let scores = q.matmul(&k.transpose()).scale(1.0 / scale);

        let scores = crate::layers::activation::softmax_rows(&scores);

        let attended = scores.matmul(&v);

        self.w_o.forward_seq(&attended)
    }
}
