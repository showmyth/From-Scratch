use crate::{layers::linear::Linear, tensor::matrix::{InitStrategy, Matrix}};

pub struct SelfAttention<const Dim: usize> {
    w_q: Linear<Dim, Dim>,
    w_k: Linear<Dim, Dim>,
    w_v: Linear<Dim, Dim>,
    w_o: Linear<Dim, Dim>,
}

impl<const Dim: usize> SelfAttention<Dim> {
    pub fn new(seed: u64, strat: Option<InitStrategy>) -> Self {
        Self {
            w_q: Linear::new(seed,          strat),
            w_k: Linear::new(seed + 1,      strat),
            w_v: Linear::new(seed + 2,      strat),
            w_o: Linear::new(seed + 3,      strat),
        }
    }

    pub fn forward<const Seq: usize>(&self, x: &Matrix<Seq, Dim>) -> Matrix<Seq, Dim> {
        let q = self.w_q.forward_seq(x);
        let k = self.w_k.forward_seq(x);
        let v = self.w_v.forward_seq(x);

        let scale = (Dim as f64).sqrt(); // could be a one liner if imo but cleaner xdd
        let scores = q.matmul(&k.transpose()).scale(1.0 / scale);
        let scores = scores.softmax_rows();

        let attended = scores.matmul(&v);

        self.w_o.forward_seq(&attended)
    }
}