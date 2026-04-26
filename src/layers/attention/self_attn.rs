use crate::{
    layers::linear::Linear,
    tensor::matrix::{InitStrategy, Matrix},
};

pub struct Attention<const Dim: usize> {
    w_q: Linear<Dim, Dim>,
    w_k: Linear<Dim, Dim>,
    w_v: Linear<Dim, Dim>,
    w_o: Linear<Dim, Dim>,
}

impl<const Dim: usize> Attention<Dim> {
    pub fn new(seed: u64, strat: Option<InitStrategy>) -> Self {
        Self {
            w_q: Linear::new(seed, strat),
            w_k: Linear::new(seed + 1, strat),
            w_v: Linear::new(seed + 2, strat),
            w_o: Linear::new(seed + 3, strat),
        }
    }

    pub fn forward<const Seq: usize>(&self, x: &Matrix<Seq, Dim>) -> Matrix<Seq, Dim> {
        let q = self.w_q.forward_seq(x);
        let k = self.w_k.forward_seq(x);
        let v = self.w_v.forward_seq(x);

        // lets not do a one liner
        // let scores = q.matmul(&k.transpose()).scale(1.0 / (Dim as f32).sqrt());

        let scale = (Dim as f32).sqrt();
        let mut scores = q.matmul(&k.transpose()).scale(1.0 / scale);

        // look before you leap
        // this looks jank as fuck ngl
        for i in 0..Seq {
            for j in (i + 1)..Seq {
                // set everything above diag to neg inf
                scores.set(i, j, f32::NEG_INFINITY);
            }
        }

        let scores = crate::layers::activation::softmax_rows(&scores);
        // TODO: why are we doing this? just do it inplace

        let attended = scores.matmul(&v);

        self.w_o.forward_seq(&attended)
    }
}
