use crate::{
    layers::activation,
    layers::linear::Linear,
    tensor::matrix::{InitStrategy, Matrix},
};

/// Two-layer feed-forward network: Linear -> GELU -> Linear
/// Hidden dimension is typically 4x the model dimension.
pub struct FeedForward<const Dim: usize, const Hidden: usize> {
    fc1: Linear<Dim, Hidden>,
    fc2: Linear<Hidden, Dim>,
}

impl<const Dim: usize, const Hidden: usize> FeedForward<Dim, Hidden> {
    pub fn new(seed: u64, strat: Option<InitStrategy>) -> Self {
        Self {
            fc1: Linear::new(seed, strat),
            fc2: Linear::new(seed + 1, strat),
        }
    }

    pub fn forward_seq<const Seq: usize>(&self, x: &Matrix<Seq, Dim>) -> Matrix<Seq, Dim> {
        let mut out = [[0.0; Dim]; Seq];

        for i in 0..Seq {
            let row = crate::tensor::array::Array { data: x.get_row(i) };
            // fc1: Dim -> Hidden
            let hidden = self.fc1.forward(&row);
            // GELU activation
            let activated = activation::gelu(&hidden);
            // fc2: Hidden -> Dim
            out[i] = self.fc2.forward(&activated).data.try_into().unwrap();
        }

        Matrix::from_arr(out)
    }
}
