use crate::tensor::array::Array;

pub struct Norm<const Dim: usize> {
    gamma: Array<Dim>,
    beta: Array<Dim>,
    eps: f32,
}

impl<const Dim: usize> Norm<Dim> {
    pub fn new() -> Self {
        let gamma = Array::from_arr([1.0; Dim]);
        let beta = Array::from_arr([0.0; Dim]);
        Self {
            gamma,
            beta,
            eps: 1e-5,
        }
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn forward(&self, x: &Array<Dim>) -> Array<Dim> {
        let mean = x.data.iter().sum::<f32>() / (Dim as f32);
        let var = x.data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (Dim as f32);
        let std_dev = (var + self.eps).sqrt();

        let mut out = [0.0; Dim];
        for i in 0..Dim {
            let norm = (x.data[i] - mean) / std_dev;
            out[i] = self.gamma.data[i] * norm + self.beta.data[i];
        }

        Array::from_arr(out)
    }
    pub fn forward_seq<const Seq: usize>(
        &self,
        x: &crate::tensor::matrix::Matrix<Seq, Dim>,
    ) -> crate::tensor::matrix::Matrix<Seq, Dim> {
        let mut out = [[0.0; Dim]; Seq];
        for i in 0..Seq {
            let arr = Array::from_arr(x.data[i]);
            let res = self.forward(&arr);
            out[i] = res.data;
        }
        crate::tensor::matrix::Matrix::from_arr(out)
    }
}
