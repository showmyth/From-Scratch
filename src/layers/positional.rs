use crate::tensor::matrix::Matrix;

// sinusidal positional encoding
pub struct PositionalEncoding<const Seq: usize, const Dim: usize> {
    pub pe: Matrix<Seq, Dim>,
}

impl<const Seq: usize, const Dim: usize> PositionalEncoding<Seq, Dim> {
    pub fn new() -> Self {
        let mut data = [[0.0; Dim]; Seq];

        // for every position in the seq
        for pos in 0..Seq {
            // for every half dim in each pos
            for i in 0..(Dim / 2) {
                // do this (paper said so...)
                let div_term = 10000.0_f32.powf((2.0 * i as f32) / Dim as f32);

                // set even indices to sin
                data[pos][2 * i] = (pos as f32 / div_term).sin();

                // set odd indices to cos
                if 2 * i + 1 < Dim {
                    data[pos][2 * i + 1] = (pos as f32 / div_term).cos();
                }
            }
        }

        Self {
            pe: Matrix { data },
        }
    }

    pub fn forward(&self, x: &Matrix<Seq, Dim>) -> Matrix<Seq, Dim> {
        // self.pe does not hold any learnable parameters so we can just add it to the input
        x.add(&self.pe)
    }
}
