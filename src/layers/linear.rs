use crate::tensor::array::Array;
use crate::tensor::matrix::{Matrix, InitStrategy};
pub struct Linear<const InputSize: usize, const OutputSize: usize> {
    pub weights: Matrix<InputSize, OutputSize>,
    pub bias: Array<OutputSize>,
    pub grad_weights: Option<Matrix<InputSize, OutputSize>>,
    pub grad_bias: Option<Array<OutputSize>>,
}

impl<const InputSize: usize, const OutputSize: usize> Linear<InputSize, OutputSize>{
    pub fn new(seed: u64, strat: Option<InitStrategy>) -> Self {
        let mut rng = seed; // we need to make some global function for seeding, 
                                 // so something like get_seed(), sort of copying
                                 // np.random.seed().
        let weights_data = Matrix::randn(&mut rng, strat);
        Self { 
            weights: weights_data,
            bias: Array { data: [0.0; OutputSize] },
            grad_weights: None,
            grad_bias: None,
        }
    }

    pub fn forward(&self, input: &Array<InputSize>) -> Array<OutputSize> {
        let mut res = [0.0; OutputSize];
        for i in 0..OutputSize {
            for j in 0..InputSize {
                res[i] += self.weights.get(i, j) * input.data[j];
            }
            res[i] += self.bias.data[i];
        }

        Array { data: res }
    }

    pub fn forward_seq<const Seq: usize>(&self, input: &Matrix<Seq, InputSize>) -> Matrix<Seq, OutputSize> {
        let mut data = [[0.0; OutputSize]; Seq];

        for s in 0..Seq {
            for i in 0..OutputSize {
                for j in 0..InputSize {
                    data[s][i] += self.weights.get(i, j) * input.data[s][j];
                }
                data[s][i] += self.bias.data[i];
            }
        }

        Matrix { data }


    }
}