use std::str::MatchIndices;

pub struct Matrix<const MatrixRows: usize, const MatrixCols: usize> {
    pub data: [[f64; MatrixCols]; MatrixRows],
}


#[derive(Clone, Copy)]
pub enum InitStrategy {
    Naive,
    Xavier,
    He,
}

impl<const MatrixRows: usize, const MatrixCols: usize> Matrix<MatrixRows, MatrixCols> {
    // basic type ops
    pub fn zeros() -> Self {
        Self { data: [[0.0; MatrixCols]; MatrixRows] }
    }

    pub fn from_arr(data: [[f64; MatrixCols]; MatrixRows]) -> Self {
        Self { data }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row][col] = val;
    }

    // randn wrapper
    pub fn randn(rng: &mut u64, strategy: Option<InitStrategy>) -> Self {
        use crate::helpers::helper_math::randn;

        let scale = match strategy.unwrap_or(InitStrategy::Naive) {
            InitStrategy::Naive => 0.01,
            InitStrategy::Xavier => (2.0 / (MatrixRows + MatrixCols) as f64).sqrt(),
            InitStrategy::He => (2.0 / MatrixRows as f64).sqrt(),
        };

        let mut data = [[0.0; MatrixCols]; MatrixRows];

        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                    data[i][j] = randn(rng) * scale;
            }
        }

        Self { data } 
    }

    // basic matrix ops
    // could do it inplace, instead of copying but thats for a later time
    pub fn transpose(&self) ->  Matrix<MatrixCols, MatrixRows> {
        let mut data = [[0.0; MatrixRows]; MatrixCols];
        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                data[j][i] = self.data[i][j];
            }
        }
        Matrix { data }
    } 

    pub fn matmul<const Other: usize>(&self, rhs: &Matrix<MatrixCols, Other>) -> Matrix<MatrixRows, Other> {
        let mut data = [[0.0; Other]; MatrixRows];
        for i in 0..MatrixRows {
            for j in 0..Other {
                for k in 0..MatrixCols {
                    data[i][j] += self.data[i][k] * rhs.data[k][j];
                }
            }
        }
        Matrix { data }
    }

    pub fn scale(&self, scalar: f64) -> Matrix<MatrixRows, MatrixCols> {
        let mut data = [[0.0; MatrixCols]; MatrixRows];
        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                data[i][j] = self.data[i][j] * scalar;
            }
        }
        Matrix { data } 
    }

    // row-wise softmax
    pub fn softmax_rows(&self) -> Matrix<MatrixRows, MatrixCols> {
        let mut data = [[0.0; MatrixCols]; MatrixRows];

        for i in 0..MatrixRows {
            let max_val = self.data[i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut exps = [0.0; MatrixCols];
            for j in 0..MatrixCols {
                exps[j] = (self.data[i][j] - max_val).exp();
            }
            let sum: f64 = exps.iter().sum();
            for j in 0..MatrixCols {
                data[i][j] = exps[j] / sum;
            }
        }
        Matrix { data }
    }
}