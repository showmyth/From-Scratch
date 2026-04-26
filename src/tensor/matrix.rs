pub struct Matrix<const MatrixRows: usize, const MatrixCols: usize> {
    // 1D flat vector instead of 2D array
    pub data: Vec<f32>,
}

#[derive(Clone, Copy)]
pub enum InitStrategy {
    Naive,
    Xavier,
    He,
}

impl<const MatrixRows: usize, const MatrixCols: usize> Matrix<MatrixRows, MatrixCols> {
    pub fn zeros() -> Self {
        Self {
            data: vec![0.0; MatrixRows * MatrixCols],
        }
    }

    // Still takes a 2D array for convenience of manual initialization
    pub fn from_arr(arr: [[f32; MatrixCols]; MatrixRows]) -> Self {
        let mut data = Vec::with_capacity(MatrixRows * MatrixCols);
        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                data.push(arr[i][j]);
            }
        }
        Self { data }
    }

    
    pub fn get_row(&self, row: usize) -> Vec<f32> {
        let start = row * MatrixCols;
        self.data[start..start + MatrixCols].to_vec()
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * MatrixCols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * MatrixCols + col] = val;
    }

    pub fn randn(rng: &mut u64, strategy: Option<InitStrategy>) -> Self {
        use crate::helpers::helper_math::randn;

        let scale = match strategy.unwrap_or(InitStrategy::Naive) {
            InitStrategy::Naive => 0.01,
            InitStrategy::Xavier => (2.0 / (MatrixRows + MatrixCols) as f32).sqrt(),
            InitStrategy::He => (2.0 / MatrixRows as f32).sqrt(),
        };

        let mut data = vec![0.0; MatrixRows * MatrixCols];

        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                data[i * MatrixCols + j] = randn(rng) as f32 * scale;
            }
        }

        Self { data }
    }

    pub fn transpose(&self) -> Matrix<MatrixCols, MatrixRows> {
        let mut data = vec![0.0; MatrixRows * MatrixCols];
        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                // write into j, i
                data[j * MatrixRows + i] = self.data[i * MatrixCols + j];
            }
        }
        Matrix { data }
    }

    pub fn matmul<const Other: usize>(
        &self,
        rhs: &Matrix<MatrixCols, Other>,
    ) -> Matrix<MatrixRows, Other> {
        let mut data = vec![0.0; MatrixRows * Other];
        for i in 0..MatrixRows {
            for j in 0..Other {
                let mut sum = 0.0;
                for k in 0..MatrixCols {
                    sum += self.data[i * MatrixCols + k] * rhs.data[k * Other + j];
                }
                data[i * Other + j] = sum;
            }
        }
        Matrix { data }
    }

    pub fn scale(&self, scalar: f32) -> Matrix<MatrixRows, MatrixCols> {
        let mut data = vec![0.0; MatrixRows * MatrixCols];
        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                data[i * MatrixCols + j] = self.data[i * MatrixCols + j] * scalar;
            }
        }
        Matrix { data }
    }

    pub fn add(&self, rhs: &Matrix<MatrixRows, MatrixCols>) -> Matrix<MatrixRows, MatrixCols> {
        let mut data = vec![0.0; MatrixRows * MatrixCols];
        for i in 0..MatrixRows {
            for j in 0..MatrixCols {
                data[i * MatrixCols + j] = self.data[i * MatrixCols + j] + rhs.data[i * MatrixCols + j];
            }
        }
        Matrix { data }
    }
}
