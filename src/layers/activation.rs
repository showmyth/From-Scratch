use crate::tensor::array::Array;
use crate::tensor::matrix::Matrix;

pub fn relu<const ArraySize: usize>(x: &Array<ArraySize>) -> Array<ArraySize> {
    let mut data = [0.0; ArraySize];
    for i in 0..ArraySize {
        data[i] = x.data[i].max(0.0);
    }
    Array { data }
}

pub fn gelu<const ArraySize: usize>(x: &Array<ArraySize>) -> Array<ArraySize> {
    let mut data = [0.0; ArraySize];
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();

    for i in 0..ArraySize {
        let val = x.data[i];
        data[i] = 0.5 * val * (1.0 + (sqrt_2_over_pi * (val + 0.044715 * val.powi(3))).tanh());
    }
    Array { data }
}

pub fn softmax<const ArraySize: usize>(x: &Array<ArraySize>) -> Array<ArraySize> {
    let max_val = x
        .data
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));

    let mut exps = [0.0; ArraySize];
    for i in 0..ArraySize {
        exps[i] = (x.data[i] - max_val).exp();
    }

    let sum_exps: f32 = exps.iter().sum();
    let mut result = [0.0; ArraySize];
    for i in 0..ArraySize {
        result[i] = exps[i] / sum_exps;
    }

    Array { data: result }
}

pub fn softmax_rows<const Rows: usize, const Cols: usize>(
    x: &Matrix<Rows, Cols>,
) -> Matrix<Rows, Cols> {
    let mut data = [[0.0; Cols]; Rows];

    for i in 0..Rows {
        let max_val = x.data[i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = [0.0; Cols];
        for j in 0..Cols {
            exps[j] = (x.data[i][j] - max_val).exp();
        }
        let sum: f32 = exps.iter().sum();
        for j in 0..Cols {
            data[i][j] = exps[j] / sum;
        }
    }
    Matrix { data }
}
