pub struct Array<const ArrSize: usize> {
    pub data: [f64; ArrSize],
}

impl<const ArrSize: usize> Array<ArrSize>{
    pub fn new(data: [f64; ArrSize]) -> Self {
        Self { data }
    }

    pub fn softmax(&self) -> Array<ArrSize> {
        let max_val =  self.data.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let mut exps = [0.0; ArrSize];
        for i in 0..ArrSize {
            exps[i] = (self.data[i] - max_val).exp();
        }

        let sum_exps: f64 = exps.iter().sum();
        let mut result = [0.0; ArrSize];
        for i in 0..ArrSize {
            result[i] = exps[i] / sum_exps;
        }

        Self { data: result }
    }
}