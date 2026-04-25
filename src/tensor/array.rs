pub struct Array<const ArrSize: usize> {
    pub data: [f64; ArrSize],
}

impl<const ArrSize: usize> Array<ArrSize> {
    pub fn new(data: [f64; ArrSize]) -> Self {
        Self { data }
    }

    pub fn from_arr(data: [f64; ArrSize]) -> Self {
        Self { data }
    }
}
