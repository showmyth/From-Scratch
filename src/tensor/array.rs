pub struct Array<const ArrSize: usize> {
    pub data: Vec<f32>,
}

impl<const ArrSize: usize> Array<ArrSize> {
    pub fn new(data: [f32; ArrSize]) -> Self {
        Self { data: data.to_vec() }
    }

    pub fn from_arr(data: [f32; ArrSize]) -> Self {
        Self { data: data.to_vec() }
    }
}
