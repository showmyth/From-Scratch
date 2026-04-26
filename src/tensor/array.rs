pub struct Array<const ArrSize: usize> {
    pub data: [f32; ArrSize],
}

impl<const ArrSize: usize> Array<ArrSize> {
    pub fn new(data: [f32; ArrSize]) -> Self {
        Self { data }
    }

    pub fn from_arr(data: [f32; ArrSize]) -> Self {
        Self { data }
    }
}
