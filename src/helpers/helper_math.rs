pub fn randn(rng: &mut u64) -> f64 {
    let u1 = 1.0 - lcg_next(rng);
    let u2 = lcg_next(rng);

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}