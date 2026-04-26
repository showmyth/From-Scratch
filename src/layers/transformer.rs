use crate::{
    layers::{decoder::Decoder, encoder::Encoder, linear::Linear, positional::PositionalEncoding},
    tensor::matrix::{InitStrategy, Matrix},
};

pub struct Transformer<
    const Dim: usize,
    const HeadDim: usize,
    const Hidden: usize,
    const VocabSize: usize,
    const SeqSrc: usize,
    const SeqTgt: usize,
> {
    pos_enc_src: PositionalEncoding<SeqSrc, Dim>,
    pos_enc_tgt: PositionalEncoding<SeqTgt, Dim>,
    encoders: Vec<Encoder<Dim, HeadDim, Hidden>>,
    decoders: Vec<Decoder<Dim, HeadDim, Hidden>>,
    fc_out: Linear<Dim, VocabSize>,
}

impl<
    const Dim: usize,
    const HeadDim: usize,
    const Hidden: usize,
    const VocabSize: usize,
    const SeqSrc: usize,
    const SeqTgt: usize,
> Transformer<Dim, HeadDim, Hidden, VocabSize, SeqSrc, SeqTgt>
{
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        seed: u64,
        strat: Option<InitStrategy>,
    ) -> Self {
        let pos_enc_src = PositionalEncoding::new();
        let pos_enc_tgt = PositionalEncoding::new();

        let encoders = (0..num_layers)
            .map(|i| Encoder::new(num_heads, seed + i as u64 * 100, strat))
            .collect();

        let decoders = (0..num_layers)
            .map(|i| Decoder::new(num_heads, seed + 5000 + i as u64 * 100, strat))
            .collect();

        // The final layer projects from the model dimension back to the vocabulary size
        let fc_out = Linear::new(seed + 9999, strat);

        Self {
            pos_enc_src,
            pos_enc_tgt,
            encoders,
            decoders,
            fc_out,
        }
    }

    pub fn forward(
        &self,
        src: &Matrix<SeqSrc, Dim>,
        tgt: &Matrix<SeqTgt, Dim>,
    ) -> Matrix<SeqTgt, VocabSize> {
        // 1. Add Positional Encoding to the embedded source sequence
        let mut enc_out = self.pos_enc_src.forward(src);

        // 2. Pass through the stack of N Encoder layers
        for encoder in &self.encoders {
            enc_out = encoder.forward(&enc_out);
        }

        // 3. Add Positional Encoding to the embedded target sequence
        let mut dec_out = self.pos_enc_tgt.forward(tgt);

        // 4. Pass through the stack of N Decoder layers
        // Cross-attention links the decoder state (dec_out) to the final encoder state (enc_out)
        for decoder in &self.decoders {
            dec_out = decoder.forward(&dec_out, &enc_out);
        }

        // 5. Final projection from Dim to VocabSize
        self.fc_out.forward_seq(&dec_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_forward() {
        // Small dimensions for a quick test
        const DIM: usize = 16;
        const HEAD_DIM: usize = 4; // 4 heads * 4 dim = 16
        const HIDDEN: usize = 64;
        const VOCAB_SIZE: usize = 100;
        const SEQ_SRC: usize = 10;
        const SEQ_TGT: usize = 10;

        let transformer = Transformer::<DIM, HEAD_DIM, HIDDEN, VOCAB_SIZE, SEQ_SRC, SEQ_TGT>::new(
            2,    // 2 encoder/decoder layers
            4,    // 4 attention heads
            42,   // seed
            None, // default init strategy
        );

        // Dummy input matrices (e.g. mock embeddings)
        let mut rng_src = 123;
        let mut rng_tgt = 456;
        let src_input = Matrix::<SEQ_SRC, DIM>::randn(&mut rng_src, None);
        let tgt_input = Matrix::<SEQ_TGT, DIM>::randn(&mut rng_tgt, None);

        // Run the forward pass!
        let output = transformer.forward(&src_input, &tgt_input);

        // Assert shape is correct (implicitly done by Rust's const generics)
        // and check that we actually got numbers back
        assert!(!output.data[0].is_nan());

        println!("Transformer forward pass completed successfully!");
        println!("Output shape: [{}, {}]", SEQ_TGT, VOCAB_SIZE);
        println!(
            "First token's first 5 vocab logits: {:?}",
            &output.data[..5]
        );
    }

    #[test]
    fn test_transformer_stress() {
        const DIM: usize = 128;
        const HEAD_DIM: usize = 32; // 4 heads * 32 dim = 128
        const HIDDEN: usize = 512; // usually 4x DIM
        const VOCAB_SIZE: usize = 5000;
        const SEQ_SRC: usize = 64;
        const SEQ_TGT: usize = 64;

        println!("Initializing 6-layer Transformer (Medium Config)...");
        let transformer = Transformer::<DIM, HEAD_DIM, HIDDEN, VOCAB_SIZE, SEQ_SRC, SEQ_TGT>::new(
            6,  // 6 layers (standard original paper)
            4,  // 4 attention heads
            99, // seed
            None,
        );

        let mut rng = 1;
        println!("Allocating input matrices...");
        let src_input = Matrix::<SEQ_SRC, DIM>::randn(&mut rng, None);
        let tgt_input = Matrix::<SEQ_TGT, DIM>::randn(&mut rng, None);

        println!("Running forward pass through all 6 layers...");
        let output = transformer.forward(&src_input, &tgt_input);

        assert!(!output.data[0].is_nan());
        println!(
            "Stress test passed! Output size: [{}, {}]",
            SEQ_TGT, VOCAB_SIZE
        );
    }
}
