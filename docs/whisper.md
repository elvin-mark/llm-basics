# Minimalistic Whisper Implementation Using Numpy

## Overview

This implementation replicates the Whisper model's core functionality for audio-to-text transcription using **NumPy** for clarity and educational purposes.

### Key Components

1. **Parameter Loading**:

   - Extracts PyTorch model weights into NumPy arrays.
   - Initializes encoder and decoder with attention, FFN, and normalization parameters.

2. **Core Functions**:

   - **GELU Activation**: Non-linear transformation.
   - **Softmax**: For attention scoring.
   - **Layer Norm**: Normalizes inputs using scale and bias.

3. **Encoder**:

   - **Convolutions**: Extract audio features.
   - **Positional Embeddings**: Adds time-series context.
   - **Transformer Blocks**: Applies attention and FFN sequentially.

4. **Decoder**:

   - Embeds tokens and applies transformer blocks, including cross-attention to use encoder outputs.

5. **Audio Processing**:

   - Converts raw audio to spectrograms.
   - Applies Mel filter banks for compact feature extraction.

6. **Inference Pipeline**:
   - Encodes audio into latent representations.
   - Decodes autoregressively to generate text tokens.
   - Translates tokens to text via a tokenizer.
