# Tiny BERT for Text Embedding

## Overview

This code implements a simplified version of BERT (Tiny BERT) for generating text embeddings. It uses transformer-based architecture to process input sentences and produce vector representations.

## Components

1. **Model Loading**:

   - Loads pre-trained weights for embeddings, transformer blocks, and pooling layers.
   - Handles hyperparameters like number of layers and attention heads.

2. **Core Functions**:

   - Activation Functions: `gelu`, `relu`
   - Normalization: `layer_norm`
   - Attention Mechanism: `mha`, `attention`
   - Feed-Forward Networks: `ffn`
   - Transformer Layer: `transformer_block`

3. **BERT Architecture**:

   - Combines token, positional, and segment embeddings.
   - Passes input through 6 transformer layers.

4. **Text Embedding**:
   - Sentences are tokenized and converted to numerical IDs.
   - Mean pooling is applied to token-level embeddings for sentence representation.
   - Embeddings are normalized for similarity comparison.

## Usage

- **Input**: A list of sentences.
- **Output**: Normalized embeddings for each sentence.
- **Similarity Calculation**: Computes cosine similarity between embeddings.

## Key Functions

- `bert`: Processes tokenized inputs through embedding layers and transformer blocks.
- `mean_pooling_and_normalization`: Aggregates token-level embeddings into a sentence-level embedding.

## Example

Input sentences:

```python
[
    "The sun is shining brightly in the sky.",
    "It’s a clear day with plenty of sunshine.",
    "I forgot to bring my umbrella, and now it’s raining heavily.",
    "The cat is sleeping peacefully on the couch.",
]
```
