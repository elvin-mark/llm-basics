# TinyBERT Implementation

## Overview

This script implements a simplified version of the BERT model for question answering tasks. It loads a pretrained TinyBERT model and defines the forward pass for tokenized inputs.

### Key Components

1. **Model Loader**:

   - Loads pretrained parameters including embeddings, attention weights, and feedforward layers.
   - Constructs a 6-layer Transformer with multi-head self-attention.

2. **Utility Functions**:

   - `gelu`, `relu`, and `softmax` for activations.
   - `layer_norm` and `linear` for processing layers.
   - `ffn` for feedforward networks.

3. **Attention Mechanism**:

   - Implements scaled dot-product attention and multi-head attention (`mha`).

4. **Transformer Block**:

   - Combines attention, feedforward, and layer normalization.

5. **Inference**:
   - Tokenizes question and context using `tokenizers`.
   - Predicts answer indices and decodes the result.

### Code Workflow

1. Load pretrained parameters and hyperparameters.
2. Tokenize the question and context.
3. Compute embeddings and pass them through the model.
4. Predict start and end indices for the answer.
5. Decode the predicted tokens to generate the output text.

### Example

**Question**: What are the primary threats to the Great Barrier Reef?  
**Context**: Contains details about the reef and mentions climate change, coral bleaching, and pollution as threats.

**Output**: _"climate change, coral bleaching, and pollution"_
