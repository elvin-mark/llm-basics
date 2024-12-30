# RWKV Model Explanation

This script implements a simplified version of the RWKV model for text generation. The RWKV model is designed for efficient sequential processing, combining ideas from transformers and RNNs.

## Components

### 1. Initialization

- **Layers and Embeddings:** `N_LAYER` (12) and `N_EMBD` (768).
- **Model and Tokenizer:** Loaded using file paths specified in environment variables.

### 2. Functions

- **Layer Normalization (`layer_norm`):** Normalizes inputs with learnable parameters.
- **Time Mixing:** Combines current token and past states to model temporal relationships.
- **Channel Mixing:** Processes individual input features with nonlinear transformations.
- **Sampling (`sample_probs`):** Samples tokens based on temperature and top-p filtering.

### 3. RWKV Implementation

- Embeds input tokens.
- Sequentially applies:
  - Time mixing for temporal dynamics.
  - Channel mixing for feature-level interactions.
- Normalizes and outputs probabilities via a softmax operation.

### 4. Inference Process

- Encodes the input text.
- Iteratively generates tokens using:
  - `RWKV` for computing probabilities.
  - `sample_probs` for token selection.
- Decodes tokens to produce coherent text.

## Example Workflow

1. **Input:** "What is the capital of Peru?"
2. **Generated Output:** "The capital of Peru is Lima."

This approach showcases the power of RWKV in text generation tasks, combining the efficiency of RNNs with the context-learning ability of transformers.
