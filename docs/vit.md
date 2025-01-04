# Vision Transformer (ViT) Implementation

### 1. **Transformer Block**

The `transformer_block` function is the core unit of a transformer model. It performs:

- **Multi-head self-attention (MHA)**: Captures relationships between different parts of the input.
- **Feed-forward neural network (FFN)**: Processes the attended features.
- **Layer Normalization (LN)**: Stabilizes training by normalizing inputs.
- **Residual Connections**: Adds the input to the output of MHA and FFN layers.

```python
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x
```

### 2. Position Embedding Interpolation

The vit_interpolation function handles resizing position embeddings when the input image size differs from the default model configuration (e.g., resizing from (224, 224) to a new size).

Position embeddings are split into a classification token (cls_pos_emb) and patch embeddings.
Resizing is performed using bicubic interpolation for consistency in patch dimensions.

```python
def vit_interpolation(
    position_embeddings,
    img_size,
    patch_size=16,
    config_image_size=(224, 224),
):
    cls_pos_emb = position_embeddings[:1]
    patch_pos_emb = position_embeddings[1:].T
    hidden_size, seq_len = patch_pos_emb.shape

    patch_height, patch_width = (
        config_image_size[0] // patch_size,
        config_image_size[1] // patch_size,
    )
    patch_pos_emb = patch_pos_emb.reshape(hidden_size, patch_height, patch_width)

    height, width = img_size
    new_patch_height, new_patch_width = (
        height // patch_size,
        width // patch_size,
    )

    patch_pos_emb = resize_bicubic(patch_pos_emb, new_patch_height, new_patch_width)

    patch_pos_emb = patch_pos_emb.reshape(hidden_size, -1).transpose(1, 0)

    scale_pos_emb = np.concatenate([cls_pos_emb, patch_pos_emb])
    return scale_pos_emb
```

### 3. Embedding Generation

The vit_embeddings function prepares input embeddings by:

- Using a convolutional layer (convolution_2d) to extract patch tokens.
- Adding a classification token (cls_token) to the token sequence.
- Incorporating position embeddings adjusted to the input size.

```python
def vit_embeddings(inputs, cls_token, position_embeddings, conv_proj):
    x = convolution_2d(
        inputs,
        conv_proj["w"],
        bias=conv_proj["b"],
        stride=16,
    )
    x = x.reshape(x.shape[0], -1).T
    x = np.vstack([cls_token, x])

    scale_pos_emb = vit_interpolation(
        position_embeddings,
        img_size=(inputs.shape[1], inputs.shape[2]),
    )
    return scale_pos_emb + x
```

### 4. Vision Transformer Model

The vit function combines embeddings, multiple transformer blocks, and a classification head:

- Embedding Generation: Input images are converted into tokens with position embeddings.
- Transformer Encoder: A series of transformer blocks process the embeddings.
- Final Layer Normalization: Stabilizes the final token representation.
- Classification Head: Maps the output to logits using a linear layer.

```python
def vit(inputs, embeddings, encoder_blocks, ln_f, classifier, n_head):
    x = vit_embeddings(inputs, **embeddings)
    for block in encoder_blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    logits = linear(x, **classifier)
    return logits[0]
```

### 5. Usage Example

The model is initialized with hyperparameters and weights using load_hparams_and_params, and an example input (x) is passed to the model for inference.

```python
hparams, params = load_hparams_and_params(os.getenv("VIT_MODEL_PATH"))

import numpy as np
np.random.seed(0)
x = np.random.rand(3, 224, 224)

logits = vit(x, **params, n_head=3)
print(logits)
```

## Key Highlights

Minimal Dependencies: The implementation relies on utility functions (mha, ffn, layer_norm, etc.).

- Flexibility: The model adjusts position embeddings to varying input sizes.
- Simplicity: A clean structure for understanding transformer-based models.
