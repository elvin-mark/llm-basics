# YOLOs Model Implementation

## Key Components

### 1. **Transformer Block**

```python
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x
```

Implements a standard transformer block with:

- Multi-Head Attention (MHA): Captures relationships between elements in the input sequence.
- Feed-Forward Network (FFN): Processes the intermediate representation.
- Layer Normalization (LN): Stabilizes training.

### 2. YOLOs Interpolation

```python
def yolos_interpolation(
    position_embeddings,
    detection_tokens,
    img_size,
    patch_size=16,
    config_image_size=(800, 1333),
):
    ...
    scale_pos_emb = np.concatenate([cls_pos_emb, patch_pos_emb, det_pos_emb])
    return scale_pos_emb
```

Handles resizing and interpolation of position embeddings to adapt to different input image sizes.
Key Steps:

- Extracts class (cls_pos_emb), patch (patch_pos_emb), and detection token embeddings (det_pos_emb).
- Reshapes and resizes the patch embeddings using bicubic interpolation to match the input image size.
- Combines all embeddings to create the final position embedding (scale_pos_emb).

### 3. YOLOs Embeddings

```python
def yolos_embeddings(
    inputs, cls_token, detection_tokens, position_embeddings, conv_proj
):
    ...
    return scale_pos_emb + x
```

Computes the initial input embeddings for the model:

- Applies a convolutional projection to downscale the input image.
- Combines class tokens, patch embeddings, and detection tokens into a unified embedding.
- Adds position embeddings (processed through yolos_interpolation).

### 4. YOLOs Model

```python
def yolos(inputs, embeddings, encoder_blocks, ln_f, clc_blocks, bbox_blocks, n_head):
    ...
    return classes, sigmoid(bboxes)
```

The main YOLOs model function. It processes the input image and generates object classes and bounding boxes:

- Embeddings: Initializes the embeddings for the input image.
- Encoder Blocks: Passes embeddings through a series of transformer blocks to process the image.
- Class Predictions:
  - Applies a series of linear layers with ReLU activations to generate class logits.
- Bounding Box Predictions:
  - Processes the embeddings through linear layers to predict bounding box coordinates.
- Sigmoid Activation: Ensures bounding box outputs are normalized to a range of [0, 1].

### 5. Loading Parameters and Running

```python
hparams, params = load_hparams_and_params(os.getenv("YOLOS_MODEL_PATH"))

np.random.seed(0)
x = np.random.rand(3, 400, 400)

a, b = yolos(x, **params, n_head=3)
print(a)
print(b)
```

- Model Parameters:
  - Loaded from a specified path using load_hparams_and_params.
- Input Data:
  - A random image tensor of shape (3, 400, 400) is generated.
- Model Execution:
  - yolos() processes the input, generating:
    - a: Predicted object classes.
    - b: Predicted bounding boxes.

## Summary of Functionality

YOLOs leverages transformer-based architectures for object detection by combining convolutional features, positional embeddings, and multi-head attention.
