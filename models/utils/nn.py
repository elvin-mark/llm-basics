import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def relu(x):
    return np.maximum(0, x)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def rms_norm(x):
    return x / np.sqrt(np.square(x).mean(-1, keepdims=True) + 1e-6)


def layer_norm(x, g, b, eps=1e-12):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj, act_fn=gelu):
    return linear(act_fn(linear(x, **c_fc)), **c_proj)


def attention(q, k, v, mask=None):
    if mask is None:
        return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head, kv_states=None, mask_enabled=False):
    x = linear(x, **c_attn)

    if kv_states is not None:
        qkv = []
        dim = c_attn["w"].shape[0]
        kv = linear(kv_states, **c_attn)
        qkv = [x[:, :dim], kv[:, dim : 2 * dim], kv[:, 2 * dim :]]
    else:
        qkv = np.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
    causal_mask = None
    if mask_enabled:
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, mask=causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x


def convolution_1d(input_tensor, weights, bias, stride=1, padding=0):
    # Get dimensions
    in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weights.shape

    # Apply padding to the input tensor
    if padding > 0:
        input_tensor = np.pad(
            input_tensor,
            ((0, 0), (padding, padding)),
            mode="constant",
            constant_values=0,
        )

    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1

    # Extract sliding windows (using strides)
    strided_indices = np.lib.stride_tricks.sliding_window_view(
        input_tensor, kernel_size, axis=1
    )
    # Shape of strided_indices: (in_channels, output_length, kernel_size)
    strided_indices = strided_indices[:, ::stride, :]  # Apply stride

    # Perform the convolution using broadcasting and summation
    output_tensor = np.tensordot(weights, strided_indices, axes=([1, 2], [0, 2]))
    # Shape of output_tensor: (out_channels, output_length)

    # Add bias to each output channel
    output_tensor += bias[:, None]  # Bias broadcasted to match output shape

    return output_tensor


def convolution_2d(image, kernel, bias=None, stride=2, padding=0):
    # Extract dimensions
    in_channels, img_width, img_height = image.shape
    in_channels_k, out_channels, k_width, k_height = kernel.shape

    # Ensure the kernel matches input channels
    if in_channels != in_channels_k:
        raise ValueError(
            "The number of input channels in the image and kernel must match."
        )

    # Add padding to the image
    if padding > 0:
        image = np.pad(
            image,
            pad_width=((0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )

    # Calculate output dimensions
    out_width = (image.shape[1] - k_width) // stride + 1
    out_height = (image.shape[2] - k_height) // stride + 1

    # Use stride tricks to create a sliding window view of the image
    shape = (in_channels, out_width, out_height, k_width, k_height)
    strides = (
        image.strides[0],
        stride * image.strides[1],
        stride * image.strides[2],
        image.strides[1],
        image.strides[2],
    )

    sliding_windows = np.lib.stride_tricks.as_strided(
        image, shape=shape, strides=strides
    )

    # Perform the convolution
    conv_result = np.einsum("cxykh,cokh->oxy", sliding_windows, kernel)

    # Add bias if provided
    if bias is not None:
        if bias.shape[0] != out_channels:
            raise ValueError("Bias shape must match the number of output channels.")
        conv_result += bias[:, None, None]

    return conv_result
