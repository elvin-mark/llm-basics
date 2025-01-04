import numpy as np
from utils.nn import (
    layer_norm,
    mha,
    ffn,
    linear,
    sigmoid,
    convolution_2d,
    relu,
    sigmoid,
)
from utils.features.img_proc import resize_bicubic
from utils.loaders.yolos import load_hparams_and_params
import os
from dotenv import load_dotenv

load_dotenv()


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def yolos_interpolation(
    position_embeddings,
    detection_tokens,
    img_size,
    patch_size=16,
    config_image_size=(800, 1333),
):
    num_detection_tokens = detection_tokens.shape[0]
    cls_pos_emb = position_embeddings[:1]
    det_pos_emb = position_embeddings[-num_detection_tokens:]

    patch_pos_emb = position_embeddings[1:-num_detection_tokens]
    patch_pos_emb = patch_pos_emb.T
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

    scale_pos_emb = np.concatenate([cls_pos_emb, patch_pos_emb, det_pos_emb])
    return scale_pos_emb


def yolos_embeddings(
    inputs, cls_token, detection_tokens, position_embeddings, conv_proj
):
    x = convolution_2d(
        inputs,
        conv_proj["w"],
        bias=conv_proj["b"],
        stride=16,
    )
    x = x.reshape(x.shape[0], -1).T
    x = np.vstack([cls_token, x, detection_tokens])

    scale_pos_emb = yolos_interpolation(
        position_embeddings,
        detection_tokens,
        img_size=(inputs.shape[1], inputs.shape[2]),
    )
    return scale_pos_emb + x


def yolos(inputs, embeddings, encoder_blocks, ln_f, clc_blocks, bbox_blocks, n_head):
    x = yolos_embeddings(inputs, **embeddings)
    for block in encoder_blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    classes = x[-100:, :]
    bboxes = x[-100:, :]
    for i, block in enumerate(clc_blocks):
        if i == len(clc_blocks) - 1:
            classes = linear(classes, **block)
        else:
            classes = relu(linear(classes, **block))
    for i, block in enumerate(bbox_blocks):
        if i == len(bbox_blocks) - 1:
            bboxes = linear(bboxes, **block)
        else:
            bboxes = relu(linear(bboxes, **block))
    return classes, sigmoid(bboxes)


hparams, params = load_hparams_and_params(os.getenv("YOLOS_MODEL_PATH"))

import numpy as np

np.random.seed(0)
x = np.random.rand(3, 400, 400)

a, b = yolos(x, **params, n_head=3)
print(a)
print(b)
