import numpy as np
from utils.nn import (
    layer_norm,
    mha,
    ffn,
    linear,
    convolution_2d,
)
from utils.features.img_proc import resize_bicubic
from utils.loaders.vit import load_hparams_and_params
import os
from dotenv import load_dotenv

load_dotenv()


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def vit_interpolation(
    position_embeddings,
    img_size,
    patch_size=16,
    config_image_size=(224, 224),
):
    cls_pos_emb = position_embeddings[:1]

    patch_pos_emb = position_embeddings[1:]
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

    scale_pos_emb = np.concatenate([cls_pos_emb, patch_pos_emb])
    return scale_pos_emb


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


def vit(inputs, embeddings, encoder_blocks, ln_f, classifier, n_head):
    x = vit_embeddings(inputs, **embeddings)
    for block in encoder_blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    logits = linear(x, **classifier)
    return logits[0]


hparams, params = load_hparams_and_params(os.getenv("VIT_MODEL_PATH"))

import numpy as np

np.random.seed(0)
x = np.random.rand(3, 224, 224)

logits = vit(x, **params, n_head=3)
print(logits)
