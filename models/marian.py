import numpy as np
from dotenv import load_dotenv
import os
from utils.nn import layer_norm, ffn, mha, linear, silu
from utils.loaders.marian import load_hparams_and_params
from utils.tokenizers.marian_tokenizer import MarianTokenizer

load_dotenv()


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head, mask_enabled=False):
    x = layer_norm(x + mha(x, **attn, n_head=n_head, mask_enabled=mask_enabled), **ln_1)
    x = layer_norm(x + ffn(x, **mlp, act_fn=silu), **ln_2)
    return x


def decoder_transformer_block(
    x, mlp, attn, encoder_attn, ln_1, ln_2, ln_3, n_head, kv_states=None
):
    x = layer_norm(x + mha(x, **attn, n_head=n_head, mask_enabled=True), **ln_1)
    if kv_states is not None:
        x = layer_norm(
            x + mha(x, **encoder_attn, kv_states=kv_states, n_head=n_head), **ln_2
        )
    x = layer_norm(x + ffn(x, **mlp, act_fn=silu), **ln_3)
    return x


def marian_encoder(input_ids, params, hparams, embed_scale=1.0):
    # Add positional embeddings
    x = (
        params["encoder"]["embed_tokens"][input_ids] * embed_scale
        + params["encoder"]["embed_positions"][: len(input_ids)]
    )
    # Transformer layers
    for layer in params["encoder"]["blocks"]:
        x = transformer_block(x, **layer, n_head=hparams["n_head"])
    return x


def marian_decoder(encoder_output, input_ids, params, hparams, embed_scale=1.0):
    # Embed tokens
    x = (
        params["decoder"]["embed_tokens"][input_ids] * embed_scale
        + params["decoder"]["embed_positions"][: len(input_ids)]
    )

    # Transformer layers
    for layer in params["decoder"]["blocks"]:
        x = decoder_transformer_block(
            x, **layer, kv_states=encoder_output, n_head=hparams["n_head"]
        )
    return x


def marian_generate(input_ids, params, hparams, n_tokens=20, embed_scale=1.0):
    # Encode audio
    encoder_output = marian_encoder(input_ids, params, hparams, embed_scale=embed_scale)
    # Initialize decoder inputs
    input_ids = [32000]  # Start of sequence token
    for _ in range(n_tokens):
        logits = marian_decoder(
            encoder_output, input_ids, params, hparams, embed_scale=embed_scale
        )
        next_token = np.argmax(linear(logits[-1], **params["lm_head"]))
        if next_token == 0:
            break
        input_ids.append(next_token)
    return input_ids


hparams, params = load_hparams_and_params(os.getenv("MARIAN_MODEL_PATH"))
tokenizer = MarianTokenizer.from_file(os.getenv("MARIAN_TOKENIZER_PATH"))

input_ids = tokenizer.encode("Hello, where are you now?").ids
ids = marian_generate(input_ids, params, hparams, n_tokens=20, embed_scale=16.0)
print(tokenizer.decode(ids))
