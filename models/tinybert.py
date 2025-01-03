import numpy as np
import torch
import os
from dotenv import load_dotenv

load_dotenv()
if os.getenv("TOKENIZER_TYPE", "deafult") == "default":
    from tokenizers import Tokenizer

    Tokenizer = Tokenizer
else:
    from utils.word_piece_tokenizer import WordPieceTokenizer

    Tokenizer = WordPieceTokenizer


def load_encoder_hparams_and_params(model_path, device="cpu"):
    n_layers = 6
    prefix = "bert."
    model = torch.load(model_path, map_location=device)
    # vocab embedding. shape [vocab_size, emb_dim] ex. [50257, 768]
    wte = model[f"{prefix}embeddings.word_embeddings.weight"].numpy()
    # context embedding. shape [ctx_len, emb_dim] ex. [1024, 768]
    wpe = model[f"{prefix}embeddings.position_embeddings.weight"].numpy()
    # Token type embedding. shape [2, emb_dim] ex. [2, 768]
    wtte = model[f"{prefix}embeddings.token_type_embeddings.weight"].numpy()

    ln_0 = {
        "g": model[f"{prefix}embeddings.LayerNorm.weight"].numpy(),
        "b": model[f"{prefix}embeddings.LayerNorm.bias"].numpy(),
    }
    blocks = []
    for i in range(n_layers):
        q = {
            "w": model[
                f"{prefix}encoder.layer.{i}.attention.self.query.weight"
            ].numpy(),
            "b": model[f"{prefix}encoder.layer.{i}.attention.self.query.bias"].numpy(),
        }
        k = {
            "w": model[f"{prefix}encoder.layer.{i}.attention.self.key.weight"].numpy(),
            "b": model[f"{prefix}encoder.layer.{i}.attention.self.key.bias"].numpy(),
        }
        v = {
            "w": model[
                f"{prefix}encoder.layer.{i}.attention.self.value.weight"
            ].numpy(),
            "b": model[f"{prefix}encoder.layer.{i}.attention.self.value.bias"].numpy(),
        }
        c_attn = {
            "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
            "b": np.hstack((q["b"], k["b"], v["b"])),
        }
        c_proj = {
            "w": model[f"{prefix}encoder.layer.{i}.attention.output.dense.weight"]
            .numpy()
            .T,
            "b": model[
                f"{prefix}encoder.layer.{i}.attention.output.dense.bias"
            ].numpy(),
        }
        attn = {"c_attn": c_attn, "c_proj": c_proj}
        ln_1 = {
            "g": model[
                f"{prefix}encoder.layer.{i}.attention.output.LayerNorm.weight"
            ].numpy(),
            "b": model[
                f"{prefix}encoder.layer.{i}.attention.output.LayerNorm.bias"
            ].numpy(),
        }

        mlp_c_fc = {
            "w": model[f"{prefix}encoder.layer.{i}.intermediate.dense.weight"]
            .numpy()
            .T,
            "b": model[f"{prefix}encoder.layer.{i}.intermediate.dense.bias"].numpy(),
        }

        mlp_c_proj = {
            "w": model[f"{prefix}encoder.layer.{i}.output.dense.weight"].numpy().T,
            "b": model[f"{prefix}encoder.layer.{i}.output.dense.bias"].numpy(),
        }

        mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

        ln_2 = {
            "g": model[f"{prefix}encoder.layer.{i}.output.LayerNorm.weight"].numpy(),
            "b": model[f"{prefix}encoder.layer.{i}.output.LayerNorm.bias"].numpy(),
        }
        block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
        blocks.append(block)
    qa = {
        "w": model[f"qa_outputs.weight"].numpy().T,
        "b": model[f"qa_outputs.bias"].numpy(),
    }

    params = {
        "ln_0": ln_0,
        "wte": wte,
        "wpe": wpe,
        "wtte": wtte,
        "blocks": blocks,
        "qa": qa,
    }
    hparams = {}
    hparams["n_head"] = 12
    hparams["n_ctx"] = 1024
    return hparams, params


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-12):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    return linear(relu(linear(x, **c_fc)), **c_proj)


def attention(q, k, v):
    return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(
        map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1))
    )
    out_heads = [attention(q, k, v) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = layer_norm(x + mha(x, **attn, n_head=n_head), **ln_1)
    x = layer_norm(x + ffn(x, **mlp), **ln_2)
    return x


def bert(inputs, segment_ids, wte, wpe, wtte, ln_0, blocks, qa, n_head):
    x = wte[inputs] + wpe[range(len(inputs))] + wtte[segment_ids]
    x = layer_norm(x, **ln_0)
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return linear(x, **qa)


hparams, params = load_encoder_hparams_and_params(
    model_path=os.getenv("BERT_MODEL_PATH")
)

tokenizer = Tokenizer.from_file(os.getenv("BERT_TOKENIZER_PATH"))

question = """
What are the primary threats to the Great Barrier Reef mentioned in the context?
"""
context = """
The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system, spanning over 2,300 kilometers. It is composed of over 2,900 individual reefs and 900 islands. The reef is home to a vast diversity of marine life, including over 1,500 species of fish, 400 species of coral, and various species of sharks, rays, and turtles. In recent decades, the reef has faced significant threats from climate change, coral bleaching, and pollution.
"""
question_ids = tokenizer.encode(question).ids
context_ids = tokenizer.encode(context).ids
input_ids = question_ids + context_ids[1:]
token_type_ids = [0] * len(question_ids) + [1] * len(context_ids[1:])
logits = bert(input_ids, token_type_ids, **params, n_head=hparams["n_head"])
idx0 = np.argmax(logits[:, 0])
idx1 = np.argmax(logits[:, 1])
output_text = tokenizer.decode(input_ids[idx0 : idx1 + 1])
print(output_text)
