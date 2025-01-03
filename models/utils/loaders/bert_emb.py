import torch
import numpy as np


def load_hparams_and_params(model_path, device="cpu"):
    n_layers = 6
    prefix = ""
    model = torch.load(model_path, map_location=device)
    # vocab embedding. shape [vocab_size, emb_dim] ex. [50257, 384]
    wte = model[f"{prefix}embeddings.word_embeddings.weight"].numpy()
    # context embedding. shape [ctx_len, emb_dim] ex. [1024, 384]
    wpe = model[f"{prefix}embeddings.position_embeddings.weight"].numpy()
    # Token type embedding. shape [2, emb_dim] ex. [2, 384]
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
    pooler = {
        "w": model[f"pooler.dense.weight"].numpy().T,
        "b": model[f"pooler.dense.bias"].numpy(),
    }

    params = {
        "ln_0": ln_0,
        "wte": wte,
        "wpe": wpe,
        "wtte": wtte,
        "blocks": blocks,
        "pooler": pooler,
    }
    hparams = {}
    hparams["n_head"] = 12
    hparams["n_ctx"] = 1024
    return hparams, params
