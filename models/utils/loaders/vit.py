import torch
import numpy as np


def load_hparams_and_params(model_path):
    n_encoder_layers = 12
    hidden_dim = 192

    model = torch.load(model_path, map_location="cpu")
    cls_token = model["vit.embeddings.cls_token"].numpy().reshape(1, hidden_dim)
    position_embeddings = (
        model["vit.embeddings.position_embeddings"].numpy().reshape(-1, hidden_dim)
    )

    conv_proj = {
        "w": model["vit.embeddings.patch_embeddings.projection.weight"]
        .numpy()
        .transpose(1, 0, 2, 3),
        "b": model["vit.embeddings.patch_embeddings.projection.bias"].numpy(),
    }

    embeddings = {
        "cls_token": cls_token,
        "position_embeddings": position_embeddings,
        "conv_proj": conv_proj,
    }

    encoder_blocks = []
    for i in range(n_encoder_layers):
        q = {
            "w": model[
                f"vit.encoder.layer.{i}.attention.attention.query.weight"
            ].numpy(),
            "b": model[f"vit.encoder.layer.{i}.attention.attention.query.bias"].numpy(),
        }
        k = {
            "w": model[f"vit.encoder.layer.{i}.attention.attention.key.weight"].numpy(),
            "b": model[f"vit.encoder.layer.{i}.attention.attention.key.bias"].numpy(),
        }
        v = {
            "w": model[
                f"vit.encoder.layer.{i}.attention.attention.value.weight"
            ].numpy(),
            "b": model[f"vit.encoder.layer.{i}.attention.attention.value.bias"].numpy(),
        }
        c_attn = {
            "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
            "b": np.hstack((q["b"], k["b"], v["b"])),
        }
        c_proj = {
            "w": model[f"vit.encoder.layer.{i}.attention.output.dense.weight"]
            .numpy()
            .T,
            "b": model[f"vit.encoder.layer.{i}.attention.output.dense.bias"].numpy(),
        }
        attn = {"c_attn": c_attn, "c_proj": c_proj}
        ln_1 = {
            "g": model[f"vit.encoder.layer.{i}.layernorm_before.weight"].numpy(),
            "b": model[f"vit.encoder.layer.{i}.layernorm_before.bias"].numpy(),
        }

        mlp_c_fc = {
            "w": model[f"vit.encoder.layer.{i}.intermediate.dense.weight"].numpy().T,
            "b": model[f"vit.encoder.layer.{i}.intermediate.dense.bias"].numpy(),
        }

        mlp_c_proj = {
            "w": model[f"vit.encoder.layer.{i}.output.dense.weight"].numpy().T,
            "b": model[f"vit.encoder.layer.{i}.output.dense.bias"].numpy(),
        }

        mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

        ln_2 = {
            "g": model[f"vit.encoder.layer.{i}.layernorm_after.weight"].numpy(),
            "b": model[f"vit.encoder.layer.{i}.layernorm_after.bias"].numpy(),
        }
        block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
        encoder_blocks.append(block)
    ln_f = {
        "g": model["vit.layernorm.weight"].numpy().T,
        "b": model["vit.layernorm.bias"].numpy(),
    }

    classifier = {
        "w": model["classifier.weight"].numpy().T,
        "b": model["classifier.bias"].numpy(),
    }

    params = {
        "embeddings": embeddings,
        "encoder_blocks": encoder_blocks,
        "ln_f": ln_f,
        "classifier": classifier,
    }
    hparams = {}
    hparams["n_head"] = 3
    hparams["n_ctx"] = 1024
    return hparams, params
