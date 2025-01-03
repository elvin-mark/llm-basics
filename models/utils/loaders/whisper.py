import torch
import numpy as np


def load_hparams_and_params(
    model_path,
    encoder_layers=4,
    decoder_layers=4,
    num_attention_heads=6,
    max_position_embeddings=1024,
    hidden_dim=384,
):
    model = torch.load(model_path, map_location="cpu")
    # Encoder
    conv1 = {
        "w": model["model.encoder.conv1.weight"].numpy(),
        "b": model["model.encoder.conv1.bias"].numpy(),
    }
    conv2 = {
        "w": model["model.encoder.conv2.weight"].numpy(),
        "b": model["model.encoder.conv2.bias"].numpy(),
    }
    embed_positions = model["model.encoder.embed_positions.weight"].numpy()
    encoder_blocks = []
    for i in range(encoder_layers):
        q = {
            "w": model[f"model.encoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
            "b": model[f"model.encoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
        }
        k = {
            "w": model[f"model.encoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
            "b": np.zeros((hidden_dim,)),
        }
        v = {
            "w": model[f"model.encoder.layers.{i}.self_attn.v_proj.weight"].numpy(),
            "b": model[f"model.encoder.layers.{i}.self_attn.v_proj.bias"].numpy(),
        }
        c_attn = {
            "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
            "b": np.hstack((q["b"], k["b"], v["b"])),
        }
        c_proj = {
            "w": model[f"model.encoder.layers.{i}.self_attn.out_proj.weight"].numpy().T,
            "b": model[f"model.encoder.layers.{i}.self_attn.out_proj.bias"].numpy(),
        }
        attn = {"c_attn": c_attn, "c_proj": c_proj}
        ln_1 = {
            "g": model[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"].numpy(),
            "b": model[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"].numpy(),
        }

        mlp_c_fc = {
            "w": model[f"model.encoder.layers.{i}.fc1.weight"].numpy().T,
            "b": model[f"model.encoder.layers.{i}.fc1.bias"].numpy(),
        }

        mlp_c_proj = {
            "w": model[f"model.encoder.layers.{i}.fc2.weight"].numpy().T,
            "b": model[f"model.encoder.layers.{i}.fc2.bias"].numpy(),
        }
        mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

        ln_2 = {
            "g": model[f"model.encoder.layers.{i}.final_layer_norm.weight"].numpy(),
            "b": model[f"model.encoder.layers.{i}.final_layer_norm.bias"].numpy(),
        }
        block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
        encoder_blocks.append(block)

    encoder_layer_norm = {
        "g": model["model.encoder.layer_norm.weight"].numpy(),
        "b": model["model.encoder.layer_norm.bias"].numpy(),
    }
    encoder = {
        "conv1": conv1,
        "conv2": conv2,
        "embed_positions": embed_positions,
        "blocks": encoder_blocks,
        "ln_f": encoder_layer_norm,
    }

    # Decoder (similar structure)
    decoder_embed_tokens = model["model.decoder.embed_tokens.weight"].numpy()
    decoder_embed_positions = model["model.decoder.embed_positions.weight"].numpy()
    decoder_blocks = []
    for i in range(decoder_layers):
        q = {
            "w": model[f"model.decoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
        }
        k = {
            "w": model[f"model.decoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
            "b": np.zeros((hidden_dim,)),
        }
        v = {
            "w": model[f"model.decoder.layers.{i}.self_attn.v_proj.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.self_attn.v_proj.bias"].numpy(),
        }
        c_attn = {
            "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
            "b": np.hstack((q["b"], k["b"], v["b"])),
        }
        c_proj = {
            "w": model[f"model.decoder.layers.{i}.self_attn.out_proj.weight"].numpy().T,
            "b": model[f"model.decoder.layers.{i}.self_attn.out_proj.bias"].numpy(),
        }
        attn = {"c_attn": c_attn, "c_proj": c_proj}
        ln_1 = {
            "g": model[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"].numpy(),
        }

        q = {
            "w": model[f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"].numpy(),
        }
        k = {
            "w": model[f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"].numpy(),
            "b": np.zeros((hidden_dim,)),
        }
        v = {
            "w": model[f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"].numpy(),
        }
        c_attn = {
            "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
            "b": np.hstack((q["b"], k["b"], v["b"])),
        }
        c_proj = {
            "w": model[f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"]
            .numpy()
            .T,
            "b": model[f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"].numpy(),
        }
        encoder_attn = {"c_attn": c_attn, "c_proj": c_proj}
        ln_2 = {
            "g": model[
                f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"
            ].numpy(),
            "b": model[
                f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"
            ].numpy(),
        }

        mlp_c_fc = {
            "w": model[f"model.decoder.layers.{i}.fc1.weight"].numpy().T,
            "b": model[f"model.decoder.layers.{i}.fc1.bias"].numpy(),
        }

        mlp_c_proj = {
            "w": model[f"model.decoder.layers.{i}.fc2.weight"].numpy().T,
            "b": model[f"model.decoder.layers.{i}.fc2.bias"].numpy(),
        }

        mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

        ln_3 = {
            "g": model[f"model.decoder.layers.{i}.final_layer_norm.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.final_layer_norm.bias"].numpy(),
        }
        block = {
            "mlp": mlp,
            "attn": attn,
            "encoder_attn": encoder_attn,
            "ln_1": ln_1,
            "ln_2": ln_2,
            "ln_3": ln_3,
        }
        decoder_blocks.append(block)
    decoder_layer_norm = {
        "g": model["model.decoder.layer_norm.weight"].numpy(),
        "b": model["model.decoder.layer_norm.bias"].numpy(),
    }
    decoder = {
        "embed_tokens": decoder_embed_tokens,
        "embed_positions": decoder_embed_positions,
        "blocks": decoder_blocks,
        "ln_f": decoder_layer_norm,
    }

    params = {
        "encoder": encoder,
        "decoder": decoder,
        "proj_out": {
            "w": model["model.decoder.embed_tokens.weight"].numpy().T,
        },
    }
    hparams = {
        "n_head": num_attention_heads,
        "n_ctx": max_position_embeddings,
    }
    return hparams, params
