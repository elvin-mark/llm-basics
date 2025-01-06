import numpy as np
from utils.functions import get_positional_encoding
from safetensors.torch import load_file


def load_hparams_and_params(
    model_path,
    encoder_layers=4,
    decoder_layers=4,
    num_attention_heads=4,
    max_position_embeddings=1024,
):
    model = load_file(model_path)
    embed_tokens = model["model.shared.weight"].numpy()
    embed_positions = get_positional_encoding(512, 256)
    encoder_blocks = []
    for i in range(encoder_layers):
        q = {
            "w": model[f"model.encoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
            "b": model[f"model.encoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
        }
        k = {
            "w": model[f"model.encoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
            "b": model[f"model.encoder.layers.{i}.self_attn.k_proj.bias"].numpy(),
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

    encoder = {
        "embed_tokens": embed_tokens,
        "embed_positions": embed_positions,
        "blocks": encoder_blocks,
    }

    # Decoder (similar structure)
    decoder_embed_tokens = model["model.shared.weight"].numpy()
    decoder_embed_positions = get_positional_encoding(512, 256)
    decoder_blocks = []
    for i in range(decoder_layers):
        q = {
            "w": model[f"model.decoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
        }
        k = {
            "w": model[f"model.decoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.self_attn.k_proj.bias"].numpy(),
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
            "b": model[f"model.decoder.layers.{i}.encoder_attn.k_proj.bias"].numpy(),
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
    decoder = {
        "embed_tokens": decoder_embed_tokens,
        "embed_positions": decoder_embed_positions,
        "blocks": decoder_blocks,
    }

    params = {
        "shared": {"w": model["model.shared.weight"].numpy().T},
        "encoder": encoder,
        "decoder": decoder,
        "lm_head": {
            "w": model["model.shared.weight"].numpy().T,
            "b": model["final_logits_bias"].numpy(),
        },
    }
    hparams = {
        "n_head": num_attention_heads,
        "n_ctx": max_position_embeddings,
    }
    return hparams, params
