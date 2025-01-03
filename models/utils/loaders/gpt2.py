import torch


def load_hparams_and_params(model_path):
    n_layers = 12
    prefix = ""
    model = torch.load(model_path, map_location="cpu")
    # vocab embedding. shape [vocab_size, emb_dim] ex. [50257, 768]
    wte = model[f"{prefix}wte.weight"].numpy()
    # context embedding. shape [ctx_len, emb_dim] ex. [1024, 768]
    wpe = model[f"{prefix}wpe.weight"].numpy()

    blocks = []
    for i in range(n_layers):
        mlp_c_fc = {
            "w": model[f"{prefix}h.{i}.mlp.c_fc.weight"].numpy(),
            "b": model[f"{prefix}h.{i}.mlp.c_fc.bias"].numpy(),
        }
        mlp_c_proj = {
            "w": model[f"{prefix}h.{i}.mlp.c_proj.weight"].numpy(),
            "b": model[f"{prefix}h.{i}.mlp.c_proj.bias"].numpy(),
        }
        mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}
        c_attn = {
            "w": model[f"{prefix}h.{i}.attn.c_attn.weight"].numpy(),
            "b": model[f"{prefix}h.{i}.attn.c_attn.bias"].numpy(),
        }
        c_proj = {
            "w": model[f"{prefix}h.{i}.attn.c_proj.weight"].numpy(),
            "b": model[f"{prefix}h.{i}.attn.c_proj.bias"].numpy(),
        }
        attn = {"c_attn": c_attn, "c_proj": c_proj}
        ln_1 = {
            "g": model[f"{prefix}h.{i}.ln_1.weight"].numpy(),
            "b": model[f"{prefix}h.{i}.ln_1.bias"].numpy(),
        }
        ln_2 = {
            "g": model[f"{prefix}h.{i}.ln_2.weight"].numpy(),
            "b": model[f"{prefix}h.{i}.ln_2.bias"].numpy(),
        }
        block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
        blocks.append(block)
    ln_f = {
        "g": model[f"{prefix}ln_f.weight"].numpy(),
        "b": model[f"{prefix}ln_f.bias"].numpy(),
    }
    if f"{prefix}lm_head.weight" in model:
        lm_head = model[f"{prefix}lm_head.weight"].numpy()
    else:
        lm_head = None
    params = {
        "wte": wte,
        "wpe": wpe,
        "blocks": blocks,
        "ln_f": ln_f,
        "lm_head": lm_head,
    }
    hparams = {}
    hparams["n_head"] = 12
    hparams["n_ctx"] = 1024
    return hparams, params
