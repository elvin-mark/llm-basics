import numpy as np
import random
import torch
import os
from dotenv import load_dotenv
from utils.tokenizers.bpe_tokenizer import BPETokenizer
from utils.nn import layer_norm, ffn, mha

load_dotenv()


def load_encoder_hparams_and_params(model_path):
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


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head, mask_enabled=True)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, lm_head, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    if lm_head:
        return layer_norm(x, **ln_f) @ lm_head.T
    return layer_norm(x, **ln_f) @ wte.T


def generate(inputs, params, n_head, n_tokens, topk=5):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = random.choice(np.argsort(logits[-1])[-topk:])
        inputs.append(int(next_id))
    return inputs


hparams, params = load_encoder_hparams_and_params(
    model_path=os.getenv("GPT2_MODEL_PATH")
)
tokenizer = BPETokenizer.from_file(os.getenv("GPT2_TOKENIZER_PATH"))
topk = 5
n_tokens = 40
prompt = "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."

input_ids = tokenizer.encode(prompt).ids
assert len(input_ids) + n_tokens < hparams["n_ctx"]
output_ids = generate(input_ids, params, hparams["n_head"], n_tokens, topk=topk)
output_text = tokenizer.decode(output_ids)
print(output_text)
