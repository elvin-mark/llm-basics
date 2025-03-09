from tqdm import tqdm
import os
from dotenv import load_dotenv
from utils.tokenizers.bpe_tokenizer import BPETokenizer
from utils.nn import layer_norm, ffn, mha_kv_cache, softmax
from utils.loaders.gpt2 import load_hparams_and_params
from utils.functions import sample_probs

load_dotenv()


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head, kv_cache=None):
    dx, kv_cache = mha_kv_cache(
        layer_norm(x, **ln_1),
        **attn,
        n_head=n_head,
        mask_enabled=True,
        kv_cache=kv_cache
    )
    x = x + dx
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x, kv_cache


def gpt2(inputs, wte, wpe, blocks, ln_f, lm_head, n_head, kv_cache=None):
    if kv_cache[0] is None:
        x = wte[inputs] + wpe[range(len(inputs))]
    else:
        x = wte[inputs[-1:]] + wpe[[len(inputs) - 1]]
    for i, block in enumerate(blocks):
        x, kv_cache[i] = transformer_block(
            x, **block, n_head=n_head, kv_cache=kv_cache[i]
        )
    if lm_head:
        return layer_norm(x, **ln_f) @ lm_head.T
    return layer_norm(x, **ln_f) @ wte.T


def generate(inputs, params, n_head, n_tokens, temperature=1.0, top_p=0.3):
    kv_cache = [None for _ in range(len(params["blocks"]))]

    for _ in tqdm(range(n_tokens), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head, kv_cache=kv_cache)
        logits = softmax(logits)
        next_id = sample_probs(logits[-1], top_p=top_p, temperature=temperature)
        inputs.append(int(next_id))
    return inputs


hparams, params = load_hparams_and_params(model_path=os.getenv("GPT2_MODEL_PATH"))
tokenizer = BPETokenizer.from_file(os.getenv("GPT2_TOKENIZER_PATH"))
temperature = 0.9
top_p = 0.7
n_tokens = 100
prompt = "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."

input_ids = tokenizer.encode(prompt).ids
assert len(input_ids) + n_tokens < hparams["n_ctx"]
output_ids = generate(
    input_ids, params, hparams["n_head"], n_tokens, temperature=temperature, top_p=top_p
)
print(output_ids)
output_text = tokenizer.decode(output_ids)
print(output_text)
