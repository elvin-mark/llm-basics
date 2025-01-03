import numpy as np
import random
import os
from dotenv import load_dotenv
from utils.tokenizers.bpe_tokenizer import BPETokenizer
from utils.nn import layer_norm, ffn, mha
from utils.loaders.gpt2 import load_hparams_and_params

load_dotenv()


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


hparams, params = load_hparams_and_params(model_path=os.getenv("GPT2_MODEL_PATH"))
tokenizer = BPETokenizer.from_file(os.getenv("GPT2_TOKENIZER_PATH"))
topk = 5
n_tokens = 40
prompt = "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."

input_ids = tokenizer.encode(prompt).ids
assert len(input_ids) + n_tokens < hparams["n_ctx"]
output_ids = generate(input_ids, params, hparams["n_head"], n_tokens, topk=topk)
output_text = tokenizer.decode(output_ids)
print(output_text)
