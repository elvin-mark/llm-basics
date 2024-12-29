import numpy as np
from torch import load
from tokenizers import Tokenizer
import os
from dotenv import load_dotenv

load_dotenv()

N_LAYER = 12
N_EMBD = 768
MODEL_FILE = os.getenv("RWKV4_MODEL_PATH")

tokenizer = Tokenizer.from_file(os.getenv("RWKV4_TOKENIZER_PATH"))

layer_norm = lambda x, w, b: (x - np.mean(x)) / np.std(x) * w + b
exp = np.exp
sigmoid = lambda x: 1 / (1 + exp(-x))


def time_mixing(
    x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout
):
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    v = Wv @ (x * mix_v + last_x * (1 - mix_v))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))

    wkv = (last_num + exp(bonus + k) * v) / (last_den + exp(bonus + k))
    rwkv = sigmoid(r) * wkv

    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return Wout @ rwkv, (x, num, den)


def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))
    vk = Wv @ np.maximum(k, 0) ** 2
    return sigmoid(r) * vk, x


def RWKV(model, token, state):
    params = lambda prefix: [
        model[key] for key in model.keys() if key.startswith(prefix)
    ]

    x = params("emb")[0][token]
    x = layer_norm(x, *params("blocks.0.ln0"))

    for i in range(N_LAYER):
        x_ = layer_norm(x, *params(f"blocks.{i}.ln1"))
        dx, state[i][:3] = time_mixing(x_, *state[i][:3], *params(f"blocks.{i}.att"))
        x = x + dx

        x_ = layer_norm(x, *params(f"blocks.{i}.ln2"))
        dx, state[i][3] = channel_mixing(x_, state[i][3], *params(f"blocks.{i}.ffn"))
        x = x + dx

    x = layer_norm(x, *params("ln_out"))
    x = params("head")[0] @ x

    e_x = exp(x - np.max(x))
    probs = e_x / e_x.sum()

    return probs, state


def sample_probs(probs, temperature=1.0, top_p=0.3):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs ** (1 / temperature)
    return np.random.choice(a=len(probs), p=probs / np.sum(probs))


weights = load(MODEL_FILE, map_location="cpu")
for k in weights.keys():
    if ".time" in k:
        weights[k] = weights[k].squeeze()
    weights[k] = weights[k].float().numpy()


interface = ":"
user = "Question"
bot = "Answer"

init_prompt = f"""{user}{interface} hi

{bot}{interface} Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.
"""

prompt = "What is the capital of Peru?"
context = init_prompt + f"""\nQuestion: {prompt}\n\nAnswer: """

# Pile
state = np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)
for token in tokenizer.encode(context).ids:
    probs, state = RWKV(weights, token, state)

for i in range(20):
    token = sample_probs(probs)
    print(tokenizer.decode([token]), end="", flush=True)
    probs, state = RWKV(weights, token, state)
