import numpy as np
from torch import load
from tokenizers import Tokenizer
import os
from dotenv import load_dotenv
from utils.nn import sigmoid, layer_norm, softmax, relu
from utils.functions import sample_probs

load_dotenv()

N_LAYER = 12
N_EMBD = 768
MODEL_FILE = os.getenv("RWKV4_MODEL_PATH")

tokenizer = Tokenizer.from_file(os.getenv("RWKV4_TOKENIZER_PATH"))

exp = np.exp


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
    vk = Wv @ relu(k) ** 2
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

    probs = softmax(x)

    return probs, state


weights = load(MODEL_FILE, map_location="cpu")
for k in weights.keys():
    if ".time" in k:
        weights[k] = weights[k].squeeze()
    weights[k] = weights[k].float().numpy()


context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

# Pile
state = np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)
for token in tokenizer.encode(context).ids:
    probs, state = RWKV(weights, token, state)

np.random.seed(0)
for i in range(100):
    token = sample_probs(probs)
    print(tokenizer.decode([token]), end="", flush=True)
    probs, state = RWKV(weights, token, state)
