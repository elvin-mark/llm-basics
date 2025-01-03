import struct
import numpy as np
from sentencepiece import SentencePieceProcessor
from dotenv import load_dotenv
import os
from utils.nn import softmax, silu, rms_norm

load_dotenv()

with open(os.getenv("LLAMA2_MODEL_PATH"), "rb") as f:
    data_ = f.read()
i = 0
j = 7  # config data
config = struct.unpack("I" * j, data_[: j * 4])
dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = config
i += j * 4
head_size = int(dim / n_heads)
j = vocab_size * dim
token_embedding_table = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((vocab_size, dim))
    .astype(np.float32)
)
i += j * 4
j = n_layers * dim
rms_att_weight = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, dim))
    .astype(np.float32)
)
i += j * 4
j = n_layers * dim * n_heads * head_size
wq = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, dim, n_heads * head_size))
    .astype(np.float32)
)
i += j * 4
j = n_layers * dim * n_kv_heads * head_size
wk = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, dim, n_kv_heads * head_size))
    .astype(np.float32)
)
i += j * 4
j = n_layers * dim * n_kv_heads * head_size
wv = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, dim, n_kv_heads * head_size))
    .astype(np.float32)
)
i += j * 4
j = n_layers * n_heads * head_size * dim
wo = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, n_heads * head_size, dim))
    .astype(np.float32)
)
i += j * 4
j = n_layers * dim
rms_ffn_weight = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, dim))
    .astype(np.float32)
)
i += j * 4
j = n_layers * hidden_dim * dim
w1 = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, hidden_dim, dim))
    .astype(np.float32)
)
i += j * 4
j = n_layers * dim * hidden_dim
w2 = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, dim, hidden_dim))
    .astype(np.float32)
)
i += j * 4
j = n_layers * hidden_dim * dim
w3 = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((n_layers, hidden_dim, dim))
    .astype(np.float32)
)
i += j * 4
j = dim
rms_final_weight = (
    np.array(struct.unpack("f" * j, data_[i : i + j * 4]))
    .reshape((dim,))
    .astype(np.float32)
)
i += j * 4
wcls = token_embedding_table


prompt = "Once upon a time,"
temperature = 0.8
n_tokens_to_generate = 100

tokenizer = SentencePieceProcessor(model_file=os.getenv("LLAMA2_TOKENIZER_PATH"))

prev_pos = 0
tokens = [tokenizer.bos_id()] + tokenizer.encode(prompt)
cache_k, cache_v = ([None] * n_layers), ([None] * n_layers)
# max_seq_len = 512
freqs_cis = np.exp(
    1j
    * np.outer(
        np.arange(2 * 512),
        (np.logspace(0, 1.0, base=1e-4, num=(dim // n_heads) // 2, endpoint=False)),
    )
).astype(np.complex64)
for cur_pos in range(len(tokens), len(tokens) + n_tokens_to_generate):
    # Embed tokens
    h = token_embedding_table[tokens[prev_pos:cur_pos], :].astype(np.float32)
    # Rotary embedding
    f = freqs_cis[prev_pos:cur_pos].reshape(-1, 1, (dim // n_heads) // 2)
    for layer in range(n_layers):
        # LayerNorm
        xn = rms_norm(h) * rms_att_weight[layer]
        # QKV projections
        xq = (xn @ wq[layer].T).reshape((-1, n_heads, (dim // n_heads)))
        xk = (xn @ wk[layer].T).reshape((-1, n_heads, (dim // n_heads)))
        xv = (xn @ wv[layer].T).reshape((-1, n_heads, (dim // n_heads)))
        # Rotary embedding
        xq = (xq.view(dtype=np.complex64) * f).view(dtype=np.float32)
        xk = (xk.view(dtype=np.complex64) * f).view(dtype=np.float32)
        # Cache
        if prev_pos == 0:
            cache_k[layer], cache_v[layer] = xk, xv
        else:
            xk, xv = cache_k[layer], cache_v[layer] = np.concatenate(
                (cache_k[layer], xk), axis=0
            ), np.concatenate((cache_v[layer], xv), axis=0)
        # Attention
        scores = np.matmul(xk, xq, axes=[(0, 2), (2, 0), (2, 1)]) / np.sqrt(
            (dim // n_heads)
        )
        # Mask
        if (cur_pos - prev_pos) > 1:
            scores += -1e10 * (1 - np.tri(cur_pos - prev_pos))
        # Attention
        h += (
            np.matmul(softmax(scores), xv, axes=[(1, 2), (0, 2), (0, 2)]).reshape(
                -1, dim
            )
        ) @ wo[layer].T
        # Feed Forward
        tmp = rms_norm(h) * rms_ffn_weight[layer]
        x1 = tmp @ w1[layer].T
        h += (silu(x1) * (tmp @ w3[layer].T)) @ w2[layer].T
    # Unembed tokens
    tokens.append(int(np.argmax(((rms_norm(h) * rms_final_weight)[-1, :] @ wcls.T))))
    prev_pos = cur_pos
print(tokenizer.decode(tokens))
