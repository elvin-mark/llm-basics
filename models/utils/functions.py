import numpy as np


def sample_probs(probs, temperature=1.0, top_p=0.3):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs ** (1 / temperature)
    return np.random.choice(a=len(probs), p=probs / np.sum(probs))


def mean_pooling_and_normalization(x):
    o = np.mean(x, axis=0)
    return o / np.linalg.norm(o)


def gauss_norm(x: np.ndarray) -> np.ndarray:
    x = (x - x.mean()) / x.std()
    return x


def get_positional_encoding(seq_len, d_model):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(seq_len)
        ]
    )
    out = np.zeros((seq_len, d_model))
    sentinel = d_model // 2 if d_model % 2 == 0 else (d_model // 2) + 1
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])
    return out
