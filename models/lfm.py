import os
import torch
import numpy as np
from dotenv import load_dotenv
from tokenizers import Tokenizer
from safetensors.torch import load_file
from utils.nn import softmax, silu

# Load environment variables (LFM_MODEL_PATH, LFM_TOKENIZER_PATH)
load_dotenv()

# Model and Tokenizer paths
MODEL_PATH = os.getenv("LFM_MODEL_PATH")
TOKENIZER_PATH = os.getenv("LFM_TOKENIZER_PATH")

print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

print(f"Loading weights from {MODEL_PATH}...")
weights = load_file(MODEL_PATH, device="cpu")


# Helper to convert PyTorch tensor to float32 NumPy array
def to_np(tensor):
    return tensor.to(torch.float32).numpy()


# Load global embeddings and norm
wte = to_np(weights["model.embed_tokens.weight"])
embedding_norm_weight = to_np(weights["model.embedding_norm.weight"])

# Load all 14 layers
print("Loading model layers...")
layers = []
for i in range(14):
    layer_params = {}
    layer_params["operator_norm"] = to_np(
        weights[f"model.layers.{i}.operator_norm.weight"]
    )
    layer_params["ffn_norm"] = to_np(weights[f"model.layers.{i}.ffn_norm.weight"])
    layer_params["w1"] = to_np(weights[f"model.layers.{i}.feed_forward.w1.weight"])
    layer_params["w2"] = to_np(weights[f"model.layers.{i}.feed_forward.w2.weight"])
    layer_params["w3"] = to_np(weights[f"model.layers.{i}.feed_forward.w3.weight"])

    # Check layer type: Conv layer vs Self-Attention layer
    if f"model.layers.{i}.self_attn.q_proj.weight" in weights:
        layer_params["type"] = "attention"
        layer_params["q_proj"] = to_np(
            weights[f"model.layers.{i}.self_attn.q_proj.weight"]
        )
        layer_params["k_proj"] = to_np(
            weights[f"model.layers.{i}.self_attn.k_proj.weight"]
        )
        layer_params["v_proj"] = to_np(
            weights[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        layer_params["out_proj"] = to_np(
            weights[f"model.layers.{i}.self_attn.out_proj.weight"]
        )
        layer_params["q_layernorm"] = to_np(
            weights[f"model.layers.{i}.self_attn.q_layernorm.weight"]
        )
        layer_params["k_layernorm"] = to_np(
            weights[f"model.layers.{i}.self_attn.k_layernorm.weight"]
        )
    else:
        layer_params["type"] = "conv"
        layer_params["conv_w"] = to_np(weights[f"model.layers.{i}.conv.conv.weight"])
        layer_params["in_proj"] = to_np(
            weights[f"model.layers.{i}.conv.in_proj.weight"]
        )
        layer_params["out_proj"] = to_np(
            weights[f"model.layers.{i}.conv.out_proj.weight"]
        )

    layers.append(layer_params)

# Architecture Hyperparameters
head_dim = 64
rope_theta = 1000000.0

# Precompute RoPE inverse frequencies
inv_freq = 1.0 / (
    rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
)


def get_rope_cos_sin(position_ids, inv_freq):
    # Compute position-wise cosine/sine values
    freqs = np.outer(position_ids, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos, sin shape: [1, seq_len, head_dim]
    # q shape: [16, seq_len, 64], k shape: [8, seq_len, 64]
    cos_exp = cos[None, :, :]
    sin_exp = sin[None, :, :]
    q_embed = (q * cos_exp) + (rotate_half(q) * sin_exp)
    k_embed = (k * cos_exp) + (rotate_half(k) * sin_exp)
    return q_embed, k_embed


def rms_norm_scaled(x, w, eps=1e-6):
    # Normalized along the last dimension and scaled by RMS weight
    norm_x = x / np.sqrt(np.square(x).mean(axis=-1, keepdims=True) + eps)
    return norm_x * w


def ffn_silu(x, w1, w3, w2):
    return (silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def lfm(inputs, layers, conv_caches, attn_caches, prev_pos, cur_pos):
    tokens_to_process = inputs[prev_pos:cur_pos]
    h = wte[tokens_to_process, :].astype(np.float32)
    seq_len = len(tokens_to_process)

    position_ids = np.arange(prev_pos, cur_pos)
    cos, sin = get_rope_cos_sin(position_ids, inv_freq)

    for layer_idx, layer in enumerate(layers):
        residual = h
        xn = rms_norm_scaled(h, layer["operator_norm"])

        if layer["type"] == "conv":
            # Project input channels
            BCx = (xn @ layer["in_proj"].T).T  # shape [3072, seq_len]
            B = BCx[0:1024, :]
            C = BCx[1024:2048, :]
            x_val = BCx[2048:3072, :]
            Bx = B * x_val  # shape [1024, seq_len]

            if prev_pos == 0:
                # Prompt mode causal convolution
                Bx_padded = np.pad(Bx, ((0, 0), (2, 0)), mode="constant")
                conv_out = (
                    layer["conv_w"][:, 0, 0][:, None] * Bx_padded[:, 0:seq_len]
                    + layer["conv_w"][:, 0, 1][:, None] * Bx_padded[:, 1 : seq_len + 1]
                    + layer["conv_w"][:, 0, 2][:, None] * Bx_padded[:, 2 : seq_len + 2]
                )
                # Store state cache
                if seq_len >= 3:
                    conv_caches[layer_idx] = Bx[:, -3:]
                else:
                    conv_caches[layer_idx] = np.pad(
                        Bx, ((0, 0), (3 - seq_len, 0)), mode="constant"
                    )
            else:
                # Incremental mode update and convolution
                conv_cache = conv_caches[layer_idx]
                conv_cache = np.concatenate([conv_cache[:, 1:], Bx], axis=-1)
                conv_caches[layer_idx] = conv_cache

                conv_out = (
                    layer["conv_w"][:, 0, 0] * conv_cache[:, 0]
                    + layer["conv_w"][:, 0, 1] * conv_cache[:, 1]
                    + layer["conv_w"][:, 0, 2] * conv_cache[:, 2]
                )
                conv_out = conv_out[:, None]

            y = C * conv_out
            attn_out = y.T @ layer["out_proj"].T

        else:
            # Project QKV
            q = xn @ layer["q_proj"].T
            k = xn @ layer["k_proj"].T
            v = xn @ layer["v_proj"].T

            # Reshape for multi-head & RMSNorm on queries/keys
            q = rms_norm_scaled(q.reshape(seq_len, 16, 64), layer["q_layernorm"])
            k = rms_norm_scaled(k.reshape(seq_len, 8, 64), layer["k_layernorm"])
            v = v.reshape(seq_len, 8, 64)

            # Transpose to [num_heads, seq_len, head_dim]
            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            # Apply RoPE Positional Embeddings
            q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

            # Update KV Cache
            if prev_pos == 0:
                attn_caches[layer_idx] = (k_embed, v)
            else:
                cache_k, cache_v = attn_caches[layer_idx]
                k_embed = np.concatenate([cache_k, k_embed], axis=1)
                v = np.concatenate([cache_v, v], axis=1)
                attn_caches[layer_idx] = (k_embed, v)

            # Repeat keys/values for Grouped Query Attention (GQA, groups = 2)
            k_repeated = np.repeat(k_embed, 2, axis=0)
            v_repeated = np.repeat(v, 2, axis=0)

            # Scaled Dot-Product Attention (scaling = 1 / sqrt(64) = 0.125)
            scores = np.matmul(q_embed, k_repeated.transpose(0, 2, 1)) * 0.125

            # Causal masking during prompt phase
            if seq_len > 1:
                total_seq_len = k_repeated.shape[1]
                causal_mask = (1.0 - np.tri(seq_len, total_seq_len)) * -1e10
                scores += causal_mask

            attn_probs = softmax(scores)
            output_states = np.matmul(attn_probs, v_repeated)
            output_states = output_states.transpose(1, 0, 2).reshape(seq_len, 1024)

            attn_out = output_states @ layer["out_proj"].T

        h = attn_out + residual
        h = h + ffn_silu(
            rms_norm_scaled(h, layer["ffn_norm"]), layer["w1"], layer["w3"], layer["w2"]
        )

    return rms_norm_scaled(h, embedding_norm_weight)


# Running prompt generation demo
prompt = "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."
input_ids = tokenizer.encode(prompt).ids
print(f"\nPrompt: {prompt}")

tokens = list(input_ids)
conv_caches = [None] * 14
attn_caches = [None] * 14

prev_pos = 0
cur_pos = len(tokens)
n_tokens = 50

print("Generating: ", end="", flush=True)
for _ in range(n_tokens):
    h = lfm(tokens, layers, conv_caches, attn_caches, prev_pos, cur_pos)
    logits = h[-1, :] @ wte.T
    next_id = int(np.argmax(logits))
    tokens.append(next_id)

    if next_id == 2:  # <|endoftext|>
        break

    print(tokenizer.decode([next_id]), end="", flush=True)
    prev_pos = cur_pos
    cur_pos = len(tokens)

print("\n\nFinal generated text:")
print(tokenizer.decode(tokens))
