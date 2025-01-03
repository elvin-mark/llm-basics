import torch
import numpy as np
from dotenv import load_dotenv
import os
from tokenizers import Tokenizer
from utils.audio_proc import pad, extract_fbank_features, mel_filter_bank

load_dotenv()


def load_whisper_parameters(
    model_path,
    encoder_layers=4,
    decoder_layers=4,
    num_attention_heads=6,
    max_position_embeddings=1024,
    hidden_dim=384,
):
    model = torch.load(model_path, map_location="cpu")
    # Encoder
    conv1 = {
        "w": model["model.encoder.conv1.weight"].numpy(),
        "b": model["model.encoder.conv1.bias"].numpy(),
    }
    conv2 = {
        "w": model["model.encoder.conv2.weight"].numpy(),
        "b": model["model.encoder.conv2.bias"].numpy(),
    }
    embed_positions = model["model.encoder.embed_positions.weight"].numpy()
    encoder_blocks = []
    for i in range(encoder_layers):
        q = {
            "w": model[f"model.encoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
            "b": model[f"model.encoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
        }
        k = {
            "w": model[f"model.encoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
            "b": np.zeros((hidden_dim,)),
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

    encoder_layer_norm = {
        "g": model["model.encoder.layer_norm.weight"].numpy(),
        "b": model["model.encoder.layer_norm.bias"].numpy(),
    }
    encoder = {
        "conv1": conv1,
        "conv2": conv2,
        "embed_positions": embed_positions,
        "blocks": encoder_blocks,
        "ln_f": encoder_layer_norm,
    }

    # Decoder (similar structure)
    decoder_embed_tokens = model["model.decoder.embed_tokens.weight"].numpy()
    decoder_embed_positions = model["model.decoder.embed_positions.weight"].numpy()
    decoder_blocks = []
    for i in range(decoder_layers):
        q = {
            "w": model[f"model.decoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
            "b": model[f"model.decoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
        }
        k = {
            "w": model[f"model.decoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
            "b": np.zeros((hidden_dim,)),
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
            "b": np.zeros((hidden_dim,)),
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
    decoder_layer_norm = {
        "g": model["model.decoder.layer_norm.weight"].numpy(),
        "b": model["model.decoder.layer_norm.bias"].numpy(),
    }
    decoder = {
        "embed_tokens": decoder_embed_tokens,
        "embed_positions": decoder_embed_positions,
        "blocks": decoder_blocks,
        "ln_f": decoder_layer_norm,
    }

    params = {
        "encoder": encoder,
        "decoder": decoder,
        "proj_out": {
            "w": model["model.decoder.embed_tokens.weight"].numpy().T,
        },
    }
    hparams = {
        "n_head": num_attention_heads,
        "n_ctx": max_position_embeddings,
    }
    return hparams, params


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)


def attention(q, k, v, mask=None):
    if mask is None:
        return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head, kv_states=None, mask_enabled=False):
    x = linear(x, **c_attn)

    if kv_states is not None:
        qkv = []
        dim = c_attn["w"].shape[0]
        kv = linear(kv_states, **c_attn)
        qkv = [x[:, :dim], kv[:, dim : 2 * dim], kv[:, 2 * dim :]]
    else:
        qkv = np.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
    causal_mask = None
    if mask_enabled:
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, mask=causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head, mask_enabled=False):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head, mask_enabled=mask_enabled)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def decoder_transformer_block(
    x, mlp, attn, encoder_attn, ln_1, ln_2, ln_3, n_head, kv_states=None
):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head, mask_enabled=True)
    if kv_states is not None:
        x = x + mha(
            layer_norm(x, **ln_2), **encoder_attn, kv_states=kv_states, n_head=n_head
        )
    x = x + ffn(layer_norm(x, **ln_3), **mlp)
    return x


def compute_spectrogram(audio, sample_rate, n_fft=400, hop_length=160):
    spectrogram = np.abs(np.fft.rfft(audio, n=n_fft, axis=-1, norm="ortho")) ** 2
    return spectrogram


def convolution(input_tensor, weights, bias, stride=1, padding=0):
    # Get dimensions
    in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weights.shape

    # Apply padding to the input tensor
    if padding > 0:
        input_tensor = np.pad(
            input_tensor,
            ((0, 0), (padding, padding)),
            mode="constant",
            constant_values=0,
        )

    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1

    # Extract sliding windows (using strides)
    strided_indices = np.lib.stride_tricks.sliding_window_view(
        input_tensor, kernel_size, axis=1
    )
    # Shape of strided_indices: (in_channels, output_length, kernel_size)
    strided_indices = strided_indices[:, ::stride, :]  # Apply stride

    # Perform the convolution using broadcasting and summation
    output_tensor = np.tensordot(weights, strided_indices, axes=([1, 2], [0, 2]))
    # Shape of output_tensor: (out_channels, output_length)

    # Add bias to each output channel
    output_tensor += bias[:, None]  # Bias broadcasted to match output shape

    return output_tensor


def whisper_encoder(audio_features, params, hparams):
    # Convolutional layers
    x = gelu(
        convolution(
            audio_features,
            params["encoder"]["conv1"]["w"],
            params["encoder"]["conv1"]["b"],
            padding=1,
        )
    )
    x = gelu(
        convolution(
            x,
            params["encoder"]["conv2"]["w"],
            params["encoder"]["conv2"]["b"],
            stride=2,
            padding=1,
        )
    )

    # Add positional embeddings
    x = x.T + params["encoder"]["embed_positions"]
    # Transformer layers
    for layer in params["encoder"]["blocks"]:
        x = transformer_block(x, **layer, n_head=hparams["n_head"])

    # Final layer norm
    x = layer_norm(x, **params["encoder"]["ln_f"])
    return x


def whisper_decoder(encoder_output, input_ids, params, hparams):
    # Embed tokens
    token_embeddings = params["decoder"]["embed_tokens"][input_ids]
    positions = np.arange(token_embeddings.shape[0])
    token_embeddings = (
        token_embeddings + params["decoder"]["embed_positions"][positions]
    )

    # Transformer layers
    x = token_embeddings
    for layer in params["decoder"]["blocks"]:
        x = decoder_transformer_block(
            x, **layer, kv_states=encoder_output, n_head=hparams["n_head"]
        )

    # Final layer norm
    x = layer_norm(x, **params["decoder"]["ln_f"])
    return x


def whisper_generate(audio_features, params, hparams, n_tokens):
    # Encode audio
    encoder_output = whisper_encoder(audio_features, params, hparams)

    # Initialize decoder inputs
    input_ids = [50257]  # Start of sequence token
    for _ in range(n_tokens):
        logits = whisper_decoder(encoder_output, input_ids, params, hparams)
        next_token = np.argmax(logits[-1] @ params["proj_out"]["w"])
        if next_token == 50256:
            break
        input_ids.append(next_token)
    return input_ids


hparams, params = load_whisper_parameters(os.getenv("WHISPER_MODEL_PATH"))
tokenizer = Tokenizer.from_file(os.getenv("WHISPER_TOKENIZER_PATH"))

# Dummy audio sample
audio_sample = np.random.randn(89600)

feature_size = 80
sampling_rate = 16000
n_fft = 400
hop_length = 160
chunk_length = 30
n_samples = chunk_length * sampling_rate

mel_filters = mel_filter_bank(
    num_frequency_bins=1 + n_fft // 2,
    num_mel_filters=feature_size,
    min_frequency=0.0,
    max_frequency=8000.0,
    sampling_rate=sampling_rate,
    norm="slaney",
    mel_scale="slaney",
)

audio_features = extract_fbank_features(
    pad([audio_sample], max_length=n_samples),
    n_fft=n_fft,
    hop_length=hop_length,
    mel_filters=mel_filters,
)
ids = whisper_generate(audio_features[0], params, hparams, 2)
print(tokenizer.decode(ids))
