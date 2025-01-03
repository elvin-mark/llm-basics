import numpy as np
from dotenv import load_dotenv
import os
from tokenizers import Tokenizer
from utils.features.audio_proc import pad, extract_fbank_features, mel_filter_bank
from utils.nn import layer_norm, ffn, mha, gelu, convolution_1d
from utils.loaders.whisper import load_hparams_and_params

load_dotenv()


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


def whisper_encoder(audio_features, params, hparams):
    # Convolutional layers
    x = gelu(
        convolution_1d(
            audio_features,
            params["encoder"]["conv1"]["w"],
            params["encoder"]["conv1"]["b"],
            padding=1,
        )
    )
    x = gelu(
        convolution_1d(
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


hparams, params = load_hparams_and_params(os.getenv("WHISPER_MODEL_PATH"))
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
ids = whisper_generate(audio_features[0], params, hparams, 20)
print(tokenizer.decode(ids))
