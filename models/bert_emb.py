import numpy as np
import os
from utils.nn import layer_norm, ffn, mha
from dotenv import load_dotenv
from utils.loaders.bert_emb import load_hparams_and_params
from utils.functions import mean_pooling_and_normalization

load_dotenv()
if os.getenv("TOKENIZER_TYPE", "default") == "default":
    from tokenizers import Tokenizer

    Tokenizer = Tokenizer
else:
    from utils.tokenizers.word_piece_tokenizer import WordPieceTokenizer

    Tokenizer = WordPieceTokenizer


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = layer_norm(x + mha(x, **attn, n_head=n_head), **ln_1)
    x = layer_norm(x + ffn(x, **mlp), **ln_2)
    return x


def bert(inputs, segment_ids, wte, wpe, wtte, ln_0, blocks, pooler, n_head):
    x = wte[inputs] + wpe[range(len(inputs))] + wtte[segment_ids]
    x = layer_norm(x, **ln_0)
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return x
    # return np.tanh(linear(x, **pooler))


hparams, params = load_hparams_and_params(model_path=os.getenv("BERT_EMB_MODEL_PATH"))

tokenizer = Tokenizer.from_file(os.getenv("BERT_EMB_TOKENIZER_PATH"))
tokenizer.no_padding()

sentences = [
    "The sun is shining brightly in the sky.",
    "It’s a clear day with plenty of sunshine.",
    "I forgot to bring my umbrella, and now it’s raining heavily.",
    "The cat is sleeping peacefully on the couch.",
]

embeddings = []
for sentence in sentences:
    sentence_ids = tokenizer.encode(sentence).ids

    logits = bert(
        sentence_ids, [0] * len(sentence_ids), **params, n_head=hparams["n_head"]
    )
    embeddings.append(mean_pooling_and_normalization(logits))

embeddings = np.vstack(embeddings)
print(embeddings @ embeddings.T)
