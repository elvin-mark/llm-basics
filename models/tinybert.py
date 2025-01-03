import numpy as np
import os
from dotenv import load_dotenv
from utils.nn import layer_norm, ffn, mha, linear, relu
from utils.loaders.tinybert import load_hparams_and_params

load_dotenv()
if os.getenv("TOKENIZER_TYPE", "deafult") == "default":
    from tokenizers import Tokenizer

    Tokenizer = Tokenizer
else:
    from utils.tokenizers.word_piece_tokenizer import WordPieceTokenizer

    Tokenizer = WordPieceTokenizer


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = layer_norm(x + mha(x, **attn, n_head=n_head), **ln_1)
    x = layer_norm(x + ffn(x, **mlp, act_fn=relu), **ln_2)
    return x


def bert(inputs, segment_ids, wte, wpe, wtte, ln_0, blocks, qa, n_head):
    x = wte[inputs] + wpe[range(len(inputs))] + wtte[segment_ids]
    x = layer_norm(x, **ln_0)
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return linear(x, **qa)


hparams, params = load_hparams_and_params(model_path=os.getenv("BERT_MODEL_PATH"))

tokenizer = Tokenizer.from_file(os.getenv("BERT_TOKENIZER_PATH"))

question = """
What are the primary threats to the Great Barrier Reef mentioned in the context?
"""
context = """
The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system, spanning over 2,300 kilometers. It is composed of over 2,900 individual reefs and 900 islands. The reef is home to a vast diversity of marine life, including over 1,500 species of fish, 400 species of coral, and various species of sharks, rays, and turtles. In recent decades, the reef has faced significant threats from climate change, coral bleaching, and pollution.
"""
question_ids = tokenizer.encode(question).ids
context_ids = tokenizer.encode(context).ids
input_ids = question_ids + context_ids[1:]
token_type_ids = [0] * len(question_ids) + [1] * len(context_ids[1:])
logits = bert(input_ids, token_type_ids, **params, n_head=hparams["n_head"])
idx0 = np.argmax(logits[:, 0])
idx1 = np.argmax(logits[:, 1])
output_text = tokenizer.decode(input_ids[idx0 : idx1 + 1])
print(output_text)
