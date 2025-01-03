import json
from utils.tokenizers.encoded_data import EncodedData


class WordPieceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}

    def from_file(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            data = json.load(f)
            return WordPieceTokenizer(data["model"]["vocab"])

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            tokens.extend(self._tokenize_word(word))
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text.lower())
        tokens = [self.vocab["[CLS]"]] + tokens + [self.vocab["[SEP]"]]
        return EncodedData(tokens)

    def decode(self, tokens):
        words = []
        current_word = ""
        tokens_ = [self.vocab_inv[token] for token in tokens]
        for token in tokens_:
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token
        if current_word:
            words.append(current_word)
        return " ".join(words)

    def no_padding(self):
        pass

    def _tokenize_word(self, word):
        subwords = []
        start = 0

        while start < len(word):
            end = len(word)
            subword = None

            while start < end:
                candidate = ("##" if start > 0 else "") + word[start:end]
                if candidate in self.vocab:
                    subword = candidate
                    break
                end -= 1

            if subword is None:
                subwords.append(self.vocab["[UNK]"])
                break

            subwords.append(self.vocab[subword])
            start = end

        return subwords
