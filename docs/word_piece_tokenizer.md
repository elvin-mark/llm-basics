# WordPiece Tokenizer

## 1. Overview

- **Purpose**: The WordPiece tokenizer splits text into subwords that are present in the vocabulary. It is particularly useful in natural language processing (NLP) tasks for handling out-of-vocabulary words efficiently.
- **Key Features**:
  - Encodes text into token IDs.
  - Decodes token IDs back into readable text.
  - Uses special tokens like `[CLS]` and `[SEP]` for compatibility with transformer models.

---

## 2. Components

### Class: `WordPieceTokenizer`

The main class encapsulates the tokenization and detokenization logic.

#### Attributes

1. **`vocab`**:
   - Maps subwords (e.g., `"hello"`, `"##llo"`) to unique token IDs.
2. **`vocab_inv`**:
   - Inverse of `vocab`, mapping token IDs back to subwords.

---

#### Methods

1. **`from_file(tokenizer_path)`**:

   - Loads the tokenizer's vocabulary from a file.
   - Returns an instance of `WordPieceTokenizer`.

2. **`tokenize(text)`**:

   - Splits input text into words and tokenizes each word into subwords based on the vocabulary.

3. **`encode(text)`**:

   - Converts the input text to lower case.
   - Tokenizes the text and adds special tokens `[CLS]` at the start and `[SEP]` at the end.
   - Returns an `EncodedData` object containing token IDs.

4. **`decode(tokens)`**:

   - Converts token IDs back to subwords using `vocab_inv`.
   - Reconstructs the original text by merging subwords, ensuring proper handling of tokens prefixed with `"##"`.

5. **`no_padding()`**:

   - Placeholder method. Could be used to disable padding functionality if implemented.

6. **`_tokenize_word(word)`**:
   - Tokenizes a single word into subwords by finding the longest matching subword in the vocabulary.
   - If no match is found, it replaces the word with the `[UNK]` token ID.

---

## 3. Workflow

### Encoding

1. Convert the text to lowercase.
2. Split the text into words and tokenize each word into subwords.
3. Map subwords to token IDs using the vocabulary.
4. Add `[CLS]` and `[SEP]` token IDs to indicate the start and end of the sequence.

### Decoding

1. Map token IDs back to subwords using `vocab_inv`.
2. Combine subwords into the original text by merging tokens prefixed with `"##"` into their preceding word.

---

## 4. Key Highlights

- **Efficiency**:
  - Uses the longest-match-first strategy to find subwords in the vocabulary, minimizing the number of subwords per word.
- **Compatibility**:
  - Includes `[CLS]` and `[SEP]` tokens for compatibility with transformer models like BERT.
- **Error Handling**:
  - Replaces out-of-vocabulary words with `[UNK]`, ensuring robustness.

---

## 5. Example

### Vocabulary Example

```json
{
  "hello": 1,
  "##llo": 2,
  "my": 3,
  "name": 4,
  "##me": 5,
  "[CLS]": 6,
  "[SEP]": 7,
  "[UNK]": 8
}
```

### Input Text:

`"Hello my name"`

### Tokenization Steps:

1. Convert to lowercase: `"hello my name"`.
2. Split into words: `["hello", "my", "name"]`.
3. Tokenize each word:
   - `"hello"` → `["hello"]`
   - `"my"` → `["my"]`
   - `"name"` → `["name"]`
4. Map to token IDs: `[6, 1, 3, 4, 7]`.

### Decoding:

1. Map token IDs back to subwords: `["[CLS]", "hello", "my", "name", "[SEP]"]`.
2. Merge subwords:
   - Reconstructs: `"hello my name"`.

### Output:

- **Encoded Tokens**: `[6, 1, 3, 4, 7]`
- **Decoded Text**: `"hello my name"`

## 6. Applications

- Used in transformer-based models like BERT and ALBERT for subword-level tokenization.
- Ideal for handling rare or unknown words by breaking them into known subwords.
