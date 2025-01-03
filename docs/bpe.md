# Byte Pair Encoding (BPE) Tokenizer

## 1. Overview

- **Purpose**: The BPE tokenizer splits text into subwords based on the most frequent pair of adjacent symbols. It is particularly useful for compressing text representation while retaining linguistic coherence.
- **Key Features**:
  - Encodes text into tokens using a learned vocabulary.
  - Decodes tokens back into the original text.
  - Handles errors in decoding gracefully.

---

## 2. Components

### Utility Functions

- **`bytes_to_unicode`**:

  - Maps UTF-8 byte values to Unicode characters for reversible encoding.
  - Ensures the BPE codes work seamlessly with Unicode strings.
  - Handles a wide range of Unicode characters to minimize unknown tokens (`UNK`).

- **`get_pairs(word)`**:
  - Extracts all adjacent symbol pairs from a given word (e.g., `("a", "b")`, `("b", "c")` for `"abc"`).

---

### Class: `BPETokenizer`

The tokenizer encapsulates all functionality for tokenizing and detokenizing text.

#### Attributes

1. **`encoder`**: Maps subwords to unique IDs.
2. **`decoder`**: Inverse of `encoder` for decoding tokens.
3. **`byte_encoder` and `byte_decoder`**:
   - Maps bytes to Unicode and vice versa.
   - Ensures compatibility with a wide character range.
4. **`bpe_ranks`**:
   - Stores the rank of BPE merge operations (from frequent to infrequent).
5. **`cache`**:
   - Caches tokenization results for performance optimization.
6. **`pat`**:
   - A regex pattern to split input text into words, numbers, and symbols.

---

#### Methods

1. **`bpe(token)`**:

   - Applies BPE to a single token by merging pairs iteratively based on `bpe_ranks`.
   - Stops merging when no valid pairs remain in the ranks.

2. **`encode(text)`**:

   - Splits the input text into tokens using the regex pattern.
   - Encodes each token into subwords using BPE and maps them to unique IDs via `encoder`.
   - Returns an `EncodedData` object containing the tokenized IDs.

3. **`decode(tokens)`**:

   - Reconstructs the original text from token IDs using `decoder` and `byte_decoder`.
   - Handles errors during decoding gracefully.

4. **`no_padding()`**:

   - Placeholder method. Could be used to disable padding functionality if implemented.

5. **`from_file(tokenizer_path)`**:
   - Loads a tokenizer configuration (encoder and BPE data) from a file.

---

## 3. Workflow

### Encoding

1. Convert text into Unicode-compatible bytes.
2. Use regex to split text into words and symbols.
3. Apply BPE to each token to generate subwords.
4. Map subwords to unique IDs using the encoder.

### Decoding

1. Map token IDs back to subwords using the decoder.
2. Combine subwords into the original text using `byte_decoder`.

---

## 4. Key Highlights

- **Efficiency**:
  - Uses caching (`self.cache`) and the `@lru_cache` decorator to speed up repetitive tokenizations.
- **Compatibility**:
  - Supports diverse character sets via Unicode mappings.
- **Extensibility**:
  - Can load custom tokenizers from files.
- **Error Handling**:
  - Decodes text gracefully with fallback mechanisms (`errors` parameter).

---

## 5. Example

### Input Text:

`"Hello, world!"`

### Tokenization Steps:

1. Split into tokens: `["Hello", ",", "world", "!"]`.
2. Encode each token into subwords using BPE:
   - `"Hello"` → `["Hel", "lo"]`
   - `"world"` → `["wo", "rld"]`
3. Map subwords to IDs using `encoder`.

### Output:

- **Encoded Tokens**: `[ID_1, ID_2, ...]`
- **Decoded Text**: `"Hello, world!"`

---

## 6. Applications

- Used in GPT-2 and other transformer-based models for efficient text representation.
- Ideal for handling diverse datasets and large vocabularies.
