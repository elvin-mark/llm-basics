# LLM Basics

This repository contains implementations and explanations of various large language models (LLMs) including GPT-2, LLama2, RWKV4, and BERT. The models are implemented in Python and utilize libraries such as `numpy`, `torch`, and `tokenizers`.

## Project Structure

- **docs/**: Contains markdown files with detailed explanations and workflows for each model.
- **models/**: Contains Python scripts implementing the different models.
- **.env**: Environment variables for model and tokenizer paths.
- **requirements.txt**: List of dependencies required to run the project.

## Setup

1. Clone the repository:

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Set up the environment variables by creating a [.env](http://_vscodecontentref_/7) file in the root directory with the following content:
   ```env
   GPT2_MODEL_PATH="path/to/gpt2.bin"
   GPT2_TOKENIZER_PATH="path/to/tokenizer.json"
   LLAMA2_MODEL_PATH="path/to/stories15M.bin"
   LLAMA2_TOKENIZER_PATH="path/to/tokenizer.model"
   RWKV4_MODEL_PATH="path/to/RWKV-4-Pile-169M-20220807-8023.pth"
   RWKV4_TOKENIZER_PATH="path/to/20B_tokenizer.json"
   BERT_MODEL_PATH="path/to/tinybert.bin"
   BERT_TOKENIZER_PATH="path/to/bert_tokenizer.json"
   ```

## Usage

### GPT-2

```sh
python models/gpt2.py
```

### LLama2

```sh
python models/llama2.py
```

### RWKV4

```sh
python models/rwkv4.py
```

### Tiny BERT

```sh
python models/tinybert.py
```
