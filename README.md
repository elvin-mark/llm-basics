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
   BERT_EMB_MODEL_PATH="path/to/bert_emb.bin"
   BERT_EMB_TOKENIZER_PATH="path/to/tokenizer.json"
   WHISPER_MODEL_PATH="path/to/whisper.bin"
   WHISPER_TOKENIZER_PATH="path/to/tokenizer.json"
   VIT_MODEL_PATH="path/to/pytorch_model.bin"
   YOLOS_MODEL_PATH="path/to/pytorch_model.bin"
   ```

## Usage of the different models

### Text Generation

#### GPT-2

- Weights and Tokenizer: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)

```sh
python models/gpt2.py
```

#### LLama2

- Weights and Tokenizer: [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)

```sh
python models/llama2.py
```

#### RWKV4

- Weights and Tokenizer: [BlinkDL/rwkv-4-pile-169m](https://huggingface.co/BlinkDL/rwkv-4-pile-169m)

```sh
python models/rwkv4.py
```

### Question and Answering

### Tiny BERT

- Weights and Tokenizer: [Intel/dynamic_tinybert](https://huggingface.co/Intel/dynamic_tinybert)

```sh
python models/tinybert.py
```

### Text Similarity

#### BERT Text Embedding

- Weights and Tokenizer: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

```sh
python models/bert_emb.py
```

### Speech Recognition

#### Whisper

- Weights and Tokenizer: [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en)

```sh
python models/whisper.py
```

### Image Classification

#### ViT

- Weights and Config: [WinKawaks/vit-tiny-patch16-224](https://huggingface.co/WinKawaks/vit-tiny-patch16-224)

```sh
python models/vit.py
```

### Object Detection

#### YOLOs

- Weights and Config: [hustvl/yolos-tiny](https://huggingface.co/hustvl/yolos-tiny)

```sh
python models/yolos.py
```
