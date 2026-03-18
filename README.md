# RAG vs Fine-Tuning Comparison Project

This project compares two approaches for adapting large language models to specific domains: **Retrieval-Augmented Generation (RAG)** and **LoRA Fine-Tuning**.

## Project Structure

```
├── data/
│   └── documents/           # Source documents for RAG
├── rag_pipeline/            # RAG implementation
│   ├── chunking.py          # Document chunking utilities
│   ├── embeddings.py        # Vector embedding generation
│   └── retrieval.py         # Vector search and RAG pipeline
├── lora_training/           # LoRA fine-tuning implementation
│   ├── config.py            # Training configuration
│   ├── dataset.py           # Dataset preparation utilities
│   ├── train_lora.py        # LoRA training script
│   └── generate_qa_dataset.py  # QA dataset generation
├── evaluation/              # Model evaluation utilities
│   └── evaluate_models.py   # Evaluation metrics
└── requirements.txt         # Project dependencies
```

## Components

### RAG Pipeline (`rag_pipeline/`)

**Chunking** (`chunking.py`)
- Loads documents from a directory
- Splits documents into overlapping chunks to preserve context
- Configurable chunk size and overlap parameters

**Embeddings** (`embeddings.py`)
- Uses Sentence Transformers (all-MiniLM-L6-v2) for semantic embeddings
- Converts text chunks into high-dimensional vector representations
- Supports batch processing

**Retrieval** (`retrieval.py`)
- FAISS-based vector database for efficient similarity search
- RAG pipeline that combines embeddings and retrieval
- Returns top-k most relevant chunks for a query

### LoRA Training (`lora_training/`)

**Config** (`config.py`)
- Centralized training configuration
- Model, LoRA, and optimization hyperparameters
- Hardware and dataset settings

**Dataset** (`dataset.py`)
- Loads QA pairs from JSON format
- Formats data for causal language model training
- Prepares dataset for the Trainer API

**Generate QA Dataset** (`generate_qa_dataset.py`)
- Generates QA pairs from document chunks using Ollama
- Creates training dataset for fine-tuning
- Requires local Ollama instance running

**Train LoRA** (`train_lora.py`)
- Fine-tunes a base model using LoRA
- Uses 4-bit quantization for memory efficiency
- Implements training loop with validation

### Evaluation (`evaluation/`)

**Evaluate Models** (`evaluate_models.py`)
- Computes Exact Match (EM) and F1 scores
- Compares RAG vs fine-tuned model performance
- Saves results to CSV

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your text documents in `data/documents/`. The chunking module will automatically load all `.txt` files.

### 3. Build RAG Index

```bash
python -c "
from rag_pipeline.chunking import load_documents, chunk_documents
from rag_pipeline.retrieval import RAGPipeline

docs = load_documents('data/documents')
chunks = chunk_documents(docs)
pipeline = RAGPipeline(chunks)
"
```

### 4. Generate QA Dataset (Optional, requires Ollama)

```bash
# Start Ollama with a model: ollama pull gemma2:2b && ollama serve
python lora_training/generate_qa_dataset.py
```

### 5. Fine-tune with LoRA (Optional)

```bash
python lora_training/train_lora.py
```

### 6. Evaluate Models

```bash
python evaluation/evaluate_models.py
```

## Key Features

✅ **Efficient Document Processing**
- Intelligent chunking with overlap to maintain context
- Semantic embeddings with Sentence Transformers

✅ **Fast Retrieval**
- FAISS vector database for efficient similarity search
- O(1) query performance with approximate nearest neighbors

✅ **Parameter-Efficient Fine-Tuning**
- LoRA reduces trainable parameters by 99%+
- 4-bit quantization for memory efficiency

✅ **Comprehensive Evaluation**
- Exact Match and F1 score metrics
- Side-by-side RAG vs fine-tuned comparison

## Troubleshooting

### Dependency Conflicts
Some package versions in requirements.txt may have compatibility issues. You can install without strict version pinning:

```bash
pip install --upgrade pip
pip install numpy==2.1.3 pandas==2.2.3 packaging==25.0 urllib3==2.1.0
```

### Missing Documents
Ensure you have `.txt` files in `data/documents/`. The project includes sample documents on:
- Transformer Architecture
- Attention Mechanisms
- RAG Systems
- Vector Embeddings
- Large Language Models
- And more...

### Ollama Not Found
To generate QA datasets, install and run Ollama:
```bash
# Install from https://ollama.ai
ollama pull gemma2:2b
ollama serve
```

## Configuration

All hyperparameters are defined in `lora_training/config.py`. Key settings:

- `BASE_MODEL`: The model to fine-tune
- `LORA_RANK`: Low-rank dimension (default: 8)
- `BATCH_SIZE`: Training batch size (default: 8)
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 512)

## References

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Sentence-Transformers](https://www.sbert.net/)

## License

This project is for educational and research purposes.