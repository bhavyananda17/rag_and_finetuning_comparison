#!/usr/bin/env python3
"""
Quick Start Script for RAG vs Fine-Tuning Comparison Project

This script demonstrates how to use both the RAG pipeline and evaluation utilities.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_pipeline.chunking import load_documents, chunk_documents
from rag_pipeline.retrieval import RAGPipeline
from evaluation.evaluate_models import evaluate_rag_vs_finetuned


def main():
    print("\n" + "="*70)
    print("  RAG vs Fine-Tuning Comparison - Quick Start")
    print("="*70 + "\n")
    
    # Step 1: Load documents
    print("[1/3] Loading and chunking documents...")
    docs_dir = project_root / "data" / "documents"
    
    if not docs_dir.exists():
        print(f"[!] Error: Documents directory not found: {docs_dir}")
        print("[*] Please create data/documents/ and add .txt files")
        return
    
    documents = load_documents(str(docs_dir))
    chunks = chunk_documents(documents, chunk_size=400, overlap=50)
    print(f"    ✓ Loaded {len(documents)} documents")
    print(f"    ✓ Generated {len(chunks)} chunks\n")
    
    # Step 2: Build RAG pipeline
    print("[2/3] Building RAG pipeline...")
    pipeline = RAGPipeline(chunks)
    print(f"    ✓ Vector database initialized with {len(chunks)} embeddings\n")
    
    # Step 3: Test retrieval with sample queries
    print("[3/3] Testing retrieval with sample queries...")
    sample_queries = [
        "What is a transformer?",
        "Explain attention mechanisms",
        "How does RAG work?",
    ]
    
    for query in sample_queries:
        print(f"\n    Query: '{query}'")
        results = pipeline.retrieve(query, k=2)
        for i, result in enumerate(results, 1):
            preview = result[:80].replace('\n', ' ') + "..."
            print(f"      [{i}] {preview}")
    
    # Step 4: Run evaluation example
    print("\n" + "-"*70)
    print("\nRunning evaluation example...")
    
    rag_predictions = [
        "Transformers use self-attention mechanisms",
        "RAG combines retrieval with generation",
    ]
    
    finetuned_predictions = [
        "Transformers are neural networks",
        "RAG retrieves documents for context",
    ]
    
    ground_truths = [
        "Transformers use self-attention mechanisms",
        "RAG combines retrieval with generation",
    ]
    
    evaluate_rag_vs_finetuned(rag_predictions, finetuned_predictions, ground_truths)
    
    print("\n" + "="*70)
    print("✓ Quick start complete!")
    print("\nNext steps:")
    print("  1. Check ISSUES_FOUND_AND_FIXED.md for detailed project information")
    print("  2. Read README.md for comprehensive documentation")
    print("  3. Run 'python lora_training/train_lora.py' to fine-tune a model")
    print("  4. Generate QA dataset with 'python lora_training/generate_qa_dataset.py'")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
