import os
import json
import requests
from typing import List, Dict
from rag_pipeline.chunking import load_documents, chunk_documents
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

# Premium aesthetics: logging and clear structure
BANNER = """
=========================================
   QA DATASET GENERATION PIPELINE
=========================================
"""

PROMPT_TEMPLATE = """
As an expert AI teacher, generate a clear, high-quality question and answer pair based ONLY on the following context.
The question should be specific and the answer should be comprehensive.

Context:
{context}

Format your output EXACTLY as:
Question: [Your question]
Answer: [Your answer]
"""

def generate_qa_pair(chunk: str) -> Dict[str, str]:
    """
    Calls the local Ollama instance to generate a QA pair for a given chunk.
    """
    prompt = PROMPT_TEMPLATE.format(context=chunk)
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json().get("response", "")
        
        # Parse the result
        lines = result.strip().split("\n")
        question = ""
        answer = ""
        
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
        
        if not question or not answer:
            # Fallback if parsing fails but content might be there
            if "Question:" in result and "Answer:" in result:
                # Basic split if it's not neat lines
                parts = result.split("Answer:")
                question = parts[0].replace("Question:", "").strip()
                answer = parts[1].strip()
            else:
                return None
                
        return {"question": question, "answer": answer, "context": chunk}
        
    except Exception as e:
        print(f"\nError generating QA for chunk: {e}")
        return None

def main():
    print(BANNER)
    
    docs_dir = "data/documents"
    output_file = "data/qa_dataset.json"
    
    # 1. Load and Chunk
    print(f"[*] Loading documents from {docs_dir}...")
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents, chunk_size=600, overlap=100) # Slightly larger chunks for better context
    
    # 2. Generate QA Pairs
    print(f"[*] Generating QA pairs using {MODEL_NAME}...")
    qa_dataset = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing Chunks")):
        if len(chunk.strip()) < 100: # Skip tiny fragments
            continue
            
        qa_pair = generate_qa_pair(chunk)
        if qa_pair:
            qa_dataset.append(qa_pair)
        
        # Periodically save progress
        if (i + 1) % 10 == 0:
            with open(output_file, "w") as f:
                json.dump(qa_dataset, f, indent=4)
                
    # 3. Final Save
    with open(output_file, "w") as f:
        json.dump(qa_dataset, f, indent=4)
    
    print(f"\n[+] Success! Generated {len(qa_dataset)} QA pairs.")
    print(f"[+] Dataset saved to {output_file}")

if __name__ == "__main__":
    main()
