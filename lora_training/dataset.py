import json
from datasets import Dataset, load_dataset
from typing import List, Dict

def format_instruction(sample: Dict[str, str]) -> Dict[str, str]:
    """
    Format the QA pair into a prompt format suitable for Causal Language Models.
    """
    prompt = f"### Instruction: {sample['question']}\n\n### Response: {sample['answer']}"
    return {"text": prompt}

def prepare_dataset(json_path: str) -> Dataset:
    """
    Load the QA dataset from JSON and format it for training.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
        
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_instruction)
    return dataset

if __name__ == "__main__":
    # Test block
    import os
    path = "data/qa_dataset.json"
    if os.path.exists(path):
        ds = prepare_dataset(path)
        print(f"Loaded dataset with {len(ds)} samples.")
        print(f"Sample prompt:\n{ds[0]['text']}")
    else:
        print(f"File {path} not found. Run generate_qa_dataset.py first.")
