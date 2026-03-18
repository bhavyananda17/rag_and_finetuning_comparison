import os
from typing import List

def load_documents(directory: str) -> List[str]:
    """
    Read all .txt files inside the given directory.
    Return a list containing the full text of each document.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        
    documents = []
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    print(f"Loaded {len(documents)} documents")
    return documents

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks using a sliding window.
    Attempts to avoid splitting in the middle of words.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap.")
        
    chunks = []
    text_length = len(text)
    
    if text_length == 0:
        return chunks
        
    start = 0
    while start < text_length:
        end = start + chunk_size
        
        # If we are not at the end of the text, try to find the last space to avoid word splitting
        if end < text_length:
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start + (chunk_size // 2):
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        if end >= text_length:
            break
            
        # Move start back by overlap, but ensure we progress
        next_start = end - overlap
        if next_start <= start:
            start = end
        else:
            start = next_start
            
    return chunks

def chunk_documents(documents: List[str], chunk_size=400, overlap=50) -> List[str]:
    """
    Apply chunk_text to every document.
    Return a flattened list of all chunks.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc, chunk_size, overlap)
        all_chunks.extend(doc_chunks)
        
    print(f"Generated {len(all_chunks)} chunks")
    return all_chunks

if __name__ == "__main__":
    directory = "data/documents"
    
    # Load documents
    docs = load_documents(directory)
    
    # Chunk documents
    chunks = chunk_documents(docs)
