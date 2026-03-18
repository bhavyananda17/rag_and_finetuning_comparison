import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingEngine:
    """
    Handles generation of vector embeddings for text chunks using Sentence Transformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[*] Initializing Embedding Engine with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of strings into a numpy array of embeddings.
        """
        if not texts:
            return np.array([])
        
        print(f"[*] Generating embeddings for {len(texts)} items...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings)

if __name__ == "__main__":
    # Test block
    engine = EmbeddingEngine()
    test_texts = ["Hello world", "Artificial intelligence is changing the world."]
    embeddings = engine.generate_embeddings(test_texts)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
