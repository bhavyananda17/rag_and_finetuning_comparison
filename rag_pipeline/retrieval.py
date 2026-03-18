import faiss
import numpy as np
from typing import List, Tuple

# Handle both relative and absolute imports
try:
    from rag_pipeline.embeddings import EmbeddingEngine
except ImportError:
    from embeddings import EmbeddingEngine

class VectorDatabase:
    """
    A simple vector store using FAISS for efficient similarity search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = [] # Store original chunks
        
    def add_documents(self, chunks: List[str], embeddings: np.ndarray):
        """
        Add chunks and their corresponding embeddings to the index.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")
            
        # Ensure embeddings are float32 for FAISS
        embeddings_f32 = embeddings.astype('float32')
        self.index.add(embeddings_f32)
        self.metadata.extend(chunks)
        print(f"[+] Added {len(chunks)} documents to the vector database.")

    def search(self, query: str, engine: EmbeddingEngine, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top k relevant chunks for a given query.
        """
        query_embedding = engine.generate_embeddings([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1: # FAISS returns -1 if no match found
                results.append((self.metadata[idx], float(dist)))
                
        return results

class RAGPipeline:
    """
    Combines EmbeddingEngine and VectorDatabase for a complete RAG retrieval system.
    """
    def __init__(self, chunks: List[str] = None):
        self.engine = EmbeddingEngine()
        self.db = None
        
        if chunks:
            self.build_index(chunks)
            
    def build_index(self, chunks: List[str]):
        embeddings = self.engine.generate_embeddings(chunks)
        dim = embeddings.shape[1]
        self.db = VectorDatabase(dim)
        self.db.add_documents(chunks, embeddings)
        
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if not self.db:
            return ["No index built yet."]
        
        results = self.db.search(query, self.engine, k)
        return [r[0] for r in results]

if __name__ == "__main__":
    # Test block
    import sys
    sys.path.insert(0, '.')
    from rag_pipeline.chunking import load_documents, chunk_documents
    
    docs = load_documents("data/documents")
    chunks = chunk_documents(docs)
    
    pipeline = RAGPipeline(chunks)
    query = "What is the attention mechanism?"
    results = pipeline.retrieve(query)
    
    print(f"\nQuery: {query}")
    print("-" * 20)
    for i, res in enumerate(results):
        print(f"Result {i+1}:\n{res[:200]}...\n")
