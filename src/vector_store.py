import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = SentenceTransformer(config['vector_store']['embedding_model'])
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=config['vector_store']['persist_directory'],
            chroma_db_impl="duckdb+parquet"
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=config['vector_store']['collection_name']
        )
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to vector store with embeddings"""
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Generate IDs
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata if metadata else [{}] * len(documents),
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        return self.collection.count()
