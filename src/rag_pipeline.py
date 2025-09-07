from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config: Dict[str, Any], vector_store):
        self.config = config
        self.vector_store = vector_store
        
        # Initialize OpenAI LLM
        self.llm = OpenAI(
            model_name=config['rag']['model_name'],
            temperature=config['rag']['temperature'],
            max_tokens=config['rag']['max_tokens'],
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG pipeline"""
        try:
            # Search for relevant documents
            results = self.vector_store.search(question, top_k=top_k)
            
            if not results or not results.get('documents'):
                return "No relevant documents found."
            
            # Format context from retrieved documents
            context = "\n\n".join([doc for doc in results['documents'][0]])
            
            # Create prompt with context
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate answer
            response = self.llm.generate([prompt])
            answer = response.generations[0][0].text.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return f"Error processing your question: {str(e)}"
