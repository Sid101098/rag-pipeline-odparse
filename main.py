#!/usr/bin/env python3

import argparse
from pathlib import Path
from src.utils import load_config, setup_logging, load_environment_variables
from src.data_loader import DataLoader
from src.parser import ODParserWrapper
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
import logging

logger = logging.getLogger(__name__)

def process_documents(config):
    """Process all documents and build vector store"""
    # Initialize components
    data_loader = DataLoader(config)
    parser = ODParserWrapper(config)
    vector_store = VectorStoreManager(config)
    
    # Discover and process files
    files = data_loader.discover_files()
    
    all_chunks = []
    all_metadata = []
    
    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        
        # Parse document
        chunks = parser.parse_document(file_path)
        
        # Create metadata for each chunk
        file_metadata = {
            'source': file_path.name,
            'file_path': str(file_path),
            'total_chunks': len(chunks)
        }
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = file_metadata.copy()
            chunk_metadata['chunk_index'] = i
            all_metadata.append(chunk_metadata)
        
        all_chunks.extend(chunks)
    
    # Add to vector store
    if all_chunks:
        vector_store.add_documents(all_chunks, all_metadata)
        logger.info(f"Processed {len(files)} files into {len(all_chunks)} chunks")
    else:
        logger.warning("No chunks were generated from the documents")

def interactive_query(config):
    """Interactive query mode for the RAG pipeline"""
    vector_store = VectorStoreManager(config)
    rag_pipeline = RAGPipeline(config, vector_store)
    
    print("RAG Pipeline Interactive Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if not question:
                continue
                
            answer = rag_pipeline.query(question)
            print(f"\nAnswer: {answer}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    setup_logging()
    
    try:
        load_environment_variables()
    except ValueError as e:
        logger.error(e)
        return
    
    # Load configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description="RAG Pipeline with od-parse")
    parser.add_argument('--process', action='store_true', help='Process documents and build vector store')
    parser.add_argument('--query', action='store_true', help='Start interactive query mode')
    
    args = parser.parse_args()
    
    if args.process:
        process_documents(config)
    elif args.query:
        interactive_query(config)
    else:
        print("Please specify either --process to build vector store or --query for interactive mode")

if __name__ == "__main__":
    main()
