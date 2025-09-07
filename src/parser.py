from odparse.parser import Parser
from odparse.core import Pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ODParserWrapper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parser = Parser()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['parser']['chunk_size'],
            chunk_overlap=config['parser']['chunk_overlap']
        )
    
    def parse_document(self, file_path: Path) -> List[str]:
        """Parse document using od-parse and split into chunks"""
        try:
            # Parse document using od-parse
            pipeline = Pipeline()
            result = pipeline.run(str(file_path))
            
            # Extract text content
            text_content = self._extract_text(result)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            logger.info(f"Parsed {file_path.name} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []
    
    def _extract_text(self, parse_result: Any) -> str:
        """Extract text content from od-parse result"""
        # This is a simplified extraction - adjust based on actual od-parse output structure
        text_content = ""
        
        if hasattr(parse_result, 'text'):
            text_content = parse_result.text
        elif isinstance(parse_result, dict) and 'text' in parse_result:
            text_content = parse_result['text']
        elif isinstance(parse_result, list):
            for item in parse_result:
                if hasattr(item, 'text'):
                    text_content += item.text + "\n"
        
        return text_content.strip()
