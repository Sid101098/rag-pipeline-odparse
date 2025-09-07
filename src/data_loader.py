import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_dir = Path(config['data']['input_directory'])
        self.supported_formats = config['data']['supported_formats']
        
    def discover_files(self) -> List[Path]:
        """Discover all supported files in the input directory"""
        files = []
        for format in self.supported_formats:
            files.extend(self.input_dir.glob(f"**/*{format}"))
        
        # Filter by file size
        max_size = self.config['parser']['max_file_size_mb'] * 1024 * 1024
        valid_files = [f for f in files if f.stat().st_size <= max_size]
        
        logger.info(f"Found {len(valid_files)} supported files")
        return valid_files
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate if file is supported and within size limits"""
        if not file_path.exists():
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            return False
        
        max_size = self.config['parser']['max_file_size_mb'] * 1024 * 1024
        if file_path.stat().st_size > max_size:
            return False
            
        return True
