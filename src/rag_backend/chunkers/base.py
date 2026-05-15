# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import re
import os

from rag_backend.config.settings import settings
from logger import logger

class DocumentChunker(ABC):

    @abstractmethod
    def chunk(self, file_path: str, parsed_content: str) -> List[Dict[str, Any]]:
        pass

    def _create_metadata(self, *, file_path: str, chunk_index: int, original_type: str,
                        chunk_size: int, overlap: int, title: str = None) -> Dict[str, Any]:
        """Create metadata dictionary for a chunk.
        
        Args:
            file_path: the document file path
            chunk_index: the index of current chunk in all chunks
            original_type: the document file type, like: pdf, word
            chunk_size: the max chunk size
            overlap: the overlap size between two chunks
            title: the chunk's title if have
        """
        import uuid
        return {
            "chunk_index": chunk_index,
            "source": file_path,
            "original_type": original_type,
            "title": title if title else " - ",
            "id": str(uuid.uuid4()),
            'chunk_size': chunk_size,
            'overlap': overlap
        }

    def get_settings(self, ext: str) -> Dict[str, Any]:
        return settings.get_chunk_config(ext)
    
    def save_chunks(self, file_path: str, chunks: List[Dict[str, Any]]) -> None:
        """save chunks to file"""
        file_name = os.path.basename(file_path)
        os.makedirs(settings.chunk_dir, exist_ok=True)
        chunk_file = os.path.join(settings.chunk_dir, file_name)
        logger.info(f"Saving chunks to file: {chunk_file}")
        logger.info(f"Chunks: {chunks}")
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in chunks]
        with open(chunk_file, "w") as f:
             json.dump(documents, f, ensure_ascii=False, indent=4)
        logger.info(f"Chunks saved to file: {chunk_file}")
        pass

    def _split_long_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """split long text by chunk_size
        
        Args:
            text: the document content
            chunk_size: the max chunk size
            overlap: the overlap size between two chunks
        Returns:
            a list of chunks
        """
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r'[\n\r]+', text)
        sentences_length = len(sentences)
        max_chunk_size = chunk_size + overlap
        chunks = []
        start_index, index = 0, 1
        while index <= sentences_length:
            tmp = "\n".join(sentences[start_index:index])
            if len(tmp) > max_chunk_size:
                index -= 1
                tmp = "\n".join(sentences[start_index:index])
                chunks.append(tmp)
                start_index = index
                index += 1
            else:
                index += 1

        if index > sentences_length:
            chunks.append("\n".join(sentences[start_index:]))
        
        return chunks

    def _split_by_chapters(self, text: str) -> List[tuple]:
        """split by chapters
        
        Args:
            text: the document content
        Returns:
            a list of tuples, each tuple is (title, content)
        """
        import re
        # ????????????1.1??2.4??10.8
        chapter_pattern = r'^\s*(\d+\.\d+)(\s|$)'
        lines = text.split('\n')
        chunks = []
        current_title = ""
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(chapter_pattern, line)
            if match:
                # ????????
                if current_text:
                    # ???›Ì????
                    chunks.append((current_title, '\n'.join(current_text)))
                    current_text = []
                # ?????
                current_title = match.group(1)
                # ????????????????
                current_text.append(line)
            else:
                # ??????
                current_text.append(line)
        
        # ?????????????
        if current_text:
            chunks.append((current_title, '\n'.join(current_text)))
        return chunks