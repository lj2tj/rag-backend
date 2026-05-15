# -*- coding: utf-8 -*-

from typing import List, Dict, Any

from langchain_core.documents import Document
# from langchain_text_splitters import CharacterTextSplitterdd
# laugchain提供的splitter，用于将文本按字符分割，但是只返回分割后的字符串数组
# 没有返回字符串的标题，所以放弃使用

from .base import DocumentChunker
from rag_backend.util import file_status_manager
from rag_backend.util import logger

class PDFChunker(DocumentChunker):
    def chunk(self, file_path: str, parsed_content: str) -> List[Dict[str, Any]]:
        file_path = file_status_manager.get_relative_path(file_path)
        chunk_settings = self.get_settings(".pdf")
        chunk_size = chunk_settings.get("chunk_size", 500)
        overlap = chunk_settings.get("overlap", 50)

        parsed_content = parsed_content.strip()
        if not parsed_content:
            return []

        chunks = []
        paragraphs = self._split_by_chapters(parsed_content)
        logger.info(f"Split into {len(paragraphs)} chapters")
        for title, content in paragraphs:
            if len(content) > chunk_size + overlap:
                text_chunks = self._split_long_text(content, chunk_size, overlap)
                for chunk_text in text_chunks:
                    metadata = self._create_metadata(
                        file_path=file_path,
                        chunk_index=len(chunks),
                        original_type="pdf",
                        chunk_size=chunk_size,
                        overlap=overlap,
                        title=title
                    )
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))
            else:
                metadata = self._create_metadata(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    original_type="pdf",
                    chunk_size=chunk_size,
                    overlap=overlap,
                    title=title
                )
                chunks.append(Document(page_content=content, metadata=metadata))

        self.save_chunks(file_path, chunks)
        return chunks