# -*- coding: utf-8 -*-
import json
from posixpath import abspath
from datetime import datetime
import os
from typing import Dict, Any

from langchain_core.documents import Document

from rag_backend.vector_store.factory import VectorStoreFactory
from rag_backend.config.settings import settings
from rag_backend.parsers.factory import DocumentParserFactory
from rag_backend.chunkers.factory import DocumentChunkerFactory

from enums import EmbedStatus
from logger import logger
from rag_backend.util import file_status_manager

async def embed_document_svc(file_status_obj: Dict[str, Any]) -> bool:
    """Embed a document."""
    file_path = file_status_obj.get('path', '')
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False

    if not DocumentParserFactory.is_supported(file_path):
        logger.error(f"Parser not supported for file: {file_path}")
        return False

    all_documents = []
    if file_status_obj.get('embeded', '') == EmbedStatus.partial_completed.name:
        logger.info(f"File {file_path} already partially embedded, restart embedding...")
        chunk_file = os.path.join(settings.chunk_dir, os.path.basename(file_path))
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            all_documents = [Document(page_content=doc.get('page_content', ''), metadata=doc.get('metadata', {})) for doc in data]
            logger.info(f"Loaded {len(all_documents)} chunks from file: {chunk_file}")
        else:
            logger.error(f"Chunk file not found: {chunk_file}")
            return False
    else:
        logger.info(f"Parsing file: {file_path}")
        parser = DocumentParserFactory.get_parser(file_path)
        parsed_content = parser.parse(file_path)
        logger.info(f"Parsed content length: {len(parsed_content)}")

        logger.info(f"Chunking file: {file_path}")
        all_documents = DocumentChunkerFactory.chunk_document(file_path, parsed_content)
        if not all_documents:
            logger.error(f"No chunks created from file: {file_path}")
            return False
        logger.info(f"Chunked content length: {len(all_documents)}")

    batch_size = settings.max_batch_size
    import threading
    lock = threading.Lock()
    j = 0
    i = 0
    if file_status_obj.get('embeded', '') == EmbedStatus.partial_completed.name:
        j = file_status_obj.get('embeded_index', 0)
        i = j * batch_size

    for i in range(0, len(all_documents), batch_size):
        try:
            logger.info(f"Embedding batch {j} of {int(len(all_documents)/batch_size)} with size {batch_size}") 
            batch = all_documents[i:i+batch_size]
            with lock:
                vector_store = VectorStoreFactory.create("default")
                vector_store.add_documents(
                    documents=batch
                )
            j += 1
        except Exception as e:
            logger.error(f"[embed_document_svc] Batch content: {batch}")
            if j > 0:
                embeded = f"{EmbedStatus.partial_completed.name}"
            else:
                embeded = EmbedStatus.not_started.name
            file_status_manager.update_file_status(file_path, 
                                    embeded=embeded, 
                                    embeded_index=j,
                                    last_update=datetime.now().isoformat(),
                                    vectorized_time=datetime.now().isoformat(),
                                    embeddings_provider=settings.embeddings_provider, 
                                    embedding_model=settings.embedding_model,
                                    vector_store_type=settings.vector_store_type)
            raise Exception(f"[embed_document_svc] Failed to embed batch {j}: {e}")
    
    embeded = EmbedStatus.completed.name
    file_status_manager.update_file_status(file_path, 
                                        embeded=embeded, 
                                        vectorized_time=datetime.now().isoformat(),
                                        embeddings_provider=settings.embeddings_provider, 
                                        embedding_model=settings.embedding_model,
                                        vector_store_type=settings.vector_store_type)
    
    logger.info("Vector store updated successfully.")
    return True
