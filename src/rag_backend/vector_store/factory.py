# -*- coding: gbk -*-
from typing import Dict, Type, Any
from rag_backend.embedding.factory import EmbeddingsFactory

from .base import BaseVectorStore
from .chroma import ChromaVectorStore

from rag_backend.config.settings import settings
from logger import logger

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    _stores: Dict[str, Type[BaseVectorStore]] = {
        'chroma': ChromaVectorStore
    }
    
    @classmethod
    def create(
        cls,
        collection_name: str,
        **kwargs: Any
    ) -> BaseVectorStore:
        """Create a vector store instance
        
        Args:
            **kwargs: Additional arguments for specific vector store implementations
            
        Returns:
            An instance of the requested vector store
            
        Raises:
            ValueError: If store_type is not supported
        """
        logger.info(f"Creating vector store instance for collection {collection_name}")

        store_type = settings.vector_store_type
        store_class = cls._stores.get(store_type.lower())
        if not store_class:
            logger.error(f"""Unsupported vector store type: {store_type}. 
                f"Supported types are: {', '.join(cls._stores.keys())}""")
            
            raise ValueError(
                f"Unsupported vector store type: {store_type}. "
                f"Supported types are: {', '.join(cls._stores.keys())}"
            )
        
        embeddings = EmbeddingsFactory.create()
        
        return store_class(
            collection_name=collection_name,
            embeddings=embeddings,
            persist_directory=settings.vector_store_path,
            **kwargs
        )
