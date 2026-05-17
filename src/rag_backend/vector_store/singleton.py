# -*- coding: utf-8 -*-
"""Singleton wrapper for vector store instances."""

from typing import Optional
from rag_backend.vector_store.factory import VectorStoreFactory
from rag_backend.vector_store.base import BaseVectorStore
from logger import logger


class VectorStoreSingleton:
    """Singleton class for managing vector store instances.
    
    This class ensures that only one instance of each vector store collection
    is created and reused throughout the application lifecycle.
    """
    
    _instances: dict[str, BaseVectorStore] = {}
    
    @classmethod
    def get_instance(cls, collection_name: str = "default") -> BaseVectorStore:
        """Get or create a vector store instance for the specified collection.
        
        Args:
            collection_name: The name of the collection. Defaults to "default".
            
        Returns:
            A vector store instance for the specified collection.
        """
        if collection_name not in cls._instances:
            logger.info(f"Creating new vector store instance for collection: {collection_name}")
            cls._instances[collection_name] = VectorStoreFactory.create(collection_name=collection_name)
        else:
            logger.info(f"Reusing existing vector store instance for collection: {collection_name}")
        
        return cls._instances[collection_name]
    
    @classmethod
    def clear_instance(cls, collection_name: str = "default") -> None:
        """Clear a specific vector store instance.
        
        Args:
            collection_name: The name of the collection to clear. Defaults to "default".
        """
        if collection_name in cls._instances:
            del cls._instances[collection_name]
            logger.info(f"Cleared vector store instance for collection: {collection_name}")
    
    @classmethod
    def clear_all_instances(cls) -> None:
        """Clear all vector store instances."""
        cls._instances.clear()
        logger.info("Cleared all vector store instances")

    @classmethod
    def refresh_instance(cls, collection_name: str = "default") -> BaseVectorStore:
        """Refresh a vector store instance for the specified collection.
        
        Args:
            collection_name: The name of the collection. Defaults to "default".
            
        Returns:
            A vector store instance for the specified collection.
        """
        if collection_name in cls._instances:
            logger.info("vector store instance for collection: {collection_name} already exists, delete it first")
            del cls._instances[collection_name]

        logger.info(f"Creating new vector store instance for collection: {collection_name}")
        cls._instances[collection_name] = VectorStoreFactory.create(collection_name=collection_name)

        return cls._instances[collection_name]