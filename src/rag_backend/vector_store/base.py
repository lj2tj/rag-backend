
# -*- coding: gbk -*-

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations"""
    
    @abstractmethod
    def __init__(self, collection_name: str, embeddings: Embeddings, **kwargs):
        """Initialize the vector store class"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> None:
        """Return a retriever interface"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """Perform a similarity search on the vector store"""
        pass
