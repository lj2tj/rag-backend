# -*- coding: utf-8 -*-
from typing import List, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from logger import logger

from .base import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation"""
    
    def __init__(self, collection_name: str, embeddings: Embeddings, persist_directory: str, **kwargs):
        """Initialize Chroma vector store"""
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.db = Chroma(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=persist_directory
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Chroma"""
        logger.info(f"Adding {len(documents)} documents to Chroma collection {self.collection_name}")
        self.db.add_documents(documents=documents)
        logger.info(f"Successfully added {len(documents)} documents")
    
    def as_retriever(self, **kwargs: Any) -> None:
        """Return a retriever interface"""
        logger.info("Loaded Chroma index for retrieval")
        return self.db.as_retriever(search_type="similarity", search_kwargs={"k": 3}, **kwargs)
    
    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """Perform a similarity search on the vector store"""
        return self.db.similarity_search(query, k=k)

    