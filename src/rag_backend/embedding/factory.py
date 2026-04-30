# -*- coding: gbk -*-

from langchain_community.embeddings import OllamaEmbeddings

from rag_backend.config.settings import settings


class EmbeddingsFactory:
    @staticmethod
    def create():
        """
        Factory method to create an embeddings instance based on .env config.
        """
        embeddings_provider = settings.embeddings_provider
        
        if embeddings_provider == "ollama":
            return OllamaEmbeddings(model=settings.embedding_model)
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider}")