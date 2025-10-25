"""
embedding_generator.py

Module for generating embeddings using OpenAI-compatible API.
"""
from typing import List

import openai

class EmbeddingGenerator:
    def __init__(self, api_base: str | None = None, api_key: str | None = None):
        """Initialize the embedding generator with optional API configuration."""
        self.configure(api_base, api_key)

    def configure(self, api_base: str | None = None, api_key: str | None = None) -> None:
        """Configure OpenAI client with custom base URL and API key."""
        if api_base:
            openai.api_base = api_base
        if api_key:
            openai.api_key = api_key

    def get_embeddings(self, texts: List[str], model: str = "text-embedding-nomic-embed-text-v1.5") -> List[List[float]]:
        """Call the embeddings endpoint and return a list of embedding vectors.

        Args:
            texts: List of text strings to embed
            model: Name of the embedding model to use

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        # The openai client will POST to {api_base}/embeddings
        resp = openai.Embedding.create(input=texts, model=model)
        # Response format: {'data': [{'embedding': [...], 'index': 0}, ...], ...}
        return [item["embedding"] for item in resp["data"]]