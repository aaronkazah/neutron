from typing import Union, Iterable
import asyncio

import numpy as np
from fastembed import TextEmbedding


class Vectors:
    def __init__(self):
        self.embedding_model = TextEmbedding()

    def _generate_sync(self, text: Union[str, Iterable[str]]) -> np.ndarray:
        """Synchronous wrapper for the embedding generation"""
        texts = [text] if isinstance(text, str) else text
        embeddings = list(self.embedding_model.embed(texts))
        return embeddings[0] if isinstance(text, str) else np.array(embeddings)

    async def generate(self, text: Union[str, Iterable[str]]) -> np.ndarray:
        """Asynchronous wrapper that runs the embedding generation in a separate thread"""
        return await asyncio.to_thread(self._generate_sync, text)


VECTORS = Vectors()
