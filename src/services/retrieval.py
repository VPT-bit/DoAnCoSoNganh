"""Retrieval services - Embedding and Reranker persistent in VRAM."""
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import bm25s
import torch
from llama_cpp import LLAMA_POOLING_TYPE_RANK
from llama_cpp.llama_embedding import LlamaEmbedding, NORM_MODE_EUCLIDEAN

from src.core.models import Corpus
from src.core.config import config
from src.core.cancellation import CancellationToken, CancelledException


class FullTextRetriever:
    """BM25-based full-text retrieval (CPU-based, no VRAM usage)."""
    
    def __init__(self):
        self._retriever: bm25s.BM25 | None = None
        self._corpus: Corpus | None = None
        self._indexed = False
    
    def process(self, corpus: Corpus) -> None:
        """Index the corpus for retrieval."""
        self._corpus = corpus
        if corpus.is_empty():
            self._indexed = False
            return
            
        texts = corpus.extract_texts()
        tokens = bm25s.tokenize(texts)
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens)
        self._indexed = True
    
    def retrieve(self, query: str, k: int = 5) -> list[int]:
        """Retrieve top-k document indices."""
        if not self._indexed or self._corpus is None:
            return []
        
        query_tokens = bm25s.tokenize(query)
        results, _ = self._retriever.retrieve(query_tokens, k=min(k, len(self._corpus.chunks)))
        return [int(idx) for idx in results[0]]
    
    def clear(self) -> None:
        """Clear indexed data."""
        self._retriever = None
        self._corpus = None
        self._indexed = False


class SemanticRetriever:
    """
    Embedding-based semantic retrieval.
    Model loaded once at startup and kept in VRAM.
    """
    
    def __init__(self, model_path: str | None = None, load_on_init: bool = True):
        self._model_path = model_path or config.models.get_embedding_path()
        self._model: LlamaEmbedding | None = None
        self._corpus: Corpus | None = None
        self._embeddings: np.ndarray | None = None
        self._indexed = False
        
        if load_on_init:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load embedding model into VRAM (called once at startup)."""
        if self._model is not None:
            return
        
        self._model = LlamaEmbedding(
            model_path=self._model_path,
            n_gpu_layers=-1,
            verbose=False,
        )
    
    def process(self, corpus: Corpus, cancel_token: Optional[CancellationToken] = None) -> None:
        """Compute embeddings for the corpus."""
        self._corpus = corpus
        if corpus.is_empty():
            self._indexed = False
            return
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before embedding")
        
        texts = corpus.extract_texts()
        self._embeddings = np.asarray(
            self._model.embed(texts, normalize=NORM_MODE_EUCLIDEAN),
            dtype=np.float32,
        )
        self._indexed = True
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[int]:
        """Retrieve top-k semantically similar document indices."""
        if not self._indexed or self._corpus is None or self._embeddings is None:
            return []
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before semantic retrieval")
        
        query_embedding = np.asarray(
            self._model.embed([query], normalize=NORM_MODE_EUCLIDEAN),
            dtype=np.float32,
        ).squeeze(axis=0)
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled after embedding query")
        
        scores = np.dot(self._embeddings, query_embedding.T).flatten()
        k = min(k, len(scores))
        top_indices = np.argsort(-scores)[:k]
        return top_indices.tolist()
    
    def release(self) -> None:
        """Release model from VRAM."""
        if self._model is not None:
            self._model.close()
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def clear(self) -> None:
        """Clear indexed data (keeps model loaded)."""
        self._corpus = None
        self._embeddings = None
        self._indexed = False


class Reranker:
    """
    Cross-encoder based reranker.
    Model loaded once at startup and kept in VRAM.
    """
    
    def __init__(self, model_path: str | None = None, load_on_init: bool = True):
        self._model_path = model_path or config.models.get_reranker_path()
        self._model: LlamaEmbedding | None = None
        self._corpus: Corpus | None = None
        
        if load_on_init:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load reranker model into VRAM (called once at startup)."""
        if self._model is not None:
            return
        
        self._model = LlamaEmbedding(
            model_path=self._model_path,
            n_gpu_layers=-1,
            pooling_type=LLAMA_POOLING_TYPE_RANK,
            verbose=False,
        )
    
    def process(self, corpus: Corpus) -> None:
        """Set corpus for reranking."""
        self._corpus = corpus
    
    def rerank(
        self,
        query: str,
        candidate_indices: list[int],
        cancel_token: Optional[CancellationToken] = None,
    ) -> list[int]:
        """Rerank candidate documents."""
        if not candidate_indices or self._corpus is None:
            return candidate_indices
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before reranking")
        
        texts = self._corpus.extract_texts_by_indices(candidate_indices)
        scores = self._model.rank(query, texts)
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled after reranking")
        
        ranked_pairs = sorted(
            zip(candidate_indices, scores),
            key=lambda x: -x[1],
        )
        return [idx for idx, _ in ranked_pairs]
    
    def release(self) -> None:
        """Release model from VRAM."""
        if self._model is not None:
            self._model.close()
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def clear(self) -> None:
        """Clear references (keeps model loaded)."""
        self._corpus = None


class RRF:
    """Reciprocal Rank Fusion for combining rankings."""
    
    def __init__(self, alpha: int = 60):
        self.alpha = alpha
    
    def aggregate(self, *rankings: list[int]) -> list[int]:
        """Aggregate multiple rankings using RRF."""
        scores: dict[int, float] = {}
        
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0) + 1 / (self.alpha + rank)
        
        sorted_indices = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_indices


class HybridRetriever:
    """
    Hybrid retrieval with persistent models in VRAM.
    
    - BM25: CPU-based (no VRAM)
    - Semantic: Embedding model in VRAM
    - Reranker: Reranker model in VRAM
    """
    
    def __init__(self, load_on_init: bool = True):
        self._corpus: Corpus | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
        self._indexed = False
        
        # Initialize retrievers
        self._ft_retriever = FullTextRetriever()
        self._rrf = RRF(alpha=config.rrf_alpha)
        self._st_retriever = SemanticRetriever(load_on_init=load_on_init)
        self._reranker = Reranker(load_on_init=load_on_init)
    
    def _process_sync(
        self,
        corpus: Corpus,
        cancel_token: Optional[CancellationToken] = None,
    ) -> None:
        """Synchronous corpus processing."""
        self._corpus = corpus
        
        # BM25 indexing (CPU)
        self._ft_retriever.process(corpus)
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled after BM25 indexing")
        
        # Semantic indexing
        self._st_retriever.process(corpus, cancel_token)
        
        # Reranker setup
        self._reranker.process(corpus)
        
        self._indexed = not corpus.is_empty()
    
    async def process(
        self,
        corpus: Corpus,
        cancel_token: Optional[CancellationToken] = None,
    ) -> None:
        """Process and index corpus asynchronously."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._process_sync,
                corpus,
                cancel_token,
            )
    
    def _get_file_indices(self, file_ids: list[str] | None) -> set[int] | None:
        """Get chunk indices belonging to specified files."""
        if file_ids is None or self._corpus is None:
            return None
        file_id_set = set(file_ids)
        return {i for i, c in enumerate(self._corpus.chunks) if c.file_id in file_id_set}
    
    def _retrieve_sync(
        self,
        query: str,
        k: int,
        file_ids: list[str] | None = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> Corpus:
        """Synchronous retrieval with optional file filtering."""
        if not self._indexed or self._corpus is None:
            return Corpus()
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before retrieval")
        
        allowed_indices = self._get_file_indices(file_ids)
        candidate_multiplier = 4
        
        # BM25 retrieval (CPU)
        ft_results = self._ft_retriever.retrieve(query, k=k * candidate_multiplier)
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled after BM25 retrieval")
        
        # Semantic retrieval
        st_results = self._st_retriever.retrieve(query, k=k * candidate_multiplier, cancel_token=cancel_token)
        
        # Filter by file if needed
        if allowed_indices is not None:
            ft_results = [i for i in ft_results if i in allowed_indices]
            st_results = [i for i in st_results if i in allowed_indices]
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled after filtering")
        
        # Fuse rankings
        fused = self._rrf.aggregate(ft_results, st_results)
        
        # Rerank
        reranked = self._reranker.rerank(query, fused[:k * candidate_multiplier], cancel_token)
        
        return self._corpus.extract_by_indices(reranked[:k])
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        file_ids: list[str] | None = None,
        cancel_token: Optional[CancellationToken] = None,
    ) -> Corpus:
        """Retrieve relevant documents asynchronously."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._retrieve_sync,
                query,
                k,
                file_ids,
                cancel_token,
            )
    
    def is_indexed(self) -> bool:
        """Check if corpus is indexed."""
        return self._indexed
    
    def release(self) -> None:
        """Release all models from VRAM."""
        self._st_retriever.release()
        self._reranker.release()
    
    def clear(self) -> None:
        """Clear indexed data (keeps models loaded)."""
        self._ft_retriever.clear()
        self._st_retriever.clear()
        self._reranker.clear()
        self._corpus = None
        self._indexed = False
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
