"""LLM service - persistent in VRAM for optimal performance."""
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Generator, Optional
import threading

import torch
from llama_cpp import Llama, llama_flash_attn_type, llama_attention_type

from src.core.config import config
from src.core.cancellation import CancellationToken, CancelledException


class LLMService:
    """
    LLM service for chat completion and summarization.
    
    Model is loaded once at startup and kept in VRAM for the entire
    server lifetime to minimize latency.
    """
    
    def __init__(self, model_path: str | None = None, load_on_init: bool = True):
        self._model_path = model_path or config.models.get_llm_path()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
        self._llm: Llama | None = None
        
        if load_on_init:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load model into VRAM (called once at startup)."""
        if self._llm is not None:
            return
        
        self._llm = Llama(
            model_path=self._model_path,
            n_gpu_layers=-1,
            attention_type=llama_attention_type.LLAMA_ATTENTION_TYPE_CAUSAL,
            flash_attn_type=llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED,
            n_ctx=config.llm_context_length,
            type_k=8,
            type_v=8,
            kv_unified=True,
            n_batch=config.llm_batch_size,
            n_ubatch=config.llm_ubatch_size,
            swa_full=True,
            last_n_tokens_size=-1,
            verbose=False,
        )
    
    def _generate_sync(
        self,
        messages: list[dict],
        cancel_token: Optional[CancellationToken],
        temperature: float = 0.1,
        top_k: int = 20,
        top_p: float = 0.85,
        min_p: float = 0.05,
        repeat_penalty: float = 1.05,
    ) -> Generator[str, None, None]:
        """Synchronous streaming generation with cancellation support."""
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before generation")
        
        output = self._llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
        )
        
        for chunk in output:
            if cancel_token and cancel_token.is_cancelled:
                raise CancelledException("Cancelled during generation")
            
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
    
    async def generate_stream(
        self,
        messages: list[dict],
        cancel_token: Optional[CancellationToken] = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with cancellation support."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            
            def _safe_next(g):
                try:
                    return next(g)
                except StopIteration:
                    return None
                except CancelledException:
                    return None
            
            gen = self._generate_sync(messages, cancel_token, temperature=temperature, **kwargs)
            
            while True:
                if cancel_token and cancel_token.is_cancelled:
                    break
                
                chunk = await loop.run_in_executor(self._executor, _safe_next, gen)
                if chunk is None:
                    break
                yield chunk
    
    def _generate_complete_sync(
        self,
        messages: list[dict],
        cancel_token: Optional[CancellationToken],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Synchronous complete generation."""
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before generation")
        
        output = self._llm.create_chat_completion(
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=20,
            top_p=0.85,
            min_p=0.05,
            repeat_penalty=1.05,
        )
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled after generation")
        
        return output["choices"][0]["message"]["content"]
    
    async def generate_complete(
        self,
        messages: list[dict],
        cancel_token: Optional[CancellationToken] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate complete response (non-streaming)."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._generate_complete_sync,
                messages,
                cancel_token,
                temperature,
                max_tokens,
            )
    
    async def summarize_chunks(
        self,
        chunks: list[str],
        file_name: str,
        cancel_token: Optional[CancellationToken] = None,
    ) -> str:
        """Generate summary from document chunks."""
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before summarization")
        
        max_chunks = 80
        combined = "\n\n---\n\n".join(chunks[:max_chunks])
        if len(combined) > 12000:
            combined = combined[:12000]
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý AI chuyên tóm tắt tài liệu. "
                    "Mỗi chunk có format [Chunk N | File | ID] và [Refs] - thể hiện thứ tự và vị trí trong file gốc. "
                    "Hãy tóm tắt rõ ràng, có cấu trúc (gạch đầu dòng), bao phủ đầy đủ các phần chính."
                ),
            },
            {
                "role": "user",
                "content": f"Tóm tắt tài liệu '{file_name}':\n\n{combined}",
            },
        ]
        
        return await self.generate_complete(messages, cancel_token, temperature=0.3, max_tokens=1024)
    
    def release(self) -> None:
        """Release model from VRAM."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()
        self._executor.shutdown(wait=False)
