"""File processing service with Docling - simple and robust."""
import gc
import asyncio
import threading
from io import BytesIO
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch

from src.core.config import config
from src.core.models import Chunk, generate_file_id
from src.core.cancellation import CancellationToken, CancelledException


class FileProcessor:
    """
    Document processor using Docling.
    
    Design:
    - On-demand loading: Docling models loaded only when processing
    - ThreadPoolExecutor for non-blocking async
    - Explicit VRAM cleanup after processing
    """
    
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docling")
        self._lock = asyncio.Lock()
        self._converter = None
        self._chunker = None
        self._is_loaded = False
        self._cancel_event = threading.Event()
    
    def _load_docling(self) -> None:
        """Load Docling models (called in worker thread)."""
        if self._is_loaded:
            return
        
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
        from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
        from docling_core.transforms.chunker.hierarchical_chunker import (
            ChunkingDocSerializer,
            ChunkingSerializerProvider,
        )
        from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
        from docling.chunking import HybridChunker
        
        class MDTableSerializerProvider(ChunkingSerializerProvider):
            def get_serializer(self, doc):
                return ChunkingDocSerializer(
                    doc=doc,
                    table_serializer=MarkdownTableSerializer(),
                )
        
        pipeline_options = ThreadedPdfPipelineOptions(
            artifacts_path=str(config.artifacts_dir),
            accelerator_options=AcceleratorOptions(
                num_threads=2,
                device=AcceleratorDevice.CUDA,
                cuda_use_flash_attention2=True,
            ),
            do_ocr=False,
            layout_batch_size=8,
            table_batch_size=4,
        )
        
        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_cls=ThreadedStandardPdfPipeline,
                ),
            },
        )
        self._converter.initialize_pipeline(InputFormat.PDF)
        
        self._chunker = HybridChunker(
            serializer_provider=MDTableSerializerProvider(),
            merge_peers=True,
            always_emit_headings=True,
        )
        
        self._is_loaded = True
    
    def preload(self) -> None:
        """Preload Docling models in background thread."""
        if self._is_loaded:
            return
        self._executor.submit(self._load_docling)
    
    def _process_sync(
        self,
        file_name: str,
        file_bytes: bytes,
    ) -> tuple[list[Chunk], str]:
        """Synchronous document processing."""
        from docling_core.types.io import DocumentStream
        from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
        
        # Load models if needed
        self._load_docling()
        
        if self._cancel_event.is_set():
            raise CancelledException("Cancelled before conversion")
        
        # Generate file ID
        file_id = generate_file_id()
        
        # Convert document
        file_stream = BytesIO(file_bytes)
        source = DocumentStream(name=file_name, stream=file_stream)
        doc = self._converter.convert(source).document
        
        if self._cancel_event.is_set():
            raise CancelledException("Cancelled after conversion")
        
        # Chunk document
        raw_chunks = list(self._chunker.chunk(dl_doc=doc))
        
        if self._cancel_event.is_set():
            raise CancelledException("Cancelled after chunking")
        
        # Build Chunk objects
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            refs = [
                str(it.self_ref)
                for it in DocChunk.model_validate(chunk).meta.doc_items
            ]
            text = self._chunker.contextualize(chunk=chunk)
            
            chunks.append(Chunk(
                file_id=file_id,
                file_name=file_name,
                chunk_id=i,
                refs=refs,
                text=text,
            ))
        
        return chunks, file_id
    
    async def process(
        self,
        file_name: str,
        file_stream: BytesIO,
        cancel_token: Optional[CancellationToken] = None,
    ) -> tuple[list[Chunk], str]:
        """
        Process document asynchronously.
        
        Args:
            file_name: Original filename
            file_stream: File content as BytesIO
            cancel_token: Optional cancellation token
            
        Returns:
            Tuple of (chunks, file_id)
        """
        async with self._lock:
            if cancel_token and cancel_token.is_cancelled:
                raise CancelledException("Cancelled before processing")
            
            file_bytes = file_stream.read()
            
            # Reset cancel event
            self._cancel_event.clear()
            
            # Register callback to set cancel event
            def on_cancel():
                self._cancel_event.set()
            
            if cancel_token:
                cancel_token.register_callback(on_cancel)
            
            loop = asyncio.get_event_loop()
            
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    self._process_sync,
                    file_name,
                    file_bytes,
                )
                return result
            except CancelledException:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to process {file_name}: {e}") from e
            finally:
                if cancel_token:
                    cancel_token.unregister_callback(on_cancel)
    
    def release(self) -> None:
        """Release Docling models and free VRAM."""
        if self._converter is not None:
            del self._converter
            self._converter = None
        
        if self._chunker is not None:
            del self._chunker
            self._chunker = None
        
        self._is_loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()
        self._executor.shutdown(wait=False)

