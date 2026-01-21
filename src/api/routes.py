"""API routes for the chatbot with cancellation support."""
import json
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

from src.core.config import config
from src.core.models import Corpus, FileSummary
from src.core.cancellation import (
    RequestManager,
    CancellationToken,
    CancelledException,
    ProcessingPhase,
)
from src.services.chat import ChatService
from src.services.file_processor import FileProcessor
from src.services.retrieval import HybridRetriever
from src.services.llm import LLMService


class AppState:
    """Application state holder with request management."""
    
    def __init__(
        self,
        corpus: Corpus,
        chat_service: ChatService,
        file_processor: FileProcessor,
        retriever: HybridRetriever,
        llm_service: LLMService,
        corpus_file: str,
    ):
        self.corpus = corpus
        self.chat_service = chat_service
        self.file_processor = file_processor
        self.retriever = retriever
        self.llm_service = llm_service
        self.corpus_file = corpus_file
        self._processing_lock = asyncio.Lock()
        self._request_manager = RequestManager()
    
    async def save_corpus(self) -> None:
        """Save corpus to file."""
        self.corpus.save(self.corpus_file)
    
    async def reload_retriever(self, cancel_token: Optional[CancellationToken] = None) -> None:
        """Reload retriever with current corpus."""
        await self.retriever.process(self.corpus, cancel_token)
    
    async def start_request(self) -> CancellationToken:
        """Start a new request (cancels any existing one)."""
        return await self._request_manager.start_request()
    
    async def cancel_current(self) -> bool:
        """Cancel current request."""
        return await self._request_manager.cancel_current()
    
    async def finish_request(self, token: CancellationToken) -> None:
        """Mark request as finished."""
        await self._request_manager.finish_request(token)
    
    def cleanup_file_processor(self) -> None:
        """Force cleanup file processor (releases Docling VRAM)."""
        self.file_processor.release()


def create_router(state: AppState) -> APIRouter:
    """Create API router with application state."""
    
    router = APIRouter(prefix="/api")
    
    @router.post("/chat")
    async def chat(
        message: str = Form(...),
        file: Optional[UploadFile] = File(None),
    ):
        """
        Handle chat message with optional file upload.
        Returns Server-Sent Events stream.
        Supports cancellation at any processing phase.
        """
        async def event_stream():
            # Start new request (cancels any existing)
            cancel_token = await state.start_request()
            
            try:
                file_summary = None
                
                # Handle file upload if present
                if file and file.filename:
                    async with state._processing_lock:
                        try:
                            # Phase: File extraction
                            cancel_token.set_phase(ProcessingPhase.FILE_EXTRACTION)
                            cancel_token.check_cancelled()
                            
                            yield f"data: {json.dumps({'status': 'Đang xử lý file...'})}\n\n"
                            
                            # Read file content
                            content = await file.read()
                            file_stream = BytesIO(content)
                            
                            yield f"data: {json.dumps({'status': 'Đang phân tích tài liệu...'})}\n\n"
                            
                            # Process file with cancellation support
                            chunks, file_id = await state.file_processor.process(
                                file.filename,
                                file_stream,
                                cancel_token,
                            )
                            
                            # Check cancellation before saving
                            cancel_token.check_cancelled()
                            
                            # Save original file
                            uploads_dir = config.uploads_dir
                            uploads_dir.mkdir(parents=True, exist_ok=True)
                            safe_filename = f"{file_id}_{file.filename}"
                            file_path = uploads_dir / safe_filename
                            with open(file_path, "wb") as f:
                                f.write(content)
                            
                            # Add chunks to corpus
                            for chunk in chunks:
                                state.corpus.add_chunk(chunk)
                            
                            # Phase: Summarization
                            cancel_token.set_phase(ProcessingPhase.SUMMARIZATION)
                            cancel_token.check_cancelled()
                            
                            yield f"data: {json.dumps({'status': 'Đang tóm tắt tài liệu...'})}\n\n"
                            
                            formatted_chunks = [c.format_for_llm() for c in chunks]
                            summary = await state.llm_service.summarize_chunks(
                                formatted_chunks,
                                file.filename,
                                cancel_token,
                            )
                            
                            # Check cancellation before finalizing
                            cancel_token.check_cancelled()
                            
                            # Create file summary
                            file_summary = FileSummary(
                                file_id=file_id,
                                file_name=file.filename,
                                summary=summary,
                                chunk_count=len(chunks),
                            )
                            
                            # Add summary to corpus
                            state.corpus.add_summary(file_summary)
                            
                            # Save and reindex
                            yield f"data: {json.dumps({'status': 'Đang lập chỉ mục...'})}\n\n"
                            await state.save_corpus()
                            await state.reload_retriever(cancel_token)
                            
                            yield f"data: {json.dumps({'status': f'Đã xử lý {file.filename} ({len(chunks)} chunks)'})}\n\n"
                            await asyncio.sleep(0.3)
                            
                        except CancelledException:
                            # Cleanup partial data on cancellation
                            if 'file_id' in dir():
                                state.corpus.remove_file(file_id)
                            yield f"data: {json.dumps({'cancelled': True, 'status': 'Đã hủy xử lý file'})}\n\n"
                            return
                
                # Build user message with file info if applicable
                if file_summary:
                    user_message = ChatService.build_user_message_with_file(message, file_summary)
                else:
                    user_message = message
                
                # Process chat with cancellation support
                async for event in state.chat_service.process_message(user_message, cancel_token):
                    yield f"data: {json.dumps(event)}\n\n"
                    
            except CancelledException:
                yield f"data: {json.dumps({'cancelled': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
            finally:
                await state.finish_request(cancel_token)
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    @router.post("/stop")
    async def stop_processing():
        """Stop current processing request."""
        cancelled = await state.cancel_current()
        # Only cleanup file processor (Docling) - LLM/Embedding/Reranker stay loaded
        state.cleanup_file_processor()
        return JSONResponse({
            "status": "ok",
            "cancelled": cancelled,
        })
    
    @router.get("/chat-history")
    async def get_chat_history():
        """Get chat history with RAG context."""
        return JSONResponse({
            "history": state.chat_service.get_history()
        })
    
    @router.post("/clear-chat")
    async def clear_chat():
        """Clear chat history and corpus."""
        # Cancel any ongoing request
        await state.cancel_current()
        
        state.chat_service.clear_history()
        
        # Delete all uploaded files
        uploads_dir = config.uploads_dir
        if uploads_dir.exists():
            for file_path in uploads_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        
        state.corpus.chunks.clear()
        state.corpus.summaries.clear()
        await state.save_corpus()
        state.retriever.clear()
        
        # Only cleanup file processor
        state.cleanup_file_processor()
        
        return JSONResponse({"status": "ok"})
    
    @router.get("/documents")
    async def list_documents():
        """List all processed documents."""
        return JSONResponse({
            "documents": state.corpus.get_file_list()
        })
    
    @router.delete("/documents/{file_id}")
    async def delete_document(file_id: str):
        """Delete a document by file ID."""
        # Get file info before deleting from corpus
        summary = state.corpus.get_summary_by_file_id(file_id)
        
        # Remove from corpus
        state.corpus.remove_file(file_id)
        await state.save_corpus()
        await state.reload_retriever()
        
        # Delete uploaded file if exists
        if summary:
            uploads_dir = config.uploads_dir
            file_path = uploads_dir / f"{file_id}_{summary.file_name}"
            if file_path.exists():
                file_path.unlink()
        
        return JSONResponse({"status": "ok"})
    
    @router.get("/documents/{file_id}/download")
    async def download_document(file_id: str):
        """Download original document."""
        summary = state.corpus.get_summary_by_file_id(file_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_path = config.uploads_dir / f"{file_id}_{summary.file_name}"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            str(file_path),
            filename=summary.file_name,
            media_type="application/octet-stream",
        )
    
    return router
