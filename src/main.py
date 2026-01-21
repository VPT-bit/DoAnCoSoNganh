"""
AI Chatbot - Main Application Entry Point

A modular RAG-based chatbot using:
- Docling for document processing
- llama-cpp-python for LLM, embedding, and reranking
- Hybrid search (BM25 + semantic) with RRF fusion
"""
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import config
from src.core.models import Corpus
from src.services import (
    FileProcessor,
    HybridRetriever,
    LLMService,
    SemanticAnalyzer,
    ChatService,
)
from src.api.routes import create_router, AppState


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Initialize services - models loaded once and kept in VRAM
    print("[INIT] Loading corpus...")
    corpus = Corpus.load(config.corpus_file)
    print(f"[INIT] Loaded {len(corpus.chunks)} chunks, {len(corpus.summaries)} documents")
    
    print("[INIT] Initializing LLM service (loading model into VRAM)...")
    llm_service = LLMService(load_on_init=True)
    
    print("[INIT] Initializing file processor...")
    file_processor = FileProcessor()  # On-demand loading
    
    print("[INIT] Initializing retriever (loading embedding + reranker into VRAM)...")
    retriever = HybridRetriever(load_on_init=True)
    
    print("[INIT] Initializing semantic analyzer...")
    analyzer = SemanticAnalyzer(llm_service)
    
    print("[INIT] Initializing chat service...")
    chat_service = ChatService(
        llm_service=llm_service,
        retriever=retriever,
        analyzer=analyzer,
        corpus=corpus,
    )
    
    # Create application state
    state = AppState(
        corpus=corpus,
        chat_service=chat_service,
        file_processor=file_processor,
        retriever=retriever,
        llm_service=llm_service,
        corpus_file=config.corpus_file,
    )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan event handler for startup/shutdown."""
        # Startup - preload Docling in background
        print("[INIT] Preloading Docling models in background...")
        file_processor.preload()
        
        if not corpus.is_empty():
            print("[INIT] Indexing existing corpus...")
            await retriever.process(corpus)
            print("[INIT] Indexing complete")
        print("[READY] Application started successfully")
        
        yield  # App is running
        
        # Shutdown - release all VRAM
        print("[SHUTDOWN] Releasing resources...")
        file_processor.release()
        retriever.release()
        llm_service.release()
        print("[SHUTDOWN] Done")
    
    app = FastAPI(
        title="AI Chatbot",
        description="RAG-based chatbot with hybrid search",
        version="2.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Include API routes
    app.include_router(create_router(state))
    
    # Serve index.html
    @app.get("/")
    async def serve_index():
        template_path = Path(__file__).parent / "templates" / "index.html"
        return FileResponse(str(template_path))
    
    return app


# Lazy initialization - only create app when needed
app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get or create the FastAPI application."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    print("=" * 50)
    print("AI Chatbot Server")
    print("=" * 50)
    print(f"[INFO] Starting server on http://{config.host}:{config.port}")
    print(f"[INFO] UI available at: http://localhost:8000")
    print("=" * 50)
    
    # Create app directly and run - avoid double initialization
    application = get_app()
    uvicorn.run(
        application,
        host=config.host,
        port=config.port,
    )