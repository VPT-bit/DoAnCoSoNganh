"""Cancellation token and request context management."""
import asyncio
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


class ProcessingPhase(Enum):
    """Current processing phase."""
    IDLE = auto()
    FILE_EXTRACTION = auto()  # Docling processing
    SUMMARIZATION = auto()     # LLM summarization
    ANALYSIS = auto()          # Query analysis
    RETRIEVAL = auto()         # RAG retrieval (embedding/reranking)
    GENERATION = auto()        # LLM response generation


@dataclass
class CancellationToken:
    """
    Thread-safe cancellation token for coordinating request cancellation.
    
    Features:
    - Cross-thread cancellation signaling
    - Phase tracking for cleanup coordination
    - Callback support for immediate cleanup
    """
    _cancelled: bool = field(default=False, repr=False)
    _phase: ProcessingPhase = field(default=ProcessingPhase.IDLE, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _callbacks: list[Callable[[], None]] = field(default_factory=list, repr=False)
    request_id: str = ""
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        with self._lock:
            return self._cancelled
    
    @property
    def phase(self) -> ProcessingPhase:
        """Get current processing phase."""
        with self._lock:
            return self._phase
    
    def cancel(self) -> None:
        """Request cancellation and invoke callbacks."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            callbacks = list(self._callbacks)
        
        # Invoke callbacks outside lock to avoid deadlock
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass  # Ignore callback errors
    
    def set_phase(self, phase: ProcessingPhase) -> None:
        """Update current processing phase."""
        with self._lock:
            self._phase = phase
    
    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called on cancellation."""
        with self._lock:
            if self._cancelled:
                # Already cancelled, invoke immediately
                callback()
            else:
                self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[], None]) -> None:
        """Unregister a cancellation callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def check_cancelled(self) -> None:
        """Raise CancelledException if cancelled."""
        if self.is_cancelled:
            raise CancelledException(f"Request cancelled in phase: {self.phase}")
    
    def reset(self) -> None:
        """Reset token for reuse."""
        with self._lock:
            self._cancelled = False
            self._phase = ProcessingPhase.IDLE
            self._callbacks.clear()


class CancelledException(Exception):
    """Raised when operation is cancelled."""
    pass


class RequestManager:
    """
    Manages active requests and their cancellation tokens.
    Ensures only one request is active at a time.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_token: Optional[CancellationToken] = None
        self._request_counter = 0
    
    async def start_request(self) -> CancellationToken:
        """
        Start a new request, cancelling any existing one.
        
        Returns:
            New CancellationToken for the request
        """
        async with self._lock:
            # Cancel existing request if any
            if self._current_token is not None:
                self._current_token.cancel()
            
            # Create new token
            self._request_counter += 1
            token = CancellationToken(request_id=f"req-{self._request_counter}")
            self._current_token = token
            return token
    
    async def cancel_current(self) -> bool:
        """
        Cancel the current request.
        
        Returns:
            True if a request was cancelled, False if no active request
        """
        async with self._lock:
            if self._current_token is not None:
                self._current_token.cancel()
                return True
            return False
    
    async def finish_request(self, token: CancellationToken) -> None:
        """Mark a request as finished."""
        async with self._lock:
            if self._current_token is token:
                self._current_token = None
    
    @property
    def is_active(self) -> bool:
        """Check if there's an active request."""
        return self._current_token is not None and not self._current_token.is_cancelled
