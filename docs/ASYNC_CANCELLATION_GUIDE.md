# Hướng dẫn Async và Cancellation trong RAG Chatbot

## Mục lục
1. [Tổng quan](#1-tổng-quan)
2. [Kiến trúc Async](#2-kiến-trúc-async)
3. [Hệ thống Cancellation](#3-hệ-thống-cancellation)
4. [Server-Sent Events (SSE)](#4-server-sent-events-sse)
5. [Pattern và Best Practices](#5-pattern-và-best-practices)
6. [Code Examples](#6-code-examples)

---

## 1. Tổng quan

### 1.1 Vấn đề cần giải quyết

Khi chuyển từ synchronous sang web application, cần giải quyết:

1. **Non-blocking I/O**: Server không bị block khi xử lý tác vụ nặng (LLM inference, document processing)
2. **Cancellation**: Cho phép người dùng hủy yêu cầu đang xử lý
3. **Streaming Response**: Gửi kết quả dần dần thay vì chờ hoàn thành
4. **Resource Management**: Quản lý VRAM và đảm bảo cleanup khi hủy

### 1.2 Giải pháp tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI (Async)                          │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ SSE Stream   │────▶│ Async/Await  │────▶│ ThreadPool   │    │
│  │ (Response)   │     │ (Non-block)  │     │ (Heavy CPU)  │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│          │                    │                    │            │
│          ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CancellationToken (Thread-safe)             │   │
│  │   - Cross-thread signaling                               │   │
│  │   - Phase tracking                                       │   │
│  │   - Callback support                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Kiến trúc Async

### 2.1 Vấn đề với Synchronous Code

```python
# ❌ Synchronous - Block entire server
def process_document(file):
    chunks = docling.convert(file)      # 30-60 giây, block!
    summary = llm.generate(chunks)      # 10-20 giây, block!
    return summary
```

Khi một request đang xử lý, server không thể nhận request khác.

### 2.2 Giải pháp: Async + ThreadPoolExecutor

```python
# ✅ Async với ThreadPool - Non-blocking
class FileProcessor:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
    
    def _process_sync(self, file_bytes: bytes) -> list[Chunk]:
        """Heavy work trong worker thread."""
        return docling.convert(file_bytes)
    
    async def process(self, file_stream: BytesIO) -> list[Chunk]:
        """Async interface - không block event loop."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._process_sync,
                file_stream.read(),
            )
            return result
```

### 2.3 Tại sao cần ThreadPoolExecutor?

| Component | Tính chất | Giải pháp |
|-----------|-----------|-----------|
| LLM Inference | CPU/GPU bound, blocking C++ calls | ThreadPoolExecutor |
| Docling Processing | CPU/GPU bound, heavy computation | ThreadPoolExecutor |
| Embedding | GPU bound | ThreadPoolExecutor |
| File I/O | I/O bound | Native async hoặc ThreadPool |
| HTTP Response | I/O bound | Native async |

**Lý do**: `llama-cpp-python` và Docling là thư viện C++/CUDA, không support native async. Cần wrap trong ThreadPool để không block Python event loop.

### 2.4 Pattern: Sync wrapper với Async interface

```python
class LLMService:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
        self._llm = Llama(...)  # Load model
    
    def _generate_sync(self, messages: list[dict]) -> Generator[str, None, None]:
        """Synchronous generation - chạy trong worker thread."""
        output = self._llm.create_chat_completion(messages, stream=True)
        for chunk in output:
            yield chunk["choices"][0]["delta"].get("content", "")
    
    async def generate_stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Async streaming interface."""
        async with self._lock:  # Chỉ 1 request tại 1 thời điểm
            loop = asyncio.get_event_loop()
            
            # Helper để gọi next() trong executor
            def _safe_next(gen):
                try:
                    return next(gen)
                except StopIteration:
                    return None
            
            gen = self._generate_sync(messages)
            
            while True:
                chunk = await loop.run_in_executor(self._executor, _safe_next, gen)
                if chunk is None:
                    break
                yield chunk
```

### 2.5 Lưu ý quan trọng

1. **Single worker**: `max_workers=1` vì LLM chỉ xử lý 1 request tại 1 thời điểm
2. **Async Lock**: Đảm bảo không có race condition khi access shared resources
3. **run_in_executor**: Bridge giữa sync code và async event loop

---

## 3. Hệ thống Cancellation

### 3.1 Thiết kế CancellationToken

```python
@dataclass
class CancellationToken:
    """Thread-safe cancellation token."""
    
    _cancelled: bool = field(default=False, repr=False)
    _phase: ProcessingPhase = field(default=ProcessingPhase.IDLE, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _callbacks: list[Callable[[], None]] = field(default_factory=list, repr=False)
    request_id: str = ""
```

### 3.2 Các thành phần chính

#### 3.2.1 ProcessingPhase - Theo dõi phase hiện tại

```python
class ProcessingPhase(Enum):
    IDLE = auto()              # Không xử lý
    FILE_EXTRACTION = auto()   # Docling đang xử lý file
    SUMMARIZATION = auto()     # LLM đang tóm tắt
    ANALYSIS = auto()          # Phân tích ngữ nghĩa
    RETRIEVAL = auto()         # Embedding/Reranking
    GENERATION = auto()        # LLM đang sinh phản hồi
```

**Tại sao cần phase?**
- Biết đang xử lý ở đâu để cleanup đúng cách
- Log/Debug dễ hơn
- UI có thể hiển thị trạng thái chi tiết

#### 3.2.2 Thread-safe Properties

```python
@property
def is_cancelled(self) -> bool:
    """Check cancellation status - thread-safe."""
    with self._lock:
        return self._cancelled

def cancel(self) -> None:
    """Request cancellation - thread-safe."""
    with self._lock:
        if self._cancelled:
            return
        self._cancelled = True
        callbacks = list(self._callbacks)  # Copy để tránh deadlock
    
    # Invoke callbacks NGOÀI lock
    for callback in callbacks:
        try:
            callback()
        except Exception:
            pass
```

**Lưu ý**: Callbacks được gọi NGOÀI lock để tránh deadlock khi callback cần acquire lock khác.

#### 3.2.3 Callback System

```python
def register_callback(self, callback: Callable[[], None]) -> None:
    """Đăng ký callback được gọi khi cancel."""
    with self._lock:
        if self._cancelled:
            callback()  # Đã cancel rồi, gọi ngay
        else:
            self._callbacks.append(callback)

def unregister_callback(self, callback: Callable[[], None]) -> None:
    """Hủy đăng ký callback."""
    with self._lock:
        if callback in self._callbacks:
            self._callbacks.remove(callback)
```

### 3.3 RequestManager - Quản lý Request đơn

```python
class RequestManager:
    """Đảm bảo chỉ 1 request active tại 1 thời điểm."""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_token: Optional[CancellationToken] = None
        self._request_counter = 0
    
    async def start_request(self) -> CancellationToken:
        """Bắt đầu request mới, hủy request cũ nếu có."""
        async with self._lock:
            # Hủy request đang chạy
            if self._current_token is not None:
                self._current_token.cancel()
            
            # Tạo token mới
            self._request_counter += 1
            token = CancellationToken(request_id=f"req-{self._request_counter}")
            self._current_token = token
            return token
    
    async def cancel_current(self) -> bool:
        """Hủy request hiện tại."""
        async with self._lock:
            if self._current_token is not None:
                self._current_token.cancel()
                return True
            return False
```

### 3.4 Sử dụng trong Code

#### 3.4.1 Check định kỳ (Polling)

```python
async def process_message(self, message: str, cancel_token: CancellationToken):
    # Check trước mỗi phase
    cancel_token.set_phase(ProcessingPhase.ANALYSIS)
    cancel_token.check_cancelled()  # Raise CancelledException nếu đã cancel
    
    analysis = await self._analyzer.analyze(...)
    
    # Check sau mỗi operation quan trọng
    cancel_token.check_cancelled()
    
    cancel_token.set_phase(ProcessingPhase.RETRIEVAL)
    results = await self._retriever.retrieve(...)
```

#### 3.4.2 Callback cho Long-running Operations

```python
async def process(self, file_name: str, file_stream: BytesIO, 
                  cancel_token: CancellationToken):
    """Process file với cancellation support."""
    
    # threading.Event để signal vào worker thread
    cancel_event = threading.Event()
    
    def on_cancel():
        """Callback được gọi khi cancel."""
        cancel_event.set()
    
    # Đăng ký callback
    cancel_token.register_callback(on_cancel)
    
    try:
        # Trong worker thread, check cancel_event
        result = await loop.run_in_executor(
            self._executor,
            self._process_sync,  # Hàm này check cancel_event.is_set()
            file_name,
            file_bytes,
        )
        return result
    finally:
        # Luôn unregister callback
        cancel_token.unregister_callback(on_cancel)
```

#### 3.4.3 Trong Sync Worker Function

```python
def _process_sync(self, file_name: str, file_bytes: bytes):
    """Sync function chạy trong worker thread."""
    
    # Check trước operation nặng
    if self._cancel_event.is_set():
        raise CancelledException("Cancelled before conversion")
    
    doc = self._converter.convert(source).document
    
    # Check sau operation nặng
    if self._cancel_event.is_set():
        raise CancelledException("Cancelled after conversion")
    
    chunks = list(self._chunker.chunk(doc))
    
    if self._cancel_event.is_set():
        raise CancelledException("Cancelled after chunking")
    
    return chunks
```

### 3.5 Pattern: Bridge giữa Async và Thread

```
┌─────────────────────────────────────────────────────────────────┐
│                     Async Context (Event Loop)                  │
│                                                                 │
│   CancellationToken                                             │
│        │                                                        │
│        │ register_callback(on_cancel)                           │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │ on_cancel() │──────────▶ threading.Event.set()              │
│   └─────────────┘                      │                        │
│                                        │                        │
└────────────────────────────────────────│────────────────────────┘
                                         │
┌────────────────────────────────────────│────────────────────────┐
│                     Worker Thread                               │
│                                        │                        │
│                                        ▼                        │
│                            threading.Event.is_set()             │
│                                        │                        │
│                                        ▼                        │
│                            if True: raise CancelledException    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Server-Sent Events (SSE)

### 4.1 Tại sao dùng SSE?

| Feature | SSE | WebSocket | Long Polling |
|---------|-----|-----------|--------------|
| Unidirectional | ✅ Server→Client | Bidirectional | Bidirectional |
| Auto-reconnect | ✅ Built-in | Manual | Manual |
| HTTP/2 support | ✅ | Limited | ✅ |
| Simplicity | ✅ Simple | Complex | Simple |
| Streaming | ✅ Native | ✅ | ❌ |

**Lý do chọn SSE**: Chatbot chỉ cần server gửi data xuống client (streaming response), không cần bidirectional.

### 4.2 Format SSE

```
data: {"status": "Đang xử lý..."}\n\n
data: {"content": "Xin "}\n\n
data: {"content": "chào"}\n\n
data: {"done": true}\n\n
```

Mỗi event:
- Bắt đầu với `data: `
- Kết thúc với `\n\n` (hai newlines)
- Content là JSON

### 4.3 Implementation trong FastAPI

```python
@router.post("/chat")
async def chat(message: str = Form(...), file: Optional[UploadFile] = File(None)):
    """Chat endpoint với SSE streaming."""
    
    async def event_stream():
        """Generator function yield SSE events."""
        cancel_token = await state.start_request()
        
        try:
            # Yield status updates
            yield f"data: {json.dumps({'status': 'Đang xử lý...'})}\n\n"
            
            # Process và yield results
            async for event in state.chat_service.process_message(message, cancel_token):
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
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
```

### 4.4 Client-side JavaScript

```javascript
async function sendMessage(message, file) {
    const formData = new FormData();
    formData.append('message', message);
    if (file) formData.append('file', file);
    
    const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData,
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                if (data.status) {
                    showStatus(data.status);
                } else if (data.content) {
                    appendContent(data.content);
                } else if (data.done) {
                    finishMessage();
                } else if (data.cancelled) {
                    showCancelled();
                } else if (data.error) {
                    showError(data.error);
                }
            }
        }
    }
}
```

### 4.5 Stop Button Implementation

```python
@router.post("/stop")
async def stop_processing():
    """Endpoint để hủy request đang xử lý."""
    cancelled = await state.cancel_current()
    state.cleanup_file_processor()  # Release Docling VRAM
    return JSONResponse({
        "status": "ok",
        "cancelled": cancelled,
    })
```

```javascript
async function stopProcessing() {
    await fetch('/api/stop', { method: 'POST' });
}
```

---

## 5. Pattern và Best Practices

### 5.1 Resource Cleanup Pattern

```python
try:
    # Setup
    cancel_token.register_callback(on_cancel)
    
    # Processing
    result = await process(...)
    
except CancelledException:
    # Cleanup partial work
    cleanup_partial_data()
    raise
    
finally:
    # Always cleanup
    cancel_token.unregister_callback(on_cancel)
```

### 5.2 Phased Processing Pattern

```python
async def process_message(self, message: str, cancel_token: CancellationToken):
    """Process với phase tracking."""
    
    # Phase 1: Analysis
    cancel_token.set_phase(ProcessingPhase.ANALYSIS)
    cancel_token.check_cancelled()
    yield {"status": "Đang phân tích..."}
    
    analysis = await self._analyzer.analyze(message, cancel_token)
    
    # Phase 2: Retrieval
    cancel_token.set_phase(ProcessingPhase.RETRIEVAL)
    cancel_token.check_cancelled()
    yield {"status": "Đang tìm kiếm..."}
    
    context = await self._retriever.retrieve(query, cancel_token)
    
    # Phase 3: Generation
    cancel_token.set_phase(ProcessingPhase.GENERATION)
    cancel_token.check_cancelled()
    yield {"status": "Đang tạo phản hồi..."}
    
    async for chunk in self._llm.generate_stream(messages, cancel_token):
        yield {"content": chunk}
```

### 5.3 Partial Save Pattern

```python
async def process_message(self, message: str, cancel_token: CancellationToken):
    full_response = ""
    
    try:
        async for chunk in self._llm.generate_stream(messages, cancel_token):
            full_response += chunk
            yield {"content": chunk}
        
        # Save complete response
        self._history.add_turn(message, full_response)
        yield {"done": True}
        
    except CancelledException:
        # Save partial response với marker
        if full_response:
            self._history.add_turn(message, full_response + "\n[Đã dừng]")
        yield {"cancelled": True}
```

### 5.4 Lock Hierarchy

```python
# Đúng: Acquire locks theo thứ tự nhất quán
async with self._processing_lock:  # Level 1
    async with self._llm._lock:     # Level 2
        await self._llm.generate(...)

# Sai: Có thể deadlock
async with self._llm._lock:         # Level 2 trước
    async with self._processing_lock:  # Level 1 sau - DEADLOCK!
        ...
```

---

## 6. Code Examples

### 6.1 Complete Cancellable Service

```python
class CancellableService:
    """Template cho service hỗ trợ cancellation."""
    
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
        self._cancel_event = threading.Event()
    
    def _heavy_work_sync(self, data: bytes, cancel_event: threading.Event) -> Result:
        """Sync heavy work với cancellation checkpoints."""
        
        # Checkpoint 1
        if cancel_event.is_set():
            raise CancelledException("Cancelled at checkpoint 1")
        
        intermediate = process_step_1(data)
        
        # Checkpoint 2
        if cancel_event.is_set():
            raise CancelledException("Cancelled at checkpoint 2")
        
        result = process_step_2(intermediate)
        
        return result
    
    async def do_work(
        self, 
        data: bytes, 
        cancel_token: Optional[CancellationToken] = None
    ) -> Result:
        """Async interface với cancellation support."""
        
        async with self._lock:
            if cancel_token and cancel_token.is_cancelled:
                raise CancelledException("Cancelled before work")
            
            # Reset event
            cancel_event = threading.Event()
            
            # Bridge: async cancel → thread event
            def on_cancel():
                cancel_event.set()
            
            if cancel_token:
                cancel_token.register_callback(on_cancel)
            
            loop = asyncio.get_event_loop()
            
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    self._heavy_work_sync,
                    data,
                    cancel_event,
                )
                return result
                
            except CancelledException:
                raise
                
            finally:
                if cancel_token:
                    cancel_token.unregister_callback(on_cancel)
```

### 6.2 SSE Event Types

```python
# Các loại event trong hệ thống
EVENT_TYPES = {
    "status": {"status": "message"},           # Trạng thái xử lý
    "content": {"content": "text chunk"},      # Nội dung streaming
    "done": {"done": True},                    # Hoàn thành
    "cancelled": {"cancelled": True},          # Đã hủy
    "error": {"error": "error message"},       # Lỗi
}

# Generator pattern
async def event_stream():
    try:
        yield f"data: {json.dumps({'status': 'Processing...'})}\n\n"
        
        async for chunk in process():
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except CancelledException:
        yield f"data: {json.dumps({'cancelled': True})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
```

### 6.3 Full Request Lifecycle

```python
@router.post("/chat")
async def chat(message: str = Form(...)):
    """Complete request lifecycle với cancellation."""
    
    async def event_stream():
        # 1. Start request - cancel existing if any
        cancel_token = await state.start_request()
        
        try:
            # 2. Phase: File Processing (if applicable)
            if file:
                cancel_token.set_phase(ProcessingPhase.FILE_EXTRACTION)
                yield f"data: {json.dumps({'status': 'Đang xử lý file...'})}\n\n"
                
                chunks, file_id = await state.file_processor.process(
                    file.filename, file_stream, cancel_token
                )
                
                # 3. Phase: Summarization
                cancel_token.set_phase(ProcessingPhase.SUMMARIZATION)
                yield f"data: {json.dumps({'status': 'Đang tóm tắt...'})}\n\n"
                
                summary = await state.llm_service.summarize_chunks(
                    chunks, file.filename, cancel_token
                )
            
            # 4. Phase: Analysis
            cancel_token.set_phase(ProcessingPhase.ANALYSIS)
            yield f"data: {json.dumps({'status': 'Đang phân tích...'})}\n\n"
            
            analysis = await analyzer.analyze(message, cancel_token)
            
            # 5. Phase: Retrieval
            cancel_token.set_phase(ProcessingPhase.RETRIEVAL)
            yield f"data: {json.dumps({'status': 'Đang tìm kiếm...'})}\n\n"
            
            context = await retriever.retrieve(query, cancel_token)
            
            # 6. Phase: Generation
            cancel_token.set_phase(ProcessingPhase.GENERATION)
            yield f"data: {json.dumps({'status': 'Đang tạo phản hồi...'})}\n\n"
            
            async for chunk in llm.generate_stream(messages, cancel_token):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except CancelledException:
            yield f"data: {json.dumps({'cancelled': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
        finally:
            # 7. Cleanup
            await state.finish_request(cancel_token)
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## Tổng kết

### Key Takeaways

1. **Async + ThreadPool**: Dùng ThreadPoolExecutor cho sync libraries (llama-cpp, Docling)

2. **CancellationToken**: Thread-safe, callback-based, phase-aware

3. **Bridge Pattern**: `threading.Event` để signal từ async context vào worker thread

4. **SSE Streaming**: Simple, effective cho unidirectional streaming

5. **Cleanup**: Always use `try/finally` để đảm bảo cleanup

### Files chính

| File | Vai trò |
|------|---------|
| `src/core/cancellation.py` | CancellationToken, RequestManager |
| `src/services/file_processor.py` | FileProcessor với ThreadPool + callback |
| `src/services/llm.py` | LLMService với streaming + cancellation |
| `src/services/retrieval.py` | HybridRetriever với async interface |
| `src/services/chat.py` | ChatService orchestration với phases |
| `src/api/routes.py` | SSE endpoints + request lifecycle |
