"""Chat service with RAG context tracking and cancellation support."""
import json
from pathlib import Path
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field, asdict

from src.core.config import config
from src.core.models import Corpus, FileSummary
from src.core.cancellation import CancellationToken, CancelledException, ProcessingPhase
from src.services.llm import LLMService
from src.services.retrieval import HybridRetriever
from src.services.semantic_analyzer import SemanticAnalyzer, QueryIntent, AnalysisResult


@dataclass
class ChatMessage:
    """
    A single chat message.
    
    Note: Only stores conversation content, not full file/chunk data.
    Query Corpus for file data when needed.
    """
    role: str  # "user", "assistant", "system"
    content: str
    rag_context: Optional[dict] = None  # RAG metadata (references only)
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {"role": self.role, "content": self.content}
        if self.rag_context:
            result["rag_context"] = self.rag_context
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """Create from dict."""
        return cls(
            role=data["role"],
            content=data["content"],
            rag_context=data.get("rag_context"),
        )


@dataclass 
class RAGContext:
    """
    RAG retrieval context metadata.
    
    Stores references to data (file_ids, chunk_ids) rather than data itself.
    Query Corpus with these references to get actual data.
    """
    intent: str
    reasoning: str
    search_query: Optional[str] = None
    summary_file_ids: list[str] = field(default_factory=list)
    chunk_file_ids: list[str] = field(default_factory=list)
    chunks_retrieved: int = 0
    chunk_refs: list[dict] = field(default_factory=list)  # {chunk_id, file_id, rank}
    
    def to_dict(self) -> dict:
        return asdict(self)


class ChatHistory:
    """
    Manages chat history with RAG reference tracking.
    
    Design principles:
    - Stores conversation content only
    - RAG context stores references (file_ids, chunk_ids), not actual data
    - Query Corpus to get actual file/chunk data when needed
    - System message contains only instructions
    - File info embedded in user messages with [FILE] tag
    """
    
    SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh hỗ trợ xử lý và tìm kiếm tài liệu.

## Khi người dùng gửi file (có tag [FILE])
- Tag [FILE] chứa: ID, Tên file, Số chunks
- Xác nhận đã nhận file và sẵn sàng hỗ trợ
- Chờ câu hỏi cụ thể từ người dùng

## Khi có ngữ cảnh tài liệu [Ngữ cảnh từ tài liệu]
- Đây là dữ liệu được trích xuất từ file để trả lời câu hỏi
- Dựa vào ngữ cảnh này để trả lời chính xác
- Trích dẫn nguồn khi cần

## Định dạng ngữ cảnh
- [SUMMARY: file_name | ID: id] - Tóm tắt file
- [Chunk N | File: name | ID: id] - Đoạn trích cụ thể

## Nguyên tắc
- Trả lời ngắn gọn, đúng trọng tâm
- Nếu không có thông tin → Nói không biết
- Câu hỏi không liên quan file → Trả lời như AI thông thường"""
    
    def __init__(self, filepath: str | None = None):
        self._filepath = filepath or config.chat_history_file
        self._messages: list[ChatMessage] = []
        self._load()
    
    def _load(self) -> None:
        """Load history from file."""
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._messages = [
                    ChatMessage.from_dict(m)
                    for m in data.get("messages", [])
                ]
        except (FileNotFoundError, json.JSONDecodeError):
            self._messages = []
        
        self._ensure_system_prompt()
        self._ensure_valid_alternation()
    
    def _ensure_system_prompt(self) -> None:
        """Ensure first message is system prompt."""
        if not self._messages or self._messages[0].role != "system":
            self._messages.insert(0, ChatMessage("system", self.SYSTEM_PROMPT))
        else:
            # Update to latest system prompt
            self._messages[0] = ChatMessage("system", self.SYSTEM_PROMPT)
    
    def _ensure_valid_alternation(self) -> None:
        """Remove trailing user message without response."""
        while len(self._messages) > 1 and self._messages[-1].role == "user":
            self._messages.pop()
        self._save()
    
    def _save(self) -> None:
        """Save history to file."""
        Path(self._filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"messages": [m.to_dict() for m in self._messages]},
                f,
                ensure_ascii=False,
                indent=2,
            )
    
    def add_turn(
        self,
        user_content: str,
        assistant_content: str,
        rag_context: Optional[RAGContext] = None,
    ) -> None:
        """Add a complete user-assistant turn with optional RAG context."""
        self._messages.append(ChatMessage("user", user_content))
        self._messages.append(ChatMessage(
            "assistant",
            assistant_content,
            rag_context=rag_context.to_dict() if rag_context else None,
        ))
        self._save()
    
    def get_messages(self) -> list[dict]:
        """Get all messages as dicts (for API response)."""
        return [m.to_dict() for m in self._messages]
    
    def get_system_prompt(self) -> str:
        """Get current system prompt."""
        return self._messages[0].content if self._messages else self.SYSTEM_PROMPT
    
    def get_recent_turns(self, n_turns: int = 3) -> list[dict]:
        """Get recent n complete turns (user-assistant pairs)."""
        conversation = self._messages[1:]  # Skip system
        n_messages = n_turns * 2
        recent = conversation[-n_messages:] if len(conversation) >= n_messages else conversation
        # Return without rag_context for LLM input
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def get_all_conversation(self) -> list[dict]:
        """Get all conversation messages (excluding system)."""
        return [{"role": m.role, "content": m.content} for m in self._messages[1:]]
    
    def clear(self) -> None:
        """Clear all history."""
        self._messages = [ChatMessage("system", self.SYSTEM_PROMPT)]
        self._save()


class ChatService:
    """
    Main chat service orchestrating RAG pipeline.
    
    Architecture:
    - SemanticAnalyzer: Uses corpus summaries + history for intelligent decisions
    - Corpus: True database - all file data queried from here
    - History: Stores conversation + references only
    
    Message Structure:
    - System: Instructions only
    - User: File info (if uploaded) + context (if retrieved) + user input
    """

    def __init__(
        self,
        llm_service: LLMService,
        retriever: HybridRetriever,
        analyzer: SemanticAnalyzer,
        corpus: Corpus,
    ):
        self._llm = llm_service
        self._retriever = retriever
        self._analyzer = analyzer
        self._corpus = corpus
        self._history = ChatHistory()
    
    def get_history(self) -> list[dict]:
        """Get chat history with RAG context references."""
        return self._history.get_messages()
    
    def clear_history(self) -> None:
        """Clear chat history."""
        self._history.clear()
    
    @staticmethod
    def build_user_message_with_file(message: str, file_summary: FileSummary) -> str:
        """
        Build user message with embedded file info.
        
        Note: Only include basic file info (ID, name, chunks).
        Summary is stored in Corpus and queried when needed.
        """
        return f"""[FILE]
- ID: {file_summary.file_id}
- Tên: {file_summary.file_name}
- Chunks: {file_summary.chunk_count}
[/FILE]

{message}"""
    
    def _build_messages(self, user_message: str, context: str | None = None) -> list[dict]:
        """
        Build message list for LLM.
        
        Structure:
        - System: Instructions only
        - Recent history: Last 3 turns
        - Current user: Context (if any) + user question
        """
        messages = [{"role": "system", "content": self._history.get_system_prompt()}]
        messages.extend(self._history.get_recent_turns(3))
        
        if context:
            user_content = f"[Ngữ cảnh từ tài liệu]\n{context}\n\n[Câu hỏi]\n{user_message}"
        else:
            user_content = user_message
        
        messages.append({"role": "user", "content": user_content})
        return messages
    
    def _get_summaries_for_files(self, file_ids: list[str]) -> str:
        """Get formatted summaries for specific files from Corpus."""
        parts = []
        for s in self._corpus.summaries:
            if s.file_id in file_ids:
                parts.append(f"[SUMMARY: {s.file_name} | ID: {s.file_id}]\n{s.summary}")
        return "\n\n---\n\n".join(parts) if parts else ""
    
    def _build_history_context_for_analyzer(self) -> str:
        """Build recent conversation context for semantic analysis."""
        conversation = self._history.get_all_conversation()
        if not conversation:
            return ""
        # Take last 6 messages, truncate long ones
        return "\n".join(
            f"[{m['role']}]: {m['content'][:500]}" 
            for m in conversation[-6:]
        )
    
    async def _retrieve_chunks(
        self,
        query: str,
        file_ids: list[str],
        cancel_token: Optional[CancellationToken],
    ) -> tuple[str, list[dict]]:
        """
        Retrieve chunks from specific files.
        
        Returns:
            Tuple of (formatted_context, chunk_references)
        """
        results = await self._retriever.retrieve(
            query,
            k=config.retrieval_top_k,
            file_ids=file_ids if file_ids else None,
            cancel_token=cancel_token,
        )
        
        if results.is_empty():
            return "", []
        
        context = "\n\n---\n\n".join(results.extract_formatted_chunks())
        chunk_refs = [
            {
                "chunk_id": chunk.chunk_id,
                "file_id": chunk.file_id,
                "rank": i + 1,
            }
            for i, chunk in enumerate(results.chunks)
        ]
        return context, chunk_refs
    
    async def _build_rag_context(
        self,
        analysis: AnalysisResult,
        user_message: str,
        cancel_token: Optional[CancellationToken],
    ) -> tuple[str | None, RAGContext]:
        """
        Build RAG context based on semantic analysis.
        
        Returns:
            Tuple of (context_string, rag_metadata)
        """
        context_parts = []
        chunk_refs: list[dict] = []
        
        # Get summaries if needed
        if analysis.summary_file_ids:
            if cancel_token:
                cancel_token.check_cancelled()
            summaries = self._get_summaries_for_files(analysis.summary_file_ids)
            if summaries:
                context_parts.append(summaries)
        
        # Get chunks if needed
        if analysis.chunk_file_ids:
            if cancel_token:
                cancel_token.set_phase(ProcessingPhase.RETRIEVAL)
                cancel_token.check_cancelled()
            
            search_query = analysis.search_query or user_message
            chunks_context, chunk_refs = await self._retrieve_chunks(
                search_query,
                analysis.chunk_file_ids,
                cancel_token,
            )
            if chunks_context:
                context_parts.append(chunks_context)
        
        # Build RAG metadata
        rag_context = RAGContext(
            intent=analysis.intent.value,
            reasoning=analysis.reasoning,
            search_query=analysis.search_query,
            summary_file_ids=analysis.summary_file_ids,
            chunk_file_ids=analysis.chunk_file_ids,
            chunks_retrieved=len(chunk_refs),
            chunk_refs=chunk_refs,
        )
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else None
        return context, rag_context
    
    async def process_message(
        self,
        user_message: str,
        cancel_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Process user message with intelligent RAG pipeline.
        
        Pipeline:
        1. Semantic Analysis: Analyze query with corpus summaries + history
        2. Context Building: Retrieve summaries/chunks based on analysis
        3. Generation: Stream LLM response with context
        
        Args:
            user_message: User's message
            cancel_token: Optional cancellation token
            
        Yields:
            Event dicts: status, content, done, error, cancelled
        """
        full_response = ""
        rag_context: Optional[RAGContext] = None
        
        try:
            # Phase: Semantic Analysis
            if cancel_token:
                cancel_token.set_phase(ProcessingPhase.ANALYSIS)
                cancel_token.check_cancelled()
            
            yield {"status": "Đang phân tích ngữ nghĩa..."}
            
            history_context = self._build_history_context_for_analyzer()
            analysis = await self._analyzer.analyze(
                user_message,
                history_context,
                self._corpus,
                cancel_token,
            )
            
            # Phase: Build RAG Context
            context = None
            if analysis.intent not in (QueryIntent.NO_RAG, QueryIntent.ACKNOWLEDGE):
                n_summary = len(analysis.summary_file_ids)
                n_chunk = len(analysis.chunk_file_ids)
                status_parts = []
                if n_summary:
                    status_parts.append(f"tóm tắt {n_summary} file")
                if n_chunk:
                    status_parts.append(f"tìm kiếm {n_chunk} file")
                yield {"status": f"Đang {', '.join(status_parts)}..."}
                
                context, rag_context = await self._build_rag_context(
                    analysis, user_message, cancel_token
                )
            else:
                # No RAG needed - just capture analysis result
                rag_context = RAGContext(
                    intent=analysis.intent.value,
                    reasoning=analysis.reasoning,
                )
            
            # Phase: Generation
            if cancel_token:
                cancel_token.set_phase(ProcessingPhase.GENERATION)
                cancel_token.check_cancelled()
            
            yield {"status": "Đang tạo phản hồi..."}
            messages = self._build_messages(user_message, context)
            
            async for chunk in self._llm.generate_stream(messages, cancel_token):
                if cancel_token and cancel_token.is_cancelled:
                    break
                full_response += chunk
                yield {"content": chunk}
            
            # Save to history with RAG references
            if full_response:
                self._history.add_turn(user_message, full_response, rag_context)
            
            yield {"done": True}
            
        except CancelledException:
            # Save partial response
            if full_response:
                self._history.add_turn(
                    user_message, 
                    full_response + "\n[Đã dừng]", 
                    rag_context
                )
            yield {"cancelled": True}
            
        except Exception as e:
            yield {"error": f"Lỗi: {str(e)}"}
