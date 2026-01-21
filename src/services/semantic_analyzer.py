"""Semantic analyzer - LLM-based query analysis."""
import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.core.models import Corpus
from src.core.cancellation import CancellationToken, CancelledException
from src.services.llm import LLMService


class QueryIntent(Enum):
    """Query intent types."""
    NO_RAG = "no_rag"           # General chat, greetings, off-topic
    ACKNOWLEDGE = "acknowledge"  # User notifies file upload, awaiting questions
    RAG_SUMMARY = "rag_summary" # Request for overview/summary
    RAG_CHUNKS = "rag_chunks"   # Request for specific information


@dataclass
class AnalysisResult:
    """Semantic analysis result."""
    intent: QueryIntent
    reasoning: str
    summary_file_ids: list[str] = field(default_factory=list)
    chunk_file_ids: list[str] = field(default_factory=list)
    search_query: Optional[str] = None


class SemanticAnalyzer:
    """
    LLM-based semantic query analyzer.
    
    Process:
    1. Build context: system prompt + chat history + corpus info
    2. LLM analyzes and returns JSON decision
    3. Parse decision and build AnalysisResult
    """
    
    SYSTEM_PROMPT = """Bạn là module phân tích ngữ nghĩa cho RAG chatbot. Nhiệm vụ: phân tích tin nhắn người dùng và đưa ra quyết định truy vấn dữ liệu.

## CORPUS HIỆN TẠI
{corpus_section}

## LỊCH SỬ HỘI THOẠI GẦN ĐÂY
{history_section}

## PHÂN LOẠI Ý ĐỊNH

### NO_RAG
Dùng khi tin nhắn KHÔNG liên quan đến nội dung file:
- Chào hỏi: "chào bạn", "hello", "hi"
- Cảm ơn: "cảm ơn", "thanks" 
- Hỏi về bot: "bạn là ai", "bạn có thể làm gì"
- Tạm biệt: "bye", "tạm biệt"
- Hội thoại xã giao không đề cập file

### ACKNOWLEDGE  
Dùng khi người dùng VỪA gửi file và CHƯA hỏi câu hỏi cụ thể:
- Tin nhắn có tag [FILE] kèm: "hãy xử lý", "đây là file", "chờ câu hỏi của tôi"
- Đặc điểm: thông báo/yêu cầu xử lý, KHÔNG có câu hỏi về nội dung

### RAG_SUMMARY
Dùng khi cần TỔNG QUAN/TÓM TẮT file:
- "tóm tắt file này", "file nói về gì", "nội dung chính là gì"
- "gồm mấy phần", "cấu trúc file", "có những phần nào"
- "khái quát nội dung", "tổng quan tài liệu"

### RAG_CHUNKS
Dùng khi cần TÌM KIẾM thông tin CỤ THỂ trong file:
- Hỏi chi tiết: "mục tiêu là gì", "phần 2 nói về gì"
- Tìm định nghĩa: "X là gì", "định nghĩa của Y"
- Trích xuất: "liệt kê các...", "những điểm chính của..."
- Bất kỳ câu hỏi cần tra cứu nội dung chi tiết

## QUY TẮC XÁC ĐỊNH FILE

Khi intent là RAG_SUMMARY hoặc RAG_CHUNKS, phải xác định file liên quan:
- "file đầu tiên", "file 1" → file đầu tiên trong corpus
- "file thứ 2", "file hai" → file thứ 2
- "file cuối", "file mới nhất" → file cuối cùng
- Đề cập tên file → file tương ứng
- Không chỉ định cụ thể → TẤT CẢ file trong corpus

## OUTPUT FORMAT

Trả về JSON duy nhất, không có text thừa:
```json
{{
  "intent": "NO_RAG|ACKNOWLEDGE|RAG_SUMMARY|RAG_CHUNKS",
  "file_ids": ["id1", "id2"],
  "search_query": "câu truy vấn tối ưu cho RAG_CHUNKS",
  "reason": "giải thích ngắn gọn"
}}
```

Quy tắc:
- NO_RAG/ACKNOWLEDGE: file_ids = [], search_query = null
- RAG_SUMMARY: file_ids = [các file cần tóm tắt], search_query = null
- RAG_CHUNKS: file_ids = [các file cần search], search_query = "query tối ưu"
- Nếu không rõ file nào, đưa TẤT CẢ file_ids vào"""

    def __init__(self, llm_service: LLMService):
        self._llm = llm_service
        self._lock = asyncio.Lock()
    
    def _build_corpus_section(self, corpus: Corpus) -> str:
        """Build corpus information section."""
        if corpus.is_empty():
            return "Corpus trống - chưa có file nào."
        
        lines = [f"Tổng: {len(corpus.summaries)} file(s)\n"]
        for i, s in enumerate(corpus.summaries, 1):
            lines.append(
                f"[File {i}]\n"
                f"  ID: {s.file_id}\n"
                f"  Tên: {s.file_name}\n"
                f"  Chunks: {s.chunk_count}\n"
                f"  Tóm tắt: {s.summary}"
            )
        return "\n\n".join(lines)
    
    def _build_history_section(self, history: str) -> str:
        """Build chat history section."""
        return history if history else "(Chưa có lịch sử)"
    
    def _parse_llm_response(self, response: str, corpus: Corpus) -> AnalysisResult:
        """
        Parse LLM JSON response into AnalysisResult.
        
        Validates file_ids against corpus and fills defaults when needed.
        """
        all_file_ids = set(corpus.get_file_ids())
        
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            # Fallback: search all files
            return AnalysisResult(
                intent=QueryIntent.RAG_CHUNKS,
                reasoning="JSON parse failed - defaulting to search",
                chunk_file_ids=list(all_file_ids),
                search_query="",
            )
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return AnalysisResult(
                intent=QueryIntent.RAG_CHUNKS,
                reasoning="JSON decode failed - defaulting to search",
                chunk_file_ids=list(all_file_ids),
                search_query="",
            )
        
        # Parse intent
        intent_str = data.get("intent", "RAG_CHUNKS").upper().strip()
        intent_map = {
            "NO_RAG": QueryIntent.NO_RAG,
            "ACKNOWLEDGE": QueryIntent.ACKNOWLEDGE,
            "RAG_SUMMARY": QueryIntent.RAG_SUMMARY,
            "RAG_CHUNKS": QueryIntent.RAG_CHUNKS,
        }
        intent = intent_map.get(intent_str, QueryIntent.RAG_CHUNKS)
        
        # Parse and validate file_ids
        raw_file_ids = data.get("file_ids", []) or []
        valid_file_ids = [fid for fid in raw_file_ids if fid in all_file_ids]
        
        # If RAG intent but no valid files → use ALL files
        if intent in (QueryIntent.RAG_SUMMARY, QueryIntent.RAG_CHUNKS):
            if not valid_file_ids:
                valid_file_ids = list(all_file_ids)
        
        # Parse search query
        search_query = data.get("search_query") or None
        
        # Build result based on intent
        result = AnalysisResult(
            intent=intent,
            reasoning=data.get("reason", ""),
        )
        
        if intent == QueryIntent.RAG_SUMMARY:
            result.summary_file_ids = valid_file_ids
        elif intent == QueryIntent.RAG_CHUNKS:
            result.chunk_file_ids = valid_file_ids
            result.search_query = search_query
        
        return result
    
    async def analyze(
        self,
        user_message: str,
        history_context: str,
        corpus: Corpus,
        cancel_token: Optional[CancellationToken] = None,
    ) -> AnalysisResult:
        """
        Analyze user query using LLM.
        
        Args:
            user_message: Current user message
            history_context: Recent chat history
            corpus: Document corpus with summaries
            cancel_token: Optional cancellation token
            
        Returns:
            AnalysisResult with intent and retrieval parameters
        """
        # Empty corpus = no RAG possible
        if corpus.is_empty():
            return AnalysisResult(QueryIntent.NO_RAG, "Corpus trống")
        
        if cancel_token and cancel_token.is_cancelled:
            raise CancelledException("Cancelled before analysis")
        
        # Build system prompt with context
        system_prompt = self.SYSTEM_PROMPT.format(
            corpus_section=self._build_corpus_section(corpus),
            history_section=self._build_history_section(history_context),
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        # Call LLM
        async with self._lock:
            try:
                response = await self._llm.generate_complete(
                    messages,
                    cancel_token,
                    temperature=0.1,
                    max_tokens=300,
                )
                return self._parse_llm_response(response, corpus)
                
            except CancelledException:
                raise
            except Exception as e:
                # On error: default to searching all files
                return AnalysisResult(
                    intent=QueryIntent.RAG_CHUNKS,
                    reasoning=f"LLM error: {e} - defaulting to search",
                    chunk_file_ids=corpus.get_file_ids(),
                    search_query=user_message,
                )
