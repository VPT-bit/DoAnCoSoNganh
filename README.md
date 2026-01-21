# Đồ án cơ sở ngành

RAG-based AI Chatbot với hybrid search sử dụng:
- **Docling** - Xử lý tài liệu (PDF, DOCX)
- **llama-cpp-python** - LLM, embedding, reranking
- **BM25 + Semantic Search** - Hybrid search với RRF fusion
- **FastAPI** - Web framework

# Cài đặt dependencies (sử dụng uv)
```bash
uv sync
```

## Models

# Tải và đặt các file model GGUF vào thư mục `models/`:
- `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` - LLM chính
- `bge-m3-Q8_0.gguf` - Embedding model
- `bge-reranker-v2-m3-Q6_K.gguf` - Reranker model

# Cài đặt các model của Docling:
```bash
docling-tools models download --output-dir artifacts
```

## Chạy ứng dụng

```bash
python -m src.main

uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Truy cập http://localhost:8000 để sử dụng giao diện chat.