# Đồ án cơ sở ngành

RAG-based AI Chatbot với hybrid search sử dụng:
- **Docling** - Xử lý tài liệu (PDF, DOCX)
- **llama-cpp-python** - LLM, embedding, reranking
- **BM25 + Semantic Search** - Hybrid search với RRF fusion
- **FastAPI** - Web framework

# Chuẩn bị llama-cpp-python
Tải [llama-cpp-python CUDA](https://github.com/JamePeng/llama-cpp-python/releases) và đặt vào thư mục `custom_packages`

# Chuẩn bị Model LLM, Embedding, Reranker
Tải và đặt các file model .gguf vào thư mục `models/`:
- [Qwen3VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF): `Qwen3VL-4B-Instruct-Q4_K_M.gguf` - LLM chính
- [BGE-M3](https://huggingface.co/gpustack/bge-m3-GGUF) `bge-m3-Q8_0.gguf` - Embedding model
- [BGE-RERANKER-V2-M3](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF) `bge-reranker-v2-m3-Q6_K.gguf` - Reranker model

# Cài đặt dependencies (sử dụng uv)
```bash
uv sync
```

# Cài đặt các model của Docling:
```bash
docling-tools models download --output-dir artifacts
```

## Chạy ứng dụng

```bash
python -m src.main

uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Truy cập http://localhost:8000 để sử dụng giao diện chat.