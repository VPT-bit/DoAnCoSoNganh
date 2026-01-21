# ĐẶC TẢ HỆ THỐNG RAG CHATBOT

## Mục lục
1. [Mô tả hệ thống](#1-mô-tả-hệ-thống)
2. [Lý thuyết công nghệ](#2-lý-thuyết-công-nghệ)
3. [Thiết kế chức năng](#3-thiết-kế-chức-năng)
4. [Thiết kế lớp](#4-thiết-kế-lớp)
5. [Đặc tả luồng hoạt động](#5-đặc-tả-luồng-hoạt-động)

---

## 1. Mô tả hệ thống

### 1.1 Tổng quan

Hệ thống RAG Chatbot là một ứng dụng trí tuệ nhân tạo kết hợp giữa **Large Language Model (LLM)** và **Retrieval-Augmented Generation (RAG)** để hỗ trợ người dùng tra cứu và tương tác với nội dung tài liệu.

### 1.2 Mục tiêu

- Cho phép người dùng upload và xử lý tài liệu (PDF, DOCX)
- Tự động tóm tắt nội dung tài liệu
- Trả lời câu hỏi dựa trên nội dung tài liệu đã upload
- Cung cấp phản hồi theo thời gian thực (streaming)
- Hỗ trợ hủy bỏ yêu cầu đang xử lý

### 1.3 Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                        │
│                    HTML/CSS/JavaScript + SSE                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                        │
│              REST Endpoints + Server-Sent Events                │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ ChatService   │      │ FileProcessor │      │HybridRetriever│
│ (Orchestrator)│      │  (Docling)    │      │ (BM25+Semantic│
└───────────────┘      └───────────────┘      │  +Reranker)   │
        │                       │              └───────────────┘
        ▼                       ▼                       │
┌───────────────┐      ┌───────────────┐               │
│SemanticAnalyzer│     │   LLMService  │◄──────────────┘
│ (Query Intent) │     │  (Generation) │
└───────────────┘      └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
           ┌───────────────┐
           │    Corpus     │
           │ (Data Storage)│
           └───────────────┘
```

### 1.4 Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Backend Framework | FastAPI |
| LLM Runtime | llama-cpp-python (GPU) |
| Document Processing | Docling |
| Embedding Model | bge-m3 (GGUF) |
| Reranker Model | bge-reranker-v2-m3 (GGUF) |
| LLM Model | Ministral-3B / Qwen3VL-4B |
| Full-text Search | BM25 (rank-bm25) |
| Frontend | HTML + JavaScript + SSE |

---

## 2. Lý thuyết công nghệ

### 2.1 Large Language Model (LLM)

#### 2.1.1 Khái niệm

Large Language Model (LLM) là mô hình ngôn ngữ quy mô lớn được huấn luyện trên lượng văn bản khổng lồ, có khả năng:
- Hiểu và sinh văn bản tự nhiên
- Tóm tắt nội dung
- Trả lời câu hỏi
- Phân loại ý định

#### 2.1.2 Kiến trúc Transformer

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<input>> LightBlue
    BackgroundColor<<attention>> LightGreen
    BackgroundColor<<ffn>> LightYellow
    BackgroundColor<<output>> LightPink
}

rectangle "Input Embedding" as input <<input>>
rectangle "Positional Encoding" as pos <<input>>

rectangle "Multi-Head\nSelf-Attention" as attn <<attention>>
rectangle "Add & Norm" as norm1 <<attention>>

rectangle "Feed-Forward\nNetwork" as ffn <<ffn>>
rectangle "Add & Norm" as norm2 <<ffn>>

rectangle "Output Linear" as output <<output>>
rectangle "Softmax" as softmax <<output>>

input --> pos
pos --> attn
attn --> norm1
norm1 --> ffn
ffn --> norm2
norm2 --> output
output --> softmax

note right of attn
  Q, K, V = Linear(X)
  Attention(Q,K,V) = 
    softmax(QK^T/√d_k)V
end note

@enduml
```

#### 2.1.3 GGUF Format

Hệ thống sử dụng định dạng GGUF (GPT-Generated Unified Format) với các đặc điểm:
- **Quantization**: Nén mô hình (Q4_K_M, Q6_K, Q8_0) giảm VRAM
- **GPU Offloading**: Đẩy layers lên GPU để tăng tốc
- **Flash Attention**: Tối ưu bộ nhớ attention
- **KV Cache Quantization**: Nén key-value cache

### 2.2 Embedding Model

#### 2.2.1 Khái niệm

Embedding model chuyển đổi văn bản thành vector số học trong không gian đa chiều, cho phép:
- Đo độ tương đồng ngữ nghĩa giữa các văn bản
- Tìm kiếm semantic (theo ý nghĩa)
- Clustering và phân loại văn bản

#### 2.2.2 BGE-M3 Model

Hệ thống sử dụng **BGE-M3** (BAAI General Embedding - Multi-Functionality, Multi-Linguality, Multi-Granularity):
- **Multi-Linguality**: Hỗ trợ 100+ ngôn ngữ bao gồm tiếng Việt
- **Multi-Granularity**: Xử lý từ câu ngắn đến đoạn văn dài (8192 tokens)
- **Multi-Functionality**: Dense + Sparse + ColBERT retrieval

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<text>> LightBlue
    BackgroundColor<<model>> LightGreen
    BackgroundColor<<vector>> LightYellow
}

rectangle "Text Input\n\"Tài liệu này nói về AI\"" as text <<text>>

rectangle "BGE-M3\nEncoder" as model <<model>> {
    rectangle "Tokenizer" as tok
    rectangle "Transformer\nLayers" as trans
    rectangle "Pooling" as pool
}

rectangle "Dense Vector\n[0.12, -0.34, 0.56, ...]" as vector <<vector>>

text --> tok
tok --> trans
trans --> pool
pool --> vector

note right of vector
  Dimension: 1024
  Normalized L2
end note

@enduml
```

### 2.3 Reranker (Cross-Encoder)

#### 2.3.1 Khái niệm

Reranker là mô hình cross-encoder đánh giá mức độ liên quan giữa query và document bằng cách xử lý cả hai cùng lúc.

#### 2.3.2 So sánh Bi-Encoder vs Cross-Encoder

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<query>> LightBlue
    BackgroundColor<<doc>> LightGreen
    BackgroundColor<<model>> LightYellow
    BackgroundColor<<score>> LightPink
}

package "Bi-Encoder (Embedding)" {
    rectangle "Query" as q1 <<query>>
    rectangle "Encoder" as e1 <<model>>
    rectangle "Vector Q" as v1 <<score>>
    
    rectangle "Document" as d1 <<doc>>
    rectangle "Encoder" as e2 <<model>>
    rectangle "Vector D" as v2 <<score>>
    
    rectangle "Cosine\nSimilarity" as cos <<score>>
    
    q1 --> e1
    e1 --> v1
    d1 --> e2
    e2 --> v2
    v1 --> cos
    v2 --> cos
}

package "Cross-Encoder (Reranker)" {
    rectangle "[CLS] Query [SEP] Document [SEP]" as input <<query>>
    rectangle "Single Encoder" as enc <<model>>
    rectangle "Relevance\nScore" as score <<score>>
    
    input --> enc
    enc --> score
}

note bottom of cos
  Fast (separate encoding)
  Lower accuracy
end note

note bottom of score
  Slow (joint encoding)
  Higher accuracy
end note

@enduml
```

#### 2.3.3 BGE-Reranker-v2-M3

Hệ thống sử dụng **bge-reranker-v2-m3**:
- Multilingual (hỗ trợ tiếng Việt)
- Lightweight (568M parameters)
- High accuracy trên MTEB benchmark

### 2.4 Hybrid Search

#### 2.4.1 Tổng quan

Hybrid Search kết hợp nhiều phương pháp tìm kiếm để tận dụng ưu điểm của từng phương pháp:

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| **BM25** | Exact match, keyword precision | Không hiểu ngữ nghĩa |
| **Semantic** | Hiểu ngữ nghĩa, synonym | Miss exact keywords |
| **Hybrid** | Kết hợp cả hai | Phức tạp hơn |

#### 2.4.2 BM25 (Best Matching 25)

BM25 là thuật toán ranking dựa trên TF-IDF cải tiến:

$$
\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$

Trong đó:
- $f(q_i, D)$: Tần suất term $q_i$ trong document $D$
- $|D|$: Độ dài document
- $\text{avgdl}$: Độ dài trung bình các document
- $k_1 = 1.5$, $b = 0.75$: Hyperparameters

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<input>> LightBlue
    BackgroundColor<<process>> LightGreen
    BackgroundColor<<output>> LightYellow
}

rectangle "Query\n\"mục tiêu của dự án\"" as query <<input>>
rectangle "Corpus Documents" as corpus <<input>>

rectangle "Tokenization\n& Preprocessing" as token <<process>>
rectangle "TF Calculation" as tf <<process>>
rectangle "IDF Calculation" as idf <<process>>
rectangle "BM25 Scoring" as bm25 <<process>>

rectangle "Ranked Results\n(by BM25 score)" as output <<output>>

query --> token
corpus --> token
token --> tf
token --> idf
tf --> bm25
idf --> bm25
bm25 --> output

@enduml
```

#### 2.4.3 Semantic Search

Tìm kiếm dựa trên độ tương đồng vector embedding:

$$
\text{similarity}(Q, D) = \frac{E_Q \cdot E_D}{||E_Q|| \cdot ||E_D||}
$$

Trong đó $E_Q$, $E_D$ là embedding vectors của query và document.

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<input>> LightBlue
    BackgroundColor<<embed>> LightGreen
    BackgroundColor<<search>> LightYellow
    BackgroundColor<<output>> LightPink
}

rectangle "Query" as query <<input>>
rectangle "Corpus Chunks" as corpus <<input>>

rectangle "Embedding\nModel" as embed <<embed>>
rectangle "Query Vector" as qvec <<embed>>
rectangle "Corpus Vectors\n(Pre-computed)" as cvec <<embed>>

rectangle "Cosine\nSimilarity" as cos <<search>>
rectangle "Top-K\nSelection" as topk <<search>>

rectangle "Semantic\nResults" as output <<output>>

query --> embed
embed --> qvec
corpus --> embed
embed --> cvec

qvec --> cos
cvec --> cos
cos --> topk
topk --> output

@enduml
```

#### 2.4.4 Reciprocal Rank Fusion (RRF)

RRF kết hợp rankings từ nhiều nguồn bằng công thức:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{\alpha + r(d)}
$$

Trong đó:
- $R$: Tập các rankings (BM25, Semantic)
- $r(d)$: Rank của document $d$ trong ranking $r$
- $\alpha = 60$: Smoothing constant

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<bm25>> LightBlue
    BackgroundColor<<semantic>> LightGreen
    BackgroundColor<<fusion>> LightYellow
    BackgroundColor<<rerank>> LightPink
}

rectangle "BM25 Results\n[D3, D1, D7, D2, ...]" as bm25 <<bm25>>
rectangle "Semantic Results\n[D1, D5, D3, D8, ...]" as semantic <<semantic>>

rectangle "RRF Fusion\nscore(d) = Σ 1/(60 + rank)" as rrf <<fusion>>

rectangle "Fused Candidates\n[D1, D3, D7, D5, ...]" as fused <<fusion>>

rectangle "Cross-Encoder\nReranker" as rerank <<rerank>>

rectangle "Final Results\n[D3, D1, D5, D7, ...]" as final <<rerank>>

bm25 --> rrf
semantic --> rrf
rrf --> fused
fused --> rerank
rerank --> final

note right of rrf
  D1: 1/61 + 1/63 = 0.032
  D3: 1/61 + 1/63 = 0.032
  D7: 1/63 + 1/65 = 0.031
end note

@enduml
```

#### 2.4.5 Pipeline Hybrid Search hoàn chỉnh

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<input>> #E3F2FD
    BackgroundColor<<bm25>> #E8F5E9
    BackgroundColor<<semantic>> #FFF3E0
    BackgroundColor<<fusion>> #F3E5F5
    BackgroundColor<<rerank>> #FFEBEE
    BackgroundColor<<output>> #E0F7FA
}

rectangle "User Query" as query <<input>>

rectangle "BM25 Retriever\n(CPU)" as bm25 <<bm25>> {
    rectangle "Tokenize" as tok1
    rectangle "TF-IDF Score" as tfidf
    rectangle "Top-K×4" as topk1
}

rectangle "Semantic Retriever\n(GPU)" as semantic <<semantic>> {
    rectangle "Embed Query" as embed
    rectangle "Cosine Similarity" as cos
    rectangle "Top-K×4" as topk2
}

rectangle "RRF Fusion" as fusion <<fusion>>

rectangle "Cross-Encoder\nReranker (GPU)" as rerank <<rerank>>

rectangle "Top-K Results" as output <<output>>

query --> bm25
query --> semantic

tok1 --> tfidf
tfidf --> topk1

embed --> cos
cos --> topk2

topk1 --> fusion
topk2 --> fusion

fusion --> rerank
rerank --> output

note bottom of output
  K = 5 (configurable)
  Final ranked chunks
  with relevance scores
end note

@enduml
```

### 2.5 Retrieval-Augmented Generation (RAG)

#### 2.5.1 Khái niệm

RAG kết hợp retrieval (tìm kiếm) với generation (sinh văn bản) để:
- Giảm hallucination của LLM
- Cung cấp thông tin cập nhật (không cần retrain)
- Trả lời dựa trên nguồn đáng tin cậy

#### 2.5.2 RAG Pipeline trong hệ thống

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<user>> #BBDEFB
    BackgroundColor<<analyze>> #C8E6C9
    BackgroundColor<<retrieve>> #FFE0B2
    BackgroundColor<<generate>> #E1BEE7
    BackgroundColor<<output>> #B2EBF2
}

rectangle "User Question" as question <<user>>

rectangle "Semantic Analysis" as analyze <<analyze>> {
    rectangle "Intent Classification" as intent
    rectangle "File Detection" as files
    rectangle "Query Optimization" as optquery
}

rectangle "Context Retrieval" as retrieve <<retrieve>> {
    rectangle "Summary Retrieval\n(RAG_SUMMARY)" as summary
    rectangle "Chunk Retrieval\n(RAG_CHUNKS)" as chunks
}

rectangle "Response Generation" as generate <<generate>> {
    rectangle "Build Prompt" as prompt
    rectangle "LLM Generation" as llm
    rectangle "Stream Response" as stream
}

rectangle "Answer + References" as output <<output>>

question --> analyze
intent --> files
files --> optquery

optquery --> summary
optquery --> chunks

summary --> prompt
chunks --> prompt
prompt --> llm
llm --> stream
stream --> output

@enduml
```

---

## 3. Thiết kế chức năng

### 3.1 Sơ đồ DFD Mức 0 (Context Diagram)

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor LightYellow
}

actor "Người dùng" as user

rectangle "RAG Chatbot\nSystem" as system

database "Document\nStorage" as storage

user --> system : Tin nhắn / File upload
system --> user : Phản hồi streaming

system <--> storage : Lưu/Đọc corpus

@enduml
```

### 3.2 Sơ đồ DFD Mức 1

```plantuml
@startuml
skinparam rectangle {
    BackgroundColor<<process>> LightGreen
    BackgroundColor<<store>> LightBlue
}

actor "Người dùng" as user

rectangle "1.0\nXử lý File" as p1 <<process>>
rectangle "2.0\nPhân tích\nNgữ nghĩa" as p2 <<process>>
rectangle "3.0\nTruy vấn\nDữ liệu" as p3 <<process>>
rectangle "4.0\nSinh phản hồi" as p4 <<process>>

database "D1: Corpus\n(chunks + summaries)" as d1 <<store>>
database "D2: Chat History" as d2 <<store>>

user --> p1 : File (PDF/DOCX)
p1 --> d1 : Chunks, Summary
p1 --> user : Xác nhận xử lý

user --> p2 : Tin nhắn
d1 --> p2 : Corpus info
d2 --> p2 : Lịch sử chat
p2 --> p3 : AnalysisResult\n(intent, file_ids, query)

d1 --> p3 : Chunks data
p3 --> p4 : Context\n(summaries/chunks)

d2 --> p4 : Recent history
p4 --> user : Streaming response
p4 --> d2 : Save turn

@enduml
```

### 3.3 Đặc tả chức năng theo Module

#### 3.3.1 Module Xử lý File (FileProcessor)

| Thuộc tính | Mô tả |
|------------|-------|
| **Chức năng** | Chuyển đổi tài liệu thành chunks văn bản |
| **Input** | File PDF hoặc DOCX |
| **Output** | List[Chunk], FileSummary |
| **Công nghệ** | Docling (GPU accelerated) |

**Quy trình:**
1. Nhận file upload từ API
2. Load Docling models (on-demand)
3. Convert document → Docling Document
4. Chunk với HybridChunker (max 512 tokens)
5. Gọi LLM tóm tắt nội dung
6. Trả về chunks + summary

#### 3.3.2 Module Phân tích Ngữ nghĩa (SemanticAnalyzer)

| Thuộc tính | Mô tả |
|------------|-------|
| **Chức năng** | Phân loại ý định và xác định chiến lược truy vấn |
| **Input** | User message, Chat history, Corpus |
| **Output** | AnalysisResult (intent, file_ids, search_query) |
| **Công nghệ** | LLM-based classification |

**Các loại Intent:**

| Intent | Mô tả | Hành động |
|--------|-------|-----------|
| NO_RAG | Chat thông thường | Không truy vấn |
| ACKNOWLEDGE | Thông báo upload file | Xác nhận |
| RAG_SUMMARY | Yêu cầu tóm tắt | Lấy summaries |
| RAG_CHUNKS | Tìm kiếm chi tiết | Hybrid search |

#### 3.3.3 Module Truy vấn Dữ liệu (HybridRetriever)

| Thuộc tính | Mô tả |
|------------|-------|
| **Chức năng** | Tìm kiếm chunks liên quan đến query |
| **Input** | Query, Corpus, File filter |
| **Output** | Top-K relevant chunks |
| **Công nghệ** | BM25 + Semantic + RRF + Reranker |

**Pipeline:**
1. BM25 retrieval → Top K×4 candidates
2. Semantic retrieval → Top K×4 candidates
3. RRF fusion → Combined ranking
4. Reranker → Final Top K

#### 3.3.4 Module Sinh phản hồi (ChatService)

| Thuộc tính | Mô tả |
|------------|-------|
| **Chức năng** | Điều phối toàn bộ pipeline và sinh phản hồi |
| **Input** | User message, Optional file |
| **Output** | Streaming response |
| **Công nghệ** | LLM streaming generation |

#### 3.3.5 Module Quản lý Dữ liệu (Corpus)

| Thuộc tính | Mô tả |
|------------|-------|
| **Chức năng** | Lưu trữ và quản lý document data |
| **Cấu trúc** | chunks: List[Chunk], summaries: List[FileSummary] |
| **Persistence** | JSON file |

### 3.4 Cơ sở dữ liệu

#### 3.4.1 Corpus (documents_index.json)

```json
{
  "chunks": [
    {
      "file_id": "abc123",
      "file_name": "document.pdf",
      "chunk_index": 0,
      "meta": {"dl_meta": {...}},
      "text": "Nội dung chunk..."
    }
  ],
  "summaries": [
    {
      "file_id": "abc123",
      "file_name": "document.pdf",
      "summary": "Tóm tắt nội dung...",
      "chunk_count": 47
    }
  ]
}
```

#### 3.4.2 Chat History (chat_history.json)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt..."
    },
    {
      "role": "user", 
      "content": "Câu hỏi..."
    },
    {
      "role": "assistant",
      "content": "Phản hồi...",
      "rag_context": {
        "intent": "rag_chunks",
        "reasoning": "...",
        "search_query": "...",
        "summary_file_ids": [],
        "chunk_file_ids": ["abc123"],
        "chunks_retrieved": 5,
        "chunk_refs": [{"chunk_id": "...", "file_id": "...", "rank": 1}]
      }
    }
  ]
}
```

---

## 4. Thiết kế lớp

### 4.1 Đặc tả các Class chính

#### 4.1.1 LLMService

**Chức năng:** Quản lý LLM model và sinh văn bản

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `_model_path` | `str` | Đường dẫn file GGUF |
| `_llm` | `Llama` | Instance llama-cpp |
| `_executor` | `ThreadPoolExecutor` | Thread pool cho async |
| `_lock` | `asyncio.Lock` | Đồng bộ hóa |

| Phương thức | Mô tả |
|-------------|-------|
| `load_model()` | Load model vào VRAM |
| `generate_stream()` | Sinh văn bản streaming (async) |
| `generate_complete()` | Sinh văn bản đầy đủ (async) |
| `summarize_chunks()` | Tóm tắt document |
| `unload_model()` | Giải phóng VRAM |

#### 4.1.2 SemanticAnalyzer

**Chức năng:** Phân tích ngữ nghĩa query bằng LLM

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `_llm` | `LLMService` | LLM service reference |
| `_lock` | `asyncio.Lock` | Đồng bộ hóa |
| `SYSTEM_PROMPT` | `str` | Prompt hướng dẫn phân tích |

| Phương thức | Mô tả |
|-------------|-------|
| `analyze()` | Phân tích query, trả về AnalysisResult |
| `_build_corpus_section()` | Xây dựng context corpus |
| `_build_history_section()` | Xây dựng context history |
| `_parse_llm_response()` | Parse JSON response từ LLM |

#### 4.1.3 HybridRetriever

**Chức năng:** Orchestrator cho hybrid search pipeline

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `_bm25` | `BM25Retriever` | BM25 search |
| `_semantic` | `SemanticRetriever` | Semantic search |
| `_reranker` | `Reranker` | Cross-encoder reranker |
| `_fusion` | `RRFFusion` | Rank fusion |
| `_corpus` | `Corpus` | Reference to corpus |

| Phương thức | Mô tả |
|-------------|-------|
| `process()` | Index corpus cho retrieval |
| `retrieve()` | Thực hiện hybrid search |
| `is_ready()` | Kiểm tra đã index chưa |
| `release_models()` | Giải phóng VRAM |
| `clear()` | Xóa index (giữ models) |

#### 4.1.4 FileProcessor

**Chức năng:** Xử lý tài liệu với Docling

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `_converter` | `DocumentConverter` | Docling converter |
| `_chunker` | `HybridChunker` | Text chunker |
| `_models_loaded` | `bool` | Trạng thái load |
| `_cancel_event` | `threading.Event` | Cancellation event |

| Phương thức | Mô tả |
|-------------|-------|
| `load_models()` | Load Docling models |
| `process()` | Xử lý file, trả về chunks |
| `release_models()` | Giải phóng VRAM |

#### 4.1.5 ChatService

**Chức năng:** Điều phối toàn bộ RAG pipeline

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `_llm` | `LLMService` | LLM service |
| `_retriever` | `HybridRetriever` | Retrieval service |
| `_analyzer` | `SemanticAnalyzer` | Query analyzer |
| `_corpus` | `Corpus` | Document corpus |
| `_history` | `ChatHistory` | Chat history |

| Phương thức | Mô tả |
|-------------|-------|
| `process_message()` | Xử lý tin nhắn (main pipeline) |
| `get_history()` | Lấy lịch sử chat |
| `clear_history()` | Xóa lịch sử |
| `build_user_message_with_file()` | Tạo message có file tag |

#### 4.1.6 Corpus

**Chức năng:** Quản lý và lưu trữ document data

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `chunks` | `list[Chunk]` | Tất cả chunks |
| `summaries` | `list[FileSummary]` | Tất cả summaries |

| Phương thức | Mô tả |
|-------------|-------|
| `add_chunk()` | Thêm chunk |
| `add_summary()` | Thêm summary |
| `get_chunks_by_file()` | Lấy chunks theo file |
| `get_file_ids()` | Lấy danh sách file IDs |
| `remove_file()` | Xóa file |
| `save()` / `load()` | Persistence |

### 4.2 Classes phụ trợ (Data Classes)

| Class | Chức năng |
|-------|-----------|
| `Chunk` | Đại diện một đoạn văn bản từ document |
| `FileSummary` | Tóm tắt thông tin của một file |
| `AnalysisResult` | Kết quả phân tích ngữ nghĩa |
| `RAGContext` | Metadata về RAG retrieval |
| `ChatMessage` | Một tin nhắn trong lịch sử |
| `CancellationToken` | Token hủy bỏ request |
| `QueryIntent` | Enum các loại intent |
| `ProcessingPhase` | Enum các phase xử lý |

### 4.3 Sơ đồ lớp (Class Diagram)

```plantuml
@startuml
skinparam classAttributeIconSize 0
skinparam classFontSize 11
skinparam classAttributeFontSize 10

' Enums
enum QueryIntent {
  NO_RAG
  ACKNOWLEDGE
  RAG_SUMMARY
  RAG_CHUNKS
}

enum ProcessingPhase {
  IDLE
  FILE_EXTRACTION
  SUMMARIZATION
  ANALYSIS
  RETRIEVAL
  GENERATION
}

' Data Classes (simplified)
class Chunk <<dataclass>> {
  file_id: str
  file_name: str
  chunk_index: int
  text: str
}

class FileSummary <<dataclass>> {
  file_id: str
  file_name: str
  summary: str
  chunk_count: int
}

class AnalysisResult <<dataclass>> {
  intent: QueryIntent
  reasoning: str
  summary_file_ids: list[str]
  chunk_file_ids: list[str]
  search_query: str
}

class RAGContext <<dataclass>> {
  intent: str
  reasoning: str
  search_query: str
  summary_file_ids: list[str]
  chunk_file_ids: list[str]
  chunks_retrieved: int
}

class CancellationToken <<dataclass>> {
  request_id: str
  +cancel()
  +check_cancelled()
  +set_phase()
}

' Main Classes
class LLMService {
  -_model_path: str
  -_llm: Llama
  -_executor: ThreadPoolExecutor
  -_lock: asyncio.Lock
  +load_model()
  +generate_stream()
  +generate_complete()
  +summarize_chunks()
  +unload_model()
}

class SemanticAnalyzer {
  -_llm: LLMService
  -_lock: asyncio.Lock
  +SYSTEM_PROMPT: str
  +analyze()
  -_build_corpus_section()
  -_build_history_section()
  -_parse_llm_response()
}

class BM25Retriever {
  -_bm25: BM25Okapi
  -_tokenized: list
  +index()
  +retrieve()
  +clear()
}

class SemanticRetriever {
  -_model_path: str
  -_llm: Llama
  -_embeddings: ndarray
  +load_model()
  +compute_embeddings()
  +retrieve()
  +unload_model()
}

class Reranker {
  -_model_path: str
  -_llm: Llama
  -_corpus: Corpus
  +load_model()
  +set_corpus()
  +rerank()
  +unload_model()
}

class RRFFusion {
  -_alpha: int
  +fuse()
}

class HybridRetriever {
  -_bm25: BM25Retriever
  -_semantic: SemanticRetriever
  -_reranker: Reranker
  -_fusion: RRFFusion
  -_corpus: Corpus
  +process()
  +retrieve()
  +is_ready()
  +release_models()
  +clear()
}

class FileProcessor {
  -_converter: DocumentConverter
  -_chunker: HybridChunker
  -_models_loaded: bool
  -_cancel_event: Event
  +load_models()
  +process()
  +release_models()
}

class Corpus {
  +chunks: list[Chunk]
  +summaries: list[FileSummary]
  +add_chunk()
  +add_summary()
  +get_chunks_by_file()
  +get_file_ids()
  +remove_file()
  +save()
  +load()
}

class ChatHistory {
  -_filepath: str
  -_messages: list[ChatMessage]
  +add_turn()
  +get_messages()
  +get_recent_turns()
  +clear()
}

class ChatService {
  -_llm: LLMService
  -_retriever: HybridRetriever
  -_analyzer: SemanticAnalyzer
  -_corpus: Corpus
  -_history: ChatHistory
  +process_message()
  +get_history()
  +clear_history()
  +build_user_message_with_file()
}

class AppState {
  +corpus: Corpus
  +chat_service: ChatService
  +file_processor: FileProcessor
  +retriever: HybridRetriever
  +llm_service: LLMService
  +request_manager: RequestManager
  +save_corpus()
  +reindex_corpus()
  +start_request()
  +cancel_request()
}

' Relationships
SemanticAnalyzer --> LLMService : uses
SemanticAnalyzer ..> AnalysisResult : creates
SemanticAnalyzer ..> QueryIntent : uses

HybridRetriever --> BM25Retriever : contains
HybridRetriever --> SemanticRetriever : contains
HybridRetriever --> Reranker : contains
HybridRetriever --> RRFFusion : contains
HybridRetriever --> Corpus : references

ChatService --> LLMService : uses
ChatService --> HybridRetriever : uses
ChatService --> SemanticAnalyzer : uses
ChatService --> Corpus : uses
ChatService --> ChatHistory : uses
ChatService ..> RAGContext : creates

Corpus --> Chunk : contains
Corpus --> FileSummary : contains

FileProcessor ..> Chunk : creates

AppState --> ChatService : contains
AppState --> FileProcessor : contains
AppState --> HybridRetriever : contains
AppState --> LLMService : contains
AppState --> Corpus : contains

CancellationToken ..> ProcessingPhase : uses

@enduml
```

---

## 5. Đặc tả luồng hoạt động

### 5.1 Luồng 1: Người dùng gửi tin nhắn (không kèm file)

**Mô tả:** Người dùng nhập tin nhắn văn bản và nhận phản hồi streaming.

**Các bước:**
1. Người dùng nhập tin nhắn và gửi
2. API nhận request, tạo CancellationToken
3. ChatService.process_message() được gọi
4. SemanticAnalyzer phân tích ý định query
5. Dựa vào intent, thực hiện retrieval (nếu cần)
6. LLMService sinh phản hồi streaming
7. Response được stream về client qua SSE
8. Lưu turn vào ChatHistory

```plantuml
@startuml
skinparam sequenceMessageAlign center
skinparam responseMessageBelowArrow true

actor User
participant "Browser\n(Frontend)" as Browser
participant "API\n(/api/chat)" as API
participant "AppState" as State
participant "ChatService" as Chat
participant "SemanticAnalyzer" as Analyzer
participant "HybridRetriever" as Retriever
participant "LLMService" as LLM
participant "ChatHistory" as History
database "Corpus" as Corpus

User -> Browser: Nhập tin nhắn
activate Browser

Browser -> API: POST /api/chat\n{message: "...", files: null}
activate API

API -> State: start_request()
activate State
State --> API: CancellationToken
deactivate State

API -> Chat: process_message(message, cancel_token)
activate Chat

== Phase 1: Semantic Analysis ==

Chat -> Chat: build_history_context_for_analyzer()
Chat -> Analyzer: analyze(message, history, corpus, token)
activate Analyzer

Analyzer -> Analyzer: _build_corpus_section(corpus)
Analyzer -> Analyzer: _build_history_section(history)
Analyzer -> LLM: generate_complete(messages, token)
activate LLM
LLM --> Analyzer: JSON response
deactivate LLM

Analyzer -> Analyzer: _parse_llm_response()
Analyzer --> Chat: AnalysisResult\n{intent, file_ids, search_query}
deactivate Analyzer

Chat --> API: yield {status: "Đang phân tích..."}
API --> Browser: SSE: status event

== Phase 2: Context Retrieval (if RAG intent) ==

alt intent == RAG_SUMMARY
    Chat -> Corpus: get_summaries_for_files(file_ids)
    Corpus --> Chat: summaries text
    
else intent == RAG_CHUNKS
    Chat --> API: yield {status: "Đang tìm kiếm..."}
    API --> Browser: SSE: status event
    
    Chat -> Retriever: retrieve(query, k, file_ids, token)
    activate Retriever
    
    Retriever -> Retriever: BM25 search (CPU)
    Retriever -> Retriever: Semantic search (GPU)
    Retriever -> Retriever: RRF fusion
    Retriever -> Retriever: Rerank (GPU)
    
    Retriever --> Chat: RetrievalResults\n{chunks, scores}
    deactivate Retriever
    
    Chat -> Chat: Format chunks as context
end

== Phase 3: Response Generation ==

Chat --> API: yield {status: "Đang tạo phản hồi..."}
API --> Browser: SSE: status event

Chat -> Chat: _build_messages(message, context)
Chat -> LLM: generate_stream(messages, token)
activate LLM

loop for each chunk
    LLM --> Chat: text chunk
    Chat --> API: yield {content: chunk}
    API --> Browser: SSE: content event
    Browser -> Browser: Append to UI
end
deactivate LLM

== Phase 4: Save History ==

Chat -> History: add_turn(user_msg, response, rag_context)
activate History
History -> History: _save()
History --> Chat: done
deactivate History

Chat --> API: yield {done: true}
deactivate Chat

API --> Browser: SSE: done event
deactivate API

Browser -> User: Hiển thị phản hồi hoàn chỉnh
deactivate Browser

@enduml
```

### 5.2 Luồng 2: Người dùng gửi tin nhắn kèm file

**Mô tả:** Người dùng upload file và gửi tin nhắn, hệ thống xử lý file trước rồi trả lời.

**Các bước:**
1. Người dùng chọn file và nhập tin nhắn
2. API nhận request với file
3. FileProcessor xử lý file (extract + chunk)
4. LLMService tóm tắt nội dung
5. Cập nhật Corpus và reindex
6. Tiếp tục xử lý tin nhắn như Luồng 1

```plantuml
@startuml
skinparam sequenceMessageAlign center
skinparam responseMessageBelowArrow true

actor User
participant "Browser" as Browser
participant "API" as API
participant "AppState" as State
participant "FileProcessor" as Processor
participant "LLMService" as LLM
participant "HybridRetriever" as Retriever
participant "ChatService" as Chat
database "Corpus" as Corpus

User -> Browser: Chọn file + Nhập tin nhắn
activate Browser

Browser -> API: POST /api/chat\n{message, files: [file]}
activate API

API -> State: start_request()
State --> API: CancellationToken

== Phase 1: File Processing ==

API --> Browser: SSE: {status: "Đang xử lý file..."}

API -> Processor: process(file_path, file_name, token)
activate Processor

Processor -> Processor: load_models() [if needed]
Processor -> Processor: Convert document (Docling)
Processor -> Processor: Chunk document

Processor --> API: List[Chunk]
deactivate Processor

== Phase 2: Summarization ==

API --> Browser: SSE: {status: "Đang tóm tắt..."}

API -> LLM: summarize_chunks(chunks, token)
activate LLM
LLM --> API: summary text
deactivate LLM

API -> API: Create FileSummary

== Phase 3: Update Corpus & Index ==

API -> Corpus: add_chunks(chunks)
API -> Corpus: add_summary(summary)
API -> State: save_corpus()

API --> Browser: SSE: {status: "Đang index..."}

API -> Retriever: process(corpus, token)
activate Retriever
Retriever -> Retriever: BM25 index
Retriever -> Retriever: Compute embeddings
Retriever --> API: done
deactivate Retriever

== Phase 4: Release Docling VRAM ==

API -> Processor: release_models()

== Phase 5: Chat Processing ==

API -> Chat: build_user_message_with_file(message, summary)
note right: Thêm [FILE] tag\nvào tin nhắn

API -> Chat: process_message(enriched_message, token)
activate Chat

ref over Chat, LLM, Retriever
  Xử lý như Luồng 1
  (Analysis → Retrieval → Generation)
end ref

loop streaming response
    Chat --> API: yield {content: chunk}
    API --> Browser: SSE: content event
end

Chat --> API: yield {done: true}
deactivate Chat

API --> Browser: SSE: done event
deactivate API

Browser -> User: Hiển thị phản hồi
deactivate Browser

@enduml
```

### 5.3 Luồng 3: Người dùng hủy yêu cầu đang xử lý

**Mô tả:** Người dùng nhấn nút Stop để hủy bỏ xử lý đang thực hiện.

```plantuml
@startuml
skinparam sequenceMessageAlign center

actor User
participant "Browser" as Browser
participant "API\n(/api/stop)" as StopAPI
participant "API\n(/api/chat)" as ChatAPI
participant "AppState" as State
participant "CancellationToken" as Token
participant "ChatService" as Chat
participant "LLMService" as LLM

User -> Browser: Click "Stop"
activate Browser

Browser -> StopAPI: POST /api/stop
activate StopAPI

StopAPI -> State: cancel_request()
activate State

State -> Token: cancel()
activate Token

Token -> Token: _cancelled = True
Token -> Token: Invoke callbacks
Token --> State: done
deactivate Token

State --> StopAPI: {cancelled: true}
deactivate State

StopAPI --> Browser: 200 OK
deactivate StopAPI

== Meanwhile in /api/chat ==

note over ChatAPI, LLM
  Đang trong quá trình xử lý...
end note

Chat -> Token: check_cancelled()
Token --> Chat: raises CancelledException

Chat -> Chat: Save partial response\n+ "[Đã dừng]"

Chat --> ChatAPI: yield {cancelled: true}

ChatAPI --> Browser: SSE: cancelled event
Browser -> Browser: Hiển thị "[Đã dừng]"

Browser -> User: Thông báo đã hủy
deactivate Browser

@enduml
```

### 5.4 Luồng 4: Người dùng xóa document

**Mô tả:** Người dùng xóa một file đã upload khỏi hệ thống.

```plantuml
@startuml
skinparam sequenceMessageAlign center

actor User
participant "Browser" as Browser
participant "API\n(/api/documents)" as API
participant "AppState" as State
participant "HybridRetriever" as Retriever
database "Corpus" as Corpus
database "Uploads\nFolder" as Uploads

User -> Browser: Click "Xóa" document
activate Browser

Browser -> API: DELETE /api/documents/{file_id}
activate API

API -> State: acquire lock
activate State

API -> Corpus: remove_file(file_id)
activate Corpus
Corpus -> Corpus: Remove chunks
Corpus -> Corpus: Remove summary
Corpus --> API: done
deactivate Corpus

API -> State: save_corpus()
State -> Corpus: save(path)

API -> Uploads: Delete original file
Uploads --> API: done

alt Corpus not empty
    API -> Retriever: clear()
    activate Retriever
    Retriever -> Retriever: Clear BM25 index
    Retriever -> Retriever: Clear embeddings
    Retriever --> API: done
    deactivate Retriever
    
    API -> State: reindex_corpus(token)
    State -> Retriever: process(corpus, token)
    Retriever --> State: done
    
else Corpus empty
    API -> Retriever: clear()
    note right: Không cần reindex
end

State --> API: release lock
deactivate State

API --> Browser: 200 OK\n{remaining_documents}
deactivate API

Browser -> Browser: Update UI\nRemove document card
Browser -> User: Thông báo xóa thành công
deactivate Browser

@enduml
```

### 5.5 Luồng 5: Người dùng xóa toàn bộ dữ liệu

**Mô tả:** Người dùng reset hệ thống, xóa tất cả chat history và documents.

```plantuml
@startuml
skinparam sequenceMessageAlign center

actor User
participant "Browser" as Browser
participant "API\n(/api/clear)" as API
participant "AppState" as State
participant "ChatService" as Chat
participant "HybridRetriever" as Retriever
database "Corpus" as Corpus
database "ChatHistory" as History
database "Uploads" as Uploads

User -> Browser: Click "Xóa tất cả"
Browser -> Browser: Confirm dialog
User -> Browser: Xác nhận
activate Browser

Browser -> API: POST /api/clear
activate API

API -> State: acquire lock
activate State

== Clear Chat History ==

API -> Chat: clear_history()
activate Chat
Chat -> History: clear()
History -> History: Reset to system prompt only
History -> History: Save empty history
Chat --> API: done
deactivate Chat

== Clear Corpus ==

API -> Corpus: clear()
activate Corpus
Corpus -> Corpus: chunks = []
Corpus -> Corpus: summaries = []
Corpus --> API: done
deactivate Corpus

API -> State: save_corpus()

== Clear Retriever Index ==

API -> Retriever: clear()
activate Retriever
Retriever -> Retriever: Clear BM25
Retriever -> Retriever: Clear embeddings
Retriever -> Retriever: Clear reranker refs
note right: Models vẫn trong VRAM
Retriever --> API: done
deactivate Retriever

== Clear Upload Files ==

API -> Uploads: Delete all files
Uploads --> API: done

State --> API: release lock
deactivate State

API --> Browser: 200 OK\n{message: "Cleared"}
deactivate API

Browser -> Browser: Reload page
Browser -> User: Giao diện sạch
deactivate Browser

@enduml
```

---

## Phụ lục

### A. Cấu trúc thư mục Project

```
chatbot/
├── main.py                 # Entry point
├── pyproject.toml          # Dependencies
├── README.md
│
├── src/
│   ├── core/
│   │   ├── config.py       # Configuration
│   │   ├── models.py       # Data models (Chunk, Corpus, FileSummary)
│   │   └── cancellation.py # Cancellation infrastructure
│   │
│   └── services/
│       ├── llm.py              # LLM service
│       ├── retrieval.py        # Hybrid retrieval
│       ├── semantic_analyzer.py # Query analysis
│       ├── file_processor.py   # Document processing
│       └── chat.py             # Chat orchestration
│
├── api/
│   └── routes.py           # FastAPI endpoints
│
├── models/                 # GGUF model files
│   ├── Ministral-3B-*.gguf
│   ├── bge-m3-*.gguf
│   └── bge-reranker-*.gguf
│
├── document_store/
│   └── documents_index.json # Corpus persistence
│
├── history/
│   └── chat_history.json   # Chat persistence
│
├── uploads/                # Uploaded files
│
├── static/                 # Frontend assets
│   ├── script.js
│   └── style.css
│
└── templates/
    └── index.html          # Main UI
```

### B. API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/chat` | Chat với optional file upload (SSE) |
| `POST` | `/api/stop` | Hủy request đang xử lý |
| `GET` | `/api/history` | Lấy lịch sử chat |
| `POST` | `/api/clear` | Xóa tất cả dữ liệu |
| `GET` | `/api/documents` | Liệt kê documents |
| `DELETE` | `/api/documents/{id}` | Xóa document |
| `GET` | `/api/documents/{id}/download` | Tải file gốc |

### C. Cấu hình hệ thống

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `n_ctx` | 12288 | Context length (tokens) |
| `n_batch` | 3072 | Batch size |
| `chunk_max_tokens` | 512 | Max tokens per chunk |
| `retrieval_top_k` | 5 | Số chunks trả về |
| `rrf_alpha` | 60 | RRF smoothing constant |
| `n_gpu_layers` | -1 | Full GPU offload |
