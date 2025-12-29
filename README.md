# ChatDocs

An intelligent document management system with RAG (Retrieval-Augmented Generation) for PDF and DOCX files.

## Overview

ChatDocs allows you to upload documents and chat with them using AI. It automatically parses, chunks, and indexes your documents, then uses retrieval-augmented generation to answer questions based on the content.

## Features

- Upload PDF and DOCX documents via drag & drop
- Automatic text extraction and chunking
- Incremental FAISS index updates
- Chat interface with adjustable retrieval parameters
- Document library with statistics and management
- Local LLM (Llama 3.2 3B) for answer generation

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open browser at `http://localhost:5000`

### Usage

1. **Upload Tab**: Drop PDF or DOCX files to upload
2. **Chat Tab**: Ask questions about your documents
3. **Library Tab**: Manage uploaded documents

## Architecture

**Strategy**: Incremental Index
- Upload: Parse → Chunk → Embed → Extend Index
- Delete: Remove → Rebuild Index
- Re-index: Rebuild complete Index

**Why Incremental?**
- Fast (only embed new documents)
- Scales better
- No re-processing of old documents

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Flask 3.0 |
| Frontend | Vanilla JS + CSS |
| PDF Parsing | PyMuPDF |
| DOCX Parsing | python-docx |
| Chunking | Token-based (Transformers) |
| Embeddings | Sentence-Transformers |
| Vector DB | FAISS (CPU) |
| LLM | Llama 3.2 3B (4-bit quantized) |

## Configuration

### RAG System

```python
# core/rag_system.py
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
llm_model = 'meta-llama/Llama-3.2-3B-Instruct'
```

### Chunking

```python
# chunking/flat_chunker.py
chunk_size = 512        # Tokens per chunk
chunk_overlap = 50      # Overlap
```

### Flask

```python
# app.py
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.run(port=5000)
```

## API Endpoints

```
POST   /api/chat              - Chat query
POST   /api/upload            - Upload document
GET    /api/documents         - List documents
DELETE /api/documents/<id>    - Delete document
POST   /api/reindex           - Rebuild index
GET    /api/statistics        - System stats
```

## Performance

- Upload + Processing: 5-30 sec (depending on file size)
- Index Update: <5 sec (incremental)
- Query Processing: 2-5 sec
- Memory: ~4-6 GB RAM (LLM)

## Deployment

### Production

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

## Security

For production deployments:
- Change `app.secret_key`
- Disable `debug=True`
- Implement authentication
- Validate file uploads
- Add rate limiting
- Use HTTPS

## Credits

- FAISS: Facebook AI Similarity Search
- Sentence Transformers: HuggingFace
- Llama 3.2: Meta AI
- PyMuPDF: Artifex Software
- Flask: Pallets Projects

## License

MIT License
