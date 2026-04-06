# AGENTS.md - Defect RAG System

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (standard)
streamlit run main.py

# Or use the Windows batch file (has hardcoded Python path)
run.bat
```

## Architecture

**App Type**: Streamlit web UI for RAG-based software defect analysis  
**Entry Point**: `main.py` → `src/ui/app.py`  
**Vector DB**: ChromaDB (`./vector_db/`)  
**API Format**: OpenAI Compatible (supports vLLM, Ollama)

### Key Modules

| Module | Purpose |
|--------|---------|
| `config/settings.yaml` | Central configuration (LLM, embedding, retrieval params) |
| `src/core/` | Data loader, embedding engine, vector store, LLM client |
| `src/chains/` | RAG chain logic and prompts |
| `src/ui/` | Streamlit components (chat, sidebar, file upload) |

## Configuration

### Priority (highest first)
1. Environment variables: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, etc.
2. `config/settings.yaml` (default location)
3. `DEFECT_RAG_CONFIG` env var to specify custom config path

### Critical Settings
```yaml
llm:
  base_url: "http://your-endpoint/v1"
  api_key: "your-key"
  model: "your-model"
  verify_ssl: false  # Set false for self-signed certs

embedding:
  base_url: "http://your-endpoint/v1"
  model: "text-embedding-3-large"
  vector_dimension: 1024  # Must match model output

retrieval:
  top_k: 5
  rerank_top_k: 20  # Retrieve more before re-ranking
  rerank_type: "llm"  # 'simple' or 'llm'
```

## Data Format

Upload JSON with defect records. Two formats supported:

1. **Sheet-wrapped**: `{"Sheet1": [{...}]}`
2. **Direct array**: `[{...}]`

Required fields for embedding: `Summary`, `PreClarification`  
Metadata fields stored: `Identifier`, `Region`, `Dept`, `Component`, `CategoryOfGaps`, etc.

## Testing

```bash
# Run integration test (no API calls)
python test_system.py

# Build index test (requires API)
python test_build_index.py
```

## Development Notes

- **Bilingual UI**: Supports `zh` (Chinese) and `en` (English) via `src/utils/lang_detector.py`
- **Session State**: App uses `st.session_state` for config, RAG chain, and index status
- **Auto-initialization**: App auto-loads config from `settings.yaml` on startup
- **Vector Store**: Persistent ChromaDB collection named "defects"
- **Reranking**: Two-phase retrieval - retrieve `rerank_top_k`, then rerank to `top_k`

## Common Tasks

| Task | Command/Action |
|------|----------------|
| Start dev server | `streamlit run main.py` |
| Change LLM model | Edit `config/settings.yaml` or env vars |
| Rebuild index | Upload new JSON in UI → Click "构建索引" |
| Run tests | `python test_system.py` |
| Clear vector DB | Delete `./vector_db/` folder |
