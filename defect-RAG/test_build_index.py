"""Test index building with actual data (requires API)."""
import sys
sys.path.insert(0, '.')

print("Testing index building...")
print("NOTE: This requires a running API endpoint")
print()

# Check if we should proceed
response = input("Do you have a running API endpoint? (yes/no): ")
if response.lower() != 'yes':
    print("Skipping index build test.")
    sys.exit(0)

from config.config_loader import get_config
from src.core.embedding_engine import OpenAICompatibleEmbedding
from src.core.vector_store import DefectVectorStore
from src.core.index_manager import IndexManager

# Load config
config = get_config()

print(f"\nAPI Configuration:")
print(f"  Embedding URL: {config.embedding.base_url}")
print(f"  Embedding Model: {config.embedding.model}")

# Initialize clients
print("\nInitializing clients...")
try:
    embedding_client = OpenAICompatibleEmbedding(
        base_url=config.embedding.base_url,
        api_key=config.embedding.api_key,
        model=config.embedding.model
    )
    print("  [OK] Embedding client initialized")
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

vector_store = DefectVectorStore(
    persist_directory=config.vector_store.persist_directory,
    collection_name=config.vector_store.collection_name
)
print(f"  [OK] Vector store ready (current count: {vector_store.count()})")

# Build index
print("\nBuilding index...")
index_manager = IndexManager(embedding_client, vector_store)

try:
    stats = index_manager.build_index(
        data_path=config.data.default_data_path,
        text_fields=config.data.text_fields,
        metadata_fields=config.data.metadata_fields,
        incremental=True
    )
    print("\nIndex build statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
