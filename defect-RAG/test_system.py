"""System integration test for Defect RAG."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Defect RAG System Integration Test")
print("=" * 60)

# Test 1: Config loading
print("\n[1/6] Testing config loading...")
try:
    from config.config_loader import get_config
    config = get_config()
    print(f"  [OK] Config loaded: LLM model={config.llm.model}")
except Exception as e:
    print(f"  [FAIL] Config error: {e}")

# Test 2: Data loading
print("\n[2/6] Testing data loading...")
try:
    from src.core.data_loader import DefectDataLoader
    loader = DefectDataLoader('./data (6).json')
    df = loader.load()
    loader.validate()
    df_processed = loader.process()
    print(f"  [OK] Data loaded: {len(df_processed)} records processed")
except Exception as e:
    print(f"  [FAIL] Data error: {e}")

# Test 3: Vector store
print("\n[3/6] Testing vector store...")
try:
    from src.core.vector_store import DefectVectorStore
    store = DefectVectorStore('./vector_db', 'defects')
    count = store.count()
    print(f"  [OK] Vector store ready: {count} records indexed")
except Exception as e:
    print(f"  [FAIL] Vector store error: {e}")

# Test 4: Prompts
print("\n[4/6] Testing prompts...")
try:
    from src.chains.prompts import get_prompts, format_similar_defects
    system, analysis = get_prompts('zh')
    print(f"  [OK] Prompts loaded (zh): system={len(system)} chars")
    system_en, analysis_en = get_prompts('en')
    print(f"  [OK] Prompts loaded (en): system={len(system_en)} chars")
except Exception as e:
    print(f"  [FAIL] Prompts error: {e}")

# Test 5: Language detection
print("\n[5/6] Testing language detection...")
try:
    from src.utils.lang_detector import detect_language
    zh_text = "这是一个中文测试"
    en_text = "This is an English test"
    print(f"  [OK] Language detection: zh_text -> {detect_language(zh_text)}")
    print(f"  [OK] Language detection: en_text -> {detect_language(en_text)}")
except Exception as e:
    print(f"  [FAIL] Language detection error: {e}")

# Test 6: RAG Chain initialization
print("\n[6/6] Testing RAG chain...")
try:
    from src.chains.rag_chain import DefectRAGChain
    from src.core.llm_client import OpenAICompatibleLLM
    from src.core.embedding_engine import OpenAICompatibleEmbedding
    from src.core.vector_store import DefectVectorStore

    # Create mock clients (won't call API without key)
    llm = OpenAICompatibleLLM(
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        model="test-model"
    )
    embedding = OpenAICompatibleEmbedding(
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        model="test-model"
    )
    store = DefectVectorStore('./vector_db', 'defects')

    rag = DefectRAGChain(llm, embedding, store, language='zh')
    print(f"  [OK] RAG chain initialized (language={rag.language})")
except Exception as e:
    print(f"  [FAIL] RAG chain error: {e}")

print("\n" + "=" * 60)
print("Integration test completed!")
print("=" * 60)
