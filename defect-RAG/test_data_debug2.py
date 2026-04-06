import sys
sys.path.insert(0, '.')

from src.core.data_loader import DefectDataLoader
from config.config_loader import get_config

# Load config
config = get_config()

# Load and process data
loader = DefectDataLoader(config.data.default_data_path)
df = loader.load()
df_processed = loader.process(
    text_fields=config.data.text_fields,
    metadata_fields=config.data.metadata_fields
)

# Check metadata
first_meta = df_processed['metadata'].iloc[0]
print("First record metadata keys:", list(first_meta.keys()))
print("\nSummary in metadata:", first_meta.get('Summary', 'NOT FOUND'))
print("\nAll metadata:")
for k, v in first_meta.items():
    v_str = str(v)[:80] if v else "None"
    print(f"  {k}: {v_str}")
