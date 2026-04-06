import sys
sys.path.insert(0, '.')

from src.core.data_loader import DefectDataLoader
from config.config_loader import get_config

# Load config
config = get_config()
print("Config text_fields:", config.data.text_fields)
print("Config metadata_fields:", config.data.metadata_fields)

# Load data
loader = DefectDataLoader(config.data.default_data_path)
df = loader.load()
print("\nData columns:", list(df.columns))
print("\nFirst row Summary:")
print(repr(df['Summary'].iloc[0]))

# Process data
df_processed = loader.process(
    text_fields=config.data.text_fields,
    metadata_fields=config.data.metadata_fields
)

print("\nFirst processed record:")
print("searchable_text (first 300 chars):")
print(df_processed['searchable_text'].iloc[0][:300])
print("\nmetadata:")
print(df_processed['metadata'].iloc[0])
