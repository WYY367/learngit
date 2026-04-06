"""Data loading and processing module for defect data."""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefectDataLoader:
    """Load and process defect data from JSON files."""

    def __init__(self, data_path: str):
        """Initialize data loader.

        Args:
            data_path: Path to JSON file containing defect data
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None

    def load(self) -> pd.DataFrame:
        """Load data from JSON file.

        Returns:
            DataFrame with defect records

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict) and 'Sheet1' in data:
            # Excel-exported JSON format
            records = data['Sheet1']
        elif isinstance(data, list):
            # Direct list format
            records = data
        elif isinstance(data, dict):
            # Single record or other structure
            records = [data]
        else:
            raise ValueError("Unsupported JSON structure")

        self.raw_data = pd.DataFrame(records)
        logger.info(f"Loaded {len(self.raw_data)} records")

        return self.raw_data

    def validate(self, df: Optional[pd.DataFrame] = None) -> bool:
        """Validate data structure and required fields.

        Args:
            df: DataFrame to validate. Uses self.raw_data if None.

        Returns:
            True if valid, raises ValueError otherwise
        """
        df = df or self.raw_data
        if df is None:
            raise ValueError("No data loaded. Call load() first.")

        # Required fields
        required_fields = ['Identifier', 'Summary']
        missing_fields = [f for f in required_fields if f not in df.columns]

        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Check for duplicates
        duplicates = df['Identifier'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate Identifier values")

        logger.info("Data validation passed")
        return True

    def process(
        self,
        text_fields: List[str] = None,
        metadata_fields: List[str] = None
    ) -> pd.DataFrame:
        """Process raw data for embedding and retrieval.

        Args:
            text_fields: Fields to include in searchable text
            metadata_fields: Fields to store as metadata

        Returns:
            Processed DataFrame with 'searchable_text' and 'metadata' columns
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load() first.")

        df = self.raw_data.copy()

        # Default fields
        text_fields = text_fields or ['Summary', 'PreClarification', 'ImprovementMeasures']
        metadata_fields = metadata_fields or [
            'Identifier', 'Summary', 'Region', 'Dept', 'Component',
            'CategoryOfGaps', 'SubCategoryOfGaps', 'Customer', 'ResolutionDate'
        ]

        # Build searchable text
        def build_searchable_text(row):
            parts = []
            for field in text_fields:
                if field in row and pd.notna(row[field]) and row[field]:
                    value = str(row[field]).strip()
                    if value and value.lower() != 'nan':
                        parts.append(f"[{field}] {value}")
            return "\n\n".join(parts)

        df['searchable_text'] = df.apply(build_searchable_text, axis=1)

        # Build metadata
        def build_metadata(row):
            metadata = {}
            for field in metadata_fields:
                if field in row:
                    value = row[field]
                    # Handle NaN and None
                    if pd.isna(value):
                        metadata[field] = None
                    else:
                        metadata[field] = value
            return metadata

        df['metadata'] = df.apply(build_metadata, axis=1)

        # Clean data
        df = df.dropna(subset=['searchable_text'])
        df = df[df['searchable_text'].str.strip() != '']

        self.processed_data = df
        logger.info(f"Processed {len(df)} records with searchable text")

        return df

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data.

        Returns:
            Dictionary with statistics
        """
        if self.raw_data is None:
            return {"error": "No data loaded"}

        stats = {
            "total_records": len(self.raw_data),
            "columns": list(self.raw_data.columns),
            "processed_records": len(self.processed_data) if self.processed_data is not None else 0
        }

        # Category statistics
        if 'CategoryOfGaps' in self.raw_data.columns:
            stats['categories'] = self.raw_data['CategoryOfGaps'].value_counts().to_dict()

        # Component statistics
        if 'Component' in self.raw_data.columns:
            stats['top_components'] = self.raw_data['Component'].value_counts().head(10).to_dict()

        # Customer statistics
        if 'Customer' in self.raw_data.columns:
            stats['customers'] = self.raw_data['Customer'].value_counts().head(10).to_dict()

        return stats


def load_defect_data(
    data_path: str,
    text_fields: List[str] = None,
    metadata_fields: List[str] = None
) -> pd.DataFrame:
    """Convenience function to load and process defect data.

    Args:
        data_path: Path to JSON file
        text_fields: Fields for searchable text
        metadata_fields: Fields for metadata

    Returns:
        Processed DataFrame
    """
    loader = DefectDataLoader(data_path)
    loader.load()
    loader.validate()
    return loader.process(text_fields, metadata_fields)
