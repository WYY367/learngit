"""Index management for defect data."""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd

from src.core.data_loader import DefectDataLoader
from src.core.embedding_engine import OpenAICompatibleEmbedding
from src.core.vector_store import DefectVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexManager:
    """Manager for building and updating defect vector index."""

    def __init__(
        self,
        embedding_client: OpenAICompatibleEmbedding,
        vector_store: DefectVectorStore
    ):
        """Initialize index manager.

        Args:
            embedding_client: Embedding client
            vector_store: Vector store
        """
        self.embedding = embedding_client
        self.vector_store = vector_store

    def build_index(
        self,
        data_path: str,
        text_fields: List[str] = None,
        metadata_fields: List[str] = None,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """Build or update vector index from data file.

        Args:
            data_path: Path to JSON data file
            text_fields: Fields to use for searchable text
            metadata_fields: Fields to store as metadata
            incremental: If True, only add/update new records

        Returns:
            Statistics about the operation
        """
        logger.info(f"Building index from {data_path}")

        # Load and process data
        loader = DefectDataLoader(data_path)
        df = loader.load()
        loader.validate()
        df = loader.process(text_fields, metadata_fields)

        # Get existing IDs if incremental
        existing_ids = set()
        if incremental:
            existing_ids = self.vector_store.get_existing_ids()
            logger.info(f"Found {len(existing_ids)} existing records")

        # Filter new records
        df['id'] = df['Identifier'].astype(str)

        if incremental and existing_ids:
            new_df = df[~df['id'].isin(existing_ids)]
            update_df = df[df['id'].isin(existing_ids)]
        else:
            new_df = df
            update_df = pd.DataFrame()

        stats = {
            "total_in_file": len(df),
            "new_records": len(new_df),
            "updated_records": len(update_df),
            "unchanged": len(existing_ids) - len(update_df) if incremental else 0
        }

        # Process new records
        if len(new_df) > 0:
            logger.info(f"Adding {len(new_df)} new records")
            self._add_records(new_df)

        # Update existing records
        if len(update_df) > 0:
            logger.info(f"Updating {len(update_df)} existing records")
            self._update_records(update_df)

        stats["total_in_index"] = self.vector_store.count()
        logger.info(f"Index build complete. Total records: {stats['total_in_index']}")

        return stats

    def _add_records(self, df: pd.DataFrame) -> None:
        """Add new records to vector store.

        Args:
            df: DataFrame with new records
        """
        texts = df['searchable_text'].tolist()
        metadatas = df['metadata'].tolist()
        ids = df['id'].tolist()

        # Generate embeddings
        embeddings = self.embedding.embed_texts(texts)

        # Add to vector store with embeddings
        self.vector_store.add_defects(texts, metadatas, ids, embeddings)

    def _update_records(self, df: pd.DataFrame) -> None:
        """Update existing records in vector store.

        Args:
            df: DataFrame with records to update
        """
        texts = df['searchable_text'].tolist()
        metadatas = df['metadata'].tolist()
        ids = df['id'].tolist()

        # Generate embeddings
        embeddings = self.embedding.embed_texts(texts)

        # Update in vector store
        self.vector_store.update_defects(texts, metadatas, ids, embeddings)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with statistics
        """
        count = self.vector_store.count()

        # Peek at some records
        sample = self.vector_store.peek(3)

        return {
            "total_records": count,
            "collection_name": self.vector_store.collection_name,
            "persist_directory": str(self.vector_store.persist_directory),
            "sample_records": sample
        }

    def reset_index(self) -> None:
        """Clear all data from index."""
        logger.warning("Resetting index - all data will be deleted")
        self.vector_store.delete_all()
        logger.info("Index reset complete")

    def search_by_identifier(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Search for a specific defect by identifier.

        Args:
            identifier: Defect identifier

        Returns:
            Defect record if found, None otherwise
        """
        # Get all records and filter
        result = self.vector_store.collection.get(
            where={"Identifier": identifier},
            include=["documents", "metadatas"]
        )

        if result['ids'] and len(result['ids']) > 0:
            return {
                'id': result['ids'][0],
                'text': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        return None
