"""Vector store using ChromaDB."""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefectVectorStore:
    """Vector store for defect data using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./vector_db",
        collection_name: str = "defects"
    ):
        """Initialize vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # Create directory if not exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with minimal configuration
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        logger.info(f"Initialized vector store: {collection_name}")

    def add_defects(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Add defect records to vector store.

        Args:
            texts: List of searchable texts
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (uses UUID if not provided)
            embeddings: Optional pre-computed embeddings

        Returns:
            List of IDs for added records
        """
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")

        if embeddings is not None and len(embeddings) != len(texts):
            raise ValueError("embeddings must have same length as texts")

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Clean metadata - ChromaDB doesn't allow None values
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for key, value in meta.items():
                if value is None:
                    cleaned[key] = ""  # Replace None with empty string
                elif isinstance(value, (int, float, bool, str)):
                    cleaned[key] = value
                else:
                    cleaned[key] = str(value)  # Convert other types to string
            cleaned_metadatas.append(cleaned)

        # Add to collection
        if embeddings is not None:
            # Use pre-computed embeddings
            self.collection.add(
                documents=texts,
                metadatas=cleaned_metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            # Let ChromaDB handle embeddings
            self.collection.add(
                documents=texts,
                metadatas=cleaned_metadatas,
                ids=ids
            )

        logger.info(f"Added {len(texts)} defects to vector store")
        return ids

    def update_defects(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """Update existing defect records.

        Args:
            texts: List of searchable texts
            metadatas: List of metadata dictionaries
            ids: List of IDs to update
            embeddings: Optional pre-computed embeddings
        """
        if len(texts) != len(metadatas) or len(texts) != len(ids):
            raise ValueError("texts, metadatas, and ids must have same length")

        if embeddings is not None and len(embeddings) != len(texts):
            raise ValueError("embeddings must have same length as texts")

        # Clean metadata
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for key, value in meta.items():
                if value is None:
                    cleaned[key] = ""
                elif isinstance(value, (int, float, bool, str)):
                    cleaned[key] = value
                else:
                    cleaned[key] = str(value)
            cleaned_metadatas.append(cleaned)

        # Update collection
        if embeddings is not None:
            self.collection.update(
                documents=texts,
                metadatas=cleaned_metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            self.collection.update(
                documents=texts,
                metadatas=cleaned_metadatas,
                ids=ids
            )

        logger.info(f"Updated {len(texts)} defects")

    def search(
        self,
        query: str,
        embeddings: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar defects.

        Args:
            query: Query text
            embeddings: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with scores and metadata
        """
        # Prepare where clause for filtering
        where_clause = None
        if filters:
            where_clause = self._build_where_clause(filters)

        # Query collection
        results = self.collection.query(
            query_embeddings=[embeddings],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)

        return formatted_results

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters.

        Args:
            filters: Dictionary of field -> value mappings

        Returns:
            ChromaDB where clause
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values - use $in operator
                conditions.append({key: {"$in": value}})
            else:
                # Single value
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    def get_existing_ids(self) -> set:
        """Get all existing defect IDs in the store.

        Returns:
            Set of existing IDs
        """
        # Get all records
        result = self.collection.get(include=[])
        return set(result['ids']) if result['ids'] else set()

    def count(self) -> int:
        """Get total number of records.

        Returns:
            Number of records
        """
        return self.collection.count()

    def delete_all(self) -> None:
        """Delete all records from collection."""
        # Get all IDs and delete
        result = self.collection.get(include=[])
        if result['ids']:
            self.collection.delete(ids=result['ids'])
            logger.info(f"Deleted {len(result['ids'])} records")

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete specific records by ID.

        Args:
            ids: List of IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} records")

    def peek(self, n: int = 5) -> List[Dict[str, Any]]:
        """Peek at n records from the collection.

        Args:
            n: Number of records to peek

        Returns:
            List of records
        """
        results = self.collection.peek(limit=n)
        formatted = []
        for i in range(len(results['ids'])):
            formatted.append({
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        return formatted
