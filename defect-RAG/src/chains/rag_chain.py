"""RAG Chain for defect analysis (without LangChain)."""
import json
import logging
from typing import List, Dict, Any, Optional
import re

from src.core.llm_client import OpenAICompatibleLLM
from src.core.embedding_engine import OpenAICompatibleEmbedding
from src.core.vector_store import DefectVectorStore
from src.core.reranker import SimpleReranker, LLMReranker, RerankConfig
from src.chains import prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefectRAGChain:
    """RAG chain for defect analysis."""

    def __init__(
        self,
        llm_client: OpenAICompatibleLLM,
        embedding_client: OpenAICompatibleEmbedding,
        vector_store: DefectVectorStore,
        language: str = "zh",
        enable_rerank: bool = True,
        rerank_top_k: int = 10,
        rerank_type: str = "simple"
    ):
        """Initialize RAG chain.

        Args:
            llm_client: LLM client
            embedding_client: Embedding client
            vector_store: Vector store
            language: 'zh' or 'en'
            enable_rerank: Whether to enable re-ranking
            rerank_top_k: Number of results to retrieve before re-ranking
            rerank_type: 'simple' or 'llm'
        """
        self.llm = llm_client
        self.embedding = embedding_client
        self.vector_store = vector_store
        self.language = language
        self.rerank_top_k = rerank_top_k
        self.enable_rerank = enable_rerank
        self.reranker_type = rerank_type  # Store reranker type for runtime updates

        # Initialize re-ranker
        rerank_config = RerankConfig(
            enable_rerank=enable_rerank,
            rerank_top_k=rerank_top_k,
            final_top_k=5  # Default final results
        )

        if rerank_type == "llm":
            self.reranker = LLMReranker(llm_client, rerank_config)
        else:
            self.reranker = SimpleReranker(rerank_config)

        logger.info(f"Initialized RAG chain with language: {language}, rerank: {enable_rerank}")

    def set_language(self, language: str) -> None:
        """Set response language.

        Args:
            language: 'zh' or 'en'
        """
        self.language = language
        logger.info(f"Language set to: {language}")

    def update_retrieval_params(
        self,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
        rerank_type: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> None:
        """Update retrieval parameters at runtime (Phase 1 enhancement).

        Args:
            top_k: Number of final results to return
            rerank_top_k: Number of candidates to retrieve before re-ranking
            enable_rerank: Whether to enable re-ranking
            rerank_type: 'simple' or 'llm'
            similarity_threshold: Minimum similarity score threshold
        """
        # Update final top_k
        if top_k is not None:
            self.reranker.config.final_top_k = top_k
            logger.info(f"Updated final_top_k: {top_k}")

        # Update rerank_top_k
        if rerank_top_k is not None:
            self.rerank_top_k = rerank_top_k
            self.reranker.config.rerank_top_k = rerank_top_k
            logger.info(f"Updated rerank_top_k: {rerank_top_k}")

        # Update enable_rerank
        if enable_rerank is not None:
            self.enable_rerank = enable_rerank
            self.reranker.config.enable_rerank = enable_rerank
            logger.info(f"Updated enable_rerank: {enable_rerank}")

        # Update rerank_type (requires re-initializing reranker)
        if rerank_type is not None:
            current_type = "llm" if isinstance(self.reranker, LLMReranker) else "simple"
            if rerank_type != current_type:
                from src.core.reranker import SimpleReranker, LLMReranker, RerankConfig

                # Create new config preserving current settings
                new_config = RerankConfig(
                    enable_rerank=self.enable_rerank,
                    rerank_top_k=self.rerank_top_k,
                    final_top_k=self.reranker.config.final_top_k
                )

                # Re-initialize reranker
                if rerank_type == "llm":
                    self.reranker = LLMReranker(self.llm, new_config)
                    logger.info("Switched to LLM-based re-ranker")
                else:
                    self.reranker = SimpleReranker(new_config)
                    logger.info("Switched to Simple rule-based re-ranker")

        # Note: similarity_threshold is used in vector_store.search()
        # It should be passed as a filter parameter when calling retrieve()
        if similarity_threshold is not None:
            logger.info(f"Similarity threshold set to: {similarity_threshold} (will be applied in retrieval)")

        logger.info("Retrieval parameters updated successfully")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        enable_rerank: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve similar defects with optional re-ranking.

        Args:
            query: User query
            top_k: Number of final results
            filters: Optional metadata filters
            enable_rerank: Override default re-rank setting

        Returns:
            List of similar defects
        """
        logger.info(f"Retrieving defects for query: {query[:50]}...")

        # Determine re-rank settings
        should_rerank = enable_rerank if enable_rerank is not None else self.enable_rerank

        # Calculate how many to retrieve before re-ranking
        if should_rerank:
            retrieve_k = max(top_k * 2, self.rerank_top_k)
        else:
            retrieve_k = top_k

        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)

        # Search vector store (retrieve more if re-ranking)
        results = self.vector_store.search(
            query=query,
            embeddings=query_embedding,
            top_k=retrieve_k,
            filters=filters
        )

        logger.info(f"Retrieved {len(results)} defects from vector store")

        # Re-rank if enabled
        if should_rerank and len(results) > top_k:
            # Update final_top_k in reranker config
            self.reranker.config.final_top_k = top_k
            results = self.reranker.rerank(query, results)
            logger.info(f"Re-ranked to top {len(results)} results")

        return results

    def analyze(
        self,
        query: str,
        similar_defects: List[Dict[str, Any]],
        llm_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze defects using LLM.

        Args:
            query: User query
            similar_defects: Retrieved similar defects
            llm_params: Optional LLM parameter overrides

        Returns:
            Analysis result dictionary
        """
        # Get prompts
        system_prompt, analysis_prompt = prompts.get_prompts(self.language)

        # Format similar defects
        formatted_defects = prompts.format_similar_defects(similar_defects, self.language)

        # Build user message
        user_message = analysis_prompt.format(
            question=query,
            similar_defects=formatted_defects
        )

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        llm_params = llm_params or {}
        response = self.llm.invoke(messages, **llm_params)

        # Parse JSON response
        return self._parse_response(response, similar_defects)

    def _parse_response(
        self,
        response: str,
        similar_defects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse LLM response to extract JSON and merge with real defect data (Scheme C).

        Args:
            response: Raw LLM response
            similar_defects: Original similar defects from retrieval

        Returns:
            Parsed result dictionary with merged defect data
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            # Parse JSON
            result = json.loads(json_str)

            # Scheme C: Merge LLM response with real retrieved defect data
            result = self._merge_defect_data(result, similar_defects)

            # Add raw response for debugging
            result['_raw_response'] = response

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Return raw response wrapped in structure with real defect data
            return {
                "error": "Failed to parse response",
                "raw_response": response,
                "similar_defects": self._format_defects_for_display(similar_defects)
            }

    def _merge_defect_data(
        self,
        llm_result: Dict[str, Any],
        retrieved_defects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge LLM analysis with real retrieved defect data.

        Args:
            llm_result: Parsed LLM response
            retrieved_defects: Original retrieved defects with real scores and metadata

        Returns:
            Merged result with accurate defect information
        """
        # Build lookup map from retrieved defects by ID
        defect_map = {}
        for defect in retrieved_defects:
            # Try multiple ID fields
            defect_id = None
            if 'metadata' in defect:
                defect_id = str(defect['metadata'].get('Identifier', ''))
            if not defect_id:
                defect_id = str(defect.get('id', ''))

            if defect_id and defect_id != 'None':
                defect_map[defect_id] = defect

        # Merge similar_defects from LLM with real data
        merged_defects = []

        if "similar_defects" in llm_result and isinstance(llm_result["similar_defects"], list):
            for llm_defect in llm_result["similar_defects"]:
                llm_id = str(llm_defect.get('id', ''))

                if llm_id and llm_id in defect_map:
                    real_defect = defect_map[llm_id]
                    meta = real_defect.get('metadata', {})

                    # Use real data for accuracy, LLM data for insight
                    merged_defect = {
                        "id": llm_id,
                        "summary": meta.get('Summary', llm_defect.get('summary', '-')),
                        "similarity_score": real_defect.get('score', 0.0),
                        "key_insight": llm_defect.get('key_insight', ''),
                        "component": meta.get('Component', ''),
                        "category": meta.get('CategoryOfGaps', ''),
                        "customer": meta.get('Customer', '')
                    }
                    merged_defects.append(merged_defect)
                else:
                    # If LLM mentions a defect not in retrieved list, use LLM data
                    logger.warning(f"LLM mentioned defect {llm_id} not found in retrieved results")
                    merged_defects.append(llm_defect)

        # If LLM didn't return any similar_defects, use all retrieved ones
        if not merged_defects and retrieved_defects:
            logger.info("LLM did not return similar_defects, using retrieved defects directly")
            merged_defects = self._format_defects_for_display(retrieved_defects)

        # Update the result with merged data
        llm_result["similar_defects"] = merged_defects

        return llm_result

    def _format_defects_for_display(
        self,
        retrieved_defects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format retrieved defects for display.

        Args:
            retrieved_defects: Raw retrieved defects

        Returns:
            Formatted defects with consistent structure
        """
        formatted = []
        for defect in retrieved_defects:
            meta = defect.get('metadata', {})
            formatted.append({
                "id": str(meta.get('Identifier', defect.get('id', '-'))),
                "summary": meta.get('Summary', '-'),
                "similarity_score": defect.get('score', 0.0),
                "key_insight": "",
                "component": meta.get('Component', ''),
                "category": meta.get('CategoryOfGaps', ''),
                "customer": meta.get('Customer', '')
            })
        return formatted

    def run(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        llm_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run complete RAG pipeline.

        Args:
            query: User query
            top_k: Number of similar defects to retrieve
            filters: Optional metadata filters
            llm_params: Optional LLM parameter overrides

        Returns:
            Complete result with retrieved defects and analysis
        """
        # Step 1: Retrieve similar defects
        similar_defects = self.retrieve(query, top_k, filters)

        if not similar_defects:
            return {
                "error": prompts.get_error_message("no_results", self.language),
                "similar_defects": [],
                "analysis": None
            }

        # Step 2: Analyze with LLM
        analysis = self.analyze(query, similar_defects, llm_params)

        # Step 3: Return combined result
        return {
            "query": query,
            "similar_defects": similar_defects,
            "analysis": analysis,
            "retrieval_count": len(similar_defects),
            "language": self.language
        }

    def chat(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        llm_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Chat with history support.

        Args:
            query: Current user query
            chat_history: Previous conversation history
            top_k: Number of similar defects to retrieve
            llm_params: Optional LLM parameter overrides

        Returns:
            Response with updated history
        """
        chat_history = chat_history or []

        # Run RAG
        result = self.run(query, top_k, llm_params=llm_params)

        # Extract answer for chat history
        if "analysis" in result and isinstance(result["analysis"], dict):
            analysis = result["analysis"]
            if "analysis" in analysis and "probable_root_cause" in analysis["analysis"]:
                answer = analysis["analysis"]["probable_root_cause"]
            else:
                answer = json.dumps(analysis, ensure_ascii=False, indent=2)
        else:
            answer = str(result.get("analysis", "No analysis available"))

        # Update history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        result["chat_history"] = chat_history

        return result
