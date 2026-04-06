"""Simple Re-ranker for retrieved defects.

Implements lightweight re-ranking strategies without external dependencies.
"""
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RerankConfig:
    """Configuration for re-ranking."""
    enable_rerank: bool = True
    rerank_top_k: int = 10  # 先召回更多，再重排序取top_k
    final_top_k: int = 5    # 最终返回的结果数
    keyword_weight: float = 0.3  # 关键词匹配权重
    vector_weight: float = 0.7   # 向量相似度权重
    min_keyword_score: float = 0.1  # 最小关键词匹配分数


class SimpleReranker:
    """Simple rule-based re-ranker.

    Combines:
    1. Vector similarity score (from original retrieval)
    2. Keyword matching score (exact + partial matches)
    3. Field importance weighting
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        """Initialize re-ranker.

        Args:
            config: Rerank configuration
        """
        self.config = config or RerankConfig()
        logger.info(f"Initialized SimpleReranker: enable={self.config.enable_rerank}")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank retrieved results.

        Args:
            query: User query
            results: Retrieved results with similarity scores

        Returns:
            Re-ranked and filtered results
        """
        if not self.config.enable_rerank or not results:
            return results[:self.config.final_top_k]

        logger.info(f"Re-ranking {len(results)} results for query: {query[:50]}...")

        # Extract keywords from query
        query_keywords = self._extract_keywords(query)

        # Score each result
        scored_results = []
        for result in results:
            # Get original vector similarity score (normalized to 0-1)
            vector_score = result.get('similarity', 0.5)

            # Calculate keyword matching score
            keyword_score = self._calculate_keyword_score(
                query_keywords, result
            )

            # Combine scores
            combined_score = (
                self.config.vector_weight * vector_score +
                self.config.keyword_weight * keyword_score
            )

            # Add score breakdown for debugging
            result['_rerank_scores'] = {
                'vector_score': round(vector_score, 4),
                'keyword_score': round(keyword_score, 4),
                'combined_score': round(combined_score, 4)
            }

            scored_results.append((combined_score, result))

        # Sort by combined score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Return top results
        reranked = [result for _, result in scored_results[:self.config.final_top_k]]

        top_score = scored_results[0][0] if scored_results else 0
        logger.info(f"Re-ranking complete. Top score: {top_score:.4f}")

        return reranked

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Normalize text
        text = text.lower().strip()

        # Split by common delimiters
        # Keep alphanumeric and Chinese characters
        words = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text)

        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'and', 'but', 'or', 'yet', 'so', 'if',
                      'because', 'although', 'though', 'while', 'where', 'when',
                      'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
                      'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                      'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
                      'our', 'their', '的', '了', '在', '是', '我', '有', '和',
                      '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
                      '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        keywords = []
        for word in words:
            if len(word) >= 2 and word not in stop_words:
                keywords.append(word)

        return keywords

    def _calculate_keyword_score(
        self,
        query_keywords: List[str],
        result: Dict[str, Any]
    ) -> float:
        """Calculate keyword matching score.

        Args:
            query_keywords: Extracted keywords from query
            result: Retrieved result with metadata and content

        Returns:
            Keyword matching score (0-1)
        """
        if not query_keywords:
            return 0.5

        # Get text fields to match against
        text_fields = []

        # Add metadata fields
        metadata = result.get('metadata', {})
        for key in ['Summary', 'Component', 'CategoryOfGaps', 'SubCategoryOfGaps']:
            if key in metadata and metadata[key]:
                text_fields.append(str(metadata[key]).lower())

        # Add content field if available
        if 'content' in result and result['content']:
            text_fields.append(str(result['content']).lower())

        if not text_fields:
            return self.config.min_keyword_score

        # Calculate matches
        total_weight = 0
        matched_weight = 0

        # Field weights - title/summary is more important
        field_weights = [1.5, 1.2, 1.0, 1.0, 0.8]  # Extra weights for important fields

        for i, field_text in enumerate(text_fields):
            weight = field_weights[i] if i < len(field_weights) else 0.5
            total_weight += weight

            # Check each keyword
            field_matched = False
            for keyword in query_keywords:
                if keyword in field_text:
                    matched_weight += weight * (len(keyword) / 10 + 0.5)  # Longer keywords get higher score
                    field_matched = True

            if field_matched:
                matched_weight += weight * 0.3  # Bonus for matching any keyword in this field

        # Calculate score
        if total_weight == 0:
            return self.config.min_keyword_score

        # Normalize by number of keywords
        keyword_score = min(1.0, matched_weight / (len(query_keywords) * 0.5))

        # Ensure minimum score
        return max(self.config.min_keyword_score, keyword_score)


class LLMReranker:
    """LLM-based re-ranker (optional, higher quality but slower).

    Uses LLM to evaluate relevance of retrieved documents.
    """

    def __init__(self, llm_client, config: Optional[RerankConfig] = None):
        """Initialize LLM re-ranker.

        Args:
            llm_client: LLM client for scoring
            config: Rerank configuration
        """
        self.llm = llm_client
        self.config = config or RerankConfig()
        logger.info(f"Initialized LLMReranker: enable={self.config.enable_rerank}")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank using LLM scoring with progress tracking (Phase 1 enhancement).

        Args:
            query: User query
            results: Retrieved results

        Returns:
            Re-ranked results
        """
        if not self.config.enable_rerank or not results:
            return results[:self.config.final_top_k]

        logger.info(f"LLM Re-ranking {len(results)} results...")

        # Score each result with LLM
        scored_results = []
        total = len(results)

        for i, result in enumerate(results, 1):
            score = self._llm_score(query, result)

            # Get original score from vector similarity (0-1 range)
            original_score = result.get('score', result.get('similarity', 0.5))

            # Adaptive weighting: trust LLM more when scores differ significantly
            score_diff = abs(score - original_score)
            llm_weight = min(0.8, 0.5 + score_diff * 0.3)  # 0.5 to 0.8 based on disagreement
            original_weight = 1 - llm_weight

            combined_score = llm_weight * score + original_weight * original_score

            result['_rerank_scores'] = {
                'llm_score': round(score, 4),
                'original_score': round(original_score, 4),
                'combined_score': round(combined_score, 4),
                'llm_weight': round(llm_weight, 2),
                'rank': i
            }

            scored_results.append((combined_score, result))

            # Log progress for large batches
            if i % 5 == 0 or i == total:
                logger.info(f"LLM scoring progress: {i}/{total}")

        # Sort by combined score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Mark final rank
        for rank, (_, result) in enumerate(scored_results[:self.config.final_top_k], 1):
            if '_rerank_scores' in result:
                result['_rerank_scores']['final_rank'] = rank

        top_score = scored_results[0][0] if scored_results else 0
        logger.info(f"LLM Re-ranking complete. Top score: {top_score:.4f}")

        return [result for _, result in scored_results[:self.config.final_top_k]]

    def _llm_score(self, query: str, result: Dict[str, Any]) -> float:
        """Get relevance score from LLM with improved prompt (Phase 1 enhancement).

        Args:
            query: User query
            result: Retrieved result

        Returns:
            Score between 0 and 1
        """
        # Get document text
        metadata = result.get('metadata', {})
        summary = metadata.get('Summary', '')
        category = metadata.get('CategoryOfGaps', '')
        subcategory = metadata.get('SubCategoryOfGaps', '')
        component = metadata.get('Component', '')
        content = result.get('text', '')[:400]  # Truncate for efficiency

        # Build improved prompt with structured scoring criteria
        prompt = f"""As a software defect analysis expert, evaluate the relevance of the following historical defect to the user's query.

## User Query
{query}

## Historical Defect
- **ID**: {metadata.get('Identifier', 'N/A')}
- **Summary**: {summary}
- **Component**: {component}
- **Category**: {category} / {subcategory}
- **Description**:
{content}

## Scoring Criteria (0-10 scale)
- **10**: Perfect match - same root cause, component, and context
- **8-9**: Highly relevant - very similar issue with applicable solution
- **6-7**: Moderately relevant - related but some key differences
- **4-5**: Weakly relevant - superficial similarity only
- **1-3**: Minimally relevant - tangentially related
- **0**: Completely irrelevant - no connection

## Instructions
1. First, briefly analyze why this defect is or isn't relevant (1-2 sentences)
2. Then provide a score from 0-10 based on the criteria above
3. Be consistent and objective in your scoring

## Output Format
Analysis: [Your analysis here]
Score: [0-10]"""

        try:
            response = self.llm.invoke(
                [{"role": "user", "content": prompt}],
                temperature=0.0,  # Ensure consistency
                max_tokens=150
            )

            # Extract score using multiple patterns for robustness
            score_patterns = [
                r'Score[:\s]+(\d+)',  # "Score: 8" or "Score 8"
                r'评分[:\s]+(\d+)',   # Chinese "评分: 8"
                r'[\(\[]?(\d+)[/\/]10[\)\]]?',  # "8/10" or "(8/10)"
                r'^(\d+)$',  # Just a number on its own line
                r'[:\s](\d+)\s*(?:分|points)?[\.\s]*$',  # "8分" or "8 points"
            ]

            for pattern in score_patterns:
                score_match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
                if score_match:
                    score = int(score_match.group(1))
                    normalized_score = max(0.0, min(1.0, score / 10.0))
                    logger.debug(f"LLM score: {score}/10 -> {normalized_score:.2f}")
                    return normalized_score

            # Fallback: find any number 0-10 in the response
            numbers = re.findall(r'\b([0-9]|10)\b', response)
            if numbers:
                score = int(numbers[-1])  # Take the last number found
                normalized_score = max(0.0, min(1.0, score / 10.0))
                logger.warning(f"Using fallback score extraction: {score}/10 -> {normalized_score:.2f}")
                return normalized_score

            logger.warning(f"Could not extract score from response: {response[:100]}...")

        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")

        return 0.5  # Default medium score on failure


def create_reranker(
    rerank_type: str = "simple",
    llm_client=None,
    config: Optional[RerankConfig] = None
):
    """Factory function to create re-ranker.

    Args:
        rerank_type: 'simple' or 'llm'
        llm_client: LLM client (required for 'llm' type)
        config: Rerank configuration

    Returns:
        Re-ranker instance
    """
    if rerank_type == "llm" and llm_client:
        return LLMReranker(llm_client, config)
    else:
        return SimpleReranker(config)
