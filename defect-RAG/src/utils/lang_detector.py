"""Language detection utility."""
import logging
from typing import Optional

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    # Set seed for reproducible results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, using simple language detection")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language of text.

    Args:
        text: Input text

    Returns:
        Language code: 'zh' for Chinese, 'en' for English, etc.
    """
    if not text or len(text.strip()) == 0:
        return "en"  # Default to English

    # Simple heuristic first (fast path)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len([c for c in text if c.isalpha()])

    if total_chars > 0 and chinese_chars / total_chars > 0.3:
        return "zh"

    # Use langdetect if available
    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            # Map to our supported languages
            if lang in ['zh-cn', 'zh-tw', 'zh-hk', 'zh']:
                return "zh"
            elif lang in ['en']:
                return "en"
            else:
                return "en"  # Default to English for unsupported languages
        except LangDetectException:
            logger.warning("Language detection failed, using default")
            return "en"
    else:
        # Fallback: check for Chinese characters
        if chinese_chars > 0:
            return "zh"
        return "en"


def get_ui_text(language: str, key: str) -> str:
    """Get UI text for specified language.

    Args:
        language: 'zh' or 'en'
        key: Text key

    Returns:
        Localized text
    """
    texts = {
        "zh": {
            "app_title": "🔍 软件缺陷分析RAG系统",
            "app_description": "基于历史缺陷数据的智能分析系统",
            "sidebar_config": "⚙️ 配置",
            "sidebar_data": "📁 数据管理",
            "chat_input_placeholder": "描述您遇到的缺陷问题...",
            "send_button": "发送",
            "analysis_result": "📊 分析结果",
            "similar_defects": "🔍 相似缺陷",
            "settings": "设置",
            "language": "语言",
            "temperature": "Temperature",
            "max_tokens": "最大Token数",
            "top_k": "检索数量",
            "enable_rerank": "启用重排序",
            "rerank_help": "使用规则或LLM重新排序检索结果，提升准确性 (LLM更准但更慢)",
            "rerank_type": "重排序类型",
            "rerank_top_k": "重排序候选数",
            "similarity_threshold": "相似度阈值",
            "retrieval_settings": "检索设置",
            "upload_file": "上传缺陷数据文件",
            "build_index": "构建索引",
            "reset_index": "重置索引",
            "index_stats": "索引统计",
            "api_config": "API配置",
            "base_url": "API地址",
            "api_key": "API密钥",
            "model_name": "模型名称",
            "save_config": "保存配置",
            "thinking": "思考中...",
            "error": "错误",
            "success": "成功",
            "warning": "警告",
            "info": "信息"
        },
        "en": {
            "app_title": "🔍 Software Defect Analysis RAG System",
            "app_description": "Intelligent defect analysis based on historical data",
            "sidebar_config": "⚙️ Configuration",
            "sidebar_data": "📁 Data Management",
            "chat_input_placeholder": "Describe your defect issue...",
            "send_button": "Send",
            "analysis_result": "📊 Analysis Result",
            "similar_defects": "🔍 Similar Defects",
            "settings": "Settings",
            "language": "Language",
            "temperature": "Temperature",
            "max_tokens": "Max Tokens",
            "top_k": "Retrieval Count",
            "enable_rerank": "Enable Re-ranking",
            "rerank_help": "Re-rank results using rules or LLM for better accuracy (LLM is slower but more accurate)",
            "rerank_type": "Re-rank Type",
            "rerank_top_k": "Re-rank Candidates",
            "similarity_threshold": "Similarity Threshold",
            "retrieval_settings": "Retrieval Settings",
            "upload_file": "Upload Defect Data File",
            "build_index": "Build Index",
            "reset_index": "Reset Index",
            "index_stats": "Index Statistics",
            "api_config": "API Configuration",
            "base_url": "API Base URL",
            "api_key": "API Key",
            "model_name": "Model Name",
            "save_config": "Save Configuration",
            "thinking": "Thinking...",
            "error": "Error",
            "success": "Success",
            "warning": "Warning",
            "info": "Info"
        }
    }

    lang = language if language in texts else "en"
    return texts[lang].get(key, key)
