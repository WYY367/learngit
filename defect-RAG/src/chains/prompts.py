"""Prompt templates for defect analysis (Chinese & English)."""

# ============================================
# Chinese Prompts
# ============================================

ZH_SYSTEM_PROMPT = """你是一个专业的软件缺陷分析专家。你的任务是基于历史缺陷数据，分析用户报告的问题，并提供专业的根因分析和建议。

分析原则：
1. 仔细理解用户描述的问题
2. 参考相似的历史缺陷，找出共同点和差异
3. 基于缺陷分类（CategoryOfGaps）推断最可能的根因
4. 提供具体、可执行的改进建议
5. 如果信息不足，明确指出需要补充的内容

请用中文回复，保持专业、清晰的语气。"""

ZH_DEFECT_ANALYSIS_PROMPT = """## 用户问题
{question}

## 检索到的相似历史缺陷（按相似度排序）
{similar_defects}

## 分析任务
请基于以上历史缺陷数据，对用户的这个问题进行全面分析：

1. **最可能的根因分析**：根据相似缺陷的CategoryOfGaps和Injection Phase，推断最可能的根因
2. **相似度评估**：给出与历史缺陷的相似度评分（高/中/低）
3. **预防和改进建议**：基于ImprovementMeasures和Non-detection Phase，提供具体建议

    ## 输出格式
请按以下JSON格式输出分析结果：

```json
{{
    "analysis": {{
        "probable_root_cause": "最可能的根因，简要说明",
        "root_cause_category": "根因分类，如：Imp: Practise, RE: Competency等",
        "confidence": "高/中/低",
        "reasoning": "推理过程的简要说明"
    }},
    "similar_defects": [
        {{
            "id": "缺陷ID",
            "summary": "必须使用检索结果中该缺陷的Summary字段内容",
            "similarity_score": 0.95,
            "key_insight": "该缺陷的关键启示"
        }}
    ],
    "recommendations": [
        "预防和改进建议1",
        "预防和改进建议2"
    ],
    "additional_info_needed": "如需要更多信息，请在此处说明，否则填null"
}}
```

重要提示：
1. similar_defects数组中的每个summary字段必须从检索到的缺陷的Summary字段中准确复制，不要自行生成描述
2. 确保JSON格式正确，可以被解析
"""

# ============================================
# English Prompts
# ============================================

EN_SYSTEM_PROMPT = """You are a professional software defect analysis expert. Your task is to analyze user-reported issues based on historical defect data and provide professional root cause analysis and recommendations.

Analysis Principles:
1. Carefully understand the problem described by the user
2. Refer to similar historical defects to find commonalities and differences
3. Infer the most likely root cause based on defect classification (CategoryOfGaps)
4. Provide specific, actionable improvement suggestions
5. If information is insufficient, clearly state what additional details are needed

Please reply in English with a professional and clear tone."""

EN_DEFECT_ANALYSIS_PROMPT = """## User Question
{question}

## Similar Historical Defects Retrieved (sorted by similarity)
{similar_defects}

## Analysis Task
Please perform a comprehensive analysis of the user's issue based on the above historical defect data:

1. **Most Probable Root Cause Analysis**: Infer the most likely root cause based on similar defects' CategoryOfGaps and Injection Phase
2. **Similarity Assessment**: Provide a similarity rating (High/Medium/Low) with historical defects
3. **Prevention and Improvement Recommendations**: Provide specific suggestions based on ImprovementMeasures and Non-detection Phase

    ## Output Format
    Please output the analysis results in the following JSON format:

```json
{{
    "analysis": {{
        "probable_root_cause": "Most probable root cause with brief explanation",
        "root_cause_category": "Root cause category, e.g., Imp: Practise, RE: Competency, etc.",
        "confidence": "High/Medium/Low",
        "reasoning": "Brief explanation of the reasoning process"
    }},
    "similar_defects": [
        {{
            "id": "Defect ID",
            "summary": "MUST use the exact Summary field from the retrieved defect",
            "similarity_score": 0.95,
            "key_insight": "Key insight from this defect"
        }}
    ],
    "recommendations": [
        "Prevention and improvement suggestion 1",
        "Prevention and improvement suggestion 2"
    ],
    "additional_info_needed": "If more information is needed, state here, otherwise use null"
}}
```

    Important Notes:
    1. The summary field in each similar_defects entry MUST be copied exactly from the Summary field of the retrieved defect, do not generate your own description
    2. Ensure the JSON format is correct and can be parsed
"""


# ============================================
# Helper Functions
# ============================================

def get_prompts(language: str = "zh"):
    """Get prompts for specified language.

    Args:
        language: 'zh' for Chinese, 'en' for English

    Returns:
        Tuple of (system_prompt, analysis_prompt)
    """
    if language.lower() in ["zh", "chinese", "中文"]:
        return ZH_SYSTEM_PROMPT, ZH_DEFECT_ANALYSIS_PROMPT
    else:
        return EN_SYSTEM_PROMPT, EN_DEFECT_ANALYSIS_PROMPT


def format_value(value, default="-"):
    """Format value, return default if None or empty."""
    if value is None or value == "" or value == "N/A":
        return default
    return str(value)


def format_similar_defects(defects: list, language: str = "zh") -> str:
    """Format similar defects for prompt.

    Args:
        defects: List of defect dictionaries with 'metadata' and 'score'
        language: Language for formatting

    Returns:
        Formatted string
    """
    if not defects:
        if language.lower() in ["zh", "chinese", "中文"]:
            return "未找到相似缺陷。"
        return "No similar defects found."

    formatted = []
    for i, defect in enumerate(defects, 1):
        meta = defect.get('metadata', {})
        text = defect.get('text', '')
        score = defect.get('score', 0)

        # Format values without N/A
        defect_id = format_value(meta.get('Identifier'))
        summary = format_value(meta.get('Summary'))
        component = format_value(meta.get('Component'))
        customer = format_value(meta.get('Customer'))
        category = format_value(meta.get('CategoryOfGaps'))
        subcategory = format_value(meta.get('SubCategoryOfGaps'))

        if language.lower() in ["zh", "chinese", "中文"]:
            section = f"""### 缺陷 {i} (相似度: {score:.2f})
- **ID**: {defect_id}
- **标题**: {summary}
- **组件**: {component}
- **客户**: {customer}
- **缺陷类型**: {category} / {subcategory}
- **详细描述**:
{text}
"""
        else:
            section = f"""### Defect {i} (Similarity: {score:.2f})
- **ID**: {defect_id}
- **Summary**: {summary}
- **Component**: {component}
- **Customer**: {customer}
- **Gap Category**: {category} / {subcategory}
- **Description**:
{text}
"""
        formatted.append(section)

    return "\n\n---\n\n".join(formatted)


# ============================================
# Error Messages
# ============================================

ZH_ERROR_MESSAGES = {
    "no_data": "未加载缺陷数据，请先上传数据文件或检查配置。",
    "api_error": "调用API时出错，请检查API配置和网络连接。",
    "no_results": "未找到相似缺陷，请尝试调整查询或上传更多数据。",
    "invalid_json": "解析响应时出错，请重试。",
    "config_error": "配置错误，请检查settings.yaml文件。"
}

EN_ERROR_MESSAGES = {
    "no_data": "No defect data loaded. Please upload a data file or check configuration.",
    "api_error": "API call failed. Please check API configuration and network connection.",
    "no_results": "No similar defects found. Please try adjusting your query or upload more data.",
    "invalid_json": "Error parsing response. Please try again.",
    "config_error": "Configuration error. Please check settings.yaml file."
}


def get_error_message(key: str, language: str = "zh") -> str:
    """Get error message for specified language."""
    if language.lower() in ["zh", "chinese", "中文"]:
        return ZH_ERROR_MESSAGES.get(key, "未知错误")
    return EN_ERROR_MESSAGES.get(key, "Unknown error")
