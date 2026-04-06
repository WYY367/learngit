"""Chat interface component."""
import streamlit as st
import json
from src.utils.lang_detector import get_ui_text


def render_chat_interface():
    """Render chat interface."""
    lang = st.session_state.language

    st.subheader("💬 " + get_ui_text(lang, "app_title"))

    # Check if RAG chain is initialized
    if st.session_state.rag_chain is None:
        st.info("👈 " + ("请先配置API设置并保存" if lang == "zh" else "Please configure API settings and save"))
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display analysis result if available
            if message.get("analysis"):
                with st.expander(get_ui_text(lang, "analysis_result")):
                    display_analysis(message["analysis"], lang)

            # Display similar defects if available
            if message.get("similar_defects"):
                with st.expander(get_ui_text(lang, "similar_defects")):
                    display_similar_defects(message["similar_defects"], lang)

    # Chat input
    if prompt := st.chat_input(get_ui_text(lang, "chat_input_placeholder")):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(get_ui_text(lang, "thinking")):
                try:
                    # Get RAG chain
                    rag_chain = st.session_state.rag_chain

                    # Get config for parameters
                    config = st.session_state.config
                    top_k = config.retrieval.top_k if config else 5

                    # Get LLM parameters
                    llm_params = config.get_llm_params() if config else {}

                    # Run RAG
                    result = rag_chain.run(
                        query=prompt,
                        top_k=top_k,
                        llm_params=llm_params
                    )

                    # Display main response
                    if "analysis" in result and isinstance(result["analysis"], dict):
                        analysis = result["analysis"]

                        # Display root cause
                        if "analysis" in analysis and "probable_root_cause" in analysis["analysis"]:
                            root_cause = analysis["analysis"]["probable_root_cause"]
                            st.markdown(f"**{'最可能的根因' if lang == 'zh' else 'Most Probable Root Cause'}:**\n{root_cause}")

                        # Display confidence
                        if "analysis" in analysis and "confidence" in analysis["analysis"]:
                            confidence = analysis["analysis"]["confidence"]
                            st.markdown(f"**{'置信度' if lang == 'zh' else 'Confidence'}:** {confidence}")

                        # Display recommendations
                        if "recommendations" in analysis and analysis["recommendations"]:
                            st.markdown(f"**{'建议' if lang == 'zh' else 'Recommendations'}:**")
                            for rec in analysis["recommendations"]:
                                st.markdown(f"- {rec}")

                    else:
                        st.markdown(str(result.get("analysis", "No analysis available")))

                    # Store in message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get("analysis", {}).get("analysis", {}).get("probable_root_cause", ""),
                        "analysis": result.get("analysis"),
                        "similar_defects": result.get("similar_defects")
                    })

                    # Show details in expanders
                    if result.get("analysis"):
                        with st.expander(get_ui_text(lang, "analysis_result")):
                            display_analysis(result["analysis"], lang)

                    if result.get("similar_defects"):
                        with st.expander(get_ui_text(lang, "similar_defects")):
                            display_similar_defects(result["similar_defects"], lang)

                except Exception as e:
                    error_msg = f"{get_ui_text(lang, 'error')}: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def display_analysis(analysis: dict, lang: str):
    """Display analysis result."""
    if not isinstance(analysis, dict):
        st.write(analysis)
        return

    # Analysis section
    if "analysis" in analysis and isinstance(analysis["analysis"], dict):
        analysis_data = analysis["analysis"]

        st.markdown("### " + ("根因分析" if lang == "zh" else "Root Cause Analysis"))

        if "probable_root_cause" in analysis_data:
            st.markdown(f"**{'最可能的根因' if lang == 'zh' else 'Root Cause'}:** {analysis_data['probable_root_cause']}")

        if "root_cause_category" in analysis_data:
            st.markdown(f"**{'根因分类' if lang == 'zh' else 'Category'}:** {analysis_data['root_cause_category']}")

        if "confidence" in analysis_data:
            st.markdown(f"**{'置信度' if lang == 'zh' else 'Confidence'}:** {analysis_data['confidence']}")

        if "reasoning" in analysis_data:
            st.markdown(f"**{'推理过程' if lang == 'zh' else 'Reasoning'}:** {analysis_data['reasoning']}")

    # Recommendations
    if "recommendations" in analysis and analysis["recommendations"]:
        st.markdown("### " + ("改进建议" if lang == "zh" else "Recommendations"))
        for i, rec in enumerate(analysis["recommendations"], 1):
            st.markdown(f"{i}. {rec}")

    # Similar defects summary
    if "similar_defects" in analysis and analysis["similar_defects"]:
        st.markdown("### " + ("相似缺陷摘要" if lang == "zh" else "Similar Defects Summary"))
        for defect in analysis["similar_defects"]:
            defect_id = format_value(defect.get('id'))
            summary = format_value(defect.get('summary'))
            score = defect.get('similarity_score', 0)
            component = format_value(defect.get('component', ''))
            category = format_value(defect.get('category', ''))

            # Ensure score is a float for formatting
            try:
                score_float = float(score) if score is not None else 0.0
            except (ValueError, TypeError):
                score_float = 0.0

            # Display with enhanced information
            st.markdown(f"- **{defect_id}**: {summary}")

            # Show additional info in smaller text
            info_parts = []
            if component != "-":
                info_parts.append(f"{'组件' if lang == 'zh' else 'Component'}: {component}")
            if category != "-":
                info_parts.append(f"{'类型' if lang == 'zh' else 'Category'}: {category}")
            info_parts.append(f"{'相似度' if lang == 'zh' else 'Similarity'}: {score_float:.3f}")

            if info_parts:
                st.markdown(f"  *{', '.join(info_parts)}*")

    # Additional info needed
    if "additional_info_needed" in analysis and analysis["additional_info_needed"]:
        st.info(f"**{'需要补充的信息' if lang == 'zh' else 'Additional Info Needed'}:** {analysis['additional_info_needed']}")

    # Raw response for debugging
    if "_raw_response" in analysis:
        with st.expander("Raw Response"):
            st.code(analysis["_raw_response"])


def format_value(value, default="-"):
    """Format value, return default if None or empty."""
    if value is None or value == "" or value == "N/A":
        return default
    return str(value)


def display_similar_defects(defects: list, lang: str):
    """Display similar defects."""
    if not defects:
        st.write("No similar defects found." if lang == "en" else "未找到相似缺陷")
        return

    for i, defect in enumerate(defects, 1):
        meta = defect.get("metadata", {})
        score = defect.get("score", 0)
        # Ensure score is a float for formatting
        try:
            score_float = float(score) if score is not None else 0.0
        except (ValueError, TypeError):
            score_float = 0.0

        # Extract key fields
        defect_id = format_value(meta.get('Identifier'))
        summary = format_value(meta.get('Summary'))
        component = format_value(meta.get('Component'))
        customer = format_value(meta.get('Customer'))
        category = format_value(meta.get('CategoryOfGaps'))
        subcategory = format_value(meta.get('SubCategoryOfGaps'))

        with st.container():
            # Title: ID + Summary (no truncation)
            title = f"{defect_id}: {summary}"
            st.markdown(f"#### {i}. {title}")

            st.markdown(f"**{'相似度' if lang == 'zh' else 'Similarity'}:** {score_float:.3f}")

            # Only show fields that have values
            info_parts = []
            if component != "-":
                info_parts.append(f"**{'组件' if lang == 'zh' else 'Component'}:** {component}")
            if customer != "-":
                info_parts.append(f"**{'客户' if lang == 'zh' else 'Customer'}:** {customer}")
            if category != "-" or subcategory != "-":
                type_str = f"{category}" if subcategory == "-" else f"{category} / {subcategory}"
                info_parts.append(f"**{'类型' if lang == 'zh' else 'Type'}:** {type_str}")

            if info_parts:
                st.markdown(" | ".join(info_parts))

            with st.expander("Details"):
                st.text(defect.get("text", ""))

            st.divider()
