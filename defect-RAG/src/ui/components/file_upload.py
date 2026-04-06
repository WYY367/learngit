"""File upload and data management component."""
import streamlit as st
import shutil
from pathlib import Path
from src.core.data_loader import DefectDataLoader
from src.core.index_manager import IndexManager
from src.core.embedding_engine import OpenAICompatibleEmbedding
from src.core.vector_store import DefectVectorStore
from src.utils.lang_detector import get_ui_text


def render_file_upload():
    """Render file upload section."""
    lang = st.session_state.language

    st.subheader("📁 " + get_ui_text(lang, "sidebar_data"))

    # Check if config is initialized
    if st.session_state.config is None:
        st.info("👈 " + ("请先配置API设置" if lang == "zh" else "Please configure API settings first"))
        return

    # File upload
    uploaded_file = st.file_uploader(
        get_ui_text(lang, "upload_file"),
        type=["json"],
        help="Upload a JSON file containing defect records"
    )

    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        st.success(f"{'文件上传成功' if lang == 'zh' else 'File uploaded'}: {uploaded_file.name}")

        # Load and preview data
        try:
            loader = DefectDataLoader(str(file_path))
            df = loader.load()

            st.markdown(f"**{'记录数' if lang == 'zh' else 'Records'}:** {len(df)}")
            st.markdown(f"**{'字段' if lang == 'zh' else 'Fields'}:** {', '.join(df.columns[:5])}...")

            # Build index button
            if st.button(get_ui_text(lang, "build_index"), type="primary"):
                with st.spinner("Building index..." if lang == "en" else "正在构建索引..."):
                    try:
                        config = st.session_state.config

                        # Initialize embedding client
                        verify_ssl = getattr(config.embedding, 'verify_ssl', True)
                        embedding_client = OpenAICompatibleEmbedding(
                            base_url=config.embedding.base_url,
                            api_key=config.embedding.api_key,
                            model=config.embedding.model,
                            verify_ssl=verify_ssl
                        )

                        # Initialize vector store
                        vector_store = DefectVectorStore(
                            persist_directory=config.vector_store.persist_directory,
                            collection_name=config.vector_store.collection_name
                        )

                        # Create index manager
                        index_manager = IndexManager(embedding_client, vector_store)

                        # Build index
                        stats = index_manager.build_index(
                            data_path=str(file_path),
                            text_fields=config.data.text_fields,
                            metadata_fields=config.data.metadata_fields,
                            incremental=True
                        )

                        # Update RAG chain's vector store
                        if st.session_state.rag_chain:
                            st.session_state.rag_chain.vector_store = vector_store

                        st.session_state.index_built = True

                        # Display stats
                        st.success(f"{'索引构建成功' if lang == 'zh' else 'Index built successfully'}!")
                        st.json(stats)

                    except Exception as e:
                        st.error(f"{'索引构建失败' if lang == 'zh' else 'Index build failed'}: {str(e)}")

        except Exception as e:
            st.error(f"{'加载文件失败' if lang == 'zh' else 'Failed to load file'}: {str(e)}")

    st.divider()

    # Index management
    st.subheader(get_ui_text(lang, "index_stats"))

    try:
        config = st.session_state.config
        vector_store = DefectVectorStore(
            persist_directory=config.vector_store.persist_directory,
            collection_name=config.vector_store.collection_name
        )

        count = vector_store.count()
        st.markdown(f"**{'当前索引记录数' if lang == 'zh' else 'Current Index Records'}:** {count}")

        if count > 0:
            # Show sample records
            with st.expander("Sample Records"):
                samples = vector_store.peek(3)
                for i, sample in enumerate(samples, 1):
                    meta = sample.get('metadata', {})
                    defect_id = meta.get('Identifier', sample.get('id', '-'))
                    summary = meta.get('Summary', '-')[:60]
                    component = meta.get('Component', '-')
                    customer = meta.get('Customer', '-')

                    # Display ID + Summary
                    st.markdown(f"**{i}. {defect_id}: {summary}...**")

                    # Optional info
                    info_parts = []
                    if component and component != '-':
                        info_parts.append(f"{'组件' if lang == 'zh' else 'Component'}: {component}")
                    if customer and customer != '-':
                        info_parts.append(f"{'客户' if lang == 'zh' else 'Customer'}: {customer}")

                    if info_parts:
                        st.markdown(f"   {' | '.join(info_parts)}")

        # Reset index button
        if count > 0:
            if st.button(get_ui_text(lang, "reset_index"), type="secondary"):
                if st.checkbox("Confirm reset? All data will be deleted." if lang == "en" else "确认重置？所有数据将被删除。"):
                    try:
                        vector_store.delete_all()
                        st.session_state.index_built = False
                        st.success("Index reset successfully!" if lang == "en" else "索引重置成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Reset failed: {str(e)}")

    except Exception as e:
        st.warning(f"Cannot access index: {str(e)}")
