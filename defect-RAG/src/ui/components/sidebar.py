"""Sidebar component for configuration."""
import streamlit as st
from config.config_loader import get_config, Config
from src.core.llm_client import OpenAICompatibleLLM
from src.core.embedding_engine import OpenAICompatibleEmbedding
from src.core.vector_store import DefectVectorStore
from src.chains.rag_chain import DefectRAGChain
from src.utils.lang_detector import get_ui_text


def render_sidebar():
    """Render sidebar with configuration options."""
    lang = st.session_state.language

    with st.sidebar:
        st.header(get_ui_text(lang, "sidebar_config"))

        # Language selection
        new_lang = st.selectbox(
            get_ui_text(lang, "language"),
            options=["zh", "en"],
            format_func=lambda x: "中文" if x == "zh" else "English",
            index=0 if lang == "zh" else 1
        )

        if new_lang != lang:
            st.session_state.language = new_lang
            st.rerun()

        st.divider()

        # API Configuration
        st.subheader(get_ui_text(lang, "api_config"))

        # Load existing config or use defaults
        config = st.session_state.config or get_config()

        # LLM API Settings
        with st.expander("LLM API", expanded=False):
            llm_base_url = st.text_input(
                f"LLM {get_ui_text(lang, 'base_url')}",
                value=config.llm.base_url if config else "http://localhost:8000/v1"
            )
            llm_api_key = st.text_input(
                f"LLM {get_ui_text(lang, 'api_key')}",
                value=config.llm.api_key if config else "",
                type="password"
            )
            llm_model = st.text_input(
                f"LLM {get_ui_text(lang, 'model_name')}",
                value=config.llm.model if config else "gpt-3.5-turbo"
            )

        # Embedding API Settings
        with st.expander("Embedding API", expanded=False):
            emb_base_url = st.text_input(
                f"Embedding {get_ui_text(lang, 'base_url')}",
                value=config.embedding.base_url if config else "http://localhost:8000/v1"
            )
            emb_api_key = st.text_input(
                f"Embedding {get_ui_text(lang, 'api_key')}",
                value=config.embedding.api_key if config else "",
                type="password"
            )
            emb_model = st.text_input(
                f"Embedding {get_ui_text(lang, 'model_name')}",
                value=config.embedding.model if config else "bge-m3"
            )

        st.divider()

        # LLM Parameters
        st.subheader(get_ui_text(lang, "settings"))

        temperature = st.slider(
            get_ui_text(lang, "temperature"),
            min_value=0.0,
            max_value=2.0,
            value=config.llm.temperature if config else 0.7,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative"
        )

        max_tokens = st.slider(
            get_ui_text(lang, "max_tokens"),
            min_value=256,
            max_value=4096,
            value=config.llm.max_tokens if config else 2048,
            step=256
        )

        top_k = st.slider(
            get_ui_text(lang, "top_k"),
            min_value=1,
            max_value=20,
            value=config.retrieval.top_k if config else 5,
            step=1,
            help="Number of final results to return"
        )

        # Phase 1: Additional retrieval parameters
        st.subheader(get_ui_text(lang, "retrieval_settings") if lang == "zh" else "Retrieval Settings")

        # Enable re-ranking toggle
        enable_rerank = st.toggle(
            get_ui_text(lang, "enable_rerank"),
            value=config.retrieval.enable_rerank if config else True,
            help=get_ui_text(lang, "rerank_help")
        )

        # Re-rank type selection
        rerank_type = st.selectbox(
            get_ui_text(lang, "rerank_type") if lang == "zh" else "Re-rank Type",
            options=["simple", "llm"],
            format_func=lambda x: "规则-based (快)" if x == "simple" and lang == "zh" else ("Rule-based (Fast)" if x == "simple" else ("LLM-based (准)" if lang == "zh" else "LLM-based (Accurate)")),
            index=0 if (config and config.retrieval.rerank_type == "simple") else 1,
            help="simple: faster but less accurate | llm: slower but more accurate"
        )

        # Re-rank top_k slider
        rerank_top_k = st.slider(
            get_ui_text(lang, "rerank_top_k") if lang == "zh" else "Re-rank Candidates",
            min_value=5,
            max_value=50,
            value=config.retrieval.rerank_top_k if config else 20,
            step=5,
            help="Number of candidates to retrieve before re-ranking (higher = better recall but slower)"
        )

        # Similarity threshold slider
        similarity_threshold = st.slider(
            get_ui_text(lang, "similarity_threshold") if lang == "zh" else "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(config.retrieval.similarity_threshold if config else 0.5),
            step=0.05,
            help="Minimum similarity score (lower = more results but may be less relevant)"
        )

        # Save configuration button
        if st.button(get_ui_text(lang, "save_config"), type="primary"):
            try:
                # Create new config
                new_config = Config()
                new_config.llm.base_url = llm_base_url
                new_config.llm.api_key = llm_api_key
                new_config.llm.model = llm_model
                new_config.llm.temperature = temperature
                new_config.llm.max_tokens = max_tokens

                new_config.embedding.base_url = emb_base_url
                new_config.embedding.api_key = emb_api_key
                new_config.embedding.model = emb_model

                new_config.retrieval.top_k = top_k
                new_config.retrieval.enable_rerank = enable_rerank
                new_config.retrieval.rerank_type = rerank_type
                new_config.retrieval.rerank_top_k = rerank_top_k
                new_config.retrieval.similarity_threshold = similarity_threshold

                # Get SSL verification setting from config
                verify_ssl_llm = getattr(new_config.llm, 'verify_ssl', True)
                verify_ssl_emb = getattr(new_config.embedding, 'verify_ssl', True)

                # Initialize clients
                llm_client = OpenAICompatibleLLM(
                    base_url=llm_base_url,
                    api_key=llm_api_key,
                    model=llm_model,
                    default_params={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": 0.9,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0
                    },
                    verify_ssl=verify_ssl_llm
                )

                embedding_client = OpenAICompatibleEmbedding(
                    base_url=emb_base_url,
                    api_key=emb_api_key,
                    model=emb_model,
                    verify_ssl=verify_ssl_emb
                )

                vector_store = DefectVectorStore(
                    persist_directory="./vector_db",
                    collection_name="defects"
                )

                # Create RAG chain with Phase 1 parameters
                rag_chain = DefectRAGChain(
                    llm_client=llm_client,
                    embedding_client=embedding_client,
                    vector_store=vector_store,
                    language=lang,
                    enable_rerank=enable_rerank,
                    rerank_top_k=rerank_top_k,
                    rerank_type=rerank_type
                )

                # Save to session state
                st.session_state.config = new_config
                st.session_state.rag_chain = rag_chain
                st.session_state.index_built = vector_store.count() > 0

                st.success(f"{get_ui_text(lang, 'success')}: Configuration saved!")

            except Exception as e:
                st.error(f"{get_ui_text(lang, 'error')}: {str(e)}")
