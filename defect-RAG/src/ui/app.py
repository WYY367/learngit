"""Streamlit UI for Defect RAG System."""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ui.components.sidebar import render_sidebar
from src.ui.components.chat import render_chat_interface
from src.ui.components.file_upload import render_file_upload
from src.utils.lang_detector import get_ui_text


def init_session_state():
    """Initialize Streamlit session state."""
    if "language" not in st.session_state:
        st.session_state.language = "zh"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "config" not in st.session_state:
        st.session_state.config = None

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "index_built" not in st.session_state:
        st.session_state.index_built = False

    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def auto_initialize():
    """Auto-initialize config and RAG chain from settings.yaml."""
    if st.session_state.initialized:
        return

    try:
        from config.config_loader import get_config
        from src.core.llm_client import OpenAICompatibleLLM
        from src.core.embedding_engine import OpenAICompatibleEmbedding
        from src.core.vector_store import DefectVectorStore
        from src.chains.rag_chain import DefectRAGChain

        # Load config
        config = get_config()

        # Initialize clients
        llm_client = OpenAICompatibleLLM(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model=config.llm.model,
            default_params={
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                "top_p": config.llm.top_p,
                "frequency_penalty": config.llm.frequency_penalty,
                "presence_penalty": config.llm.presence_penalty
            },
            verify_ssl=getattr(config.llm, 'verify_ssl', True)
        )

        embedding_client = OpenAICompatibleEmbedding(
            base_url=config.embedding.base_url,
            api_key=config.embedding.api_key,
            model=config.embedding.model,
            verify_ssl=getattr(config.embedding, 'verify_ssl', True)
        )

        vector_store = DefectVectorStore(
            persist_directory=config.vector_store.persist_directory,
            collection_name=config.vector_store.collection_name
        )

        # Create RAG chain
        rag_chain = DefectRAGChain(
            llm_client=llm_client,
            embedding_client=embedding_client,
            vector_store=vector_store,
            language=st.session_state.language
        )

        # Save to session state
        st.session_state.config = config
        st.session_state.rag_chain = rag_chain
        st.session_state.index_built = vector_store.count() > 0
        st.session_state.initialized = True

    except Exception as e:
        # Auto-initialization failed, user will need to configure manually
        st.session_state.initialized = True  # Mark as attempted
        print(f"Auto-initialization failed: {e}")


def main():
    """Main application."""
    # Initialize session state
    init_session_state()

    # Auto-initialize config and clients
    auto_initialize()

    # Page config
    lang = st.session_state.language
    st.set_page_config(
        page_title=get_ui_text(lang, "app_title"),
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.title(get_ui_text(lang, "app_title"))
    st.markdown(f"*{get_ui_text(lang, 'app_description')}*")

    # Render sidebar
    render_sidebar()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Chat interface
        render_chat_interface()

    with col2:
        # File upload and data management
        render_file_upload()


if __name__ == "__main__":
    main()
