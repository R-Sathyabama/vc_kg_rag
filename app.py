"""
Hybrid RAG Application - Clean UI with Verbose Terminal Logging
Production-ready interface for document Q&A
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

from document_processor import DocumentProcessor
from hybrid_rag import HybridRAGEngine
from config import RAGConfig

# Configure logging for verbose terminal output
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean and minimal
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .kg-box {
        background-color: #f0f7f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'config' not in st.session_state:
        st.session_state.config = RAGConfig()


def sidebar_configuration():
    """Render minimal sidebar - only API key"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.config.openai_api_key,
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            st.session_state.config.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")
            if st.session_state.rag_engine:
                stats = st.session_state.rag_engine.get_statistics()
                if 'knowledge_graph' in stats:
                    kg_stats = stats['knowledge_graph']
                    st.metric("Entities", kg_stats.get('num_nodes', 0))
                    st.metric("Relations", kg_stats.get('num_edges', 0))
        else:
            st.info("📄 Upload documents to begin")
        
        st.divider()
        st.caption("💡 View terminal for detailed processing logs")


def process_uploaded_files(uploaded_files):
    """Process uploaded files with verbose terminal logging"""
    if not st.session_state.config.openai_api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar")
        return
    
    try:
        logger.info("=" * 80)
        logger.info("DOCUMENT PROCESSING STARTED")
        logger.info("=" * 80)
        
        with st.spinner("Processing documents... Check terminal for details"):
            # Initialize processor
            processor = DocumentProcessor(
                chunk_size=st.session_state.config.chunk_size,
                chunk_overlap=st.session_state.config.chunk_overlap
            )
            
            logger.info(f"📄 Processing {len(uploaded_files)} file(s)")
            logger.info(f"⚙️  Chunk Size: {st.session_state.config.chunk_size}")
            logger.info(f"⚙️  Chunk Overlap: {st.session_state.config.chunk_overlap}")
            
            # Process all files
            all_documents = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                logger.info(f"\n{'─' * 80}")
                logger.info(f"📎 File {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                st.text(f"Processing: {uploaded_file.name}")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process document
                try:
                    logger.info("🔄 Extracting text...")
                    documents = processor.process_document(file_path=tmp_path)
                    logger.info(f"✅ Extracted {len(documents)} chunks")
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if not all_documents:
                logger.error("❌ No documents were successfully processed")
                st.error("No documents were successfully processed")
                return
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"📊 TOTAL: {len(all_documents)} chunks from {len(uploaded_files)} files")
            logger.info(f"{'=' * 80}\n")
            
            # Initialize RAG engine
            logger.info("🚀 Initializing Hybrid RAG Engine...")
            rag_engine = HybridRAGEngine(st.session_state.config)
            
            # Index documents
            logger.info("\n📥 Starting indexing process...")
            rag_engine.index_documents(all_documents)
            
            # Save to session state
            st.session_state.rag_engine = rag_engine
            st.session_state.documents_processed = True
            
            logger.info(f"\n{'=' * 80}")
            logger.info("✅ DOCUMENT PROCESSING COMPLETE")
            logger.info(f"{'=' * 80}\n")
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} files with {len(all_documents)} chunks!")
            
    except Exception as e:
        logger.error(f"❌ PROCESSING ERROR: {str(e)}", exc_info=True)
        st.error(f"Error processing files: {str(e)}")


def display_answer(response):
    """Display answer with KG visualization"""
    # Answer
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Answer")
    st.write(response['answer'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Query Type", response['query_type'].upper())
    with col2:
        st.metric("Vector Docs", response['num_vector_docs'])
    with col3:
        st.metric("Graph Entities", response['num_graph_entities'])
    with col4:
        st.metric("Graph Relations", response['num_graph_relationships'])
    
    # Knowledge Graph Insights
    if response.get('graph_data') and (response['graph_data'].get('entities') or response['graph_data'].get('relationships')):
        st.markdown('<div class="kg-box">', unsafe_allow_html=True)
        st.markdown("### 🕸️ Knowledge Graph Insights")
        
        graph_col1, graph_col2 = st.columns(2)
        
        with graph_col1:
            if response['graph_data'].get('entities'):
                st.markdown("**📍 Entities Found:**")
                for entity in response['graph_data']['entities'][:10]:
                    entity_type = entity.get('type', 'UNKNOWN')
                    entity_name = entity.get('name', 'Unknown')
                    st.markdown(f"- **{entity_name}** `({entity_type})`")
        
        with graph_col2:
            if response['graph_data'].get('relationships'):
                st.markdown("**🔗 Relationships:**")
                for rel in response['graph_data']['relationships'][:10]:
                    source = rel.get('source', '?')
                    target = rel.get('target', '?')
                    relation = rel.get('relation', 'RELATED_TO')
                    st.markdown(f"- {source} **→** `{relation}` **→** {target}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sources
    if response['sources']:
        with st.expander("📚 Sources", expanded=False):
            for i, source in enumerate(response['sources'], 1):
                st.markdown(
                    f'<div class="source-box">'
                    f'<strong>Source {i}:</strong> {source["file_name"]} '
                    f'(Page: {source["page"]}, Chunk: {source["chunk_id"]})'
                    f'</div>',
                    unsafe_allow_html=True
                )


def main():
    """Main application"""
    initialize_session_state()
    sidebar_configuration()
    
    # Header
    st.markdown('<div class="main-header">🤖 Hybrid RAG Q&A System</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666;">Advanced Document Analysis with Vector Search + Knowledge Graph</p>',
        unsafe_allow_html=True
    )
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])
    
    with tab1:
        st.header("Upload Documents")
        st.markdown("Upload PDF files or images to analyze")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload documents to process"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("🚀 Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
    
    with tab2:
        st.header("Ask Questions")
        
        if not st.session_state.documents_processed:
            st.info("👆 Please upload and process documents first")
        else:
            # Query input
            query = st.text_input(
                "Enter your question",
                placeholder="What is this document about?",
                help="Ask any question about your uploaded documents"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.button("Ask", type="primary")
            with col2:
                clear_button = st.button("Clear History")
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
            
            if ask_button and query:
                try:
                    logger.info("\n" + "=" * 80)
                    logger.info(f"USER QUESTION: {query}")
                    logger.info("=" * 80)
                    
                    with st.spinner("Processing query... Check terminal for detailed flow"):
                        response = st.session_state.rag_engine.query(query)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": response,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Display answer
                        display_answer(response)
                    
                    logger.info(f"\n{'=' * 80}")
                    logger.info("✅ QUERY PROCESSING COMPLETE")
                    logger.info(f"{'=' * 80}\n")
                        
                except Exception as e:
                    logger.error(f"❌ QUERY ERROR: {str(e)}", exc_info=True)
                    st.error(f"Error: {str(e)}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.divider()
                st.subheader("Chat History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                    with st.expander(f"[{chat['timestamp']}] {chat['query'][:50]}..."):
                        st.markdown(f"**Question:** {chat['query']}")
                        st.markdown(f"**Answer:** {chat['response']['answer']}")
    
    # Footer
    st.divider()
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
        'Powered by LangChain, OpenAI, and NetworkX | Hybrid RAG System v1.1'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()