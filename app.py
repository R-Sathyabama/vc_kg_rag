"""
Hybrid RAG Application - Main Streamlit App
Production-ready interface for document Q&A
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Optional

from document_processor import DocumentProcessor
from hybrid_rag import HybridRAGEngine
from config import RAGConfig

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
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
    """Render sidebar configuration"""
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
        
        # Model Settings
        st.subheader("Model Settings")
        
        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Model for answer generation"
        )
        st.session_state.config.llm_model = llm_model
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0,
            help="Model for document embeddings"
        )
        st.session_state.config.embedding_model = embedding_model
        
        st.divider()
        
        # RAG Settings
        st.subheader("RAG Settings")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=st.session_state.config.chunk_size,
            step=100,
            help="Size of text chunks for processing"
        )
        st.session_state.config.chunk_size = chunk_size
        
        top_k = st.slider(
            "Top K Retrieval",
            min_value=3,
            max_value=10,
            value=st.session_state.config.top_k_retrieval,
            step=1,
            help="Number of documents to retrieve"
        )
        st.session_state.config.top_k_retrieval = top_k
        
        st.divider()
        
        # Advanced RAG Techniques
        st.subheader("Advanced Techniques")
        
        st.info("ℹ️ Hybrid Mode (Vector + KG) enabled by default")
        
        rag_fusion = st.checkbox(
            "RAG Fusion",
            value=st.session_state.config.rag_fusion,
            help="Generate multiple query variations for better retrieval"
        )
        st.session_state.config.rag_fusion = rag_fusion
        
        adaptive_retrieval = st.checkbox(
            "Adaptive Retrieval",
            value=st.session_state.config.adaptive_retrieval,
            help="Adjust retrieval based on query complexity"
        )
        st.session_state.config.adaptive_retrieval = adaptive_retrieval
        
        corrective_rag = st.checkbox(
            "Corrective RAG",
            value=st.session_state.config.corrective_rag,
            help="Evaluate and refine retrieved documents"
        )
        st.session_state.config.corrective_rag = corrective_rag
        
        use_kg = st.checkbox(
            "Knowledge Graph",
            value=st.session_state.config.use_knowledge_graph,
            help="Enable knowledge graph for entity relationships"
        )
        st.session_state.config.use_knowledge_graph = use_kg
        
        st.divider()
        
        # System Info
        st.subheader("System Info")
        if st.session_state.documents_processed:
            st.success("✅ Documents Indexed")
            if st.session_state.rag_engine:
                stats = st.session_state.rag_engine.get_statistics()
                st.markdown(f"**Vector Store:** Initialized")
                if 'knowledge_graph' in stats:
                    kg_stats = stats['knowledge_graph']
                    st.markdown(f"**Entities:** {kg_stats.get('num_nodes', 0)}")
                    st.markdown(f"**Relations:** {kg_stats.get('num_edges', 0)}")
        else:
            st.info("📄 No documents indexed yet")


def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    if not st.session_state.config.openai_api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar")
        return
    
    try:
        with st.spinner("Processing documents... This may take a few minutes."):
            # Initialize processor
            processor = DocumentProcessor(
                chunk_size=st.session_state.config.chunk_size,
                chunk_overlap=st.session_state.config.chunk_overlap
            )
            
            # Process all files
            all_documents = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.text(f"Processing: {uploaded_file.name}")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process document
                try:
                    documents = processor.process_document(file_path=tmp_path)
                    all_documents.extend(documents)
                except Exception as e:
                    st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if not all_documents:
                st.error("No documents were successfully processed")
                return
            
            st.text(f"Processed {len(all_documents)} chunks from {len(uploaded_files)} files")
            
            # Initialize RAG engine
            st.text("Initializing RAG engine...")
            rag_engine = HybridRAGEngine(st.session_state.config)
            
            # Index documents
            st.text("Indexing documents...")
            rag_engine.index_documents(all_documents)
            
            # Save to session state
            st.session_state.rag_engine = rag_engine
            st.session_state.documents_processed = True
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} files with {len(all_documents)} chunks!")
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")


def display_answer(response):
    """Display answer with formatting and KG visualization"""
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Answer")
    st.write(response['answer'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Query Type", response['query_type'].upper())
    
    with col2:
        st.metric("Vector Docs", response['num_vector_docs'])
    
    with col3:
        st.metric("Graph Entities", response['num_graph_entities'])
    
    with col4:
        st.metric("Graph Relations", response['num_graph_relationships'])
    
    # Display Knowledge Graph Data (always shown when available)
    if response.get('graph_data') and (response['graph_data'].get('entities') or response['graph_data'].get('relationships')):
        st.markdown("---")
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
    
    # Display sources
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
        '<div class="sub-header">Advanced Document Analysis with Vector Search + Knowledge Graph</div>',
        unsafe_allow_html=True
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "💬 Ask Questions", "📊 System Stats"])
    
    with tab1:
        st.header("Upload Documents")
        st.markdown("Upload PDF files or images (PNG, JPG) to analyze")
        
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
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_engine.query(query)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": response
                        })
                        
                        # Display answer
                        display_answer(response)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.divider()
                st.subheader("Chat History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                    with st.expander(f"Q{len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                        st.markdown(f"**Question:** {chat['query']}")
                        st.markdown(f"**Answer:** {chat['response']['answer']}")
    
    with tab3:
        st.header("System Statistics")
        
        if st.session_state.documents_processed and st.session_state.rag_engine:
            stats = st.session_state.rag_engine.get_statistics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Vector Store")
                st.markdown(f"**Status:** {'✅ Initialized' if stats['vector_store']['initialized'] else '❌ Not Initialized'}")
            
            with col2:
                if 'knowledge_graph' in stats:
                    st.subheader("Knowledge Graph")
                    kg_stats = stats['knowledge_graph']
                    st.metric("Entities", kg_stats.get('num_nodes', 0))
                    st.metric("Relationships", kg_stats.get('num_edges', 0))
                    
                    if kg_stats.get('entity_types'):
                        st.markdown("**Entity Types:**")
                        for entity_type, count in kg_stats['entity_types'].items():
                            st.write(f"- {entity_type}: {count}")
                    
                    if kg_stats.get('relation_types'):
                        st.markdown("**Relation Types:**")
                        for relation_type, count in kg_stats['relation_types'].items():
                            st.write(f"- {relation_type}: {count}")
        else:
            st.info("Process documents to see statistics")
    
    # Footer
    st.divider()
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
        'Powered by LangChain, OpenAI, and NetworkX | Hybrid RAG System v1.0'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()