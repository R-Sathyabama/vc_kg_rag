"""
Configuration Management for Hybrid RAG System
"""
import os
from typing import Optional
from pydantic import BaseModel, Field

class RAGConfig(BaseModel):
    """Configuration for RAG system"""
    
    # OpenAI Settings
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model for generation")
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    
    # Vector Store Settings
    vector_store_type: str = Field(default="chroma", description="Vector store type")
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")
    top_k_retrieval: int = Field(default=5, description="Number of documents to retrieve")
    
    # Knowledge Graph Settings
    use_knowledge_graph: bool = Field(default=True, description="Enable knowledge graph")
    neo4j_uri: Optional[str] = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: Optional[str] = Field(default="neo4j", description="Neo4j username")
    neo4j_password: Optional[str] = Field(default="password", description="Neo4j password")
    
    # Hybrid RAG Settings
    vector_weight: float = Field(default=0.5, description="Weight for vector search")
    graph_weight: float = Field(default=0.5, description="Weight for graph search")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    use_hybrid_by_default: bool = Field(default=True, description="Always use hybrid approach")
    
    # RAG Techniques - All enabled by default for best results
    rag_fusion: bool = Field(default=True, description="Enable RAG fusion")
    adaptive_retrieval: bool = Field(default=True, description="Enable adaptive retrieval")
    self_query: bool = Field(default=False, description="Enable self-query")
    corrective_rag: bool = Field(default=True, description="Enable corrective RAG")
    
    # System Prompts
    system_prompt_vector: str = Field(
        default="""You are a precise AI assistant analyzing documents. 
        Use ONLY the provided context to answer questions accurately.
        If the answer is not in the context, say so clearly.
        Provide specific quotes when possible.""",
        description="System prompt for vector RAG"
    )
    
    system_prompt_graph: str = Field(
        default="""You are a knowledge graph expert.
        Use the entity relationships and properties to answer questions.
        Explain connections between entities when relevant.
        Be precise about the relationships you identify.""",
        description="System prompt for graph RAG"
    )
    
    system_prompt_hybrid: str = Field(
        default="""You are an advanced AI assistant with access to both semantic search 
        and knowledge graph capabilities. Synthesize information from both sources 
        to provide comprehensive, accurate answers. Prioritize facts from the documents
        and explain entity relationships when they add value.""",
        description="System prompt for hybrid RAG"
    )

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password")
        )

# Default configuration
DEFAULT_CONFIG = RAGConfig()