"""
Hybrid RAG Engine
Intelligently combines vector search and knowledge graph for optimal retrieval
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from vector_store import VectorStoreManager
from knowledge_graph import KnowledgeGraphManager
from config import RAGConfig


class HybridRAGEngine:
    """Hybrid RAG system combining vector search and knowledge graph"""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize hybrid RAG engine
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.vector_store = VectorStoreManager(
            embedding_model=config.embedding_model,
            vector_store_type=config.vector_store_type,
            openai_api_key=config.openai_api_key
        )
        self.knowledge_graph = KnowledgeGraphManager(
            openai_api_key=config.openai_api_key
        )
        
        # Initialize LLM with correct parameters
        import os
        if config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = config.openai_api_key
        
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=1000  # Limit answer to max 1000 tokens
        )
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents in both vector store and knowledge graph
        
        Args:
            documents: List of documents to index
        """
        try:
            print("Indexing documents in vector store...")
            self.vector_store.create_vector_store(documents)
            
            if self.config.use_knowledge_graph:
                print("Building knowledge graph...")
                self.knowledge_graph.build_graph_from_documents(documents)
            
            print("Indexing complete!")
        except Exception as e:
            raise Exception(f"Error indexing documents: {str(e)}")
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine if query needs vector search, graph search, or both
        
        Args:
            query: User query
            
        Returns:
            Query type: 'vector', 'graph', or 'hybrid'
        """
        try:
            analysis_prompt = PromptTemplate(
                template="""Analyze this question and determine the best retrieval strategy.
                
                - Choose 'vector' if the question asks for general information, facts, or content from documents
                - Choose 'graph' if the question asks about relationships, connections, or entities
                - Choose 'hybrid' if the question needs both contextual information and relationships
                
                Question: {question}
                
                Strategy (respond with only one word - vector/graph/hybrid):""",
                input_variables=["question"]
            )
            
            # Use new RunnableSequence API
            chain = analysis_prompt | self.llm
            result = chain.invoke({"question": query})
            
            # Extract content from result
            if hasattr(result, 'content'):
                result = result.content.strip().lower()
            else:
                result = str(result).strip().lower()
            
            if 'vector' in result:
                return 'vector'
            elif 'graph' in result:
                return 'graph'
            else:
                return 'hybrid'
                
        except Exception as e:
            print(f"Error determining query type: {str(e)}")
            return 'hybrid'  # Default to hybrid
    
    def _retrieve_from_vector(self, query: str) -> List[Document]:
        """
        Retrieve documents from vector store
        
        Args:
            query: Search query
            
        Returns:
            Retrieved documents
        """
        return self.vector_store.retrieve(
            query=query,
            k=self.config.top_k_retrieval,
            use_fusion=self.config.rag_fusion,
            use_adaptive=self.config.adaptive_retrieval,
            use_corrective=self.config.corrective_rag
        )
    
    def _retrieve_from_graph(self, query: str) -> Dict[str, Any]:
        """
        Retrieve information from knowledge graph
        
        Args:
            query: Search query
            
        Returns:
            Graph query results
        """
        return self.knowledge_graph.query_graph(query, max_hops=2)
    
    def retrieve(
        self, 
        query: str,
        force_type: Optional[str] = None
    ) -> Tuple[List[Document], Dict[str, Any], str]:
        """
        Retrieve information using hybrid approach
        
        Args:
            query: User query
            force_type: Force specific retrieval type
            
        Returns:
            Tuple of (vector_docs, graph_data, query_type)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Force hybrid mode if configured
            if self.config.use_hybrid_by_default and not force_type:
                query_type = 'hybrid'
                logger.info("🔀 Hybrid mode enabled by default")
            elif force_type:
                query_type = force_type
                logger.info(f"🎯 Forced query type: {query_type}")
            else:
                logger.info("🤔 Analyzing query type...")
                query_type = self._determine_query_type(query)
                logger.info(f"✅ Determined type: {query_type}")
            
            vector_docs = []
            graph_data = {"entities": [], "relationships": [], "context": ""}
            
            # Retrieve based on query type
            if query_type in ['vector', 'hybrid']:
                logger.info(f"\n{'─' * 80}")
                logger.info("STEP 2: VECTOR SEARCH PIPELINE")
                logger.info(f"{'─' * 80}")
                logger.info("🔍 Retrieving from vector store...")
                vector_docs = self._retrieve_from_vector(query)
                logger.info(f"✅ Retrieved {len(vector_docs)} documents from vector store")
            
            if query_type in ['graph', 'hybrid'] and self.config.use_knowledge_graph:
                logger.info(f"\n{'─' * 80}")
                logger.info("STEP 3: GRAPH SEARCH PIPELINE")
                logger.info(f"{'─' * 80}")
                logger.info("🕸️  Retrieving from knowledge graph...")
                graph_data = self._retrieve_from_graph(query)
                logger.info(f"✅ Retrieved {len(graph_data.get('entities', []))} entities, "
                          f"{len(graph_data.get('relationships', []))} relationships")
            
            logger.info(f"\n{'─' * 80}")
            logger.info("STEP 4: CONTEXT COMBINATION")
            logger.info(f"{'─' * 80}")
            logger.info("🔗 Combining vector and graph contexts...")
            logger.info(f"   ├─ Vector chunks: {len(vector_docs)}")
            logger.info(f"   ├─ Graph entities: {len(graph_data.get('entities', []))}")
            logger.info(f"   └─ Graph relations: {len(graph_data.get('relationships', []))}")
            
            return vector_docs, graph_data, query_type
            
        except Exception as e:
            logger.error(f"❌ Error in retrieval: {str(e)}", exc_info=True)
            raise Exception(f"Error in retrieval: {str(e)}")
    
    def _build_context(
        self, 
        vector_docs: List[Document], 
        graph_data: Dict[str, Any],
        query_type: str
    ) -> str:
        """
        Build context from retrieved information
        
        Args:
            vector_docs: Retrieved documents from vector store
            graph_data: Retrieved data from knowledge graph
            query_type: Type of query
            
        Returns:
            Combined context string
        """
        context_parts = []
        
        # Add vector context
        if vector_docs:
            context_parts.append("=== DOCUMENT CONTEXT ===")
            for i, doc in enumerate(vector_docs, 1):
                source = doc.metadata.get('file_name', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(
                    f"\n[Document {i} - Source: {source}, Page: {page}]\n{doc.page_content}\n"
                )
        
        # Add graph context
        if graph_data and (graph_data.get('entities') or graph_data.get('relationships')):
            context_parts.append("\n=== KNOWLEDGE GRAPH CONTEXT ===")
            
            if graph_data.get('entities'):
                context_parts.append("\nEntities:")
                for entity in graph_data['entities'][:10]:  # Limit to 10
                    context_parts.append(
                        f"- {entity['name']} ({entity['type']})"
                    )
            
            if graph_data.get('relationships'):
                context_parts.append("\nRelationships:")
                for rel in graph_data['relationships'][:10]:  # Limit to 10
                    context_parts.append(
                        f"- {rel['source']} {rel['relation']} {rel['target']}"
                    )
            
            if graph_data.get('context'):
                context_parts.append(f"\n{graph_data['context']}")
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self, 
        query: str,
        vector_docs: List[Document],
        graph_data: Dict[str, Any],
        query_type: str
    ) -> str:
        """
        Generate answer using retrieved context
        
        Args:
            query: User query
            vector_docs: Retrieved documents
            graph_data: Graph query results
            query_type: Type of query
            
        Returns:
            Generated answer
        """
        try:
            # Build context
            context = self._build_context(vector_docs, graph_data, query_type)
            
            if not context or context.strip() == "":
                return "I don't have enough information in the uploaded documents to answer this question."
            
            # Select appropriate system prompt
            if query_type == 'vector':
                system_prompt = self.config.system_prompt_vector
            elif query_type == 'graph':
                system_prompt = self.config.system_prompt_graph
            else:
                system_prompt = self.config.system_prompt_hybrid
            
            # Generate answer
            answer_prompt = PromptTemplate(
                template="""
{system_prompt}

Context:
{context}

Question: {question}

Answer (provide a natural, comprehensive response based ONLY on the context above):""",
                input_variables=["system_prompt", "context", "question"]
            )
            
            # Use new RunnableSequence API
            chain = answer_prompt | self.llm
            result = chain.invoke({
                "system_prompt": system_prompt,
                "context": context,
                "question": query
            })
            
            # Extract content from result
            if hasattr(result, 'content'):
                answer = result.content.strip()
            else:
                answer = str(result).strip()
            
            return answer
            
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Main query method combining retrieval and generation
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing answer and metadata
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"\n{'─' * 80}")
            logger.info("STEP 1: QUERY ANALYSIS")
            logger.info(f"{'─' * 80}")
            logger.info(f"🎯 Analyzing query type...")
            
            # Retrieve information
            vector_docs, graph_data, query_type = self.retrieve(query)
            
            logger.info(f"🎯 Query Type Determined: {query_type.upper()}")
            logger.info(f"   → Will use: {'Vector + Graph' if query_type == 'hybrid' else query_type.title()}")
            
            # Generate answer
            logger.info(f"\n{'─' * 80}")
            logger.info("STEP 5: ANSWER GENERATION")
            logger.info(f"{'─' * 80}")
            
            # Limit context to avoid token overflow
            logger.info("📏 Limiting final context to max 1000 tokens...")
            answer = self.generate_answer(query, vector_docs, graph_data, query_type)
            
            logger.info(f"✅ Generated answer ({len(answer.split())} words)")
            
            # Prepare response
            response = {
                "answer": answer,
                "query_type": query_type,
                "num_vector_docs": len(vector_docs),
                "num_graph_entities": len(graph_data.get('entities', [])),
                "num_graph_relationships": len(graph_data.get('relationships', [])),
                "graph_data": graph_data,
                "sources": [
                    {
                        "file_name": doc.metadata.get('file_name', 'Unknown'),
                        "page": doc.metadata.get('page', 'N/A'),
                        "chunk_id": doc.metadata.get('chunk_id', 0)
                    }
                    for doc in vector_docs
                ]
            }
            
            logger.info(f"\n📊 FINAL STATISTICS:")
            logger.info(f"   ├─ Query Type: {query_type.upper()}")
            logger.info(f"   ├─ Vector Documents Used: {len(vector_docs)}")
            logger.info(f"   ├─ Graph Entities Found: {len(graph_data.get('entities', []))}")
            logger.info(f"   └─ Graph Relationships: {len(graph_data.get('relationships', []))}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}", exc_info=True)
            raise Exception(f"Error processing query: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the indexed data"""
        stats = {
            "vector_store": {
                "initialized": self.vector_store.vector_store is not None
            }
        }
        
        if self.config.use_knowledge_graph:
            stats["knowledge_graph"] = self.knowledge_graph.get_graph_statistics()
        
        return stats