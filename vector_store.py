"""
Vector Store Module with Advanced RAG Techniques
Implements RAG Fusion, Adaptive Retrieval, and Corrective RAG
"""
from typing import List, Dict, Any, Optional
import numpy as np

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class VectorStoreManager:
    """Manages vector store operations with advanced RAG techniques"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        vector_store_type: str = "chroma",
        openai_api_key: str = ""
    ):
        """
        Initialize vector store manager
        
        Args:
            embedding_model: OpenAI embedding model name
            vector_store_type: Type of vector store (chroma or faiss)
            openai_api_key: OpenAI API key
        """
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        
        # Initialize embeddings with correct parameters
        import os
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model
        )
        self.vector_store = None
        
        # Initialize LLM with correct parameters
        import os
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to index
        """
        try:
            if self.vector_store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory="./chroma_db"
                )
            elif self.vector_store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            # Get documents with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            
            # For ChromaDB, lower distance = higher similarity
            # Convert to similarity scores (0-100%)
            if self.vector_store_type == "chroma":
                # Chroma returns distances (lower is better)
                filtered_docs = [doc for doc, score in docs_with_scores]
            else:
                # FAISS returns similarity scores (higher is better)
                filtered_docs = [
                    doc for doc, score in docs_with_scores 
                    if score >= threshold
                ]
            
            return filtered_docs[:k]
        except Exception as e:
            raise Exception(f"Error in similarity search: {str(e)}")
    
    def rag_fusion(self, query: str, k: int = 5) -> List[Document]:
        """
        RAG Fusion: Generate multiple queries and combine results
        
        Args:
            query: Original query
            k: Number of documents to retrieve
            
        Returns:
            Fused list of documents
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Generate multiple query perspectives
            query_generation_prompt = PromptTemplate(
                template="""You are a helpful AI assistant. Generate 3 different versions 
                of the given question to retrieve relevant documents from a vector database.
                Provide these alternative questions separated by newlines.
                
                Original question: {question}
                
                Alternative questions:""",
                input_variables=["question"]
            )
            
            # Use new RunnableSequence API
            chain = query_generation_prompt | self.llm
            result = chain.invoke({"question": query})
            
            # Extract content from result
            if hasattr(result, 'content'):
                result = result.content
            else:
                result = str(result)
            
            # Parse generated queries
            generated_queries = [q.strip() for q in result.split('\n') if q.strip()]
            all_queries = [query] + generated_queries[:3]
            
            logger.info(f"\n      🔀 RAG FUSION - Generated Query Variations:")
            logger.info(f"      {'─' * 70}")
            for i, q in enumerate(all_queries, 1):
                prefix = "Original" if i == 1 else f"Variant {i-1}"
                logger.info(f"      {prefix:>12}: {q}")
            logger.info(f"      {'─' * 70}\n")
            
            # Retrieve documents for each query
            all_docs = []
            doc_scores = {}
            
            logger.info(f"      🔎 Searching with {len(all_queries)} query variations...")
            for idx, q in enumerate(all_queries):
                docs = self.similarity_search(q, k=k)
                logger.info(f"         Query {idx+1}: Found {len(docs)} documents")
                for doc in docs:
                    doc_id = doc.page_content[:100]  # Use first 100 chars as ID
                    if doc_id in doc_scores:
                        doc_scores[doc_id]['score'] += 1
                    else:
                        doc_scores[doc_id] = {'doc': doc, 'score': 1}
            
            # Sort by score and return top k
            sorted_docs = sorted(
                doc_scores.values(), 
                key=lambda x: x['score'], 
                reverse=True
            )
            
            logger.info(f"\n      ✅ Fusion Complete:")
            logger.info(f"         • Total unique chunks: {len(sorted_docs)}")
            logger.info(f"         • Top {min(k, len(sorted_docs))} selected")
            logger.info(f"         • Fusion scores: {[d['score'] for d in sorted_docs[:5]]}")
            
            return [item['doc'] for item in sorted_docs[:k]]
            
        except Exception as e:
            logger.warning(f"      ⚠️  RAG Fusion failed: {str(e)}")
            logger.info(f"      ↩️  Falling back to regular search")
            return self.similarity_search(query, k=k)
    
    def adaptive_retrieval(
        self, 
        query: str, 
        context: str = "", 
        k: int = 5
    ) -> List[Document]:
        """
        Adaptive Retrieval: Adjust retrieval strategy based on query complexity
        
        Args:
            query: Search query
            context: Additional context
            k: Number of documents to retrieve
            
        Returns:
            Retrieved documents
        """
        try:
            # Analyze query complexity
            complexity_prompt = PromptTemplate(
                template="""Analyze the complexity of this question and rate it as 'simple', 'medium', or 'complex'.
                
                Question: {question}
                
                Rating (respond with only one word - simple/medium/complex):""",
                input_variables=["question"]
            )
            
            # Use new RunnableSequence API
            chain = complexity_prompt | self.llm
            result = chain.invoke({"question": query})
            
            # Extract content from result
            if hasattr(result, 'content'):
                complexity = result.content.strip().lower()
            else:
                complexity = str(result).strip().lower()
            
            # Adjust k based on complexity
            if 'simple' in complexity:
                adjusted_k = max(3, k - 2)
                logger.info(f"      🎯 Query Complexity: SIMPLE")
                logger.info(f"      📊 Adjusted retrieval: {adjusted_k} documents (reduced)")
            elif 'complex' in complexity:
                adjusted_k = k + 3
                logger.info(f"      🎯 Query Complexity: COMPLEX")
                logger.info(f"      📊 Adjusted retrieval: {adjusted_k} documents (increased)")
            else:
                adjusted_k = k
                logger.info(f"      🎯 Query Complexity: MEDIUM")
                logger.info(f"      📊 Adjusted retrieval: {adjusted_k} documents (standard)")
            
            # Retrieve documents
            return self.similarity_search(query, k=adjusted_k)
            
        except Exception as e:
            print(f"Adaptive retrieval failed, using default: {str(e)}")
            return self.similarity_search(query, k=k)
    
    def corrective_rag(
        self, 
        query: str, 
        documents: List[Document], 
        k: int = 5
    ) -> List[Document]:
        """
        Corrective RAG: Evaluate and refine retrieved documents
        
        Args:
            query: Search query
            documents: Initially retrieved documents
            k: Number of documents to return
            
        Returns:
            Refined list of documents
        """
        try:
            if not documents:
                return []
            
            # Evaluate relevance of each document
            relevance_prompt = PromptTemplate(
                template="""Evaluate if the following document is relevant to answer the question.
                Respond with only 'yes' or 'no'.
                
                Question: {question}
                
                Document: {document}
                
                Is this document relevant?""",
                input_variables=["question", "document"]
            )
            
            # Use new RunnableSequence API
            chain = relevance_prompt | self.llm
            
            relevant_docs = []
            logger.info(f"      🔬 Evaluating {len(documents)} documents for relevance...")
            
            for idx, doc in enumerate(documents, 1):
                try:
                    result = chain.invoke({
                        "question": query, 
                        "document": doc.page_content[:500]
                    })
                    
                    # Extract content from result
                    if hasattr(result, 'content'):
                        response = result.content.strip().lower()
                    else:
                        response = str(result).strip().lower()
                    
                    is_relevant = 'yes' in response
                    status = "✓ Relevant" if is_relevant else "✗ Not relevant"
                    logger.info(f"         Doc {idx}: {status}")
                    
                    if is_relevant:
                        relevant_docs.append(doc)
                except:
                    # If evaluation fails, include the document
                    logger.info(f"         Doc {idx}: ⚠ Evaluation failed, including by default")
                    relevant_docs.append(doc)
            
            # If we have too few relevant docs, retrieve more
            if len(relevant_docs) < k:
                additional_docs = self.similarity_search(query, k=k*2)
                for doc in additional_docs:
                    if doc not in relevant_docs:
                        relevant_docs.append(doc)
                        if len(relevant_docs) >= k:
                            break
            
            return relevant_docs[:k]
            
        except Exception as e:
            print(f"Corrective RAG failed, returning original docs: {str(e)}")
            return documents[:k]
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5,
        use_fusion: bool = True,
        use_adaptive: bool = True,
        use_corrective: bool = True
    ) -> List[Document]:
        """
        Retrieve documents using configured RAG techniques
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            use_fusion: Enable RAG fusion
            use_adaptive: Enable adaptive retrieval
            use_corrective: Enable corrective RAG
            
        Returns:
            Retrieved documents
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Step 1: Initial retrieval
            if use_fusion:
                logger.info(f"\n   🔀 RAG FUSION: Generating multiple query variations...")
                documents = self.rag_fusion(query, k=k*2)
                logger.info(f"   ✅ Retrieved {len(documents)} docs using fusion")
            elif use_adaptive:
                logger.info(f"\n   🎯 ADAPTIVE RETRIEVAL: Analyzing query complexity...")
                documents = self.adaptive_retrieval(query, k=k*2)
                logger.info(f"   ✅ Retrieved {len(documents)} docs adaptively")
            else:
                logger.info(f"\n   🔍 STANDARD SEARCH: Basic similarity search...")
                documents = self.similarity_search(query, k=k*2)
                logger.info(f"   ✅ Retrieved {len(documents)} docs")
            
            # Step 2: Corrective refinement
            if use_corrective and documents:
                logger.info(f"\n   🔬 CORRECTIVE RAG: Evaluating relevance...")
                original_count = len(documents)
                documents = self.corrective_rag(query, documents, k=k)
                logger.info(f"   ✅ Refined from {original_count} to {len(documents)} relevant docs")
            
            logger.info(f"\n   📊 VECTOR SEARCH COMPLETE: {len(documents[:k])} final documents")
            return documents[:k]
            
        except Exception as e:
            logger.error(f"   ❌ Error in retrieval: {str(e)}")
            raise Exception(f"Error in retrieval: {str(e)}")