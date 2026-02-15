"""
Knowledge Graph Module
Handles entity extraction, relationship building, and graph querying
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import networkx as nx
from collections import defaultdict

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class KnowledgeGraphManager:
    """Manages knowledge graph operations"""
    
    def __init__(self, openai_api_key: str = ""):
        """
        Initialize knowledge graph manager
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.graph = nx.MultiDiGraph()
        
        # Initialize LLM with correct parameters
        import os
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        self.entity_properties = defaultdict(dict)
    
    def extract_entities_and_relationships(
        self, 
        document: Document
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from document using LLM
        
        Args:
            document: Document to process
            
        Returns:
            Dictionary containing entities and relationships
        """
        try:
            extraction_prompt = PromptTemplate(
                template="""Extract entities and their relationships from the following text.
                Return ONLY a valid JSON object with this exact structure:
                {{
                    "entities": [
                        {{"name": "entity_name", "type": "entity_type", "properties": {{"key": "value"}}}}
                    ],
                    "relationships": [
                        {{"source": "entity1", "target": "entity2", "relation": "relationship_type", "properties": {{"key": "value"}}}}
                    ]
                }}
                
                Entity types can be: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, CONCEPT, PRODUCT, etc.
                Relationship types can be: WORKS_FOR, LOCATED_IN, CREATED, RELATED_TO, PART_OF, etc.
                
                Text: {text}
                
                JSON:""",
                input_variables=["text"]
            )
            
            # Use new RunnableSequence API
            chain = extraction_prompt | self.llm
            result = chain.invoke({"text": document.page_content[:3000]})
            
            # Extract content from result
            if hasattr(result, 'content'):
                result = result.content
            else:
                result = str(result)
            
            # Clean the response
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            # Parse JSON
            try:
                extracted_data = json.loads(result)
            except json.JSONDecodeError:
                # If JSON parsing fails, return empty structure
                extracted_data = {"entities": [], "relationships": []}
            
            # Add source metadata
            extracted_data['source'] = document.metadata.get('file_name', 'unknown')
            extracted_data['chunk_id'] = document.metadata.get('chunk_id', 0)
            
            return extracted_data
            
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return {"entities": [], "relationships": [], "source": "unknown"}
    
    def build_graph_from_documents(self, documents: List[Document]) -> None:
        """
        Build knowledge graph from documents
        
        Args:
            documents: List of documents to process
        """
        try:
            print(f"Building knowledge graph from {len(documents)} documents...")
            
            for i, doc in enumerate(documents):
                print(f"Processing document {i+1}/{len(documents)}...")
                
                # Extract entities and relationships
                extracted = self.extract_entities_and_relationships(doc)
                
                # Add entities to graph
                for entity in extracted.get('entities', []):
                    entity_name = entity.get('name', '').strip()
                    if not entity_name:
                        continue
                    
                    entity_type = entity.get('type', 'UNKNOWN')
                    properties = entity.get('properties', {})
                    
                    # Add node - use entity_type instead of type to avoid conflict
                    self.graph.add_node(
                        entity_name,
                        entity_type=entity_type,
                        **properties
                    )
                    
                    # Store properties
                    self.entity_properties[entity_name].update(properties)
                    self.entity_properties[entity_name]['entity_type'] = entity_type
                
                # Add relationships to graph
                for rel in extracted.get('relationships', []):
                    source = rel.get('source', '').strip()
                    target = rel.get('target', '').strip()
                    relation = rel.get('relation', 'RELATED_TO')
                    properties = rel.get('properties', {})
                    
                    if source and target:
                        self.graph.add_edge(
                            source,
                            target,
                            relation=relation,
                            **properties
                        )
            
            print(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")
            
        except Exception as e:
            raise Exception(f"Error building knowledge graph: {str(e)}")
    
    def query_graph(
        self, 
        query: str, 
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Query knowledge graph
        
        Args:
            query: Natural language query
            max_hops: Maximum number of hops for graph traversal
            
        Returns:
            Dictionary containing query results
        """
        try:
            # Extract entities from query
            entity_extraction_prompt = PromptTemplate(
                template="""Extract the main entities mentioned in this question.
                Return ONLY a JSON list of entity names.
                
                Question: {question}
                
                JSON list:""",
                input_variables=["question"]
            )
            
            # Use new RunnableSequence API
            chain = entity_extraction_prompt | self.llm
            result = chain.invoke({"question": query})
            
            # Extract content from result
            if hasattr(result, 'content'):
                result = result.content
            else:
                result = str(result)
            
            # Parse entities
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]
            result = result.strip()
            
            try:
                query_entities = json.loads(result)
                if not isinstance(query_entities, list):
                    query_entities = []
            except:
                query_entities = []
            
            # Find entities in graph (fuzzy matching)
            found_entities = []
            for entity in query_entities:
                entity_lower = entity.lower()
                for node in self.graph.nodes():
                    if entity_lower in node.lower() or node.lower() in entity_lower:
                        found_entities.append(node)
                        break
            
            if not found_entities:
                # If no entities found, return graph summary
                return self._get_graph_summary()
            
            # Get subgraph around found entities
            subgraph_nodes = set(found_entities)
            for entity in found_entities:
                # Add neighbors within max_hops
                for neighbor in nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=max_hops
                ).keys():
                    subgraph_nodes.add(neighbor)
            
            # Build result
            result_data = {
                "entities": [],
                "relationships": [],
                "context": ""
            }
            
            # Add entity information
            for node in subgraph_nodes:
                entity_info = {
                    "name": node,
                    "type": self.graph.nodes[node].get('entity_type', 'UNKNOWN'),
                    "properties": dict(self.graph.nodes[node])
                }
                result_data["entities"].append(entity_info)
            
            # Add relationship information
            for source, target, data in self.graph.edges(data=True):
                if source in subgraph_nodes and target in subgraph_nodes:
                    rel_info = {
                        "source": source,
                        "target": target,
                        "relation": data.get('relation', 'RELATED_TO'),
                        "properties": {k: v for k, v in data.items() if k != 'relation'}
                    }
                    result_data["relationships"].append(rel_info)
            
            # Generate context summary
            result_data["context"] = self._generate_context_summary(result_data)
            
            return result_data
            
        except Exception as e:
            print(f"Error querying graph: {str(e)}")
            return {"entities": [], "relationships": [], "context": ""}
    
    def _get_graph_summary(self) -> Dict[str, Any]:
        """Get summary of entire graph"""
        summary = {
            "entities": [],
            "relationships": [],
            "context": f"Knowledge graph contains {self.graph.number_of_nodes()} entities "
                      f"and {self.graph.number_of_edges()} relationships."
        }
        
        # Get most connected nodes
        if self.graph.number_of_nodes() > 0:
            node_degrees = dict(self.graph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for node, degree in top_nodes:
                summary["entities"].append({
                    "name": node,
                    "type": self.graph.nodes[node].get('entity_type', 'UNKNOWN'),
                    "properties": dict(self.graph.nodes[node])
                })
        
        return summary
    
    def _generate_context_summary(self, data: Dict[str, Any]) -> str:
        """Generate human-readable context from graph data"""
        context_parts = []
        
        # Add entities
        if data["entities"]:
            entities_str = ", ".join([
                f"{e['name']} ({e['type']})" 
                for e in data["entities"][:5]
            ])
            context_parts.append(f"Relevant entities: {entities_str}")
        
        # Add relationships
        if data["relationships"]:
            rels_str = "; ".join([
                f"{r['source']} {r['relation']} {r['target']}"
                for r in data["relationships"][:5]
            ])
            context_parts.append(f"Relationships: {rels_str}")
        
        return ". ".join(context_parts) if context_parts else ""
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if self.graph.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "entity_types": {},
                "relation_types": {}
            }
        
        # Count entity types
        entity_types = defaultdict(int)
        for node in self.graph.nodes():
            entity_type = self.graph.nodes[node].get('entity_type', 'UNKNOWN')
            entity_types[entity_type] += 1
        
        # Count relation types
        relation_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            relation = data.get('relation', 'UNKNOWN')
            relation_types[relation] += 1
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types)
        }