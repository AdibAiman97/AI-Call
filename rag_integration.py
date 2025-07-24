"""
RAG Integration for Gemini Live API
Integrates MongoDB Atlas Vector Search with Gemini Live conversations
"""

import os
import logging
from typing import Optional, Dict, Any
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import certifi
# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGService:
    """RAG service for MongoDB Atlas Vector Search integration with Gemini Live"""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.vector_store: Optional[MongoDBAtlasVectorSearch] = None
        self.retriever = None
        self.chain = None
        self.llm = None
        self.embeddings = None
        
        # Configuration
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.MONGODB_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        self.DB_NAME = os.getenv("RAG_DB_NAME", "test_db")
        self.COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "voxis_ai")
        self.INDEX_NAME = os.getenv("RAG_INDEX_NAME", "voxis_ai_vector_index")
        
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        if not self.MONGODB_URI:
            raise ValueError("MONGODB_ATLAS_CLUSTER_URI environment variable not set")
    
    async def initialize(self):
        """Initialize RAG components"""
        try:
            logger.info("Initializing RAG service...")
            
            # Initialize Gemini components
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                api_key=self.GEMINI_API_KEY
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=self.GEMINI_API_KEY
            )
            
            # Initialize MongoDB client
            self.client = MongoClient(self.MONGODB_URI,tlsCAFile=certifi.where())
            collection = self.client[self.DB_NAME][self.COLLECTION_NAME]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB Atlas")
            
            # Check collection exists and has documents
            doc_count = collection.count_documents({})
            logger.info(f"Found {doc_count} documents in {self.DB_NAME}.{self.COLLECTION_NAME}")
            
            # Initialize vector store
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embeddings,
                index_name=self.INDEX_NAME,
                text_key="text",  # Field containing the text content  
                embedding_key="embedding",  # Field containing the embedding vectors (from MongoDB test)
                relevance_score_fn="cosine",
            )
            
            # Create retriever with more permissive settings for testing
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": 5,  # Top 5 relevant documents
                    # Remove score threshold to get all results
                }
            )
            
            # Create RAG chain
            system_prompt = """
            You are a helpful AI assistant with access to a knowledge database about Gamuda Cove residential projects.
            Use the given context to answer the user's question accurately and concisely.
            
            Guidelines:
            - Always use the provided context to answer questions about properties, prices, layouts, and amenities
            - Provide specific details from the context such as prices, sizes, and features
            - Keep responses conversational and natural since this is a voice conversation
            - Aim for 2-3 sentences maximum unless more detail is specifically requested
            - Be direct and helpful with the information available
            - Only say you don't have information if the context is truly empty or irrelevant
            
            Context from knowledge base: {context}
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            self.chain = create_retrieval_chain(self.retriever, document_chain)
            
            logger.info("RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            return False
    
    async def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process user query through RAG pipeline
        
        Args:
            user_input: User's question/query
            
        Returns:
            Dict containing answer and metadata
        """
        try:
            if not self.chain:
                raise ValueError("RAG service not initialized")
            
            logger.info(f"Processing RAG query: {user_input}")
            
            # Execute RAG chain
            result = self.chain.invoke({"input": user_input})
            
            # Extract information
            answer = result.get("answer", "I couldn't process your question.")
            context_docs = result.get("context", [])
            
            # Prepare response
            response = {
                "answer": answer,
                "sources_count": len(context_docs),
                "sources": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in context_docs
                ],
                "query": user_input
            }
            
            logger.info(f"RAG query processed successfully, found {len(context_docs)} relevant sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return {
                "answer": "I encountered an error while searching my knowledge base. Please try again.",
                "sources_count": 0,
                "sources": [],
                "query": user_input,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check RAG service health"""
        try:
            status = {
                "rag_initialized": bool(self.chain),
                "mongodb_connected": False,
                "vector_store_ready": bool(self.vector_store),
                "document_count": 0
            }
            
            if self.client:
                # Test MongoDB connection
                self.client.admin.command('ping')
                status["mongodb_connected"] = True
                
                # Get document count
                collection = self.client[self.DB_NAME][self.COLLECTION_NAME]
                status["document_count"] = collection.count_documents({})
            
            return status
            
        except Exception as e:
            logger.error(f"RAG health check failed: {e}")
            return {
                "rag_initialized": False,
                "mongodb_connected": False,
                "vector_store_ready": False,
                "document_count": 0,
                "error": str(e)
            }
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Global RAG service instance
rag_service = RAGService()

async def initialize_rag() -> bool:
    """Initialize the global RAG service"""
    return await rag_service.initialize()

async def process_rag_query(query: str) -> Dict[str, Any]:
    """Process a query through RAG"""
    return await rag_service.query(query)

async def get_rag_health() -> Dict[str, Any]:
    """Get RAG service health status"""
    return await rag_service.health_check()