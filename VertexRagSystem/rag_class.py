import os
import json
import ssl
from typing import List, Dict, Optional, Tuple
import asyncio
from dataclasses import dataclass

# GCP Vertex AI imports
import vertexai
from vertexai.language_models import TextEmbeddingModel

# LangChain imports
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# MongoDB imports
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import certifi

# Configuration
@dataclass
class RAGConfig:
    project_id: str = "your-gcp-project-id"
    location: str = "us-central1"  # or your preferred region
    embedding_model: str = "text-embedding-005"
    llm_model: str = "gemini-2.0-flash-001"
    max_output_tokens: int = 1000
    temperature: float = 0.7
    top_k_docs: int = 3
    # Memory configuration
    max_token_limit: int = 2000  # Token limit for memory buffer
    return_messages: bool = True  # Return messages format for memory
    # MongoDB configuration
    mongo_db_connection_string: str = ""
    db_name: str = "test_db"
    collection_name: str = "test_collection_pdf"
    atlas_vector_search_index_name: str = "test_index_pdf"

class VertexRAGSystem:
    """Complete RAG system using MongoDB Atlas Vector Search with LangChain and Vertex AI"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm_model = None
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
        # LangChain memory components
        self.chat_model = None  # For memory summarization
        self.memory = None
        
        # MongoDB components
        self.mongo_client = None
        self.mongodb_collection = None
        
    async def initialize(self):
        """Initialize MongoDB vector store, embedding model, LLM, and memory"""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.config.project_id, location=self.config.location)
            print(f"✅ Vertex AI initialized for project: {self.config.project_id}")
            
            # Initialize MongoDB client
            if not self.config.mongo_db_connection_string:
                raise ValueError("MongoDB connection string not provided in config")
            
            # Try secure connection first
            try:
                print("🔄 Attempting secure MongoDB connection...")
                self.mongo_client = MongoClient(
                    self.config.mongo_db_connection_string, 
                    tlsCAFile=certifi.where()
                )
                # Test the connection immediately
                self.mongo_client.admin.command('ping')
                print("✅ Secure MongoDB connection established")
            except Exception as ssl_error:
                print(f"⚠️ Secure connection failed: {ssl_error}")
                print("🔄 Trying simplified connection...")
                try:
                    # Try without explicit TLS settings (rely on connection string)
                    self.mongo_client = MongoClient(self.config.mongo_db_connection_string)
                    self.mongo_client.admin.command('ping')
                    print("✅ MongoDB connection established (simplified)")
                except Exception as fallback_error:
                    print(f"❌ All connection attempts failed: {fallback_error}")
                    raise Exception(f"MongoDB connection failed: {fallback_error}")
            self.mongodb_collection = self.mongo_client[self.config.db_name][self.config.collection_name]
            print(f"✅ MongoDB connected to database: {self.config.db_name}, collection: {self.config.collection_name}")
            
            # Initialize Vertex AI embeddings through LangChain
            self.embeddings = VertexAIEmbeddings(
                model_name=self.config.embedding_model,
                project=self.config.project_id,
                location=self.config.location
            )
            print(f"✅ Vertex AI embeddings model '{self.config.embedding_model}' initialized through LangChain")
            
            # Initialize MongoDB Atlas Vector Search
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.mongodb_collection,
                embedding=self.embeddings,
                index_name=self.config.atlas_vector_search_index_name,
                relevance_score_fn="cosine"
            )
            print(f"✅ MongoDB Atlas Vector Search initialized with index: {self.config.atlas_vector_search_index_name}")
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.top_k_docs}
            )
            print(f"✅ Vector store retriever created with top_k={self.config.top_k_docs}")
            
            # Initialize LangChain ChatVertexAI for LLM and memory management
            self.chat_model = ChatVertexAI(
                model_name=self.config.llm_model,
                project=self.config.project_id,
                location=self.config.location,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens
            )
            print(f"✅ LangChain ChatVertexAI model '{self.config.llm_model}' initialized")
            
            # Initialize ConversationSummaryBufferMemory
            self.memory = ConversationSummaryBufferMemory(
                llm=self.chat_model,
                max_token_limit=self.config.max_token_limit,
                return_messages=self.config.return_messages
            )
            print(f"✅ ConversationSummaryBufferMemory initialized with token limit: {self.config.max_token_limit}")
            
            # Initialize RAG chain
            await self._setup_rag_chain()
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    async def _setup_rag_chain(self):
        """Setup the RAG chain with prompt template"""
        try:
            # Define system prompt
            system_prompt = """
            Use the given context to answer the question.
            If you don't know the answer, say you don't know.

            MUST NOT GENERATE ** or any other symbols other than period (.) and comma (,)
            
            Context documents: {context}
            """
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}")
            ])
            
            # Create document chain
            document_chain = create_stuff_documents_chain(self.chat_model, prompt)
            
            # Create retrieval chain
            self.chain = create_retrieval_chain(self.retriever, document_chain)
            
            print("✅ RAG chain initialized with retrieval and document processing")
            
        except Exception as e:
            print(f"❌ Failed to setup RAG chain: {e}")
            raise
    
    # MEMORY MANAGEMENT METHODS
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history from memory"""
        if not self.memory:
            return ""
        
        try:
            # Get the buffer (recent messages + summary if exists)
            messages = self.memory.chat_memory.messages
            
            if not messages:
                return ""
            
            # Format messages for context
            history_parts = []
            
            # Add summary if it exists
            if hasattr(self.memory, 'moving_summary_buffer') and self.memory.moving_summary_buffer:
                history_parts.append(f"Previous conversation summary: {self.memory.moving_summary_buffer}")
            
            # Add recent messages
            for message in messages:
                if isinstance(message, HumanMessage):
                    history_parts.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    history_parts.append(f"Assistant: {message.content}")
            
            return "\n".join(history_parts)
            
        except Exception as e:
            print(f"⚠️ Error getting conversation history: {e}")
            return ""
    
    def add_to_memory(self, human_input: str, ai_response: str):
        """Add human input and AI response to memory"""
        if not self.memory:
            print("⚠️ Memory not initialized")
            return
        
        try:
            self.memory.save_context(
                inputs={"input": human_input},
                outputs={"output": ai_response}
            )
            # print(f"💾 Added conversation to memory")
        except Exception as e:
            print(f"⚠️ Error adding to memory: {e}")
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            print("🗑️ Memory cleared")
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        if not self.memory:
            return {"status": "Memory not initialized"}
        
        try:
            messages = self.memory.chat_memory.messages
            return {
                "total_messages": len(messages),
                "has_summary": bool(hasattr(self.memory, 'moving_summary_buffer') and self.memory.moving_summary_buffer),
                "memory_initialized": True
            }
        except Exception as e:
            return {"error": str(e), "memory_initialized": False}

    # RAG RETRIEVAL METHODS
    
    async def retrieve_relevant_docs(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents using MongoDB Atlas Vector Search"""
        if not self.retriever:
            raise ValueError("Retriever not initialized. Run initialize() first.")
        
        try:
            # Update retriever with custom top_k if provided
            if top_k:
                self.retriever.search_kwargs["k"] = top_k
            
            # Retrieve documents
            docs = await asyncio.get_event_loop().run_in_executor(
                None, self.retriever.invoke, query
            )
            
            # Format results to match existing interface
            results = []
            for i, doc in enumerate(docs):
                results.append({
                    "doc_id": doc.metadata.get("_id", f"doc_{i}"),
                    "title": doc.metadata.get("title", "Document"),
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "similarity": 1.0  # MongoDB Atlas doesn't return scores by default
                })
            
            print(f"🔍 Retrieved {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            print(f"❌ Failed to retrieve documents: {e}")
            return []
    
    # LLM GENERATION METHODS
    
    async def generate_response(
        self, 
        prompt: str,
        context: Optional[str] = None,
        use_memory: bool = True
    ) -> str:
        """Generate response using the RAG chain"""
        if not self.chain:
            raise ValueError("RAG chain not initialized")
        
        try:
            # Add conversation history to prompt if memory is enabled
            full_prompt = prompt
            if use_memory and self.memory:
                conversation_history = self.get_conversation_history()
                if conversation_history:
                    full_prompt = f"Conversation History:\n{conversation_history}\n\nCurrent Question: {prompt}"
            
            # Use the RAG chain to get response
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.chain.invoke, {"input": full_prompt}
            )
            
            ai_response = result.get("answer", "Sorry, I couldn't generate a response.")
            
            # Add to memory if enabled
            if use_memory:
                self.add_to_memory(prompt, ai_response)
            
            return ai_response
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return f"Error generating response: {str(e)}"

    async def generate_response_stream(
        self, 
        prompt: str, 
        context: Optional[str] = None, 
        use_memory: bool = True,
        call_summary: Optional[str] = None
    ):
        """Generate streaming response with memory support and MongoDB document retrieval"""
        if not self.chat_model:
            raise ValueError("LLM model not initialized")

        try:
            # print(f"🔍 Starting document retrieval for query: '{prompt[:50]}...'")
            
            # Get conversation history if memory is enabled
            conversation_history = ""
            if use_memory and self.memory:
                conversation_history = self.get_conversation_history()

            # Step 1: Test MongoDB connection
            if not self.test_connection():
                print("❌ MongoDB connection failed, proceeding without document context")
                retrieved_docs = []
            else:
                # print("✅ MongoDB connection verified")
                
                # Step 2: Retrieve relevant documents with detailed logging
                retrieved_docs = []
                try:
                    if not self.retriever:
                        print("❌ Retriever not initialized")
                        retrieved_docs = []
                    else:
                        # print(f"📚 Retrieving documents from MongoDB collection: {self.config.collection_name}")
                        
                        # Check collection stats first
                        doc_count = self.mongodb_collection.count_documents({})
                        # print(f"📊 Total documents in collection: {doc_count}")
                        
                        if doc_count == 0:
                            print("⚠️ Collection is empty - no documents to retrieve")
                            retrieved_docs = []
                        else:
                            # Perform the actual retrieval
                            docs = await asyncio.get_event_loop().run_in_executor(
                                None, self.retriever.invoke, prompt
                            )
                            retrieved_docs = docs
                            print(f"✅ Retrieved {len(retrieved_docs)} documents from MongoDB")
                            
                            # Log document details for debugging
                            # for i, doc in enumerate(retrieved_docs[:3]):  # Log first 2 docs
                                # print(f"📄 Doc {i+1}: {doc.page_content[:100]}...")
                            
                except Exception as retrieval_error:
                    print(f"❌ Document retrieval failed: {retrieval_error}")
                    retrieved_docs = []

            # Step 3: Prepare context from retrieved documents
            context_from_docs = ""
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs):
                    context_parts.append(f"Document {i+1}:\nContent: {doc.page_content}\n")
                context_from_docs = "\n".join(context_parts)
                # print(f"📝 Prepared context from {len(retrieved_docs)} documents")
            else:
                print("⚠️ No documents retrieved - proceeding without document context")

            # Step 4: Build the complete prompt
            prompt_parts = []
            
            if conversation_history:
                prompt_parts.append(f"Conversation History:\n{conversation_history}\n")
                # print("💭 Added conversation history to prompt")
            
            if call_summary:
                print(f"📋 Call summary received: {call_summary[:100]}...")
                prompt_parts.append(f"Previous Call Summary:\n{call_summary}\n")

            if context_from_docs:
                prompt_parts.append(f"Retrieved Documents Context:\n{context_from_docs}\n")
                # print("📚 Added document context to prompt")
            
            prompt_parts.append(f"Current Question: {prompt}")
            
            # Add instructions based on whether we have context
            if context_from_docs:
                prompt_parts.append("""\n
                Instructions: Please answer the question based on the retrieved documents and conversation history above. 
                If the documents don't contain relevant information for the question, clearly state that.
                Keep your responses concise and accurate.
                """)
            else:
                prompt_parts.append("""\n
                Instructions: No relevant documents were found in the knowledge base for this query. 
                Please provide a helpful response based on your general knowledge, but mention that no specific documents were available.
                Keep your response concise.
                """)
            
            full_prompt = "\n".join(prompt_parts)
            # print(f"🤖 Sending prompt to LLM (length: {len(full_prompt)} chars)")

            # Step 5: Stream the LLM response
            full_response = ""
            chunk_count = 0
            
            async for chunk in self.chat_model.astream(full_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    chunk_count += 1
                    yield chunk.content
            
            # print(f"✅ Streaming complete: {chunk_count} chunks, {len(full_response)} total chars")
            
            # Step 6: Add to memory if enabled
            if use_memory and full_response:
                self.add_to_memory(prompt, full_response)
                # print("💾 Response saved to conversation memory")
                
        except Exception as e:
            error_msg = f"\n[Error in generate_response_stream: {str(e)}]"
            print(f"❌ Stream generation error: {e}")
            yield error_msg
    
    async def rag_query(self, query: str, include_sources: bool = True, use_memory: bool = True) -> Dict:
        """Complete RAG pipeline using LangChain retrieval chain"""
        try:
            # Add conversation history to query if memory is enabled
            full_query = query
            if use_memory and self.memory:
                conversation_history = self.get_conversation_history()
                if conversation_history:
                    full_query = f"Conversation History:\n{conversation_history}\n\nCurrent Question: {query}"
            
            # Use the RAG chain
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.chain.invoke, {"input": full_query}
            )
            
            answer = result.get("answer", "I couldn't generate a response.")
            
            # Extract source information if requested
            sources = []
            if include_sources and "context" in result:
                for i, doc in enumerate(result["context"]):
                    sources.append({
                        "doc_id": doc.metadata.get("_id", f"doc_{i}"),
                        "title": doc.metadata.get("title", "Document"),
                        "similarity": 1.0
                    })
            
            # Add to memory if enabled
            if use_memory:
                self.add_to_memory(query, answer)
            
            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "error": None,
                "memory_stats": self.get_memory_stats() if use_memory else None
            }
            
        except Exception as e:
            error_result = {
                "query": query,
                "answer": "An error occurred while processing your query.",
                "sources": [],
                "error": str(e)
            }
            
            # Add error to memory if enabled
            if use_memory:
                self.add_to_memory(query, error_result["answer"])
            
            return error_result
    
    # UTILITY METHODS
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            if not self.mongodb_collection:
                return {"error": "MongoDB collection not initialized"}
            
            doc_count = self.mongodb_collection.count_documents({})
            return {
                "total_documents": doc_count,
                "database": self.config.db_name,
                "collection": self.config.collection_name,
                "index_name": self.config.atlas_vector_search_index_name
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        try:
            if self.mongo_client:
                self.mongo_client.admin.command('ping')
                return True
            return False
        except Exception as e:
            print(f"❌ MongoDB connection test failed: {e}")
            return False
