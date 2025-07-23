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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    max_output_tokens: int = 200
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
            # Define simplified system prompt for real estate sales assistant
            system_prompt = """
            VERY IMPORTANT: 
You are a voice assistant Gina for GAMUDA COVE sales gallery. A sales gallery located at  Bandar Gamuda Cove, Kuala Langat in Selangor.

You are tasked with answering questions about the township development, specific property details, and booking appointments. 
This is a voice conversation, so keep your responses short, like in a real conversation. Don't ramble for too long.
Keep all your responses short and simple. Use casual language, phrases like "Umm...", "Well...", and "I mean" are preferred.
If they wish to book an appointment, your goal is to gather necessary information from callers in a friendly and efficient manner like follows:

1. Ask for their full name.
2. Ask for what kind of property they are interested in. Semi-detached, Terrace, Bungalow, Apartments, etc.
3. Ask what is the purpose of their purchase. For investment, for own stay, for family, etc.
4. When talking about the property, talk about in order of the features, benefits, and pricing of the properties. 

Key guidelines:
- NEVER say "I don't have information" or "I don't know" about anything
- For unclear questions, ask clarifying questions like "Do you mean..." or "Are you asking about..."
- For any topic you're unsure about, always redirect professionally: "That's a great question! I'd love to discuss that when you visit our sales gallery. When would work best for you?"
- Use simple, conversational language (like talking to a friend)
- Never use symbols like asterisks, dashes, or brackets
- Mirror the customer's speaking style - if they're formal, be formal; if casual, be casual
- Always try to guide the conversation toward scheduling a viewing
- Respond naturally, one topic at a time based on the conversation flow

For first-time callers: Greet them warmly and introduce yourself as Gina from Gamuda Cove.

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
    
    def _should_retrieve_documents(self, query: str) -> bool:
        """
        Determine if we should retrieve documents based on the query type
        Returns False for greetings and simple conversational queries
        """
        # Convert to lowercase for easier matching
        query_lower = query.lower().strip()
        
        # Define patterns that DON'T need document retrieval
        greeting_patterns = [
            "hi", "hello", "hey", "good morning", "good afternoon", 
            "good evening", "how are you", "what's up", "greetings"
        ]
        
        simple_conversational = [
            "thank you", "thanks", "okay", "ok", "yes", "no", 
            "goodbye", "bye", "see you", "have a good day"
        ]
        
        # Check if query is just a greeting or simple response
        if query_lower in greeting_patterns + simple_conversational:
            return False
        
        # Check if query is very short and likely conversational
        if len(query.split()) <= 2 and any(pattern in query_lower for pattern in greeting_patterns):
            return False
        
        # For everything else, retrieve documents
        return True
    
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
            return "That's a great question, and I'd love to discuss that in more detail when you visit our sales gallery. When would you like to set an appointment with us?"

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

            # Check if we should retrieve documents
            should_retrieve = self._should_retrieve_documents(prompt)
            
            if should_retrieve:
                # Step 1: Test MongoDB connection
                if not self.test_connection():
                    print("❌ MongoDB connection failed, proceeding without document context")
                    retrieved_docs = []
                else:
                    # Step 2: Retrieve relevant documents with detailed logging
                    retrieved_docs = []
                    try:
                        if not self.retriever:
                            print("❌ Retriever not initialized")
                            retrieved_docs = []
                        else:
                            # Check collection stats first
                            doc_count = self.mongodb_collection.count_documents({})
                            
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
                                
                    except Exception as retrieval_error:
                        print(f"❌ Document retrieval failed: {retrieval_error}")
                        retrieved_docs = []
            else:
                print(f"📝 Skipping document retrieval for greeting/conversational query: '{prompt}'")
                retrieved_docs = []

            # Step 3: Prepare context from retrieved documents
            context_from_docs = ""
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs):
                    context_parts.append(f"Document {i+1}:\nContent: {doc.page_content}\n")
                context_from_docs = "\n".join(context_parts)
                print(f"📝 Prepared context from {len(retrieved_docs)} documents")
            else:
                if should_retrieve:
                    print("⚠️ No documents retrieved - proceeding without document context")
                # If we intentionally skipped retrieval for greetings, no warning needed

            # Step 4: Build the complete prompt
            prompt_parts = []

            if conversation_history:
                prompt_parts.append(f"Conversation History:\n{conversation_history}\n")

            if call_summary:
                print(f"📋 Call summary received: {call_summary[:100]}...")
                prompt_parts.append(f"Previous Call Summary:\n{call_summary}\n")

            if context_from_docs:
                prompt_parts.append(f"Retrieved Documents Context:\n{context_from_docs}\n")

            prompt_parts.append(f"Current Question: {prompt}")

            # Add simplified instructions based on query type
            if should_retrieve and context_from_docs:
                prompt_parts.append("""\n
                RESPONSE INSTRUCTIONS: 
                - Use simple, friendly language
                - Never use symbols (no asterisks, dashes, brackets, etc.)
                - Convert all numbers to words (e.g., "2214" becomes "two thousand two hundred and fourteen")
                - Explain as such the person can explain within 3 sentences.
                """)
            elif should_retrieve and not context_from_docs:
                prompt_parts.append("""\n
                RESPONSE INSTRUCTIONS:
                - Use simple, friendly language
                - Never use symbols (no asterisks, dashes, brackets, etc.)
                - Since no specific documents were found, use this professional redirect: "That's a great question! I'd love to discuss that when you visit our sales gallery. When would work best for you?"
                """)
            else:
                # For greetings and simple conversational queries
                prompt_parts.append("""\n
                RESPONSE INSTRUCTIONS:
                - Use simple, friendly language
                - Never use symbols (no asterisks, dashes, brackets, etc.)
                - Keep response natural and conversational
                - Focus on greeting the customer warmly and asking how you can help
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
    
    
    
    # UTILITY METHODS
    
    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            if self.mongodb_collection is None:
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
            if self.mongo_client is not None:
                self.mongo_client.admin.command('ping')
                return True
            return False
        except Exception as e:
            print(f"❌ MongoDB connection test failed: {e}")
            return False

    async def upload_pdf_to_mongodb(self, pdf_file_path: str) -> Dict:
        """
        Upload PDF to MongoDB with text embedding using Vertex AI
        Uses the same embedding model as the main RAG system for consistency
        """
        try:
            print(f"📄 Starting PDF upload process for: {pdf_file_path}")
            
            # Step 1: Verify that embeddings are initialized
            if not self.embeddings:
                raise ValueError("Vertex AI embeddings not initialized. Please run initialize() first.")
            
            print("🔧 Using existing Vertex AI embeddings...")
            
            # Step 2: Use the existing vector store (same embeddings, same collection)
            # This ensures consistency with the main RAG system
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please run initialize() first.")
            
            print("✅ Using existing MongoDB vector store...")
            
            # Step 3: Initialize text splitter
            print("📝 Initializing text splitter...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            # Step 4: Load and split PDF
            print(f"📖 Loading PDF from: {pdf_file_path}")
            loader = PyPDFLoader(pdf_file_path)
            docs = loader.load_and_split(text_splitter)
            
            if not docs:
                return {
                    "success": False,
                    "message": "No content found in PDF",
                    "documents_processed": 0
                }
            
            print(f"📄 PDF split into {len(docs)} chunks")
            
            # Step 5: Add metadata to documents
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "source_file": os.path.basename(pdf_file_path),
                    "chunk_index": i,
                    "total_chunks": len(docs),
                    "upload_timestamp": asyncio.get_event_loop().time(),
                    "embedding_model": f"vertex-ai-{self.config.embedding_model}",
                    "document_type": "pdf"
                })
            
            # Step 6: Add documents to vector store
            print("🔄 Adding documents to MongoDB vector store...")
            await asyncio.get_event_loop().run_in_executor(
                None, self.vector_store.add_documents, docs
            )
            
            # Step 7: The retriever automatically uses the updated vector store
            print("✅ Documents added to existing vector store - no retriever update needed")
            
            # Step 8: Verify vector search index
            try:
                # Get embedding dimension from the model
                embedding_dim = 768  # Default for most Vertex AI embedding models
                if hasattr(self.embeddings, 'model_name'):
                    # Different models might have different dimensions
                    if 'gecko' in self.embeddings.model_name:
                        embedding_dim = 768
                    elif 'textembedding-gecko@003' in self.embeddings.model_name:
                        embedding_dim = 768
                    
                self.vector_store.create_vector_search_index(dimensions=embedding_dim)
                print("✅ Vector search index verified")
            except Exception as index_error:
                print(f"⚠️ Vector search index note: {index_error}")
            
            print(f"✅ PDF upload completed successfully!")
            
            return {
                "success": True,
                "message": f"PDF '{os.path.basename(pdf_file_path)}' uploaded successfully",
                "documents_processed": len(docs),
                "chunks_created": len(docs),
                "embedding_model": f"vertex-ai-{self.config.embedding_model}",
                "vector_store_stats": self.get_vector_store_stats()
            }
            
        except Exception as e:
            error_msg = f"PDF upload failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "message": error_msg,
                "documents_processed": 0,
                "error": str(e)
            }
    
    async def upload_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> Dict:
        """
        Upload PDF from bytes data (useful for FastAPI file uploads)
        Uses Vertex AI embeddings for consistency with the main RAG system
        """
        try:
            # Step 1: Save bytes to temporary file
            temp_dir = "/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"temp_{filename}")
            
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(pdf_bytes)
            
            print(f"📄 Temporary PDF saved to: {temp_file_path}")
            
            # Step 2: Process the PDF using Vertex AI
            result = await self.upload_pdf_to_mongodb(temp_file_path)
            
            # Step 3: Clean up temporary file
            try:
                os.remove(temp_file_path)
                print("🧹 Temporary file cleaned up")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
            
            return result
            
        except Exception as e:
            error_msg = f"PDF upload from bytes failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "message": error_msg,
                "documents_processed": 0,
                "error": str(e)
            }
