import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import asyncio
from dataclasses import dataclass

# GCP Vertex AI imports
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Part, Content

# LangChain imports for memory management
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage

# Configuration
@dataclass
class RAGConfig:
    project_id: str = "your-gcp-project-id"
    location: str = "us-central1"  # or your preferred region
    embedding_model: str = "text-embedding-004"
    llm_model: str = "gemini-2.0-flash-001"
    max_output_tokens: int = 1000
    temperature: float = 0.7
    top_k_docs: int = 3
    # Memory configuration
    max_token_limit: int = 2000  # Token limit for memory buffer
    return_messages: bool = True  # Return messages format for memory

class VertexRAGSystem:
    """Complete RAG system using GCP Vertex AI with LangChain memory management"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = None
        self.llm_model = None
        self.knowledge_base = []
        self.embedded_docs = []
        
        # LangChain memory components
        self.chat_model = None  # For memory summarization
        self.memory = None
        
    async def initialize(self):
        """Initialize both embedding and LLM models along with memory"""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.config.project_id, location=self.config.location)
            
            # Initialize embedding model
            self.embedding_model = TextEmbeddingModel.from_pretrained(self.config.embedding_model)
            print(f"✅ Embedding model '{self.config.embedding_model}' initialized")
            
            # Initialize LLM model (direct VertexAI for streaming)
            self.llm_model = GenerativeModel(self.config.llm_model)
            print(f"✅ LLM model '{self.config.llm_model}' initialized")
            
            # Initialize LangChain ChatVertexAI for memory management
            self.chat_model = ChatVertexAI(
                model_name=self.config.llm_model,
                project=self.config.project_id,
                location=self.config.location,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens
            )
            print(f"✅ LangChain ChatVertexAI model initialized for memory")
            
            # Initialize ConversationSummaryBufferMemory
            self.memory = ConversationSummaryBufferMemory(
                llm=self.chat_model,
                max_token_limit=self.config.max_token_limit,
                return_messages=self.config.return_messages
            )
            print(f"✅ ConversationSummaryBufferMemory initialized with token limit: {self.config.max_token_limit}")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
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
            print(f"💾 Added conversation to memory")
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

    # EMBEDDING FUNCTIONS
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to knowledge base
        Expected format: [{"id": "doc1", "title": "Title", "content": "Content..."}]
        """
        self.knowledge_base.extend(documents)
        print(f"📚 Added {len(documents)} documents to knowledge base")
    
    async def embed_documents(self, batch_size: int = 10):
        """Generate embeddings for all documents in batches"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        if not self.knowledge_base:
            raise ValueError("No documents in knowledge base")
            
        print(f"🔄 Embedding {len(self.knowledge_base)} documents...")
        self.embedded_docs = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(self.knowledge_base), batch_size):
            batch = self.knowledge_base[i:i + batch_size]
            batch_texts = [doc["content"] for doc in batch]
            
            try:
                # Get embeddings for batch
                embeddings_response = self.embedding_model.get_embeddings(batch_texts)
                
                # Store embeddings with metadata
                for j, embedding in enumerate(embeddings_response):
                    doc_index = i + j
                    self.embedded_docs.append({
                        "doc_id": batch[j]["id"],
                        "title": batch[j]["title"],
                        "content": batch[j]["content"],
                        "embedding": np.array(embedding.values)
                    })
                
                print(f"  ✅ Embedded batch {i//batch_size + 1}/{(len(self.knowledge_base)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"  ❌ Failed to embed batch starting at index {i}: {e}")
                continue
        
        print(f"🎉 Successfully embedded {len(self.embedded_docs)} documents")
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
            
        try:
            embedding_response = self.embedding_model.get_embeddings([query])
            return np.array(embedding_response[0].values)
        except Exception as e:
            print(f"❌ Failed to embed query: {e}")
            raise
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    async def retrieve_relevant_docs(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve most relevant documents for a query"""
        if not self.embedded_docs:
            raise ValueError("No embedded documents available. Run embed_documents() first.")
            
        top_k = top_k or self.config.top_k_docs
        
        # Get query embedding
        query_embedding = await self.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for doc in self.embedded_docs:
            similarity = self.cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((similarity, doc))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k documents with their similarity scores
        results = []
        for i in range(min(top_k, len(similarities))):
            similarity_score, doc = similarities[i]
            results.append({
                "similarity": similarity_score,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "content": doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"]
            })
        
        print(f"🔍 Retrieved {len(results)} relevant documents for query: '{query[:50]}...'")
        return results
    
    # LLM FUNCTIONS
    
    async def generate_response(
        self, 
        prompt: str,
        context: Optional[str] = None,
        use_memory: bool = True
    ) -> str:
        """Generate response using LLM with optional context and conversation memory"""
        if not self.llm_model:
            raise ValueError("LLM model not initialized")
        
        # Get conversation history if memory is enabled
        conversation_history = ""
        if use_memory and self.memory:
            conversation_history = self.get_conversation_history()
        
        # Prepare the full prompt with context and memory
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append(f"Conversation History:\n{conversation_history}\n")
        
        if context:
            prompt_parts.append(f"Context information:\n{context}\n")
        
        prompt_parts.append(f"Current Question: {prompt}")
        
        if context:
            prompt_parts.append("\nPlease answer the question based on the context provided above and the conversation history. If the context doesn't contain relevant information, please say so.")
        
        full_prompt = "\n".join(prompt_parts)
        
        try:
            response = await self.llm_model.generate_content_async(
                full_prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens
                }
            )
            
            if response.candidates and response.candidates[0].content:
                ai_response = response.candidates[0].content.parts[0].text
                
                # Add to memory if enabled
                if use_memory:
                    self.add_to_memory(prompt, ai_response)
                
                return ai_response
            else:
                return "Sorry, I couldn't generate a response."
                
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return f"Error generating response: {str(e)}"

    async def generate_response_stream(self, prompt: str, context: Optional[str] = None, use_memory: bool = True):
        """Generate streaming response with memory support"""
        if not self.llm_model:
            raise ValueError("LLM model not initialized")

        # Get conversation history if memory is enabled
        conversation_history = ""
        if use_memory and self.memory:
            conversation_history = self.get_conversation_history()

        # Prepare the full prompt with context and memory
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append(f"Conversation History:\n{conversation_history}\n")
        
        if context:
            prompt_parts.append(f"Context information:\n{context}\n")
        
        prompt_parts.append(f"Current Question: {prompt}")
        
        if context:
            prompt_parts.append("\nPlease answer the question based on the context provided above and the conversation history. If the context doesn't contain relevant information, please say so.")
        
        full_prompt = "\n".join(prompt_parts)

        try:
            # Collect the full response for memory storage
            full_response = ""
            
            # Stream content from the LLM
            response = self.llm_model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens
                },
                stream=True
            )
                
            for chunk in response:
                if chunk.candidates and chunk.candidates[0].content:
                    text = chunk.candidates[0].content.parts[0].text
                    full_response += text
                    yield text
            
            # Add complete conversation to memory after streaming is done
            if use_memory and full_response:
                self.add_to_memory(prompt, full_response)
                
        except Exception as e:
            error_msg = f"\n[Error generating response: {str(e)}]"
            yield error_msg
    
    async def rag_query(self, query: str, include_sources: bool = True, use_memory: bool = True) -> Dict:
        """Complete RAG pipeline: retrieve + generate with memory support"""
        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = await self.retrieve_relevant_docs(query)
            
            if not relevant_docs:
                result = {
                    "query": query,
                    "answer": "I couldn't find any relevant information in the knowledge base.",
                    "sources": [],
                    "error": None
                }
                
                # Still add to memory even if no docs found
                if use_memory:
                    self.add_to_memory(query, result["answer"])
                
                return result
            
            # Step 2: Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"Document {i+1} (ID: {doc['doc_id']}):\nTitle: {doc['title']}\nContent: {doc['content']}\n")
                sources.append({
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "similarity": doc["similarity"]
                })
            
            context = "\n".join(context_parts)
            
            # Step 3: Generate response using LLM with memory
            answer = await self.generate_response(query, context, use_memory=use_memory)
            
            result = {
                "query": query,
                "answer": answer,
                "sources": sources if include_sources else [],
                "error": None,
                "memory_stats": self.get_memory_stats() if use_memory else None
            }
            
            return result
            
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
    
    # UTILITY FUNCTIONS
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to file for faster startup"""
        if not self.embedded_docs:
            print("⚠️ No embeddings to save")
            return
            
        # Convert numpy arrays to lists for JSON serialization
        serializable_docs = []
        for doc in self.embedded_docs:
            doc_copy = doc.copy()
            doc_copy["embedding"] = doc["embedding"].tolist()
            serializable_docs.append(doc_copy)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_docs, f)
        
        print(f"💾 Saved {len(serializable_docs)} embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings from file"""
        try:
            with open(filepath, 'r') as f:
                serializable_docs = json.load(f)
            
            # Convert lists back to numpy arrays
            self.embedded_docs = []
            for doc in serializable_docs:
                doc["embedding"] = np.array(doc["embedding"])
                self.embedded_docs.append(doc)
            
            print(f"📂 Loaded {len(self.embedded_docs)} embeddings from {filepath}")
            
        except FileNotFoundError:
            print(f"⚠️ Embeddings file {filepath} not found")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
