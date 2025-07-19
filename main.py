from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import json
import numpy as np
import logging
from typing import List, Dict, Optional
from pymilvus import MilvusClient
import openai
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with error handling
openai_client = None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openai_client = client  # Ensure openai_client is set on success
    print("✅ OpenAI client initialized successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not initialize OpenAI client: {e}")
    print("   The app will run in fallback mode with simple responses.")
    openai_client = None

# Milvus configuration from environment variables
MILVUS_HOST = os.getenv("MILVUS_HOST", "in03-874be76b9aa0be7.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "268c3796886a41827afcee6560f083fbfc4992ae7265598b4d3582979748054380929293cd76ea79244845abf9773e4e9128de0e")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "youtubeCreaterVideos")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))


class MilvusVectorDB:
    def __init__(
        self,
        collection_name: str = MILVUS_COLLECTION,
        embedding_model: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION,
        host: str = MILVUS_HOST,
        token: str = MILVUS_TOKEN
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.host = host
        self.token = token
        self.collection = None
        self.milvus_client = None
        self._connect_to_milvus()
        self._init_collection()

    def _connect_to_milvus(self):
        """Connect to Zilliz Cloud using MilvusClient and token-based authentication"""
        try:
            self.milvus_client = MilvusClient(
                uri=f"https://{self.host}",
                token=self.token
            )
            logger.info(f"✅ Connected to Zilliz Cloud at https://{self.host} using MilvusClient")
        except Exception as e:
            logger.error(f"Failed to connect to Zilliz Cloud: {e}")
            logger.info("⚠️  Zilliz Cloud not available. Using local fallback mode.")
            self.milvus_client = None

    def _init_collection(self):
        """Initialize the Milvus collection using MilvusClient methods"""
        if not self.milvus_client:
            self.collection = None
            return
        try:
            if not self.milvus_client.has_collection(self.collection_name):
                schema = {
                    "fields": [
                        {"name": "primary_key", "description": "PK", "type": "INT64", "is_primary": True, "autoID": True},
                        {"name": "content", "description": "Content", "type": "VARCHAR", "max_length": 65535},
                        {"name": "metadata", "description": "Metadata", "type": "VARCHAR", "max_length": 65535},
                        {"name": "vector", "description": "Embedding vector", "type": "FLOAT_VECTOR", "dim": self.dimension}
                    ],
                    "description": "Chat documents collection"
                }
                self.milvus_client.create_collection(self.collection_name, schema=schema)
                logger.info(f"✅ Created Milvus collection: {self.collection_name}")
                # Create index
                index_params = {
                    "field_name": "vector",
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128}
                }
                self.milvus_client.create_index(self.collection_name, index_params)
            else:
                logger.info(f"✅ Loaded existing Milvus collection: {self.collection_name}")
            self.collection = self.collection_name
        except Exception as e:
            logger.error(f"Failed to create/load collection: {e}")
            self.collection = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using OpenAI API"""
        try:
            response = client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            arr = np.array(embedding, dtype=np.float32)
            logger.info(f"Embedding shape: {arr.shape}, dtype: {arr.dtype}, first 5: {arr[:5]}")
            if arr.shape != (self.dimension,):
                logger.error(f"Embedding dimension mismatch: got {arr.shape}, expected {self.dimension}")
            if np.isnan(arr).any():
                logger.error("Embedding contains NaN values!")
            return arr
        except Exception as e:
            logger.error(f"Error getting embedding from OpenAI: {e}")
            return np.zeros(self.dimension)

    def add_document(self, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a document to the Milvus collection using MilvusClient"""
        if not self.collection or not self.milvus_client:
            logger.warning("Milvus collection not available")
            return False
        try:
            embedding = self._get_embedding(content)
            metadata_str = json.dumps(metadata or {})
            # Generate a unique primary key (e.g., using current timestamp in ms)
            primary_key = int(time.time() * 1000)
            data = {
                "primary_key": primary_key,
                "vector": embedding.tolist(),
                "content": [content],
                "metadata": [metadata_str],
                "channel_name": metadata.get("channel_name", "")  # Add channel_name as top-level field
            }
            self.milvus_client.insert(self.collection, data)
            logger.info(f"✅ Added document to Milvus (embedding dim: {len(embedding)})")
            return True
        except Exception as e:
            logger.error(f"Error adding document to Milvus: {e}")
            return False

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar documents in Milvus using MilvusClient"""
        if not self.collection or not self.milvus_client:
            logger.warning("Milvus collection not available")
            return []
        try:
            query_embedding = self._get_embedding(query) # 'numpy.ndarray'            
            print(f"DEBUG: Query embedding shape: {query_embedding.shape}, first 5 values: {query_embedding[:5]}")
            print(f"DEBUG: Using embedding model: {self.embedding_model}, dimension: {self.dimension}")
            
            results = self.milvus_client.search(
                collection_name = self.collection, 
                data = [query_embedding],  
                limit = top_k,
                output_fields = ["content", "metadata"]
                )
            print(f"DEBUG: Raw search results: {results}")
            
            formatted_results = []
            for hit in results[0]:             
                try:
                    # Handle metadata that might be stored as array
                    metadata_raw = hit.get("metadata", "{}")
                    if isinstance(metadata_raw, list) and len(metadata_raw) > 0:
                        metadata_raw = metadata_raw[0]
                    metadata = json.loads(metadata_raw)
                except:
                    metadata = {}
                
                # Handle content that might be stored as array
                content_raw = hit.get("content", "")
                if isinstance(content_raw, list) and len(content_raw) > 0:
                    content_raw = content_raw[0]
                
                formatted_results.append({
                    'document': {
                        'id': hit.get("primary_key", None),
                        'content': content_raw,
                        'metadata': metadata
                    },
                    'score': 1.0 - float(hit.get("distance", 1.0))  # Convert distance to similarity score
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching Milvus: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get Milvus collection statistics using MilvusClient"""
        if not self.collection or not self.milvus_client:
            return {
                'status': 'not_connected',
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.dimension
            }
        try:
            stats = self.milvus_client.get_collection_stats(self.collection)
            return {
                'status': 'connected',
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.dimension,
                'total_entities': stats.get('row_count', 0),
                'milvus_host': f"{self.host}"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'status': 'error',
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.dimension,
                'error': str(e)
            }

    def clear_db(self) -> bool:
        """Clear all documents from the collection using MilvusClient"""
        if not self.collection or not self.milvus_client:
            return False
        try:
            self.milvus_client.drop_collection(self.collection)
            logger.info(f"✅ Dropped Milvus collection: {self.collection}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


# Global vector database instance
vector_db = MilvusVectorDB()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: list = []


class DocumentRequest(BaseModel):
    content: str
    metadata: str = "{}"


class DocumentResponse(BaseModel):
    success: bool
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up FastAPI application...")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("   Please set your OpenAI API key in a .env file or environment variable.")
        print("   The chat will fall back to simple responses.")
    else:
        print("✅ OpenAI API key found. ChatGPT integration enabled.")
    
    print("✅ Vector database initialized. RAG functionality enabled.")
    
    yield
    # Shutdown
    print("Shutting down FastAPI application...")


app = FastAPI(
    title="ChatGPT Chat API with RAG",
    description="A FastAPI application with ChatGPT integration, RAG functionality, and beautiful chat interface",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root() -> FileResponse:
    """Serve the chat interface."""
    return FileResponse("static/index.html")


@app.get("/hello")
async def hello_world() -> dict[str, str]:
    """Return a simple hello world message."""
    return {"message": "Hello, World!"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/vector-stats")
async def get_vector_stats() -> dict:
    """Get vector database statistics."""
    try:
        stats = vector_db.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vector stats: {str(e)}")


@app.post("/add-document", response_model=DocumentResponse)
async def add_document(request: DocumentRequest) -> DocumentResponse:
    """Add a document to the vector database for RAG."""
    try:
        metadata = {}
        if request.metadata:
            try:
                metadata = json.loads(request.metadata)
            except json.JSONDecodeError:
                metadata = {"raw_metadata": request.metadata}
        
        success = vector_db.add_document(request.content, metadata)
        if success:
            return DocumentResponse(success=True, message="Document added successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to add document")
    except Exception as e:
        import traceback
        print(f"Failed to add document: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest) -> ChatResponse:
    """Process a chat message and return a response using ChatGPT with RAG."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        sources = []
        # Try to use ChatGPT API with RAG
        if openai_client:
            response, sources = await get_chatgpt_response_with_rag(request.message, return_sources=True)
        else:
            # Fallback to simple responses if no API key
            response = generate_fallback_response(request.message.lower())
        return ChatResponse(response=response, sources=sources)
    except Exception as e:
        print(f"Error in chat processing: {e}")
        # Fallback to simple response on error
        fallback_response = generate_fallback_response(request.message.lower())
        return ChatResponse(response=fallback_response, sources=[])


async def get_chatgpt_response_with_rag(message: str, return_sources: bool = False):
    """Get a response from ChatGPT API with RAG context and optionally return sources."""
    if not openai_client:
        return generate_fallback_response(message.lower()), [] if return_sources else generate_fallback_response(message.lower())
    try:
        # Get relevant context from vector database
        context = ""
        similar_docs = vector_db.search(message, top_k=1000)
        print("similar_docs = ", similar_docs)
        sources = []
        if similar_docs:
            context_parts = []
            for result in similar_docs:
                video_title = result['document']['metadata'].get('video_title', 'Unknown video')
                context_parts.append(f"From video '{video_title}': {result['document']['content']}")
                sources.append({
                    "content": result['document']['content'],
                    "metadata": result['document']['metadata'],
                    "score": result['score']
                })
            context = "\n\n".join(context_parts)
            context = f"\n\nSOURCE MATERIAL - Use this information to answer the question:\n{context}\n\n"
        # Prepare system message with RAG context
        system_message = """You are a helpful, friendly AI assistant with access to a knowledge base. 

CRITICAL INSTRUCTIONS:
- When context is provided, you MUST use it as your only source for answering
- Always reference specific details, numbers, names, and quotes from the context

Guidelines:
- If context is not relevant or no context is provided, just say you don't know.
- Keep responses concise, helpful, and engaging"""

        # Prepare messages for ChatGPT
        messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]
        
        # Add context and question together in a single user message for better integration
        if context:
            combined_content = f"""SOURCE MATERIAL - You MUST use this information to answer the question:

{context}

IMPORTANT: Use specific details, numbers, and quotes from the source material above. Do not give generic responses.

User question: {message}"""
            print(f"DEBUG: Sending to ChatGPT:\n{combined_content}")
            messages.append({
                "role": "user",
                "content": combined_content
            })
        else:
            print(f"DEBUG: No context found, sending question only: {message}")
            messages.append({
                "role": "user",
                "content": message
            })
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        if return_sources:
            return response.choices[0].message.content.strip(), sources
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        if return_sources:
            return generate_fallback_response(message.lower()), []
        return generate_fallback_response(message.lower())


async def get_chatgpt_response(message: str) -> str:
    """Get a response from ChatGPT API (without RAG)."""
    if not openai_client:
        return generate_fallback_response(message.lower())
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, friendly AI assistant. Keep your responses concise, helpful, and engaging."
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return generate_fallback_response(message.lower())


def generate_fallback_response(message: str) -> str:
    """Generate a fallback response when ChatGPT API is not available."""
    import random
    
    # Simple keyword-based responses
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    questions = ["how are you", "what's up", "how's it going"]
    thanks = ["thank you", "thanks", "thx"]
    weather = ["weather", "temperature", "forecast"]
    time = ["time", "clock", "hour"]
    
    if any(greeting in message for greeting in greetings):
        responses = [
            "Hello! How can I help you today?",
            "Hi there! Nice to meet you!",
            "Hey! I'm here to assist you.",
            "Greetings! What can I do for you?"
        ]
        return random.choice(responses)
    
    elif any(q in message for q in questions):
        responses = [
            "I'm doing great, thanks for asking! How about you?",
            "I'm functioning perfectly! Ready to help with anything you need.",
            "All systems operational! What can I assist you with today?"
        ]
        return random.choice(responses)
    
    elif any(t in message for t in thanks):
        responses = [
            "You're welcome! Is there anything else I can help you with?",
            "My pleasure! Let me know if you need anything else.",
            "Glad I could help! Feel free to ask more questions."
        ]
        return random.choice(responses)
    
    elif any(w in message for w in weather):
        return "I can't check the weather in real-time, but I'd recommend checking a weather app or website for the most accurate forecast!"
    
    elif any(t in message for t in time):
        from datetime import datetime
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
    
    elif "help" in message:
        return "I can help you with general questions, greetings, and basic information. Just ask me anything!"
    
    elif "bye" in message or "goodbye" in message:
        responses = [
            "Goodbye! Have a great day!",
            "See you later! Feel free to come back anytime.",
            "Take care! I'll be here when you need me."
        ]
        return random.choice(responses)
    
    else:
        responses = [
            "That's interesting! Tell me more about that.",
            "I'm not sure I understand. Could you rephrase that?",
            "Interesting question! I'm still learning, but I'd love to help where I can.",
            "I'm here to help! Could you provide more context?",
            "That's a great point! What specifically would you like to know?"
        ]
        return random.choice(responses)


@app.get("/status")
async def status():
    """Return Milvus/OpenAI/collection status for UI display."""
    stats = vector_db.get_stats()
    openai_status = bool(openai_client)
    return {
        "milvus": stats,
        "openai": openai_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Mount static files LAST to avoid overriding API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static") 
