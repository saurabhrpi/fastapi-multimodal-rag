# FastAPI Multimodal RAG - ChatGPT Chat with RAG

A modern FastAPI application featuring a beautiful chat interface with ChatGPT integration and Retrieval-Augmented Generation (RAG) capabilities using Milvus vector database.

## Features

- 🚀 **FastAPI Backend** - Modern, fast Python web framework
- 💬 **ChatGPT Integration** - Real AI-powered conversations
- 🔍 **RAG (Retrieval-Augmented Generation)** - Enhanced responses with knowledge base
- 🗄️ **Milvus Vector Database** - High-performance vector search and storage
- 🧠 **Sentence Transformers** - Semantic embeddings using all-MiniLM-L6-v2
- 🎨 **Beautiful Chat Interface** - Modern, responsive UI
- 📚 **Document Management** - Add and search through your knowledge base
- 🔧 **Simple Setup** - Easy installation and configuration

## Quick Start

### 1. Start Milvus (Required)

First, start the Milvus vector database using Docker:

```bash
docker-compose up -d
```

Wait for all services to be healthy (check with `docker-compose ps`).

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### 4. Run the Application

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the Application

- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Milvus Console**: http://localhost:9091

## API Endpoints

### Core Endpoints

- `GET /` - Serve the chat interface
- `GET /hello` - Simple hello world endpoint
- `GET /health` - Health check endpoint

### Chat & RAG Endpoints

- `POST /ask` - Send a message and get AI response with RAG
- `POST /add-document` - Add a document to the knowledge base
- `GET /vector-stats` - Get Milvus collection statistics

### Example Usage

#### Add a Document to Knowledge Base

```bash
curl -X POST "http://localhost:8000/add-document" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.",
       "metadata": "{\"source\": \"documentation\", \"topic\": \"web frameworks\"}"
     }'
```

#### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What is FastAPI?"
     }'
```

#### Get Vector Database Stats

```bash
curl "http://localhost:8000/vector-stats"
```

## How RAG Works

1. **Document Storage**: Documents are converted to semantic embeddings using sentence transformers
2. **Vector Storage**: Embeddings are stored in Milvus vector database with efficient indexing
3. **Semantic Search**: When you ask a question, Milvus finds the most semantically similar documents
4. **Enhanced Response**: ChatGPT receives both your question and relevant context
5. **Better Answers**: Responses are more accurate and contextual

## Project Structure

```
fastapi-vibe-coding/
├── main.py              # FastAPI application
├── vector_db.py         # Milvus vector database implementation
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create from env.example)
├── env.example          # Environment variables template
├── static/
│   └── index.html       # Chat interface
└── README.md           # This file
```

## Dependencies

- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **OpenAI** - ChatGPT API integration
- **PyMilvus** - Milvus Python client
- **Sentence Transformers** - Semantic embeddings
- **NumPy** - Numerical computing
- **python-dotenv** - Environment variable management

## Milvus Configuration

The application uses Milvus with the following configuration:

- **Host**: localhost
- **Port**: 19530
- **Collection**: chat_documents
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Index Type**: IVF_FLAT with COSINE similarity
- **Search Parameters**: nprobe=10, top_k=3

## Troubleshooting

### Milvus Connection Issues

If you see "Milvus not available" messages:

1. **Check Docker services**: `docker-compose ps`
2. **Start Milvus**: `docker-compose up -d`
3. **Wait for health checks**: All services should show "healthy"
4. **Check logs**: `docker-compose logs standalone`

### OpenAI API Key Issues

If you see fallback responses instead of ChatGPT responses:
1. Check that your `.env` file exists and contains `OPENAI_API_KEY`
2. Verify your API key is valid and has credits
3. Check the console for any error messages

### Installation Issues

If you encounter installation problems:
1. Make sure you're using Python 3.8 or higher
2. Try installing packages individually: `pip install fastapi uvicorn openai pymilvus sentence-transformers python-dotenv numpy`
3. Use a virtual environment: `python -m venv venv && source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)

### Vector Database Issues

The vector database uses Milvus with sentence transformers. If you encounter issues:
1. Ensure Milvus is running: `docker-compose up -d`
2. Check Milvus health: `curl http://localhost:9091/healthz`
3. Verify the collection exists in Milvus console: http://localhost:9091
4. Restart the application if needed

## Docker Commands

### Start Milvus
```bash
docker-compose up -d
```

### Stop Milvus
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Reset Milvus Data
```bash
docker-compose down -v
docker-compose up -d
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
