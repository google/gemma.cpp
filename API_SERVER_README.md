# Gemma.cpp API Server

This is an HTTP API server for gemma.cpp that implements the Google API protocol, allowing you to interact with Gemma models through REST API endpoints compatible with the Google API format.

## Features

- **API-compatible**: Implements Google API endpoints
- **Unified client/server**: Single codebase supports both local and public API modes
- **Text generation**: Support for `generateContent` endpoint
- **Streaming support**: Server-Sent Events (SSE) for `streamGenerateContent`
- **Model management**: Support for `/v1beta/models` endpoint
- **Session management**: Maintains conversation context with KV cache
- **JSON responses**: All responses in Google API format
- **Error handling**: Proper HTTP status codes and error messages

## Building

The API server is built alongside the main gemma.cpp project:

```bash
# Configure the build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the API server and client
cmake --build build --target gemma_api_server gemma_api_client -j 8
```

The binaries will be created at:
- `build/gemma_api_server` - Local API server
- `build/gemma_api_client` - Unified client for both local and public APIs

## Usage

### Starting the Local API Server

```bash
./build/gemma_api_server \
  --tokenizer path/to/tokenizer.spm \
  --weights path/to/model.sbs \
  --port 8080
```

**Required arguments:**
- `--tokenizer`: Path to the tokenizer file (`.spm`)
- `--weights`: Path to the model weights file (`.sbs`)

**Optional arguments:**
- `--port`: Port to listen on (default: 8080)
- `--model`: Model name for API endpoints (default: gemma3-4b)

### Using the Unified Client

#### With Local Server
```bash
# Interactive chat with local server
./build/gemma_api_client --interactive 1 --host localhost --port 8080

# Single prompt with local server
./build/gemma_api_client --prompt "Hello, how are you?"
```

#### With Public Google API
```bash
# Set API key and use public API
export GOOGLE_API_KEY="your-api-key-here"
./build/gemma_api_client --interactive 1

# Or pass API key directly
./build/gemma_api_client --api_key "your-api-key" --interactive 1
```

## API Endpoints

The server implements Google API endpoints:

### 1. Generate Content - `POST /v1beta/models/gemma3-4b:generateContent`

Generate a response for given content (non-streaming).

**Request:**
```json
{
  "contents": [
    {
      "parts": [
        {"text": "Why is the sky blue?"}
      ]
    }
  ],
  "generationConfig": {
    "temperature": 0.9,
    "topK": 1,
    "maxOutputTokens": 1024
  }
}
```

**Response:**
```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {"text": "The sky appears blue because..."}
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "index": 0
    }
  ],
  "promptFeedback": {
    "safetyRatings": []
  },
  "usageMetadata": {
    "promptTokenCount": 5,
    "candidatesTokenCount": 25,
    "totalTokenCount": 30
  }
}
```

### 2. Stream Generate Content - `POST /v1beta/models/gemma3-4b:streamGenerateContent`

Generate a response with Server-Sent Events (SSE) streaming.

**Request:** Same as above

**Response:** Stream of SSE events:
```
data: {"candidates":[{"content":{"parts":[{"text":"The"}],"role":"model"},"index":0}],"promptFeedback":{"safetyRatings":[]}}

data: {"candidates":[{"content":{"parts":[{"text":" sky"}],"role":"model"},"index":0}],"promptFeedback":{"safetyRatings":[]}}

data: [DONE]
```

### 3. List Models - `GET /v1beta/models`

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "models/gemma3-4b",
      "displayName": "Gemma3 4B", 
      "description": "Gemma3 4B model running locally"
    }
  ]
}
```

## Example Usage

### Using curl with Local Server

```bash
# Generate content (non-streaming)
curl -X POST http://localhost:8080/v1beta/models/gemma3-4b:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "Hello, how are you?"}]}],
    "generationConfig": {"temperature": 0.9, "topK": 1, "maxOutputTokens": 1024}
  }'

# Stream generate content (SSE)
curl -X POST http://localhost:8080/v1beta/models/gemma3-4b:streamGenerateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "Tell me a story"}]}],
    "generationConfig": {"temperature": 0.9, "topK": 1, "maxOutputTokens": 1024}
  }'

# List models
curl http://localhost:8080/v1beta/models
```

### Multi-turn Conversation with curl

```bash
# First message
curl -X POST http://localhost:8080/v1beta/models/gemma3-4b:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"parts": [{"text": "Hi, my name is Alice"}]}
    ]
  }'

# Follow-up message with conversation history
curl -X POST http://localhost:8080/v1beta/models/gemma3-4b:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"parts": [{"text": "Hi, my name is Alice"}]},
      {"parts": [{"text": "Hello Alice! Nice to meet you."}]},
      {"parts": [{"text": "What is my name?"}]}
    ]
  }'
```

### Using Python

```python
import requests

# Generate content
response = requests.post('http://localhost:8080/v1beta/models/gemma3-4b:generateContent',
  json={
    'contents': [{'parts': [{'text': 'Explain quantum computing in simple terms'}]}],
    'generationConfig': {
      'temperature': 0.9,
      'topK': 1,
      'maxOutputTokens': 1024
    }
  }
)

result = response.json()
if 'candidates' in result and result['candidates']:
    text = result['candidates'][0]['content']['parts'][0]['text']
    print(text)
```

## Configuration Options

The Google API supports various generation configuration options:

- **temperature**: Controls randomness (0.0 to 2.0, default: 1.0)
- **topK**: Top-K sampling parameter (default: 1)
- **maxOutputTokens**: Maximum number of tokens to generate (default: 8192)

## Key Features

- **Unified Implementation**: Same codebase handles both local server and public API
- **Session Management**: Maintains conversation context using KV cache
- **Streaming Support**: Real-time token generation via Server-Sent Events
- **Error Handling**: Comprehensive error responses and HTTP status codes
- **Memory Efficient**: Optimized token processing and caching

## Compatibility

This implementation is compatible with:
- Google API format and endpoints
- Standard HTTP clients (curl, browsers, Python requests, etc.)
- Server-Sent Events (SSE) for streaming responses
- JSON request/response format
