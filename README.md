# Akash Chat API Plugin for ElizaOS

This plugin integrates the Akash Chat API with ElizaOS, providing a complete replacement for OpenAI API functionality. It supports text generation, embeddings, tokenization, and object generation using language models hosted on the Akash Network.

## Features

- **Text Generation**: Generate text using various LLMs hosted on Akash Network
- **Embeddings**: Generate text embeddings for semantic search and vector storage
- **Tokenization**: Encode and decode text using appropriate tokenizers
- **Object Generation**: Generate structured JSON objects from text prompts
- **Model Support**: Supports multiple models for different use cases and performance requirements

## Available Models

### Chat & Completion Models
- DeepSeek-R1-0528
- DeepSeek-R1-Distill-Llama-70B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- Meta-Llama-3-1-8B-Instruct-FP8
- Meta-Llama-3-2-3B-Instruct
- Meta-Llama-3-3-70B-Instruct
- Meta-Llama-4-Maverick-17B-128E-Instruct-FP8
- Qwen3-235B-A22B-Instruct-2507-FP8

### Embedding Models
- BAAI-bge-large-en-v1-5

## Setup Instructions

1. **Create a `.env` file in the plugin directory**:
   ```
   cp .env.example .env
   ```

2. **Add your Akash Chat API key to the `.env` file**:
   ```
   AKASH_CHAT_API_KEY=your_akash_chat_api_key_here
   ```

3. **Configure model selections (optional)**:
   You can customize which models to use by setting these environment variables in your `.env` file:
   ```
   # Small model for cost-effective, fast responses
   AKASH_CHAT_SMALL_MODEL=Meta-Llama-3-1-8B-Instruct-FP8
   
   # Large model for complex tasks requiring more reasoning
   AKASH_CHAT_LARGE_MODEL=Meta-Llama-3-3-70B-Instruct
   
   # Embedding model for vector operations
   AKASHCHAT_EMBEDDING_MODEL=BAAI-bge-large-en-v1-5
   ```

## Usage in Character Files

To use this plugin with a character, add it to your character's `.character.json` file:

```json
{
  "plugins": ["akashchat"],
  "modelProvider": "akashchat"
}
```

## Configuration Options

### Required
- `AKASH_CHAT_API_KEY`: Your API key for the Akash Chat API

### Optional Model Configuration
- `AKASH_CHAT_SMALL_MODEL`: Model to use for TEXT_SMALL operations (default: Meta-Llama-3-1-8B-Instruct-FP8)
- `AKASH_CHAT_LARGE_MODEL`: Model to use for TEXT_LARGE operations (default: Meta-Llama-3-3-70B-Instruct)
- `AKASHCHAT_EMBEDDING_MODEL`: Model to use for text embeddings (default: BAAI-bge-large-en-v1-5)
- `AKASHCHAT_EMBEDDING_DIMENSIONS`: Dimension size for embeddings (default: 1024)

### Optional Gateway Configuration
- `CLOUDFLARE_GW_ENABLED`: Set to 'true' to route requests through Cloudflare AI Gateway
- `CLOUDFLARE_AI_ACCOUNT_ID`: Cloudflare account ID (required if Cloudflare gateway is enabled)
- `CLOUDFLARE_AI_GATEWAY_ID`: Cloudflare gateway ID (required if Cloudflare gateway is enabled)

### Optional Debug Settings
- `AKASH_CHAT_EXPERIMENTAL_TELEMETRY`: Set to 'true' to enable experimental telemetry

## Troubleshooting

If you encounter issues:

1. **API Key Issues**:
   - Make sure the API key is correctly set in your `.env` file
   - Verify that your API key has the necessary permissions

2. **Model Configuration**:
   - Check that the modelProvider in your character file is set to "akashchat"
   - Verify that the model names in your environment variables match available models

3. **Connection Issues**:
   - Check your internet connection
   - Verify the API endpoint is accessible
   - Check logs for any API connection errors

## License

MIT
