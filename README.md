# akash-chat Plugin

This plugin provides integration with Akash Chat's models through the ElizaOS v2 platform.

## Usage

Add the plugin to your character configuration:

```json
"plugins": ["@elizaos/plugin-akash-chat"]
```

## Configuration

The plugin requires these environment variables (can be set in .env file or character settings):

```json
"settings": {
  "AKASH_CHAT_API_KEY": "your_akash-chat_api_key",
  "AKASH_CHAT_BASE_URL": "optional_custom_endpoint",
  "AKASH_CHAT_SMALL_MODEL": "Meta-Llama-3-1-8B-Instruct-FP8",
  "AKASH_CHAT_LARGE_MODEL": "Meta-Llama-3-3-70B-Instruct",
  "AKASH_CHAT_EMBEDDING_MODEL": "BAAI-bge-large-en-v1-5",
  "AKASH_CHAT_EMBEDDING_DIMENSIONS": "1024"
}
```

Or in `.env` file:

```env
AKASH_CHAT_API_KEY=your_akash-chat_api_key
# Optional overrides:
AKASH_CHAT_BASE_URL=optional_custom_endpoint
AKASH_CHAT_SMALL_MODEL=Meta-Llama-3-1-8B-Instruct-FP8
AKASH_CHAT_LARGE_MODEL=Meta-Llama-3-3-70B-Instruct
AKASH_CHAT_EMBEDDING_MODEL=BAAI-bge-large-en-v1-5
AKASH_CHAT_EMBEDDING_DIMENSIONS=1024
```

### Configuration Options

- `AKASH_CHAT_API_KEY` (required): Your Akash Chat API credentials
- `AKASH_CHAT_BASE_URL`: Custom API endpoint (default: https://chatapi.akash.network/api/v1)
- `AKASH_CHAT_SMALL_MODEL`: Defaults to Llama 3.1 ("Meta-Llama-3-1-8B-Instruct-FP8")
- `AKASH_CHAT_LARGE_MODEL`: Defaults to Llama 3.3 ("Meta-Llama-3-3-70B-Instruct")
- `AKASH_CHAT_EMBEDDING_MODEL`: Defaults to BAAI-bge-large-en-v1-5 ("BAAI-bge-large-en-v1-5")
- `AKASH_CHAT_EMBEDDING_DIMENSIONS`: Defaults to 1024 (1024)

The plugin provides these model classes:

- `TEXT_SMALL`: Optimized for fast, cost-effective responses
- `TEXT_LARGE`: For complex tasks requiring deeper reasoning
- `TEXT_TOKENIZER_ENCODE`: Text tokenization
- `TEXT_TOKENIZER_DECODE`: Token decoding
- `TEXT_EMBEDDING`: Text embedding generation

## Additional Features

### Text Embeddings

```js
const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, 'text to embed');
```

### Text Generation

```js
// Small model for quick responses
const smallResponse = await runtime.useModel(ModelType.TEXT_SMALL, {
  prompt: 'Your prompt here'
});

// Large model for complex tasks
const largeResponse = await runtime.useModel(ModelType.TEXT_LARGE, {
  prompt: 'Your complex prompt here',
  temperature: 0.7,
  maxTokens: 8192
});
```

### Tokenization

```js
// Encode text to tokens
const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, {
  prompt: 'Text to tokenize'
});

// Decode tokens back to text
const text = await runtime.useModel(ModelType.TEXT_TOKENIZER_DECODE, {
  tokens: tokens
});
```
