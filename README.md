# Akash Chat Plugin for ElizaOS

This plugin provides integration with the Akash Chat API through the ElizaOS platform, offering a cost-effective alternative to OpenAI with similar capabilities powered by the Akash Supercloud.

## Features

- **OpenAI-Compatible API**: Seamless integration using the OpenAI SDK
- **Multiple Model Types**: Support for small and large language models
- **Text Embeddings**: Vector embeddings for semantic search and RAG
- **Object Generation**: Structured JSON output generation
- **Token Management**: Text tokenization and detokenization
- **Comprehensive Testing**: Built-in test suite for validation
- **Model Usage Tracking**: Token usage monitoring and event emission
- **Error Handling**: Robust error handling with proper fallbacks

## Optimizations

This plugin has been optimized based on the structure and best practices from the OpenAI plugin:

### Performance Improvements
- **Simplified Client Creation**: Removed unnecessary caching complexity
- **Direct OpenAI SDK Usage**: Leverages the proven `@ai-sdk/openai` package
- **Efficient Error Handling**: Streamlined error handling without over-engineered retry logic
- **Optimized Embeddings**: Clean vector generation without unnecessary fallback vectors

### Code Quality
- **TypeScript Compliance**: Full TypeScript support with proper typing
- **ElizaOS Standards**: Follows ElizaOS plugin conventions and patterns
- **Modular Architecture**: Well-organized helper functions and clean separation of concerns
- **Comprehensive Documentation**: JSDoc comments for all functions

### Developer Experience
- **Better Configuration**: Cleaner environment variable management
- **Improved Logging**: Enhanced logging with proper categorization
- **Test Coverage**: Comprehensive test suite including tokenization tests
- **Error Messages**: Clear, actionable error messages

## Installation

```bash
npm install @eliza-plugins/plugin-akash-chat
```

## Usage

Add the plugin to your character configuration:

```json
{
  "plugins": ["@eliza-plugins/plugin-akash-chat"]
}
```

## Configuration

### Environment Variables

```bash
# Required
AKASH_CHAT_API_KEY=your_akash_chat_api_key

# Optional - Model Configuration
AKASH_CHAT_BASE_URL=https://chatapi.akash.network/api/v1
AKASH_CHAT_SMALL_MODEL=Meta-Llama-3-1-8B-Instruct-FP8
AKASH_CHAT_LARGE_MODEL=Meta-Llama-3-3-70B-Instruct
AKASH_CHAT_EMBEDDING_MODEL=BAAI-bge-large-en-v1-5
AKASH_CHAT_EMBEDDING_DIMENSIONS=1024
```

### Character Settings

Alternatively, configure via character settings:

```json
{
  "settings": {
    "AKASH_CHAT_API_KEY": "your_akash_chat_api_key",
    "AKASH_CHAT_BASE_URL": "https://chatapi.akash.network/api/v1",
    "AKASH_CHAT_SMALL_MODEL": "Meta-Llama-3-1-8B-Instruct-FP8",
    "AKASH_CHAT_LARGE_MODEL": "Meta-Llama-3-3-70B-Instruct",
    "AKASH_CHAT_EMBEDDING_MODEL": "BAAI-bge-large-en-v1-5",
    "AKASH_CHAT_EMBEDDING_DIMENSIONS": "1024"
  }
}
```

## Available Models

### Chat + Completions

The plugin supports various Akash Chat models:

- **DeepSeek-R1-0528**: Advanced reasoning model
- **DeepSeek-R1-Distill-Llama-70B**: Large distilled model
- **Meta-Llama-3-1-8B-Instruct-FP8**: Fast, efficient model (default small)
- **Meta-Llama-3-3-70B-Instruct**: High-quality large model (default large)
- **Meta-Llama-4-Maverick-17B-128E-Instruct-FP8**: Latest Llama model
- **Qwen3-235B-A22B-FP8**: Large multilingual model

### Embeddings

- **BAAI-bge-large-en-v1-5**: High-quality English embeddings (default)

## API Usage

### Text Generation

```javascript
// Small model for fast responses
const response = await runtime.useModel(ModelType.TEXT_SMALL, {
  prompt: "Explain the Akash Network",
  temperature: 0.7,
  maxTokens: 1000
});

// Large model for complex reasoning
const response = await runtime.useModel(ModelType.TEXT_LARGE, {
  prompt: "Write a detailed analysis of decentralized computing",
  temperature: 0.8,
  maxTokens: 2000
});
```

### Embeddings

```javascript
const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
  text: "Akash Network is a decentralized cloud computing marketplace"
});
```

### Object Generation

```javascript
const structuredData = await runtime.useModel(ModelType.OBJECT_SMALL, {
  prompt: "Generate a JSON object describing the Akash Network",
  temperature: 0.3
});
```

### Tokenization

```javascript
// Encode text to tokens
const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, {
  prompt: "Hello, Akash Network!"
});

// Decode tokens back to text
const text = await runtime.useModel(ModelType.TEXT_TOKENIZER_DECODE, {
  tokens: tokens
});
```

## Model Types

- `TEXT_SMALL`: Optimized for fast, cost-effective responses
- `TEXT_LARGE`: For complex tasks requiring deeper reasoning  
- `TEXT_EMBEDDING`: Text embedding with 1024 dimensions
- `OBJECT_SMALL`: Structured object generation (fast)
- `OBJECT_LARGE`: Structured object generation (high-quality)
- `TEXT_TOKENIZER_ENCODE`: Text tokenization
- `TEXT_TOKENIZER_DECODE`: Token decoding

## Configuration Options

- `AKASH_CHAT_API_KEY` (required): Your Akash Chat API credentials
- `AKASH_CHAT_BASE_URL`: Custom API endpoint (default: https://chatapi.akash.network/api/v1)
- `AKASH_CHAT_SMALL_MODEL`: Small model name (default: Meta-Llama-3-1-8B-Instruct-FP8)
- `AKASH_CHAT_LARGE_MODEL`: Large model name (default: Meta-Llama-3-3-70B-Instruct)
- `AKASH_CHAT_EMBEDDING_MODEL`: Embedding model (default: BAAI-bge-large-en-v1-5)
- `AKASH_CHAT_EMBEDDING_DIMENSIONS`: Embedding dimensions (default: 1024)

## Testing

The plugin includes comprehensive tests that can be run independently:

```bash
# Run unit/integration tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run E2E tests (requires ElizaOS monorepo structure)
npm run test:e2e
```

### Test Types

1. **Unit Tests**: Test individual functions and components
2. **Integration Tests**: Test plugin integration with mock runtime
3. **E2E Tests**: Full end-to-end testing (requires monorepo setup)

### Available Test Cases

- **Configuration Tests**: Plugin metadata and config validation
- **Initialization Tests**: API key validation and setup
- **Model Handler Tests**: All model types (text, embedding, tokenization, objects)
- **Integration Tests**: API validation and embedding generation

### Running Tests in Standalone Mode

For standalone plugin development, use:

```bash
# Standard test run (recommended)
npm test

# Skip E2E tests that require monorepo
vitest run
```

The `npm run test:e2e` command will attempt to run the full ElizaOS test suite but may fail with "Could not find monorepo root" error when run outside the ElizaOS monorepo. This is expected behavior for standalone plugins.

## Benefits of Akash Network

- **Cost-Effective**: Competitive pricing compared to traditional cloud providers
- **Decentralized**: Permissionless marketplace for cloud resources
- **OpenAI Compatible**: Easy migration from OpenAI API
- **Community Support**: Active community available on Discord and GitHub

## Getting API Key

1. Visit [Akash Chat API](https://chatapi.akash.network)
2. Sign up for an account
3. Generate your API key
4. Add it to your environment variables or character settings

## Support

- [GitHub Discussions](https://github.com/orgs/akash-network/discussions)
- [Akash Discord](https://discord.com/invite/akash)
- [Documentation](https://chatapi.akash.network/documentation)

## License

This plugin is distributed under the same license as ElizaOS.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Changelog

### v1.0.0-beta.52
- Complete rewrite based on OpenAI plugin structure
- Improved error handling and performance
- Added comprehensive model usage tracking
- Enhanced TypeScript support
- Streamlined configuration management
- Added tokenization test coverage
- Removed unnecessary complexity and over-engineering
- Better integration with ElizaOS standards