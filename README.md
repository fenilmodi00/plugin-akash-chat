# Akash Chat Plugin

A plugin for integrating with the Akash Chat API, providing text generation and embedding capabilities.

## Features

- Text generation using different model sizes (small and large)
- Text embedding generation
- Object generation capabilities
- Error handling and retry mechanisms
- Configuration management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plugin-akash-chat.git
cd plugin-akash-chat
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Then edit `.env` and add your Akash Chat API key.

## Configuration

The following environment variables are required:

- `AKASH_CHAT_API_KEY`: Your Akash Chat API key
- `AKASH_CHAT_SMALL_MODEL`: Model name for small text generation tasks
- `AKASH_CHAT_LARGE_MODEL`: Model name for large text generation tasks
- `AKASH_CHAT_EMBEDDING_MODEL`: Model name for embedding generation

## Usage

```typescript
import { akashChatPlugin } from 'plugin-akash-chat';

// Initialize the plugin
await akashChatPlugin.init(config, runtime);

// Use the plugin for text generation
const result = await runtime.useModel(ModelType.TEXT_SMALL, {
  prompt: "Your prompt here"
});
```

## Development

1. Make your changes
2. Run tests:
```bash
npm test
```
3. Build the project:
```bash
npm run build
```

## License

MIT
