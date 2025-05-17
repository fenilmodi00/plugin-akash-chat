# Akash Chat API Plugin for Eliza

This plugin integrates the Akash Chat API with Eliza, allowing characters to use language models hosted on the Akash Network.

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
   You can customize which models to use by setting these environment variables:
   ```
   AKASH_CHAT_SMALL_MODEL=Meta-Llama-3-1-8B-Instruct-FP8
   AKASH_CHAT_LARGE_MODEL=Meta-Llama-3-3-70B-Instruct
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

## Troubleshooting

If you encounter model downloading issues:

1. Make sure the API key is correctly set in your `.env` file
2. Verify that the modelProvider in your character file is set to "akashchat"
3. Check logs for any API connection errors

## Available Models

The plugin supports various Akash Chat API models:
- Text generation (small): Meta-Llama-3-1-8B-Instruct-FP8
- Text generation (large): Meta-Llama-3-3-70B-Instruct
- Embeddings: BAAI-bge-large-en-v1-5

## License

MIT