<<<<<<< HEAD
# Akash Chat Plugin for ElizaOS
=======
# Akash Chat API Plugin for Eliza
>>>>>>> beta.1.0.41

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

<<<<<<< HEAD
1. Install the plugin:
```bash
npm install @eliza-plugins/plugin-akash-chat
```

2. Configure your environment variables:
```
AKASH_CHAT_API_KEY=your_api_key_here
```

3. Add to your agent configuration:
```json
{
  "name": "MyAgent",
  "plugins": ["@eliza-plugins/plugin-akash-chat"],
  "settings": {
    "secrets": {
      "API_KEY": "your_api_key_here"
    }
  }
=======
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
>>>>>>> beta.1.0.41
}
```

## Troubleshooting

<<<<<<< HEAD
| Setting | Description | Default |
|---------|-------------|---------|
| `API_KEY` | Your Akash Chat API key | (Required) |
| `AKASH_CHAT_SMALL_MODEL` | Model to use for small text generation | Meta-Llama-3-1-8B-Instruct-FP8 |
| `AKASH_CHAT_MEDIUM_MODEL` | Model to use for medium text generation | Meta-Llama-3-2-3B-Instruct |
| `AKASH_CHAT_LARGE_MODEL` | Model to use for large text generation | Meta-Llama-3-3-70B-Instruct |
| `AKASHCHAT_EMBEDDING_MODEL` | Model to use for embeddings | BAAI-bge-large-en-v1-5 |
| `AKASHCHAT_EMBEDDING_DIMENSIONS` | Dimensions for embeddings | 1024 |

## Available Models

The plugin supports all models available on Akash Chat API, including:
- Meta-Llama-3-1-8B-Instruct-FP8
- Meta-Llama-3-2-3B-Instruct
- Meta-Llama-3-3-70B-Instruct
- Meta-Llama-3-3-8B-Instruct
- DeepSeek-R1-Distill-Llama-70B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- BAAI-bge-large-en-v1-5 (embeddings)
=======
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
>>>>>>> beta.1.0.41
