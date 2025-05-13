cat > README.md << EOL
# Akash Chat Plugin for ElizaOS

This plugin integrates Akash Chat API with ElizaOS, providing a complete replacement for OpenAI API functionality. It supports text generation, embeddings, tokenization, and object generation.

## Features

- Text generation (small, medium, and large models)
- Text embeddings using BAAI-bge-large-en-v1-5
- Tokenization (encode/decode)
- JSON object generation
- Rate limit handling with automatic retries
- Cloudflare Gateway support (optional)

## Setup

1. Install the plugin:
\`\`\`bash
npm install @elizaos/plugin-akash-chat
\`\`\`

2. Configure your environment variables:
\`\`\`
AKASH_CHAT_API_KEY=your_api_key_here
\`\`\`

3. Add to your agent configuration:
\`\`\`json
{
  "name": "MyAgent",
  "plugins": ["@elizaos/plugin-akash-chat"],
  "settings": {
    "secrets": {
      "AKASH_CHAT_API_KEY": "your_api_key_here"
    }
  }
}
\`\`\`

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| \`AKASH_CHAT_API_KEY\` | Your Akash Chat API key | (Required) |
| \`AKASH_CHAT_SMALL_MODEL\` | Model to use for small text generation | Meta-Llama-3-1-8B-Instruct-FP8 |
| \`AKASH_CHAT_MEDIUM_MODEL\` | Model to use for medium text generation | Meta-Llama-3-2-3B-Instruct |
| \`AKASH_CHAT_LARGE_MODEL\` | Model to use for large text generation | Meta-Llama-3-3-70B-Instruct |
| \`AKASHCHAT_EMBEDDING_MODEL\` | Model to use for embeddings | BAAI-bge-large-en-v1-5 |
| \`AKASHCHAT_EMBEDDING_DIMENSIONS\` | Dimensions for embeddings | 1024 |

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
EOL