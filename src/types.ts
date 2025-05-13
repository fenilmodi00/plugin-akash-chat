export interface AkashChatResponse {
  choices: {
    message: {
      content: string;
      role: string;
    };
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface EmbeddingResponse {
  data: {
    embedding: number[];
  }[];
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface TokenizationResponse {
  tokens: number[];
  token_strings?: string[];
}

export interface AkashChatOptions {
  modelName?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
  stopSequences?: string[];
}

export interface EmbeddingOptions {
  modelName?: string;
  dimensions?: number;
} 