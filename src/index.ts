/**
 * Akash Chat Plugin
 * 
 * This plugin provides integration with the Akash Chat API, allowing for:
 * - Text generation using different model sizes
 * - Text embedding generation
 * - Error handling and retry mechanisms
 * - Configuration management
 */

import type {
  AgentRuntime,
  ModelTypeName,
  ObjectGenerationParams,
  Plugin,
  TextEmbeddingParams,
} from '@elizaos/core';
import {
  type DetokenizeTextParams,
  type GenerateTextParams,
  ModelType,
  type TokenizeTextParams,
  logger,
} from '@elizaos/core';
import fetch, { RequestInit, Response } from 'node-fetch';
import FormData from 'form-data';

/**
 * Constants
 */
const AKASH_CHAT_BASE_URL = 'https://chatapi.akash.network/api/v1';

/**
 * Model Types Configuration
 * 
 * Defines the available model categories:
 * - SMALL: For smaller, faster models
 * - LARGE: For larger, more capable models
 * - EMBEDDING: For text embedding models
 */
const MODEL_TYPES = {
  SMALL: 'SMALL',
  LARGE: 'LARGE',
  EMBEDDING: 'EMBEDDING',
} as const;

type ModelType = keyof typeof MODEL_TYPES;
type ModelName = string;

/**
 * Error Types
 * 
 * Defines different categories of errors that can occur:
 * - CONFIG_ERROR: Issues with configuration settings
 * - AUTH_ERROR: Authentication/authorization failures
 * - API_ERROR: General API request failures
 * - NETWORK_ERROR: Network connectivity issues
 * - VALIDATION_ERROR: Input validation failures
 * - RATE_LIMIT_ERROR: API rate limit exceeded
 * - SERVER_ERROR: Server-side errors
 */
const ErrorCodes = {
  CONFIG_ERROR: 'CONFIG_ERROR',
  AUTH_ERROR: 'AUTH_ERROR',
  API_ERROR: 'API_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  RATE_LIMIT_ERROR: 'RATE_LIMIT_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
} as const;

type ErrorCode = typeof ErrorCodes[keyof typeof ErrorCodes];

/**
 * Response Interfaces
 * 
 * Defines the structure of API responses:
 * - ChatResponse: Structure for chat completion responses
 * - EmbeddingResponse: Structure for embedding generation responses
 */
interface ChatResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
}

interface EmbeddingResponse {
  data: Array<{
    embedding: number[];
  }>;
}

/**
 * Custom Error Class
 * 
 * Extends the standard Error class to include:
 * - status: HTTP status code (if applicable)
 * - code: Error category from ErrorCodes
 * - details: Additional error context
 */
class AkashChatError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly code: ErrorCode = 'API_ERROR',
    public readonly details?: Record<string, any>
  ) {
    super(message);
    this.name = 'AkashChatError';
  }

  /**
   * Determines if an error should be retried
   * @param code The error code to check
   * @returns true if the error is retryable
   */
  static isRetryable(code: ErrorCode): boolean {
    return ['NETWORK_ERROR', 'SERVER_ERROR', 'RATE_LIMIT_ERROR'].includes(code);
  }
}

/**
 * Configuration Management
 * 
 * Helper functions for managing configuration settings:
 * - getSetting: Retrieves settings with fallback to environment variables
 * - getBaseURL: Validates and returns the API base URL
 * - getApiKey: Validates and returns the API key
 * - getModel: Retrieves and validates model configuration
 */

/**
 * Retrieves a setting value with fallback to environment variables
 * @param runtime The runtime context
 * @param key The setting key to retrieve
 * @param defaultValue Optional default value if setting is not found
 * @returns The setting value
 * @throws AkashChatError if the setting is required but not found
 */
function getSetting(runtime: any, key: string, defaultValue?: string): string | undefined {
  const value = runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
  if (value === undefined) {
    throw new AkashChatError(
      `Configuration missing: ${key}`,
      undefined,
      'CONFIG_ERROR',
      { key, defaultValue }
    );
  }
  return value;
}

/**
 * Retrieves and validates the base URL for API requests
 * @param runtime The runtime context
 * @returns The validated base URL
 */
function getBaseURL(_runtime: any): string {
  return AKASH_CHAT_BASE_URL;
}

/**
 * Retrieves and validates the API key
 * @param runtime The runtime context
 * @returns The validated API key
 * @throws AkashChatError if the API key is missing
 */
function getApiKey(runtime: any): string {
  try {
    const key = getSetting(runtime, 'AKASH_CHAT_API_KEY');
    if (!key) {
      throw new AkashChatError(
        'API key is required. Please set AKASH_CHAT_API_KEY environment variable',
        undefined,
        'CONFIG_ERROR'
      );
    }
    return key;
  } catch (error) {
    if (error instanceof AkashChatError) throw error;
    throw new AkashChatError(
      'Failed to get API key',
      undefined,
      'CONFIG_ERROR',
      { error: error.message }
    );
  }
}

/**
 * Retrieves and validates the model configuration
 * @param runtime The runtime context
 * @param type The type of model to retrieve
 * @returns The validated model name
 * @throws AkashChatError if the model configuration is invalid
 */
function getModel(runtime: any, type: ModelType): ModelName {
  try {
    const envKey = `AKASH_CHAT_${type}_MODEL`;
    const model = getSetting(runtime, envKey);

    if (!model || typeof model !== 'string') {
      throw new AkashChatError(
        `Invalid model configuration for ${type}`,
        undefined,
        'VALIDATION_ERROR',
        { type, model }
      );
    }

    return model;
  } catch (error) {
    if (error instanceof AkashChatError) throw error;
    throw new AkashChatError(
      `Failed to get model for ${type}`,
      undefined,
      'CONFIG_ERROR',
      { error: error.message, type }
    );
  }
}

/**
 * API Request Management
 * 
 * Helper functions for making API requests:
 * - fetchWithRetry: Makes API requests with retry logic and error handling
 */

/**
 * Makes an API request with retry logic and error handling
 * @param url The URL to request
 * @param options The fetch options
 * @param maxRetries Maximum number of retry attempts
 * @param retryDelay Initial delay between retries (in milliseconds)
 * @returns The API response
 * @throws AkashChatError if all retries fail
 */
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries = 3,
  retryDelay = 1000
): Promise<Response> {
  let lastError: AkashChatError | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);

      if (response.ok) {
        return response;
      }

      const errorMessage = await response.text().catch(() => response.statusText);
      const errorDetails = { status: response.status, attempt, url };

      // Handle specific HTTP status codes
      if (response.status === 401) {
        throw new AkashChatError(
          'Authentication failed',
          response.status,
          'AUTH_ERROR',
          errorDetails
        );
      }

      if (response.status === 429) {
        throw new AkashChatError(
          'Rate limit exceeded',
          response.status,
          'RATE_LIMIT_ERROR',
          errorDetails
        );
      }

      if (response.status >= 500) {
        throw new AkashChatError(
          `Server error: ${errorMessage}`,
          response.status,
          'SERVER_ERROR',
          errorDetails
        );
      }

      throw new AkashChatError(
        `Request failed: ${errorMessage}`,
        response.status,
        'API_ERROR',
        errorDetails
      );
    } catch (error) {
      if (error instanceof AkashChatError) {
        lastError = error;
        if (!AkashChatError.isRetryable(error.code) || attempt >= maxRetries - 1) {
          throw error;
        }
      } else {
        lastError = new AkashChatError(
          'Network error occurred',
          undefined,
          'NETWORK_ERROR',
          { error: error.message, attempt }
        );
      }

      // Exponential backoff for retries
      if (attempt < maxRetries - 1) {
        const delay = retryDelay * Math.pow(2, attempt);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new AkashChatError('Max retries exceeded', undefined, 'NETWORK_ERROR');
}

/**
 * Defines the Akash Chat plugin with its name, description, and configuration options.
 * @type {Plugin}
 */
export const akashChatPlugin: Plugin = {
  name: 'akash-chat',
  description: 'A plugin for accessing the Akash Chat API.',
  config: {
    AKASH_CHAT_API_KEY: process.env.AKASH_CHAT_API_KEY,
    AKASH_CHAT_BASE_URL: process.env.AKASH_CHAT_BASE_URL,
    AKASH_CHAT_SMALL_MODEL: process.env.AKASH_CHAT_SMALL_MODEL,
    AKASH_CHAT_LARGE_MODEL: process.env.AKASH_CHAT_LARGE_MODEL,
    AKASH_CHAT_EMBEDDING_MODEL: process.env.AKASH_CHAT_EMBEDDING_MODEL,
  },
  async init(_config, runtime) {
    try {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);

      const response = await fetchWithRetry(`${baseURL}/models`, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });

      if (!response.ok) {
        throw new AkashChatError(
          `API key validation failed: ${response.statusText}`,
          response.status
        );
      }

      logger.info('Akash Chat plugin initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize Akash Chat plugin:', error);
      throw error;
    }
  },
  models: {
    [ModelType.TEXT_SMALL]: async (runtime, { prompt, stopSequences = [] }: GenerateTextParams) => {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, 'SMALL');

      const response = await fetchWithRetry(`${baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
          max_tokens: 8192,
          frequency_penalty: 0.7,
          presence_penalty: 0.7,
          stop: stopSequences,
        }),
      });

      const data = (await response.json()) as ChatResponse;
      if (!data.choices?.[0]?.message?.content) {
        throw new AkashChatError('Invalid response format from Akash Chat API');
      }
      return data.choices[0].message.content;
    },
    [ModelType.TEXT_LARGE]: async (
      runtime,
      {
        prompt,
        stopSequences = [],
        maxTokens = 8192,
        temperature = 0.7,
        frequencyPenalty = 0.7,
        presencePenalty = 0.7,
      }: GenerateTextParams
    ) => {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, 'LARGE');

      const response = await fetchWithRetry(`${baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature,
          max_tokens: maxTokens,
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
          stop: stopSequences,
        }),
      });

      const data = (await response.json()) as ChatResponse;
      if (!data.choices?.[0]?.message?.content) {
        throw new AkashChatError('Invalid response format from Akash Chat API');
      }
      return data.choices[0].message.content;
    },
    [ModelType.TEXT_EMBEDDING]: async (runtime, params?: TextEmbeddingParams) => {
      if (!params?.text) {
        return new Array(1024).fill(0);
      }

      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, 'EMBEDDING');

      const response = await fetchWithRetry(`${baseURL}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          input: params.text,
        }),
      });

      const data = (await response.json()) as EmbeddingResponse;
      if (!data.data?.[0]?.embedding) {
        throw new AkashChatError('Invalid response format from Akash Chat API');
      }
      return data.data[0].embedding;
    },
    [ModelType.OBJECT_SMALL]: async (runtime, params: ObjectGenerationParams) => {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, 'SMALL');

      // Enhanced prompt to ensure valid JSON generation
      const prompt = `You are a JSON generator. Your task is to generate a valid JSON object that strictly matches the following schema. Only return the JSON object, no other text.

Schema:
${JSON.stringify(params.schema, null, 2)}

Requirements:
1. The response must be valid JSON
2. The object must match the schema exactly
3. Do not include any explanations or text outside the JSON
4. Ensure all required properties are included
5. Use appropriate data types as specified in the schema`;

      const response = await fetchWithRetry(`${baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.3, // Lower temperature for more consistent output
          max_tokens: 8192,
          frequency_penalty: 0.0,
          presence_penalty: 0.0,
        }),
      });

      const data = (await response.json()) as ChatResponse;
      if (!data.choices?.[0]?.message?.content) {
        throw new AkashChatError('Invalid response format from Akash Chat API');
      }

      let content = data.choices[0].message.content.trim();
      
      // Try to extract JSON if it's wrapped in backticks or has additional text
      try {
        // Remove any markdown code block markers
        content = content.replace(/^```json\n?/, '').replace(/\n?```$/, '');
        
        // Try to find JSON-like content if there's surrounding text
        const jsonMatch = content.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
        if (jsonMatch) {
          content = jsonMatch[0];
        }

        const parsedObject = JSON.parse(content);

        // Validate the generated object against the schema
        // This is a basic check - you might want to add more thorough schema validation
        if (typeof parsedObject !== 'object' || parsedObject === null) {
          throw new Error('Generated content is not a valid object');
        }

        return parsedObject;
      } catch (error) {
        throw new AkashChatError(
          'Failed to parse generated object',
          undefined,
          'VALIDATION_ERROR',
          { 
            error: error.message, 
            content,
            schema: params.schema 
          }
        );
      }
    },
  },
  tests: [
    {
      name: 'akash_chat_plugin_tests',
      tests: [
        {
          name: 'akash_chat_test_url_and_api_key_validation',
          fn: async (runtime) => {
            const baseURL = getBaseURL(runtime);
            const response = await fetchWithRetry(`${baseURL}/models`, {
              headers: { Authorization: `Bearer ${getApiKey(runtime)}` },
            });
            const data = await response.json();
            logger.info('Models Available:', (data as any)?.data.length);
            if (!response.ok) {
              throw new AkashChatError(
                `Failed to validate Akash Chat API key: ${response.statusText}`,
                response.status
              );
            }
          },
        },
        {
          name: 'akash_chat_test_text_large',
          fn: async (runtime) => {
            const text = await runtime.useModel(ModelType.TEXT_LARGE, {
              prompt: 'What is the nature of reality in 10 words?',
            });
            if (text.length === 0) {
              throw new AkashChatError('Failed to generate text');
            }
            logger.info('Generated text:', text);
          },
        },
        {
          name: 'akash_chat_test_embeddings',
          fn: async (runtime) => {
            const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
              text: 'This is a test sentence for embedding generation',
            });
            if (!Array.isArray(embedding) || embedding.length === 0) {
              throw new AkashChatError('Failed to generate embeddings');
            }
            logger.info('Generated embedding with length:', embedding.length);
          },
        },
      ],
    },
  ],
};

export default akashChatPlugin;
