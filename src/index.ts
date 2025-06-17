import { createOpenAI } from '@ai-sdk/openai';
import type {
  DetokenizeTextParams,
  GenerateTextParams,
  IAgentRuntime,
  ModelTypeName,
  ObjectGenerationParams,
  Plugin,
  TextEmbeddingParams,
  TokenizeTextParams,
} from '@elizaos/core';
import {
  EventType,
  logger,
  ModelType,
  VECTOR_DIMS,
} from '@elizaos/core';
import {
  generateObject,
  generateText,
  JSONParseError,
  type JSONValue,
  type LanguageModelUsage,
} from 'ai';
import { encodingForModel, type TiktokenModel } from 'js-tiktoken';
import { fetch } from 'undici';

/**
 * Retrieves a configuration setting from the runtime, falling back to environment variables or a default value if not found.
 *
 * @param key - The name of the setting to retrieve.
 * @param defaultValue - The value to return if the setting is not found in the runtime or environment.
 * @returns The resolved setting value, or defaultValue if not found.
 */
function getSetting(
  runtime: IAgentRuntime,
  key: string,
  defaultValue?: string
): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

/**
 * Retrieves the Akash Chat API base URL from runtime settings, environment variables, or defaults.
 *
 * @returns The resolved base URL for Akash Chat API requests.
 */
function getBaseURL(runtime: IAgentRuntime): string {
  const baseURL = getSetting(runtime, 'AKASH_CHAT_BASE_URL', 'https://chatapi.akash.network/api/v1') as string;
  logger.debug(`[AkashChat] Using base URL: ${baseURL}`);
  return baseURL;
}

/**
 * Helper function to get the API key for Akash Chat
 *
 * @param runtime The runtime context
 * @returns The configured API key
 */
function getApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'AKASH_CHAT_API_KEY');
}

/**
 * Helper function to get the small model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured small model name
 */
function getSmallModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'AKASH_CHAT_SMALL_MODEL', 'Meta-Llama-3-1-8B-Instruct-FP8') as string;
}

/**
 * Helper function to get the large model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured large model name
 */
function getLargeModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'AKASH_CHAT_LARGE_MODEL', 'Meta-Llama-3-3-70B-Instruct') as string;
}

/**
 * Helper function to get the embedding model name
 *
 * @param runtime The runtime context
 * @returns The configured embedding model name
 */
function getEmbeddingModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, 'AKASH_CHAT_EMBEDDING_MODEL', 'BAAI-bge-large-en-v1-5') as string;
}

/**
 * Create an Akash Chat client with proper configuration
 *
 * @param runtime The runtime context
 * @returns Configured OpenAI client pointing to Akash Chat
 */
function createAkashChatClient(runtime: IAgentRuntime) {
  return createOpenAI({
    apiKey: getApiKey(runtime),
    baseURL: getBaseURL(runtime),
  });
}

/**
 * Asynchronously tokenizes the given text based on the specified model and prompt.
 *
 * @param model - The type of model to use for tokenization.
 * @param prompt - The text prompt to tokenize.
 * @returns An array of tokens representing the encoded prompt.
 */
async function tokenizeText(model: ModelTypeName, prompt: string): Promise<number[]> {
  try {
    // Use OpenAI-compatible model names for tokenization since Akash models aren't supported by js-tiktoken
    const modelName = model === ModelType.TEXT_SMALL ? 'gpt-4o-mini' : 'gpt-4o';
    const encoding = encodingForModel(modelName as TiktokenModel);
    return encoding.encode(prompt);
  } catch (error) {
    logger.error('Error in tokenizeText:', error);
    return [];
  }
}

/**
 * Detokenize a sequence of tokens back into text using the specified model.
 *
 * @param model - The type of model to use for detokenization.
 * @param tokens - The sequence of tokens to detokenize.
 * @returns The detokenized text.
 */
async function detokenizeText(model: ModelTypeName, tokens: number[]): Promise<string> {
  try {
    // Use OpenAI-compatible model names for tokenization since Akash models aren't supported by js-tiktoken
    const modelName = model === ModelType.TEXT_SMALL ? 'gpt-4o-mini' : 'gpt-4o';
    const encoding = encodingForModel(modelName as TiktokenModel);
    return encoding.decode(tokens);
  } catch (error) {
    logger.error('Error in detokenizeText:', error);
    return '';
  }
}

/**
 * Helper function to generate objects using specified model type
 */
async function generateObjectByModelType(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
  modelType: string,
  getModelFn: (runtime: IAgentRuntime) => string
): Promise<JSONValue> {
  const akashChat = createAkashChatClient(runtime);
  const modelName = getModelFn(runtime);
  logger.log(`[AkashChat] Using ${modelType} model: ${modelName}`);
  const temperature = params.temperature ?? 0;

  try {
    const { object, usage } = await generateObject({
      model: akashChat.languageModel(modelName),
      output: 'no-schema',
      prompt: params.prompt,
      temperature: temperature,
      experimental_repairText: getJsonRepairFunction(),
    });

    if (usage) {
      emitModelUsageEvent(runtime, modelType as ModelTypeName, params.prompt, usage);
    }
    return object;
  } catch (error: unknown) {
    if (error instanceof JSONParseError) {
      logger.error(`[generateObject] Failed to parse JSON: ${error.message}`);

      const repairFunction = getJsonRepairFunction();
      const repairedJsonString = await repairFunction({
        text: error.text,
        error,
      });

      if (repairedJsonString) {
        try {
          const repairedObject = JSON.parse(repairedJsonString);
          logger.info('[generateObject] Successfully repaired JSON.');
          return repairedObject;
        } catch (repairParseError: unknown) {
          const message = repairParseError instanceof Error ? repairParseError.message : String(repairParseError);
          logger.error(`[generateObject] Failed to parse repaired JSON: ${message}`);
          throw repairParseError;
        }
      } else {
        logger.error('[generateObject] JSON repair failed.');
        throw error;
      }
    } else {
      const message = error instanceof Error ? error.message : String(error);
      logger.error(`[generateObject] Unknown error: ${message}`);
      throw error;
    }
  }
}

/**
 * Returns a function to repair JSON text
 */
function getJsonRepairFunction(): (params: {
  text: string;
  error: unknown;
}) => Promise<string | null> {
  return async ({ text, error }: { text: string; error: unknown }) => {
    try {
      if (error instanceof JSONParseError) {
        const cleanedText = text.replace(/```json\n|\n```|```/g, '');
        JSON.parse(cleanedText);
        return cleanedText;
      }
      return null;
    } catch (jsonError: unknown) {
      const message = jsonError instanceof Error ? jsonError.message : String(jsonError);
      logger.warn(`Failed to repair JSON text: ${message}`);
      return null;
    }
  };
}

/**
 * Emits a model usage event
 * @param runtime The runtime context
 * @param type The model type
 * @param prompt The prompt used
 * @param usage The LLM usage data
 */
function emitModelUsageEvent(
  runtime: IAgentRuntime,
  type: ModelTypeName,
  prompt: string,
  usage: LanguageModelUsage
) {
  runtime.emitEvent(EventType.MODEL_USED, {
    provider: 'akashchat',
    type,
    prompt,
    tokens: {
      prompt: usage.promptTokens,
      completion: usage.completionTokens,
      total: usage.totalTokens,
    },
  });
}

export const akashchatPlugin: Plugin = {
  name: 'akashchat',
  description: 'Akash Chat API plugin for language model capabilities via Akash Network',
  
  config: {
    AKASH_CHAT_API_KEY: process.env.AKASH_CHAT_API_KEY,
    AKASH_CHAT_BASE_URL: process.env.AKASH_CHAT_BASE_URL,
    AKASH_CHAT_SMALL_MODEL: process.env.AKASH_CHAT_SMALL_MODEL,
    AKASH_CHAT_LARGE_MODEL: process.env.AKASH_CHAT_LARGE_MODEL,
    AKASH_CHAT_EMBEDDING_MODEL: process.env.AKASH_CHAT_EMBEDDING_MODEL,
    AKASH_CHAT_EMBEDDING_DIMENSIONS: process.env.AKASH_CHAT_EMBEDDING_DIMENSIONS,
  },
  
  async init(_config: Record<string, string>, runtime: IAgentRuntime) {
    try {
      const apiKey = getApiKey(runtime);
      logger.log(`[AkashChat Init] API Key present: ${!!apiKey}`);
      
      if (!apiKey) {
        logger.warn(
          'AKASH_CHAT_API_KEY is not set - Akash Chat functionality will be limited'
        );
        return;
      }

      try {
        const baseURL = getBaseURL(runtime);
        logger.log(`[AkashChat Init] Using base URL: ${baseURL}`);
        logger.log(`[AkashChat Init] Making request to: ${baseURL}/models`);
        
        const response = await fetch(`${baseURL}/models`, {
          headers: { 
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
          },
        });
        
        logger.log(`[AkashChat Init] Response status: ${response.status}`);
        
        if (!response.ok) {
          logger.warn(`Akash Chat API key validation failed: ${response.statusText}`);
          logger.warn('Akash Chat functionality will be limited until a valid API key is provided');
        } else {
          const data = await response.json();
          logger.log(`✅ Akash Chat API connected successfully. Models available: ${(data as any)?.data?.length || 0}`);
        }
      } catch (fetchError: unknown) {
        const message = fetchError instanceof Error ? fetchError.message : String(fetchError);
        logger.warn(`Error validating Akash Chat API key: ${message}`);
        logger.warn('Akash Chat functionality will be limited until a valid API key is provided');
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      logger.warn(
        `Akash Chat plugin configuration issue: ${message} - You need to configure the AKASH_CHAT_API_KEY in your environment variables`
      );
    }
  },
  
  models: {
    [ModelType.TEXT_EMBEDDING]: async (
      runtime: IAgentRuntime,
      params: TextEmbeddingParams | string | null
    ): Promise<number[]> => {
      const embeddingModelName = getEmbeddingModel(runtime);
      const embeddingDimension = parseInt(
        getSetting(runtime, 'AKASH_CHAT_EMBEDDING_DIMENSIONS', '1024') || '1024',
        10
      ) as (typeof VECTOR_DIMS)[keyof typeof VECTOR_DIMS];
      
      logger.debug(
        `[AkashChat] Using embedding model: ${embeddingModelName} with dimension: ${embeddingDimension}`
      );

      if (!Object.values(VECTOR_DIMS).includes(embeddingDimension)) {
        const errorMsg = `Invalid embedding dimension: ${embeddingDimension}. Must be one of: ${Object.values(VECTOR_DIMS).join(', ')}`;
        logger.error(errorMsg);
        throw new Error(errorMsg);
      }

      if (params === null) {
        logger.debug('Creating test embedding for initialization');
        const testVector = Array(embeddingDimension).fill(0);
        testVector[0] = 0.1;
        return testVector;
      }

      let text: string;
      if (typeof params === 'string') {
        text = params;
      } else if (typeof params === 'object' && params.text) {
        text = params.text;
      } else {
        logger.warn('Invalid input format for embedding');
        const fallbackVector = Array(embeddingDimension).fill(0);
        fallbackVector[0] = 0.2;
        return fallbackVector;
      }

      if (!text.trim()) {
        logger.warn('Empty text for embedding');
        const emptyVector = Array(embeddingDimension).fill(0);
        emptyVector[0] = 0.3;
        return emptyVector;
      }

      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);

      if (!apiKey) {
        throw new Error('Akash Chat API key not configured');
      }

      try {
        const response = await fetch(`${baseURL}/embeddings`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: embeddingModelName,
            input: text,
          }),
        });

        if (!response.ok) {
          logger.error(`Akash Chat API error: ${response.status} - ${response.statusText}`);
          const errorVector = Array(embeddingDimension).fill(0);
          errorVector[0] = 0.4;
          return errorVector;
        }

        const data = (await response.json()) as {
          data: [{ embedding: number[] }];
          usage?: { prompt_tokens: number; total_tokens: number };
        };

        if (!data?.data?.[0]?.embedding) {
          logger.error('API returned invalid structure');
          const errorVector = Array(embeddingDimension).fill(0);
          errorVector[0] = 0.5;
          return errorVector;
        }

        const embedding = data.data[0].embedding;

        if (data.usage) {
          const usage = {
            promptTokens: data.usage.prompt_tokens,
            completionTokens: 0,
            totalTokens: data.usage.total_tokens,
          };

          emitModelUsageEvent(runtime, ModelType.TEXT_EMBEDDING, text, usage);
        }

        logger.log(`Got valid embedding with length ${embedding.length}`);
        return embedding;
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(`Error generating embedding: ${message}`);
        const errorVector = Array(embeddingDimension).fill(0);
        errorVector[0] = 0.6;
        return errorVector;
      }
    },
    
    [ModelType.TEXT_TOKENIZER_ENCODE]: async (
      _runtime: IAgentRuntime,
      { prompt, modelType = ModelType.TEXT_LARGE }: TokenizeTextParams
    ) => {
      return await tokenizeText(modelType ?? ModelType.TEXT_LARGE, prompt);
    },
    
    [ModelType.TEXT_TOKENIZER_DECODE]: async (
      _runtime: IAgentRuntime,
      { tokens, modelType = ModelType.TEXT_LARGE }: DetokenizeTextParams
    ) => {
      return await detokenizeText(modelType ?? ModelType.TEXT_LARGE, tokens);
    },
    
    [ModelType.TEXT_SMALL]: async (
      runtime: IAgentRuntime,
      { 
        prompt, 
        stopSequences = [],
        maxTokens = 8192,
        temperature = 0.7,
        frequencyPenalty = 0.7,
        presencePenalty = 0.7,
      }: GenerateTextParams
    ) => {
      const akashChat = createAkashChatClient(runtime);
      const modelName = getSmallModel(runtime);
      
      logger.log(`[AkashChat] Using TEXT_SMALL model: ${modelName}`);
      logger.log(prompt);

      const { text: response, usage } = await generateText({
        model: akashChat.languageModel(modelName),
        prompt: prompt,
        system: runtime.character.system ?? undefined,
        temperature: temperature,
        maxTokens: maxTokens,
        frequencyPenalty: frequencyPenalty,
        presencePenalty: presencePenalty,
        stopSequences: stopSequences,
      });

      if (usage) {
        emitModelUsageEvent(runtime, ModelType.TEXT_SMALL, prompt, usage);
      }

      return response;
    },
    
    [ModelType.TEXT_LARGE]: async (
      runtime: IAgentRuntime,
      {
        prompt,
        stopSequences = [],
        maxTokens = 8192,
        temperature = 0.7,
        frequencyPenalty = 0.7,
        presencePenalty = 0.7,
      }: GenerateTextParams
    ) => {
      const akashChat = createAkashChatClient(runtime);
      const modelName = getLargeModel(runtime);
      
      logger.log(`[AkashChat] Using TEXT_LARGE model: ${modelName}`);
      logger.log(prompt);

      const { text: response, usage } = await generateText({
        model: akashChat.languageModel(modelName),
        prompt: prompt,
        system: runtime.character.system ?? undefined,
        temperature: temperature,
        maxTokens: maxTokens,
        frequencyPenalty: frequencyPenalty,
        presencePenalty: presencePenalty,
        stopSequences: stopSequences,
      });

      if (usage) {
        emitModelUsageEvent(runtime, ModelType.TEXT_LARGE, prompt, usage);
      }

      return response;
    },
    
    [ModelType.OBJECT_SMALL]: async (runtime: IAgentRuntime, params: ObjectGenerationParams) => {
      return generateObjectByModelType(runtime, params, ModelType.OBJECT_SMALL, getSmallModel);
    },
    
    [ModelType.OBJECT_LARGE]: async (runtime: IAgentRuntime, params: ObjectGenerationParams) => {
      return generateObjectByModelType(runtime, params, ModelType.OBJECT_LARGE, getLargeModel);
    },
  },
  
  tests: [
    {
      name: 'akashchat_plugin_tests',
      tests: [
        {
          name: 'akashchat_test_url_and_api_key_validation',
          fn: async (runtime: IAgentRuntime) => {
            const baseURL = getBaseURL(runtime);
            const response = await fetch(`${baseURL}/models`, {
              headers: {
                Authorization: `Bearer ${getApiKey(runtime)}`,
              },
            });
            const data = await response.json();
            logger.log('Models Available:', (data as { data?: unknown[] })?.data?.length ?? 'N/A');
            if (!response.ok) {
              throw new Error(`Failed to validate Akash Chat API key: ${response.statusText}`);
            }
          },
        },
        {
          name: 'akashchat_test_text_embedding',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
                text: 'Hello, world!',
              });
              logger.log('embedding', embedding);
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_text_embedding: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'akashchat_test_text_large',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_LARGE, {
                prompt: 'What is the nature of reality in 10 words?',
              });
              if (text.length === 0) {
                throw new Error('Failed to generate text');
              }
              logger.log('generated with test_text_large:', text);
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_text_large: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'akashchat_test_text_small',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_SMALL, {
                prompt: 'What is the nature of reality in 10 words?',
              });
              if (text.length === 0) {
                throw new Error('Failed to generate text');
              }
              logger.log('generated with test_text_small:', text);
            } catch (error: unknown) {
              const message = error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_text_small: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'akashchat_test_text_tokenizer_encode',
          fn: async (runtime: IAgentRuntime) => {
            const prompt = 'Hello tokenizer encode!';
            const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, { prompt });
            if (!Array.isArray(tokens) || tokens.length === 0) {
              throw new Error('Failed to tokenize text: expected non-empty array of tokens');
            }
            logger.log('Tokenized output:', tokens);
          },
        },
        {
          name: 'akashchat_test_text_tokenizer_decode',
          fn: async (runtime: IAgentRuntime) => {
            const prompt = 'Hello tokenizer decode!';
            const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, { prompt });
            const decodedText = await runtime.useModel(ModelType.TEXT_TOKENIZER_DECODE, { tokens });
            if (decodedText !== prompt) {
              throw new Error(
                `Decoded text does not match original. Expected "${prompt}", got "${decodedText}"`
              );
            }
            logger.log('Decoded text:', decodedText);
          },
        },
      ],
    },
  ],
};

export default akashchatPlugin;