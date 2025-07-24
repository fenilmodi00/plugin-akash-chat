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
  safeReplacer,
  ServiceType,
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
import { fetch, FormData } from 'undici';

/**
 * Retrieves a configuration setting from the runtime, falling back to environment variables or a default value if not found.
 *
 * @param key - The name of the setting to retrieve.
 * @param defaultValue - The value to return if the setting is not found in the runtime or environment.
 * @returns The resolved setting value, or {@link defaultValue} if not found.
 */
function getSetting(
  runtime: IAgentRuntime,
  key: string,
  defaultValue?: string
): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

/**
 * Retrieves the AkashChat API base URL from runtime settings, environment variables, or defaults.
 *
 * @returns The resolved base URL for AkashChat API requests.
 */
function getBaseURL(runtime: IAgentRuntime): string {
  const baseURL = getSetting(
    runtime,
    'AKASH_CHAT_BASE_URL',
    'https://chatapi.akash.network/api/v1'
  ) as string;
  logger.debug(`[AkashChat] Base URL: ${baseURL}`);
  return baseURL;
}

/**
 * Retrieves the AkashChat API base URL for embeddings, falling back to the general base URL.
 *
 * @returns The resolved base URL for AkashChat embedding requests.
 */
function getEmbeddingBaseURL(runtime: IAgentRuntime): string {
  const embeddingURL = getSetting(runtime, 'AKASH_CHAT_EMBEDDING_URL');
  if (embeddingURL) {
    logger.debug(`[AkashChat] Using specific embedding base URL: ${embeddingURL}`);
    return embeddingURL;
  }
  logger.debug('[AkashChat] Falling back to general base URL for embeddings.');
  return getBaseURL(runtime);
}

/**
 * Helper function to get the API key for AkashChat
 *
 * @param runtime The runtime context
 * @returns The configured API key
 */
function getApiKey(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, 'AKASH_CHAT_API_KEY');
}

/**
 * Helper function to get the embedding API key for AkashChat, falling back to the general API key if not set.
 *
 * @param runtime The runtime context
 * @returns The configured API key
 */
function getEmbeddingApiKey(runtime: IAgentRuntime): string | undefined {
  const embeddingApiKey = getSetting(runtime, 'AKASH_CHAT_EMBEDDING_API_KEY');
  if (embeddingApiKey) {
    logger.debug(
      `[AkashChat] Using specific embedding API key: ${embeddingApiKey}`
    );
    return embeddingApiKey;
  }
  logger.debug('[AkashChat] Falling back to general API key for embeddings.');
  return getApiKey(runtime);
}

/**
 * Helper function to get the small model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured small model name
 */
function getSmallModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'AKASH_CHAT_SMALL_MODEL') ??
    (getSetting(runtime, 'SMALL_MODEL', 'Meta-Llama-3-1-8B-Instruct-FP8') as string)
  );
}

/**
 * Helper function to get the large model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured large model name
 */
function getLargeModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'AKASH_CHAT_LARGE_MODEL') ??
    (getSetting(runtime, 'LARGE_MODEL', 'Meta-Llama-3-3-70B-Instruct') as string)
  );
}

/**
 * Helper function to get the embedding model name with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured embedding model name
 */
function getEmbeddingModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, 'AKASHCHAT_EMBEDDING_MODEL') ??
    'BAAI-bge-large-en-v1-5'
  );
}

/**
 * Helper function to get the embedding dimensions with fallbacks
 *
 * @param runtime The runtime context
 * @returns The configured embedding dimensions
 */
function getEmbeddingDimensions(runtime: IAgentRuntime): 384 | 512 | 768 | 1024 | 1536 | 3072 {
  const value = Number.parseInt(
    getSetting(runtime, 'AKASHCHAT_EMBEDDING_DIMENSIONS', '1024') || '1024',
    10
  );
  // Validate that the value is one of the allowed dimensions
  if (![384, 512, 768, 1024, 1536, 3072].includes(value)) {
    logger.warn(`Invalid embedding dimension: ${value}. Using default 1024.`);
    return 1024;
  }
  return value as (typeof VECTOR_DIMS)[keyof typeof VECTOR_DIMS];
}

/**
 * Helper function to get experimental telemetry setting
 *
 * @param runtime The runtime context
 * @returns Whether experimental telemetry is enabled
 */
function getExperimentalTelemetry(runtime: IAgentRuntime): boolean {
  const setting = getSetting(runtime, 'AKASH_CHAT_EXPERIMENTAL_TELEMETRY', 'false');
  // Convert to string and check for truthy values
  const normalizedSetting = String(setting).toLowerCase();
  const result = normalizedSetting === 'true';
  logger.debug(
    `[AkashChat] Experimental telemetry: "${setting}" (type: ${typeof setting}, normalized: "${normalizedSetting}", result: ${result})`
  );
  return result;
}

/**
 * Create an AkashChat client with proper configuration
 *
 * @param runtime The runtime context
 * @returns Configured AkashChat client
 */
function createAkashChatClient(runtime: IAgentRuntime) {
  return createOpenAI({
    apiKey: getApiKey(runtime),
    baseURL: getBaseURL(runtime),
    fetch: runtime.fetch,
  });
}

/**
 * Asynchronously tokenizes the given text based on the specified model and prompt.
 *
 * @param {ModelTypeName} model - The type of model to use for tokenization.
 * @param {string} prompt - The text prompt to tokenize.
 * @returns {number[]} - An array of tokens representing the encoded prompt.
 */
async function tokenizeText(model: ModelTypeName, prompt: string) {
  const modelName =
    model === ModelType.TEXT_SMALL
      ? getSmallModel({} as IAgentRuntime)
      : getLargeModel({} as IAgentRuntime);
  const encoding = encodingForModel(modelName as TiktokenModel);
  const tokens = encoding.encode(prompt);
  return tokens;
}

/**
 * Detokenize a sequence of tokens back into text using the specified model.
 *
 * @param {ModelTypeName} model - The type of model to use for detokenization.
 * @param {number[]} tokens - The sequence of tokens to detokenize.
 * @returns {string} The detokenized text.
 */
async function detokenizeText(model: ModelTypeName, tokens: number[]) {
  const modelName =
    model === ModelType.TEXT_SMALL
      ? getSmallModel({} as IAgentRuntime)
      : getLargeModel({} as IAgentRuntime);
  const encoding = encodingForModel(modelName as TiktokenModel);
  return encoding.decode(tokens);
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
  const akashchat = createAkashChatClient(runtime);
  const modelName = getModelFn(runtime);
  logger.log(`[AkashChat] Using ${modelType} model: ${modelName}`);
  const temperature = params.temperature ?? 0;
  const schemaPresent = !!params.schema;

  if (schemaPresent) {
    logger.info(
      `Using ${modelType} without schema validation (schema provided but output=no-schema)`
    );
  }

  try {
    const { object, usage } = await generateObject({
      model: akashchat.languageModel(modelName),
      output: 'no-schema',
      prompt: params.prompt,
      temperature: temperature,
      experimental_repairText: getJsonRepairFunction(),
    });

    if (usage) {
      emitModelUsageEvent(
        runtime,
        modelType as ModelTypeName,
        params.prompt,
        usage
      );
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
          const message =
            repairParseError instanceof Error
              ? repairParseError.message
              : String(repairParseError);
          logger.error(
            `[generateObject] Failed to parse repaired JSON: ${message}`
          );
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
      const message =
        jsonError instanceof Error ? jsonError.message : String(jsonError);
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

/**
 * Handles rate limit errors with exponential backoff
 */
async function handleRateLimitError(error: Error, retryFn: () => Promise<unknown>, retryCount = 0) {
  if (!error.message.includes('Rate limit')) {
    throw error;
  }
  
  // Extract retry delay from error message if possible
  let retryDelay = Math.min(10000 * Math.pow(1.5, retryCount), 60000); // Exponential backoff with 1 minute max
  const delayMatch = error.message.match(/try again in (\d+\.?\d*)s/i);
  
  if (delayMatch?.[1]) {
    // Convert to milliseconds and add a small buffer
    retryDelay = Math.ceil(Number.parseFloat(delayMatch[1]) * 1000) + 500;
  }
  
  logger.info(`Rate limit reached. Retrying after ${retryDelay}ms (attempt ${retryCount + 1})`);
  await new Promise((resolve) => setTimeout(resolve, retryDelay));
  
  try {
    return await retryFn();
  } catch (retryError: any) {
    if (retryError.message.includes('Rate limit') && retryCount < 3) {
      return handleRateLimitError(retryError, retryFn, retryCount + 1);
    }
    throw retryError;
  }
}

/**
 * Generate text using AkashChat API with optimized handling
 */
async function generateAkashChatText(
  akashchat: ReturnType<typeof createOpenAI>,
  model: string,
  params: {
    prompt: string;
    system?: string;
    temperature: number;
    maxTokens: number;
    frequencyPenalty: number;
    presencePenalty: number;
    stopSequences: string[];
  }
) {
  try {
    const { text, usage } = await generateText({
      model: akashchat.languageModel(model),
      prompt: params.prompt,
      system: params.system,
      temperature: params.temperature,
      maxTokens: params.maxTokens,
      frequencyPenalty: params.frequencyPenalty,
      presencePenalty: params.presencePenalty,
      stopSequences: params.stopSequences,
      experimental_telemetry: {
        isEnabled: getExperimentalTelemetry(akashchat as unknown as IAgentRuntime),
      },
    });
    
    if (usage) {
      emitModelUsageEvent(
        akashchat as unknown as IAgentRuntime,
        model.includes('8B') ? ModelType.TEXT_SMALL : ModelType.TEXT_LARGE,
        params.prompt,
        usage
      );
    }
    
    return text;
  } catch (error: unknown) {
    if (error instanceof Error && error.message.includes('Rate limit')) {
      return handleRateLimitError(error, () => 
        generateAkashChatText(akashchat, model, params)
      ) as Promise<string>;
    }
    
    logger.error('Error generating text:', error);
    return 'Error generating text. Please try again later.';
  }
}

export const akashchatPlugin: Plugin = {
  name: 'akashchat',
  description: 'AkashChat API plugin for language model capabilities via Akash Network',
  
  config: {
    AKASH_CHAT_API_KEY: process.env.AKASH_CHAT_API_KEY,
    AKASH_CHAT_BASE_URL: process.env.AKASH_CHAT_BASE_URL,
    AKASH_CHAT_SMALL_MODEL: process.env.AKASH_CHAT_SMALL_MODEL,
    AKASH_CHAT_LARGE_MODEL: process.env.AKASH_CHAT_LARGE_MODEL,
    SMALL_MODEL: process.env.SMALL_MODEL,
    LARGE_MODEL: process.env.LARGE_MODEL,
    AKASHCHAT_EMBEDDING_MODEL: process.env.AKASHCHAT_EMBEDDING_MODEL,
    AKASHCHAT_EMBEDDING_API_KEY: process.env.AKASHCHAT_EMBEDDING_API_KEY,
    AKASHCHAT_EMBEDDING_URL: process.env.AKASHCHAT_EMBEDDING_URL,
    AKASHCHAT_EMBEDDING_DIMENSIONS: process.env.AKASHCHAT_EMBEDDING_DIMENSIONS,
    AKASH_CHAT_EXPERIMENTAL_TELEMETRY: process.env.AKASH_CHAT_EXPERIMENTAL_TELEMETRY,
  },
  
  async init(_config, runtime) {
    // do check in the background
    new Promise<void>(async (resolve) => {
      resolve();
      try {
        if (!getApiKey(runtime)) {
          logger.warn(
            'AKASH_CHAT_API_KEY is not set in environment - AkashChat functionality will be limited'
          );
          return;
        }
        try {
          const baseURL = getBaseURL(runtime);
          const response = await fetch(`${baseURL}/models`, {
            headers: { Authorization: `Bearer ${getApiKey(runtime)}` },
          });
          if (!response.ok) {
            logger.warn(
              `AkashChat API key validation failed: ${response.statusText}`
            );
            logger.warn(
              'AkashChat functionality will be limited until a valid API key is provided'
            );
          } else {
            logger.log('AkashChat API key validated successfully');
          }
        } catch (fetchError: unknown) {
          const message =
            fetchError instanceof Error
              ? fetchError.message
              : String(fetchError);
          logger.warn(`Error validating AkashChat API key: ${message}`);
          logger.warn(
            'AkashChat functionality will be limited until a valid API key is provided'
          );
        }
      } catch (error: unknown) {
        const message =
          (error as { errors?: Array<{ message: string }> })?.errors
            ?.map((e) => e.message)
            .join(', ') ||
          (error instanceof Error ? error.message : String(error));
        logger.warn(
          `AkashChat plugin configuration issue: ${message} - You need to configure the AKASH_CHAT_API_KEY in your environment variables`
        );
      }
    });
  },

  models: {
    [ModelType.TEXT_EMBEDDING]: async (
      runtime: IAgentRuntime,
      params: TextEmbeddingParams | string | null
    ): Promise<number[]> => {
      const embeddingModelName = getEmbeddingModel(runtime);
      const embeddingDimension = getEmbeddingDimensions(runtime);

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

      const embeddingBaseURL = getEmbeddingBaseURL(runtime);
      const apiKey = getEmbeddingApiKey(runtime);

      if (!apiKey) {
        throw new Error('AkashChat API key not configured');
      }

      try {
        const response = await fetch(`${embeddingBaseURL}/embeddings`, {
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

        const responseClone = response.clone();
        const rawResponseBody = await responseClone.text();

        if (!response.ok) {
          logger.error(
            `AkashChat API error: ${response.status} - ${response.statusText}`
          );
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
      _runtime,
      { prompt, modelType = ModelType.TEXT_LARGE }: TokenizeTextParams
    ) => {
      return await tokenizeText(modelType ?? ModelType.TEXT_LARGE, prompt);
    },
    [ModelType.TEXT_TOKENIZER_DECODE]: async (
      _runtime,
      { tokens, modelType = ModelType.TEXT_LARGE }: DetokenizeTextParams
    ) => {
      return await detokenizeText(modelType ?? ModelType.TEXT_LARGE, tokens);
    },
    [ModelType.TEXT_SMALL]: async (
      runtime: IAgentRuntime,
      { prompt, stopSequences = [] }: GenerateTextParams
    ) => {
      const temperature = 0.7;
      const frequencyPenalty = 0.7;
      const presencePenalty = 0.7;
      const max_response_length = 8192;

      const akashchat = createAkashChatClient(runtime);
      const modelName = getSmallModel(runtime);
      const experimentalTelemetry = getExperimentalTelemetry(runtime);

      logger.log(`[AkashChat] Using TEXT_SMALL model: ${modelName}`);
      logger.log(prompt);

      const { text: akashchatResponse, usage } = await generateText({
        model: akashchat.languageModel(modelName),
        prompt: prompt,
        system: runtime.character.system ?? undefined,
        temperature: temperature,
        maxTokens: max_response_length,
        frequencyPenalty: frequencyPenalty,
        presencePenalty: presencePenalty,
        stopSequences: stopSequences,
        experimental_telemetry: {
          isEnabled: experimentalTelemetry,
        },
      });

      if (usage) {
        emitModelUsageEvent(runtime, ModelType.TEXT_SMALL, prompt, usage);
      }

      return akashchatResponse;
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
      const akashchat = createAkashChatClient(runtime);
      const modelName = getLargeModel(runtime);
      const experimentalTelemetry = getExperimentalTelemetry(runtime);

      logger.log(`[AkashChat] Using TEXT_LARGE model: ${modelName}`);
      logger.log(prompt);

      const { text: akashchatResponse, usage } = await generateText({
        model: akashchat.languageModel(modelName),
        prompt: prompt,
        system: runtime.character.system ?? undefined,
        temperature: temperature,
        maxTokens: maxTokens,
        frequencyPenalty: frequencyPenalty,
        presencePenalty: presencePenalty,
        stopSequences: stopSequences,
        experimental_telemetry: {
          isEnabled: experimentalTelemetry,
        },
      });

      if (usage) {
        emitModelUsageEvent(runtime, ModelType.TEXT_LARGE, prompt, usage);
      }

      return akashchatResponse;
    },
    [ModelType.OBJECT_SMALL]: async (
      runtime: IAgentRuntime,
      params: ObjectGenerationParams
    ) => {
      return generateObjectByModelType(
        runtime,
        params,
        ModelType.OBJECT_SMALL,
        getSmallModel
      );
    },
    [ModelType.OBJECT_LARGE]: async (
      runtime: IAgentRuntime,
      params: ObjectGenerationParams
    ) => {
      return generateObjectByModelType(
        runtime,
        params,
        ModelType.OBJECT_LARGE,
        getLargeModel
      );
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
            logger.log(
              'Models Available:',
              (data as { data?: unknown[] })?.data?.length ?? 'N/A'
            );
            if (!response.ok) {
              throw new Error(
                `Failed to validate AkashChat API key: ${response.statusText}`
              );
            }
          },
        },
        {
          name: 'akashchat_test_text_embedding',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const embedding = await runtime.useModel(
                ModelType.TEXT_EMBEDDING,
                {
                  text: 'Hello, world!',
                }
              );
              logger.log('embedding', embedding);
            } catch (error: unknown) {
              const message =
                error instanceof Error ? error.message : String(error);
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
              const message =
                error instanceof Error ? error.message : String(error);
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
              const message =
                error instanceof Error ? error.message : String(error);
              logger.error(`Error in test_text_small: ${message}`);
              throw error;
            }
          },
        },
        {
          name: 'akashchat_test_image_generation',
          fn: async (runtime: IAgentRuntime) => {
            logger.log('akashchat_test_image_generation');
            try {
              await runtime.useModel(ModelType.IMAGE, {
                prompt: 'A beautiful sunset over a calm ocean',
                n: 1,
                size: '1024x1024',
              });
              throw new Error('Image generation should not be supported');
            } catch (error: unknown) {
              const message =
                error instanceof Error ? error.message : String(error);
              if (message.includes('not supported')) {
                logger.log('Image generation correctly not supported');
              } else {
                logger.error(`Unexpected error in image generation test: ${message}`);
                throw error;
              }
            }
          },
        },
        {
          name: 'image-description',
          fn: async (runtime: IAgentRuntime) => {
            try {
              logger.log('akashchat_test_image_description');
              try {
                const result = await runtime.useModel(
                  ModelType.IMAGE_DESCRIPTION,
                  'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Vitalik_Buterin_TechCrunch_London_2015_%28cropped%29.jpg/537px-Vitalik_Buterin_TechCrunch_London_2015_%28cropped%29.jpg'
                );

                if (
                  result &&
                  typeof result === 'object' &&
                  'title' in result &&
                  'description' in result
                ) {
                  logger.log('Image description:', result);
                } else {
                  logger.error(
                    'Invalid image description result format:',
                    result
                  );
                }
              } catch (e: unknown) {
                const message = e instanceof Error ? e.message : String(e);
                logger.error(`Error in image description test: ${message}`);
              }
            } catch (e: unknown) {
              const message = e instanceof Error ? e.message : String(e);
              logger.error(
                `Error in akashchat_test_image_description: ${message}`
              );
            }
          },
        },
        {
          name: 'akashchat_test_transcription',
          fn: async (runtime: IAgentRuntime) => {
            logger.log('akashchat_test_transcription');
            try {
              await runtime.useModel(
                ModelType.TRANSCRIPTION,
                Buffer.from('test')
              );
              throw new Error('Transcription should not be supported');
            } catch (error: unknown) {
              const message =
                error instanceof Error ? error.message : String(error);
              if (message.includes('not supported')) {
                logger.log('Transcription correctly not supported');
              } else {
                logger.error(`Unexpected error in transcription test: ${message}`);
                throw error;
              }
            }
          },
        },
        {
          name: 'akashchat_test_text_tokenizer_encode',
          fn: async (runtime: IAgentRuntime) => {
            const prompt = 'Hello tokenizer encode!';
            const tokens = await runtime.useModel(
              ModelType.TEXT_TOKENIZER_ENCODE,
              { prompt }
            );
            if (!Array.isArray(tokens) || tokens.length === 0) {
              throw new Error(
                'Failed to tokenize text: expected non-empty array of tokens'
              );
            }
            logger.log('Tokenized output:', tokens);
          },
        },
        {
          name: 'akashchat_test_text_tokenizer_decode',
          fn: async (runtime: IAgentRuntime) => {
            const prompt = 'Hello tokenizer decode!';
            const tokens = await runtime.useModel(
              ModelType.TEXT_TOKENIZER_ENCODE,
              { prompt }
            );
            const decodedText = await runtime.useModel(
              ModelType.TEXT_TOKENIZER_DECODE,
              { tokens }
            );
            if (decodedText !== prompt) {
              throw new Error(
                `Decoded text does not match original. Expected "${prompt}", got "${decodedText}"`
              );
            }
            logger.log('Decoded text:', decodedText);
          },
        },
        {
          name: 'akashchat_test_text_to_speech',
          fn: async (runtime: IAgentRuntime) => {
            try {
              const text = 'Hello, this is a test for text-to-speech.';
              throw new Error('Text-to-speech should not be supported');
            } catch (error: unknown) {
              const message =
                error instanceof Error ? error.message : String(error);
              if (message.includes('not supported')) {
                logger.log('Text-to-speech correctly not supported');
              } else {
                logger.error(`Unexpected error in text-to-speech test: ${message}`);
                throw error;
              }
            }
          },
        },
      ],
    },
  ],
};
export default akashchatPlugin;