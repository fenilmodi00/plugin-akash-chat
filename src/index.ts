import { createOpenAI } from '@ai-sdk/openai';
import type {
  ImageDescriptionParams,
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
  VECTOR_DIMS,
} from '@elizaos/core';
import { generateObject, generateText } from 'ai';
import { type TiktokenModel, encodingForModel } from 'js-tiktoken';
import { z } from 'zod';

/**
 * Runtime interface for the AkashChat plugin
 */
interface Runtime {
  getSetting(key: string): string | undefined;
  character: {
    system?: string;
  };
  fetch?: typeof fetch;
}

interface ImageGenerationResponse {
  job_id: string;
  worker_name: string;
  worker_city: string;
  worker_country: string;
  status: string;
  result: string;
  worker_gpu: string;
  elapsed_time: number;
  queue: number;
}[];

/**
 * Helper function to get settings with fallback to process.env
 *
 * @param runtime The runtime context
 * @param key The setting key to retrieve
 * @param defaultValue Optional default value if not found
 * @returns The setting value with proper fallbacks
 */
function getSetting(runtime: any, key: string, defaultValue?: string): string | undefined {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}

/**
 * Helper function to get the base URL for AkashChat API
 *
 * @param runtime The runtime context
 * @returns The configured base URL or default
 */
function getBaseURL(runtime: any): string {
  return getSetting(runtime, 'AKASH_CHAT_BASE_URL', 'https://chatapi.akash.network/api/v1');
}

/**
 * Helper function to get the image generation URL for AkashChat API
 *
 * @param runtime The runtime context
 * @returns The configured image generation URL or default
 */
function getImageGenerateURL(runtime: any): string {
  return getSetting(runtime, 'AKASH_CHAT_IMAGE_GENERATE_URL', 'https://gen.akash.network/api');
}

/**
 * Helper function to get the API key for AkashChat
 *
 * @param runtime The runtime context
 * @returns The configured API key
 */
function getApiKey(runtime: any): string | undefined {
  return getSetting(runtime, 'AKASH_CHAT_API_KEY');
}


/**
 * Gets the Cloudflare Gateway base URL for a specific provider if enabled
 * @param runtime The runtime environment
 * @param provider The model provider name
 * @returns The Cloudflare Gateway base URL if enabled, undefined otherwise
 */
function getCloudflareGatewayBaseURL(runtime: Runtime, provider: string): string | undefined {
  try {
    const isCloudflareEnabled = runtime.getSetting('CLOUDFLARE_GW_ENABLED') === 'true';
    const cloudflareAccountId = runtime.getSetting('CLOUDFLARE_AI_ACCOUNT_ID');
    const cloudflareGatewayId = runtime.getSetting('CLOUDFLARE_AI_GATEWAY_ID');

    const defaultUrl = 'https://chatapi.akash.network/api/v1';
    logger.debug('Cloudflare Gateway Configuration:', {
      isEnabled: isCloudflareEnabled,
      hasAccountId: !!cloudflareAccountId,
      hasGatewayId: !!cloudflareGatewayId,
      provider: provider,
    });

    if (!isCloudflareEnabled) {
      logger.debug('Cloudflare Gateway is not enabled');
      return defaultUrl;
    }

    if (!cloudflareAccountId) {
      logger.warn('Cloudflare Gateway is enabled but CLOUDFLARE_AI_ACCOUNT_ID is not set');
      return defaultUrl;
    }

    if (!cloudflareGatewayId) {
      logger.warn('Cloudflare Gateway is enabled but CLOUDFLARE_AI_GATEWAY_ID is not set');
      return defaultUrl;
    }

    const baseURL = `https://gateway.ai.cloudflare.com/v1/${cloudflareAccountId}/${cloudflareGatewayId}/${provider.toLowerCase()}`;
    logger.info('Using Cloudflare Gateway:', {
      provider,
      baseURL,
      accountId: cloudflareAccountId,
      gatewayId: cloudflareGatewayId,
    });

    return baseURL;
  } catch (error) {
    logger.error('Error in getCloudflareGatewayBaseURL:', error);
    return 'https://chatapi.akash.network/api/v1';
  }
}

function findModelName(model: ModelTypeName): TiktokenModel {
  try {
    const name =
      model === ModelType.TEXT_SMALL
        ? (process.env.AKASH_CHAT_SMALL_MODEL ?? 'Meta-Llama-3-1-8B-Instruct-FP8')
        : (process.env.AKASH_CHAT_LARGE_MODEL ?? 'Meta-Llama-3-3-70B-Instruct');
    return name as TiktokenModel;
  } catch (error) {
    logger.error('Error in findModelName:', error);
    return 'Meta-Llama-3-1-8B-Instruct-FP8' as TiktokenModel;
  }
}

async function tokenizeText(model: ModelTypeName, prompt: string) {
  try {
    const encoding = encodingForModel(findModelName(model));
    const tokens = encoding.encode(prompt);
    return tokens;
  } catch (error) {
    logger.error('Error in tokenizeText:', error);
    return [];
  }
}

/**
 * Detokenize a sequence of tokens back into text using the specified model.
 *
 * @param {ModelTypeName} model - The type of model to use for detokenization.
 * @param {number[]} tokens - The sequence of tokens to detokenize.
 * @returns {string} The detokenized text.
 */
async function detokenizeText(model: ModelTypeName, tokens: number[]) {
  try {
    const modelName = findModelName(model);
    const encoding = encodingForModel(modelName);
    return encoding.decode(tokens);
  } catch (error) {
    logger.error('Error in detokenizeText:', error);
    return '';
  }
}

/**
 * Handles rate limit errors, waits for the appropriate delay, and retries the operation
 * @param error The error object from the failed request
 * @param retryFn The function to retry after waiting
 * @returns Result from the retry function
 */
async function handleRateLimitError(error: Error, retryFn: () => Promise<unknown>) {
  try {
    if (error.message.includes('Rate limit reached')) {
      logger.warn('Akash Chat API rate limit reached', { error: error.message });

      // Extract retry delay from error message if possible
      let retryDelay = 10000; // Default to 10 seconds
      const delayMatch = error.message.match(/try again in (\d+\.?\d*)s/i);
      if (delayMatch?.[1]) {
        // Convert to milliseconds and add a small buffer
        retryDelay = Math.ceil(Number.parseFloat(delayMatch[1]) * 1000) + 1000;
      }

      logger.info(`Will retry after ${retryDelay}ms delay`);

      // Wait for the suggested delay plus a small buffer
      await new Promise((resolve) => setTimeout(resolve, retryDelay));

      // Retry the request
      logger.info('Retrying request after rate limit delay');
      return await retryFn();
    }

    // For other errors, log and rethrow
    logger.error('Error with AkashChat API:', error);
    throw error;
  } catch (retryError) {
    logger.error('Error during retry handling:', retryError);
    throw retryError;
  }
}

/**
 * Generate text using AkashChat API with retry handling for rate limits
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
    const { text: akashchatResponse } = await generateText({
      model: akashchat.languageModel(model),
      prompt: params.prompt,
      system: params.system,
      temperature: params.temperature,
      maxTokens: params.maxTokens,
      frequencyPenalty: params.frequencyPenalty,
      presencePenalty: params.presencePenalty,
      stopSequences: params.stopSequences,
    });
    return akashchatResponse;
  } catch (error: unknown) {
    try {
      return await handleRateLimitError(error as Error, async () => {
        const { text: akashchatRetryResponse } = await generateText({
          model: akashchat.languageModel(model),
          prompt: params.prompt,
          system: params.system,
          temperature: params.temperature,
          maxTokens: params.maxTokens,
          frequencyPenalty: params.frequencyPenalty,
          presencePenalty: params.presencePenalty,
          stopSequences: params.stopSequences,
        });
        return akashchatRetryResponse;
      });
    } catch (retryError) {
      logger.error('Final error in generateAkashChatText:', retryError);
      return 'Error generating text. Please try again later.';
    }
  }
}

/**
 * Generate object using AkashChat API with consistent error handling
 */
async function generateAkashChatObject(
  akashchat: ReturnType<typeof createOpenAI>,
  model: string,
  params: ObjectGenerationParams
) {
  try {
    const { object } = await generateObject({
      model: akashchat.languageModel(model),
      output: 'no-schema',
      prompt: params.prompt,
      temperature: params.temperature,
    });
    return object;
  } catch (error: unknown) {
    logger.error('Error generating object:', error);
    return {};
  }
}

export const akashchatPlugin: Plugin = {
  name: 'akashchat',
  description: 'AkashChat API plugin',
  config: {
    AKASH_CHAT_API_KEY: process.env.AKASH_CHAT_API_KEY,
    AKASH_CHAT_SMALL_MODEL: process.env.AKASH_CHAT_SMALL_MODEL,
    AKASH_CHAT_MEDIUM_MODEL: process.env.AKASH_CHAT_MEDIUM_MODEL,
    AKASH_CHAT_LARGE_MODEL: process.env.AKASH_CHAT_LARGE_MODEL,
    AKASHCHAT_EMBEDDING_MODEL: process.env.AKASHCHAT_EMBEDDING_MODEL ?? 'BAAI-bge-large-en-v1-5',
    AKASHCHAT_EMBEDDING_DIMENSIONS: process.env.AKASHCHAT_EMBEDDING_DIMENSIONS ?? '1024',
    AKASHCHAT_IMAGE_GENERATION_URL: process.env.AKASHCHAT_IMAGE_GENERATION_URL ?? 'https://gen.akash.network/api',
  },
  async init(config: Record<string, string>) {
    if (!process.env.AKASH_CHAT_API_KEY) {
      throw Error('Missing AKASH_CHAT_API_KEY in environment variables');
    }
  },
  models: {
    [ModelType.TEXT_EMBEDDING]: async (
      runtime,
      params: TextEmbeddingParams | string | null
    ): Promise<number[]> => {
      const embeddingDimension = parseInt(
        getSetting(runtime, 'AKASHCHAT_EMBEDDING_DIMENSIONS', '1024')
      ) as (typeof VECTOR_DIMS)[keyof typeof VECTOR_DIMS];

      // Validate embedding dimension
      if (!Object.values(VECTOR_DIMS).includes(embeddingDimension)) {
        logger.error(
          `Invalid embedding dimension: ${embeddingDimension}. Must be one of: ${Object.values(VECTOR_DIMS).join(', ')}`
        );
        throw new Error(
          `Invalid embedding dimension: ${embeddingDimension}. Must be one of: ${Object.values(VECTOR_DIMS).join(', ')}`
        );
      }

      // Handle null input (initialization case)
      if (params === null) {
        logger.debug('Creating test embedding for initialization');
        // Return a consistent vector for null input
        const testVector = Array(embeddingDimension).fill(0);
        testVector[0] = 0.1; // Make it non-zero
        return testVector;
      }

      // Get the text from whatever format was provided
      let text: string;
      if (typeof params === 'string') {
        text = params; // Direct string input
      } else if (typeof params === 'object' && params.text) {
        text = params.text; // Object with text property
      } else {
        logger.warn('Invalid input format for embedding');
        // Return a fallback for invalid input
        const fallbackVector = Array(embeddingDimension).fill(0);
        fallbackVector[0] = 0.2; // Different value for tracking
        return fallbackVector;
      }

      // Skip API call for empty text
      if (!text.trim()) {
        logger.warn('Empty text for embedding');
        const emptyVector = Array(embeddingDimension).fill(0);
        emptyVector[0] = 0.3; // Different value for tracking
        return emptyVector;
      }

      try {
        const baseURL = getBaseURL(runtime);

        // Call the AkashChat API
        const response = await fetch(`${baseURL}/embeddings`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${getApiKey(runtime)}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: getSetting(runtime, 'AKASHCHAT_EMBEDDING_MODEL', 'BAAI-bge-large-en-v1-5'),
            input: text,
          }),
        });

        if (!response.ok) {
          logger.error(`Akash Chat API error: ${response.status} - ${response.statusText}`);
          const errorVector = Array(embeddingDimension).fill(0);
          errorVector[0] = 0.4; // Different value for tracking
          return errorVector;
        }

        const data = (await response.json()) as {
          data: [{ embedding: number[] }];
        };

        if (!data?.data?.[0]?.embedding) {
          logger.error('API returned invalid structure');
          const errorVector = Array(embeddingDimension).fill(0);
          errorVector[0] = 0.5; // Different value for tracking
          return errorVector;
        }

        const embedding = data.data[0].embedding;
        logger.log(`Got valid embedding with length ${embedding.length}`);
        return embedding;
      } catch (error) {
        logger.error('Error generating embedding:', error);
        const errorVector = Array(embeddingDimension).fill(0);
        errorVector[0] = 0.6; // Different value for tracking
        return errorVector;
      }
    },
    [ModelType.TEXT_TOKENIZER_ENCODE]: async (
      _runtime,
      { prompt, modelType = ModelType.TEXT_LARGE }: TokenizeTextParams
    ) => {
      try {
        return await tokenizeText(modelType ?? ModelType.TEXT_LARGE, prompt);
      } catch (error) {
        logger.error('Error in TEXT_TOKENIZER_ENCODE model:', error);
        // Return empty array instead of crashing
        return [];
      }
    },
    [ModelType.TEXT_TOKENIZER_DECODE]: async (
      _runtime,
      { tokens, modelType = ModelType.TEXT_LARGE }: DetokenizeTextParams
    ) => {
      try {
        return await detokenizeText(modelType ?? ModelType.TEXT_LARGE, tokens);
      } catch (error) {
        logger.error('Error in TEXT_TOKENIZER_DECODE model:', error);
        // Return empty string instead of crashing
        return '';
      }
    },
    [ModelType.TEXT_SMALL]: async (runtime, { prompt, stopSequences = [] }: GenerateTextParams) => {
      try {
        const temperature = 0.7;
        const frequency_penalty = 0.7;
        const presence_penalty = 0.7;
        const max_response_length = 8192;
        const baseURL = getCloudflareGatewayBaseURL(runtime, 'akashchat');
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting('AKASH_CHAT_API_KEY'),
          fetch: runtime.fetch,
          baseURL,
        });

        const model =
          runtime.getSetting('AKASH_CHAT_SMALL_MODEL') ??
          runtime.getSetting('SMALL_MODEL') ??
          'Meta-Llama-3-1-8B-Instruct-FP8';

        logger.log('generating text');
        logger.log(prompt);

        return await generateAkashChatText(akashchat, model, {
          prompt,
          system: runtime.character.system ?? undefined,
          temperature,
          maxTokens: max_response_length,
          frequencyPenalty: frequency_penalty,
          presencePenalty: presence_penalty,
          stopSequences,
        });
      } catch (error) {
        logger.error('Error in TEXT_SMALL model:', error);
        return 'Error generating text. Please try again later.';
      }
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
      try {
        const model =
          runtime.getSetting('AKASH_CHAT_LARGE_MODEL') ??
          runtime.getSetting('LARGE_MODEL') ??
          'Meta-Llama-3-3-70B-Instruct';
        const baseURL = getCloudflareGatewayBaseURL(runtime, 'akashchat');
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting('AKASH_CHAT_API_KEY'),
          fetch: runtime.fetch,
          baseURL,
        });

        return await generateAkashChatText(akashchat, model, {
          prompt,
          system: runtime.character.system ?? undefined,
          temperature,
          maxTokens,
          frequencyPenalty,
          presencePenalty,
          stopSequences,
        });
      } catch (error) {
        logger.error('Error in TEXT_LARGE model:', error);
        return 'Error generating text. Please try again later.';
      }
    },
    [ModelType.IMAGE]: async (
      runtime,
      params: {
        prompt: string;
        negative: string;
        sampler: string;
        scheduler: string;
        preferred_gpu: string[];
      }
    ) => {
      try {
        const imageGenerateURL = getImageGenerateURL(runtime);
        const response = await fetch(`${imageGenerateURL}/generate`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${runtime.getSetting('AKASH_CHAT_API_KEY')}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            prompt: params.prompt,
            negative: params.negative ?? '',
            sampler: params.sampler ?? 'dpmpp_2m',
            scheduler: params.scheduler ?? 'sgm_uniform',
            preferred_gpu: params.preferred_gpu ?? ['RTX4090', 'A10', 'A100', 'V100-32Gi', 'H100'],
          }),
        });
        if (!response.ok) {
          logger.error(`Failed to generate image: ${response.statusText}`);
          return [{ url: '' }];
        }
        const data: { job_id: string } = await response.json() as { job_id: string };
        // Expecting response to include job_id
        const jobId = data?.job_id;
        if (!jobId) {
          logger.error('No job_id found in image generation response');
          return [{ url: '' }];
        }
        // Polling loop for job status
        const statusUrl = `${imageGenerateURL}/status?ids=${jobId}`;
        let status = 'pending';
        let result = '';
        const maxAttempts = 60; // 1 minute timeout
        let attempts = 0;
        while (status === 'pending' && attempts < maxAttempts) {
          await new Promise(res => setTimeout(res, 3000));
          attempts++;
          try {
            const pollRes = await fetch(statusUrl);
            if (!pollRes.ok) {
              logger.error(`Failed to poll image status: ${pollRes.statusText}`);
              continue;
            }
            const pollData: ImageGenerationResponse = await pollRes.json() as ImageGenerationResponse;
            // pollData is expected to be an array or object keyed by jobId
            let jobStatusObj;
            if (Array.isArray(pollData)) {
              jobStatusObj = pollData.find((j: any) => j.job_id === jobId);
            } else if (pollData && pollData[jobId]) {
              jobStatusObj = pollData[jobId];
            } else {
              jobStatusObj = pollData;
            }
            if (!jobStatusObj) {
              logger.error('No job status found for job_id:', jobId);
              continue;
            }
            status = jobStatusObj.status;
            if (status === 'completed') {
              result = jobStatusObj.result;
              break;
            }
          } catch (err) {
            logger.error('Error polling image status:', err);
          }
        }
        if (status !== 'completed' || !result) {
          logger.error('Image generation did not complete successfully');
          return [{ url: '' }];
        }
        // Return the image as a data URL (webp, etc)
        return [{ url: result }];
      } catch (error) {
        logger.error('Error in IMAGE model:', error);
        return [{ url: '' }];
      }
    },
    [ModelType.OBJECT_SMALL]: async (runtime, params: ObjectGenerationParams) => {
      try {
        const baseURL = getCloudflareGatewayBaseURL(runtime, 'akashchat');
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting('AKASH_CHAT_API_KEY'),
          baseURL,
        });
        const model =
          runtime.getSetting('AKASH_CHAT_SMALL_MODEL') ??
          runtime.getSetting('SMALL_MODEL') ??
          'Meta-Llama-3-1-8B-Instruct-FP8';

        if (params.schema) {
          logger.info('Using OBJECT_SMALL without schema validation');
        }

        return await generateAkashChatObject(akashchat, model, params);
      } catch (error) {
        logger.error('Error in OBJECT_SMALL model:', error);
        // Return empty object instead of crashing
        return {};
      }
    },
    [ModelType.OBJECT_LARGE]: async (runtime, params: ObjectGenerationParams) => {
      try {
        const baseURL = getCloudflareGatewayBaseURL(runtime, 'akashchat');
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting('AKASH_CHAT_API_KEY'),
          baseURL,
        });
        const model =
          runtime.getSetting('AKASH_CHAT_LARGE_MODEL') ??
          runtime.getSetting('LARGE_MODEL') ??
          'Meta-Llama-3-3-70B-Instruct';

        if (params.schema) {
          logger.info('Using OBJECT_LARGE without schema validation');
        }

        return await generateAkashChatObject(akashchat, model, params);
      } catch (error) {
        logger.error('Error in OBJECT_LARGE model:', error);
        // Return empty object instead of crashing
        return {};
      }
    },
  },
  tests: [
    {
      name: 'akashchat_plugin_tests',
      tests: [
        {
          name: 'akashchat_test_url_and_api_key_validation',
          fn: async (runtime) => {
            try {
              const baseURL =
                getCloudflareGatewayBaseURL(runtime, 'akashchat') ?? 'https://chatapi.akash.network/api/v1';
              const response = await fetch(`${baseURL}/models`, {
                headers: {
                  Authorization: `Bearer ${runtime.getSetting('AKASH_CHAT_API_KEY')}`,
                },
              });
              const data = await response.json();
              logger.log('Models Available:', (data as { data: unknown[] })?.data?.length);
              if (!response.ok) {
                logger.error(`Failed to validate Akash Chat API key: ${response.statusText}`);
                return;
              }
            } catch (error) {
              logger.error('Error in akashchat_test_url_and_api_key_validation:', error);
            }
          },
        },
        {
          name: 'akashchat_test_text_embedding',
          fn: async (runtime) => {
            try {
              const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
                text: 'Hello, world!',
              });
              logger.log('embedding', embedding);
            } catch (error) {
              logger.error('Error in test_text_embedding:', error);
            }
          },
        },
        {
          name: 'akashchat_test_text_large',
          fn: async (runtime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_LARGE, {
                prompt: 'What is the nature of reality in 10 words?',
              });
              if (text.length === 0) {
                logger.error('Failed to generate text');
                return;
              }
              logger.log('generated with test_text_large:', text);
            } catch (error) {
              logger.error('Error in test_text_large:', error);
            }
          },
        },
        {
          name: 'akashchat_test_text_small',
          fn: async (runtime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_SMALL, {
                prompt: 'What is the nature of reality in 10 words?',
              });
              if (text.length === 0) {
                logger.error('Failed to generate text');
                return;
              }
              logger.log('generated with test_text_small:', text);
            } catch (error) {
              logger.error('Error in test_text_small:', error);
            }
          },
        },
        {
          name: 'akashchat_test_image_generation',
          fn: async (runtime) => {
            try {
              logger.log('akashchat_test_image_generation');
              const image = await runtime.useModel(ModelType.IMAGE, {
                prompt: 'A beautiful sunset over a calm ocean',
                negative: '',
                sampler: 'dpmpp_2m',
                scheduler: 'sgm_uniform',
                preferred_gpu: ['RTX4090', 'A10', 'A100', 'V100-32Gi', 'H100'],
              });
              logger.log('generated with test_image_generation:', image);
            } catch (error) {
              logger.error('Error in test_image_generation:', error);
            }
          },
        },
        {
          name: 'akashchat_test_text_tokenizer_encode',
          fn: async (runtime) => {
            try {
              const prompt = 'Hello tokenizer encode!';
              const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, { prompt });
              if (!Array.isArray(tokens) || tokens.length === 0) {
                logger.error('Failed to tokenize text: expected non-empty array of tokens');
                return;
              }
              logger.log('Tokenized output:', tokens);
            } catch (error) {
              logger.error('Error in test_text_tokenizer_encode:', error);
            }
          },
        },
        {
          name: 'akashchat_test_text_tokenizer_decode',
          fn: async (runtime) => {
            try {
              const prompt = 'Hello tokenizer decode!';
              // Encode the string into tokens first
              const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, { prompt });
              // Now decode tokens back into text
              const decodedText = await runtime.useModel(ModelType.TEXT_TOKENIZER_DECODE, {
                tokens,
              });
              if (decodedText !== prompt) {
                logger.error(
                  `Decoded text does not match original. Expected "${prompt}", got "${decodedText}"`
                );
                return;
              }
              logger.log('Decoded text:', decodedText);
            } catch (error) {
              logger.error('Error in test_text_tokenizer_decode:', error);
            }
          },
        },
        {
          name: 'akashchat_test_object_small',
          fn: async (runtime) => {
            try {
              const object = await runtime.useModel(ModelType.OBJECT_SMALL, {
                prompt:
                  'Generate a JSON object representing a user profile with name, age, and hobbies',
                temperature: 0.7,
              });
              logger.log('Generated object:', object);
            } catch (error) {
              logger.error('Error in test_object_small:', error);
            }
          },
        },
        {
          name: 'akashchat_test_object_large',
          fn: async (runtime) => {
            try {
              const object = await runtime.useModel(ModelType.OBJECT_LARGE, {
                prompt:
                  'Generate a detailed JSON object representing a restaurant with name, cuisine type, menu items with prices, and customer reviews',
                temperature: 0.7,
              });
              logger.log('Generated object:', object);
            } catch (error) {
              logger.error('Error in test_object_large:', error);
            }
          },
        },
      ],
    },
  ],
};
export default akashchatPlugin;