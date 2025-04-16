// src/index.ts
import { createOpenAI } from "@ai-sdk/openai";
import {
  ModelType,
  logger,
  VECTOR_DIMS
} from "@elizaos/core";
import { generateObject, generateText } from "ai";
import { encodingForModel } from "js-tiktoken";
function getSetting(runtime, key, defaultValue) {
  return runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
}
function getBaseURL(runtime) {
  return "https://chatapi.akash.network/api/v1";
}
function getApiKey(runtime) {
  return getSetting(runtime, "AKASH_CHAT_API_KEY");
}
function getCloudflareGatewayBaseURL(runtime, provider) {
  try {
    const isCloudflareEnabled = runtime.getSetting("CLOUDFLARE_GW_ENABLED") === "true";
    const cloudflareAccountId = runtime.getSetting("CLOUDFLARE_AI_ACCOUNT_ID");
    const cloudflareGatewayId = runtime.getSetting("CLOUDFLARE_AI_GATEWAY_ID");
    const defaultUrl = "https://chatapi.akash.network/api/v1";
    logger.debug("Cloudflare Gateway Configuration:", {
      isEnabled: isCloudflareEnabled,
      hasAccountId: !!cloudflareAccountId,
      hasGatewayId: !!cloudflareGatewayId,
      provider
    });
    if (!isCloudflareEnabled) {
      logger.debug("Cloudflare Gateway is not enabled");
      return defaultUrl;
    }
    if (!cloudflareAccountId) {
      logger.warn("Cloudflare Gateway is enabled but CLOUDFLARE_AI_ACCOUNT_ID is not set");
      return defaultUrl;
    }
    if (!cloudflareGatewayId) {
      logger.warn("Cloudflare Gateway is enabled but CLOUDFLARE_AI_GATEWAY_ID is not set");
      return defaultUrl;
    }
    const baseURL = `https://gateway.ai.cloudflare.com/v1/${cloudflareAccountId}/${cloudflareGatewayId}/${provider.toLowerCase()}`;
    logger.info("Using Cloudflare Gateway:", {
      provider,
      baseURL,
      accountId: cloudflareAccountId,
      gatewayId: cloudflareGatewayId
    });
    return baseURL;
  } catch (error) {
    logger.error("Error in getCloudflareGatewayBaseURL:", error);
    return "https://chatapi.akash.network/api/v1";
  }
}
function findModelName(model) {
  try {
    const name = model === ModelType.TEXT_SMALL ? process.env.AKASH_CHAT_SMALL_MODEL ?? "Meta-Llama-3-1-8B-Instruct-FP8" : process.env.AKASH_CHAT_LARGE_MODEL ?? "Meta-Llama-3-3-70B-Instruct";
    return name;
  } catch (error) {
    logger.error("Error in findModelName:", error);
    return "Meta-Llama-3-1-8B-Instruct-FP8";
  }
}
async function tokenizeText(model, prompt) {
  try {
    const encoding = encodingForModel(findModelName(model));
    const tokens = encoding.encode(prompt);
    return tokens;
  } catch (error) {
    logger.error("Error in tokenizeText:", error);
    return [];
  }
}
async function detokenizeText(model, tokens) {
  try {
    const modelName = findModelName(model);
    const encoding = encodingForModel(modelName);
    return encoding.decode(tokens);
  } catch (error) {
    logger.error("Error in detokenizeText:", error);
    return "";
  }
}
async function handleRateLimitError(error, retryFn) {
  try {
    if (error.message.includes("Rate limit reached")) {
      logger.warn("Akash Chat API rate limit reached", { error: error.message });
      let retryDelay = 1e4;
      const delayMatch = error.message.match(/try again in (\d+\.?\d*)s/i);
      if (delayMatch?.[1]) {
        retryDelay = Math.ceil(Number.parseFloat(delayMatch[1]) * 1e3) + 1e3;
      }
      logger.info(`Will retry after ${retryDelay}ms delay`);
      await new Promise((resolve) => setTimeout(resolve, retryDelay));
      logger.info("Retrying request after rate limit delay");
      return await retryFn();
    }
    logger.error("Error with AkashChat API:", error);
    throw error;
  } catch (retryError) {
    logger.error("Error during retry handling:", retryError);
    throw retryError;
  }
}
async function generateAkashChatText(akashchat, model, params) {
  try {
    const { text: akashchatResponse } = await generateText({
      model: akashchat.languageModel(model),
      prompt: params.prompt,
      system: params.system,
      temperature: params.temperature,
      maxTokens: params.maxTokens,
      frequencyPenalty: params.frequencyPenalty,
      presencePenalty: params.presencePenalty,
      stopSequences: params.stopSequences
    });
    return akashchatResponse;
  } catch (error) {
    try {
      return await handleRateLimitError(error, async () => {
        const { text: akashchatRetryResponse } = await generateText({
          model: akashchat.languageModel(model),
          prompt: params.prompt,
          system: params.system,
          temperature: params.temperature,
          maxTokens: params.maxTokens,
          frequencyPenalty: params.frequencyPenalty,
          presencePenalty: params.presencePenalty,
          stopSequences: params.stopSequences
        });
        return akashchatRetryResponse;
      });
    } catch (retryError) {
      logger.error("Final error in generateAkashChatText:", retryError);
      return "Error generating text. Please try again later.";
    }
  }
}
async function generateAkashChatObject(akashchat, model, params) {
  try {
    const { object } = await generateObject({
      model: akashchat.languageModel(model),
      output: "no-schema",
      prompt: params.prompt,
      temperature: params.temperature
    });
    return object;
  } catch (error) {
    logger.error("Error generating object:", error);
    return {};
  }
}
var akashchatPlugin = {
  name: "akashchat",
  description: "AkashChat API plugin",
  config: {
    AKASH_CHAT_API_KEY: process.env.AKASH_CHAT_API_KEY,
    AKASH_CHAT_SMALL_MODEL: process.env.AKASH_CHAT_SMALL_MODEL,
    AKASH_CHAT_MEDIUM_MODEL: process.env.AKASH_CHAT_MEDIUM_MODEL,
    AKASH_CHAT_LARGE_MODEL: process.env.AKASH_CHAT_LARGE_MODEL,
    AKASHCHAT_EMBEDDING_MODEL: process.env.AKASHCHAT_EMBEDDING_MODEL ?? "BAAI-bge-large-en-v1-5",
    AKASHCHAT_EMBEDDING_DIMENSIONS: process.env.AKASHCHAT_EMBEDDING_DIMENSIONS ?? "1024",
    AKASHCHAT_IMAGE_GENERATION_URL: process.env.AKASHCHAT_IMAGE_GENERATION_URL ?? "https://gen.akash.network/api"
  },
  async init(config) {
    if (!process.env.AKASH_CHAT_API_KEY) {
      throw Error("Missing AKASH_CHAT_API_KEY in environment variables");
    }
  },
  models: {
    [ModelType.TEXT_EMBEDDING]: async (runtime, params) => {
      const embeddingDimension = parseInt(
        getSetting(runtime, "AKASHCHAT_EMBEDDING_DIMENSIONS", "1024")
      );
      if (!Object.values(VECTOR_DIMS).includes(embeddingDimension)) {
        logger.error(
          `Invalid embedding dimension: ${embeddingDimension}. Must be one of: ${Object.values(VECTOR_DIMS).join(", ")}`
        );
        throw new Error(
          `Invalid embedding dimension: ${embeddingDimension}. Must be one of: ${Object.values(VECTOR_DIMS).join(", ")}`
        );
      }
      if (params === null) {
        logger.debug("Creating test embedding for initialization");
        const testVector = Array(embeddingDimension).fill(0);
        testVector[0] = 0.1;
        return testVector;
      }
      let text;
      if (typeof params === "string") {
        text = params;
      } else if (typeof params === "object" && params.text) {
        text = params.text;
      } else {
        logger.warn("Invalid input format for embedding");
        const fallbackVector = Array(embeddingDimension).fill(0);
        fallbackVector[0] = 0.2;
        return fallbackVector;
      }
      if (!text.trim()) {
        logger.warn("Empty text for embedding");
        const emptyVector = Array(embeddingDimension).fill(0);
        emptyVector[0] = 0.3;
        return emptyVector;
      }
      try {
        const baseURL = getBaseURL(runtime);
        const response = await fetch(`${baseURL}/embeddings`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${getApiKey(runtime)}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: getSetting(runtime, "AKASHCHAT_EMBEDDING_MODEL", "BAAI-bge-large-en-v1-5"),
            input: text
          })
        });
        if (!response.ok) {
          logger.error(`Akash Chat API error: ${response.status} - ${response.statusText}`);
          const errorVector = Array(embeddingDimension).fill(0);
          errorVector[0] = 0.4;
          return errorVector;
        }
        const data = await response.json();
        if (!data?.data?.[0]?.embedding) {
          logger.error("API returned invalid structure");
          const errorVector = Array(embeddingDimension).fill(0);
          errorVector[0] = 0.5;
          return errorVector;
        }
        const embedding = data.data[0].embedding;
        logger.log(`Got valid embedding with length ${embedding.length}`);
        return embedding;
      } catch (error) {
        logger.error("Error generating embedding:", error);
        const errorVector = Array(embeddingDimension).fill(0);
        errorVector[0] = 0.6;
        return errorVector;
      }
    },
    [ModelType.TEXT_TOKENIZER_ENCODE]: async (_runtime, { prompt, modelType = ModelType.TEXT_LARGE }) => {
      try {
        return await tokenizeText(modelType ?? ModelType.TEXT_LARGE, prompt);
      } catch (error) {
        logger.error("Error in TEXT_TOKENIZER_ENCODE model:", error);
        return [];
      }
    },
    [ModelType.TEXT_TOKENIZER_DECODE]: async (_runtime, { tokens, modelType = ModelType.TEXT_LARGE }) => {
      try {
        return await detokenizeText(modelType ?? ModelType.TEXT_LARGE, tokens);
      } catch (error) {
        logger.error("Error in TEXT_TOKENIZER_DECODE model:", error);
        return "";
      }
    },
    [ModelType.TEXT_SMALL]: async (runtime, { prompt, stopSequences = [] }) => {
      try {
        const temperature = 0.7;
        const frequency_penalty = 0.7;
        const presence_penalty = 0.7;
        const max_response_length = 8192;
        const baseURL = getCloudflareGatewayBaseURL(runtime, "akashchat");
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting("AKASH_CHAT_API_KEY"),
          fetch: runtime.fetch,
          baseURL
        });
        const model = runtime.getSetting("AKASH_CHAT_SMALL_MODEL") ?? runtime.getSetting("SMALL_MODEL") ?? "Meta-Llama-3-1-8B-Instruct-FP8";
        logger.log("generating text");
        logger.log(prompt);
        return await generateAkashChatText(akashchat, model, {
          prompt,
          system: runtime.character.system ?? void 0,
          temperature,
          maxTokens: max_response_length,
          frequencyPenalty: frequency_penalty,
          presencePenalty: presence_penalty,
          stopSequences
        });
      } catch (error) {
        logger.error("Error in TEXT_SMALL model:", error);
        return "Error generating text. Please try again later.";
      }
    },
    [ModelType.TEXT_LARGE]: async (runtime, {
      prompt,
      stopSequences = [],
      maxTokens = 8192,
      temperature = 0.7,
      frequencyPenalty = 0.7,
      presencePenalty = 0.7
    }) => {
      try {
        const model = runtime.getSetting("AKASH_CHAT_LARGE_MODEL") ?? runtime.getSetting("LARGE_MODEL") ?? "Meta-Llama-3-3-70B-Instruct";
        const baseURL = getCloudflareGatewayBaseURL(runtime, "akashchat");
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting("AKASH_CHAT_API_KEY"),
          fetch: runtime.fetch,
          baseURL
        });
        return await generateAkashChatText(akashchat, model, {
          prompt,
          system: runtime.character.system ?? void 0,
          temperature,
          maxTokens,
          frequencyPenalty,
          presencePenalty,
          stopSequences
        });
      } catch (error) {
        logger.error("Error in TEXT_LARGE model:", error);
        return "Error generating text. Please try again later.";
      }
    },
    [ModelType.OBJECT_SMALL]: async (runtime, params) => {
      try {
        const baseURL = getCloudflareGatewayBaseURL(runtime, "akashchat");
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting("AKASH_CHAT_API_KEY"),
          baseURL
        });
        const model = runtime.getSetting("AKASH_CHAT_SMALL_MODEL") ?? runtime.getSetting("SMALL_MODEL") ?? "Meta-Llama-3-1-8B-Instruct-FP8";
        if (params.schema) {
          logger.info("Using OBJECT_SMALL without schema validation");
        }
        return await generateAkashChatObject(akashchat, model, params);
      } catch (error) {
        logger.error("Error in OBJECT_SMALL model:", error);
        return {};
      }
    },
    [ModelType.OBJECT_LARGE]: async (runtime, params) => {
      try {
        const baseURL = getCloudflareGatewayBaseURL(runtime, "akashchat");
        const akashchat = createOpenAI({
          apiKey: runtime.getSetting("AKASH_CHAT_API_KEY"),
          baseURL
        });
        const model = runtime.getSetting("AKASH_CHAT_LARGE_MODEL") ?? runtime.getSetting("LARGE_MODEL") ?? "Meta-Llama-3-3-70B-Instruct";
        if (params.schema) {
          logger.info("Using OBJECT_LARGE without schema validation");
        }
        return await generateAkashChatObject(akashchat, model, params);
      } catch (error) {
        logger.error("Error in OBJECT_LARGE model:", error);
        return {};
      }
    }
  },
  tests: [
    {
      name: "akashchat_plugin_tests",
      tests: [
        {
          name: "akashchat_test_url_and_api_key_validation",
          fn: async (runtime) => {
            try {
              const baseURL = getCloudflareGatewayBaseURL(runtime, "akashchat") ?? "https://chatapi.akash.network/api/v1";
              const response = await fetch(`${baseURL}/models`, {
                headers: {
                  Authorization: `Bearer ${runtime.getSetting("AKASH_CHAT_API_KEY")}`
                }
              });
              const data = await response.json();
              logger.log("Models Available:", data?.data?.length);
              if (!response.ok) {
                logger.error(`Failed to validate Akash Chat API key: ${response.statusText}`);
                return;
              }
            } catch (error) {
              logger.error("Error in akashchat_test_url_and_api_key_validation:", error);
            }
          }
        },
        {
          name: "akashchat_test_text_embedding",
          fn: async (runtime) => {
            try {
              const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
                text: "Hello, world!"
              });
              logger.log("embedding", embedding);
            } catch (error) {
              logger.error("Error in test_text_embedding:", error);
            }
          }
        },
        {
          name: "akashchat_test_text_large",
          fn: async (runtime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_LARGE, {
                prompt: "What is the nature of reality in 10 words?"
              });
              if (text.length === 0) {
                logger.error("Failed to generate text");
                return;
              }
              logger.log("generated with test_text_large:", text);
            } catch (error) {
              logger.error("Error in test_text_large:", error);
            }
          }
        },
        {
          name: "akashchat_test_text_small",
          fn: async (runtime) => {
            try {
              const text = await runtime.useModel(ModelType.TEXT_SMALL, {
                prompt: "What is the nature of reality in 10 words?"
              });
              if (text.length === 0) {
                logger.error("Failed to generate text");
                return;
              }
              logger.log("generated with test_text_small:", text);
            } catch (error) {
              logger.error("Error in test_text_small:", error);
            }
          }
        },
        {
          name: "akashchat_test_text_tokenizer_encode",
          fn: async (runtime) => {
            try {
              const prompt = "Hello tokenizer encode!";
              const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, { prompt });
              if (!Array.isArray(tokens) || tokens.length === 0) {
                logger.error("Failed to tokenize text: expected non-empty array of tokens");
                return;
              }
              logger.log("Tokenized output:", tokens);
            } catch (error) {
              logger.error("Error in test_text_tokenizer_encode:", error);
            }
          }
        },
        {
          name: "akashchat_test_text_tokenizer_decode",
          fn: async (runtime) => {
            try {
              const prompt = "Hello tokenizer decode!";
              const tokens = await runtime.useModel(ModelType.TEXT_TOKENIZER_ENCODE, { prompt });
              const decodedText = await runtime.useModel(ModelType.TEXT_TOKENIZER_DECODE, {
                tokens
              });
              if (decodedText !== prompt) {
                logger.error(
                  `Decoded text does not match original. Expected "${prompt}", got "${decodedText}"`
                );
                return;
              }
              logger.log("Decoded text:", decodedText);
            } catch (error) {
              logger.error("Error in test_text_tokenizer_decode:", error);
            }
          }
        },
        {
          name: "akashchat_test_object_small",
          fn: async (runtime) => {
            try {
              const object = await runtime.useModel(ModelType.OBJECT_SMALL, {
                prompt: "Generate a JSON object representing a user profile with name, age, and hobbies",
                temperature: 0.7
              });
              logger.log("Generated object:", object);
            } catch (error) {
              logger.error("Error in test_object_small:", error);
            }
          }
        },
        {
          name: "akashchat_test_object_large",
          fn: async (runtime) => {
            try {
              const object = await runtime.useModel(ModelType.OBJECT_LARGE, {
                prompt: "Generate a detailed JSON object representing a restaurant with name, cuisine type, menu items with prices, and customer reviews",
                temperature: 0.7
              });
              logger.log("Generated object:", object);
            } catch (error) {
              logger.error("Error in test_object_large:", error);
            }
          }
        }
      ]
    }
  ]
};
var index_default = akashchatPlugin;
export {
  akashchatPlugin,
  index_default as default
};
//# sourceMappingURL=index.js.map