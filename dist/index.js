// src/index.ts
import {
  ModelType,
  logger
} from "@elizaos/core";
import fetch from "node-fetch";
var AKASH_CHAT_BASE_URL = "https://chatapi.akash.network/api/v1";
var AkashChatError = class extends Error {
  constructor(message, status, code = "API_ERROR", details) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
    this.name = "AkashChatError";
  }
  /**
   * Determines if an error should be retried
   * @param code The error code to check
   * @returns true if the error is retryable
   */
  static isRetryable(code) {
    return ["NETWORK_ERROR", "SERVER_ERROR", "RATE_LIMIT_ERROR"].includes(code);
  }
};
function getSetting(runtime, key, defaultValue) {
  const value = runtime.getSetting(key) ?? process.env[key] ?? defaultValue;
  if (value === void 0) {
    throw new AkashChatError(
      `Configuration missing: ${key}`,
      void 0,
      "CONFIG_ERROR",
      { key, defaultValue }
    );
  }
  return value;
}
function getBaseURL(_runtime) {
  return AKASH_CHAT_BASE_URL;
}
function getApiKey(runtime) {
  try {
    const key = getSetting(runtime, "AKASH_CHAT_API_KEY");
    if (!key) {
      throw new AkashChatError(
        "API key is required. Please set AKASH_CHAT_API_KEY environment variable",
        void 0,
        "CONFIG_ERROR"
      );
    }
    return key;
  } catch (error) {
    if (error instanceof AkashChatError) throw error;
    throw new AkashChatError(
      "Failed to get API key",
      void 0,
      "CONFIG_ERROR",
      { error: error.message }
    );
  }
}
function getModel(runtime, type) {
  try {
    const envKey = `AKASH_CHAT_${type}_MODEL`;
    const model = getSetting(runtime, envKey);
    if (!model || typeof model !== "string") {
      throw new AkashChatError(
        `Invalid model configuration for ${type}`,
        void 0,
        "VALIDATION_ERROR",
        { type, model }
      );
    }
    return model;
  } catch (error) {
    if (error instanceof AkashChatError) throw error;
    throw new AkashChatError(
      `Failed to get model for ${type}`,
      void 0,
      "CONFIG_ERROR",
      { error: error.message, type }
    );
  }
}
async function fetchWithRetry(url, options, maxRetries = 3, retryDelay = 1e3) {
  let lastError = null;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      if (response.ok) {
        return response;
      }
      const errorMessage = await response.text().catch(() => response.statusText);
      const errorDetails = { status: response.status, attempt, url };
      if (response.status === 401) {
        throw new AkashChatError(
          "Authentication failed",
          response.status,
          "AUTH_ERROR",
          errorDetails
        );
      }
      if (response.status === 429) {
        throw new AkashChatError(
          "Rate limit exceeded",
          response.status,
          "RATE_LIMIT_ERROR",
          errorDetails
        );
      }
      if (response.status >= 500) {
        throw new AkashChatError(
          `Server error: ${errorMessage}`,
          response.status,
          "SERVER_ERROR",
          errorDetails
        );
      }
      throw new AkashChatError(
        `Request failed: ${errorMessage}`,
        response.status,
        "API_ERROR",
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
          "Network error occurred",
          void 0,
          "NETWORK_ERROR",
          { error: error.message, attempt }
        );
      }
      if (attempt < maxRetries - 1) {
        const delay = retryDelay * Math.pow(2, attempt);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }
  throw lastError || new AkashChatError("Max retries exceeded", void 0, "NETWORK_ERROR");
}
var akashChatPlugin = {
  name: "akash-chat",
  description: "A plugin for accessing the Akash Chat API.",
  config: {
    AKASH_CHAT_API_KEY: process.env.AKASH_CHAT_API_KEY,
    AKASH_CHAT_BASE_URL: process.env.AKASH_CHAT_BASE_URL,
    AKASH_CHAT_SMALL_MODEL: process.env.AKASH_CHAT_SMALL_MODEL,
    AKASH_CHAT_LARGE_MODEL: process.env.AKASH_CHAT_LARGE_MODEL,
    AKASH_CHAT_EMBEDDING_MODEL: process.env.AKASH_CHAT_EMBEDDING_MODEL
  },
  async init(_config, runtime) {
    try {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const response = await fetchWithRetry(`${baseURL}/models`, {
        headers: { Authorization: `Bearer ${apiKey}` }
      });
      if (!response.ok) {
        throw new AkashChatError(
          `API key validation failed: ${response.statusText}`,
          response.status
        );
      }
      logger.info("Akash Chat plugin initialized successfully");
    } catch (error) {
      logger.error("Failed to initialize Akash Chat plugin:", error);
      throw error;
    }
  },
  models: {
    [ModelType.TEXT_SMALL]: async (runtime, { prompt, stopSequences = [] }) => {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, "SMALL");
      const response = await fetchWithRetry(`${baseURL}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.7,
          max_tokens: 8192,
          frequency_penalty: 0.7,
          presence_penalty: 0.7,
          stop: stopSequences
        })
      });
      const data = await response.json();
      if (!data.choices?.[0]?.message?.content) {
        throw new AkashChatError("Invalid response format from Akash Chat API");
      }
      return data.choices[0].message.content;
    },
    [ModelType.TEXT_LARGE]: async (runtime, {
      prompt,
      stopSequences = [],
      maxTokens = 8192,
      temperature = 0.7,
      frequencyPenalty = 0.7,
      presencePenalty = 0.7
    }) => {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, "LARGE");
      const response = await fetchWithRetry(`${baseURL}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature,
          max_tokens: maxTokens,
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
          stop: stopSequences
        })
      });
      const data = await response.json();
      if (!data.choices?.[0]?.message?.content) {
        throw new AkashChatError("Invalid response format from Akash Chat API");
      }
      return data.choices[0].message.content;
    },
    [ModelType.TEXT_EMBEDDING]: async (runtime, params) => {
      if (!params?.text) {
        return new Array(1024).fill(0);
      }
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, "EMBEDDING");
      const response = await fetchWithRetry(`${baseURL}/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model,
          input: params.text
        })
      });
      const data = await response.json();
      if (!data.data?.[0]?.embedding) {
        throw new AkashChatError("Invalid response format from Akash Chat API");
      }
      return data.data[0].embedding;
    },
    [ModelType.OBJECT_SMALL]: async (runtime, params) => {
      const baseURL = getBaseURL(runtime);
      const apiKey = getApiKey(runtime);
      const model = getModel(runtime, "SMALL");
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
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.3,
          // Lower temperature for more consistent output
          max_tokens: 8192,
          frequency_penalty: 0,
          presence_penalty: 0
        })
      });
      const data = await response.json();
      if (!data.choices?.[0]?.message?.content) {
        throw new AkashChatError("Invalid response format from Akash Chat API");
      }
      let content = data.choices[0].message.content.trim();
      try {
        content = content.replace(/^```json\n?/, "").replace(/\n?```$/, "");
        const jsonMatch = content.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
        if (jsonMatch) {
          content = jsonMatch[0];
        }
        const parsedObject = JSON.parse(content);
        if (typeof parsedObject !== "object" || parsedObject === null) {
          throw new Error("Generated content is not a valid object");
        }
        return parsedObject;
      } catch (error) {
        throw new AkashChatError(
          "Failed to parse generated object",
          void 0,
          "VALIDATION_ERROR",
          {
            error: error.message,
            content,
            schema: params.schema
          }
        );
      }
    }
  },
  tests: [
    {
      name: "akash_chat_plugin_tests",
      tests: [
        {
          name: "akash_chat_test_url_and_api_key_validation",
          fn: async (runtime) => {
            const baseURL = getBaseURL(runtime);
            const response = await fetchWithRetry(`${baseURL}/models`, {
              headers: { Authorization: `Bearer ${getApiKey(runtime)}` }
            });
            const data = await response.json();
            logger.info("Models Available:", data?.data.length);
            if (!response.ok) {
              throw new AkashChatError(
                `Failed to validate Akash Chat API key: ${response.statusText}`,
                response.status
              );
            }
          }
        },
        {
          name: "akash_chat_test_text_large",
          fn: async (runtime) => {
            const text = await runtime.useModel(ModelType.TEXT_LARGE, {
              prompt: "What is the nature of reality in 10 words?"
            });
            if (text.length === 0) {
              throw new AkashChatError("Failed to generate text");
            }
            logger.info("Generated text:", text);
          }
        },
        {
          name: "akash_chat_test_embeddings",
          fn: async (runtime) => {
            const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
              text: "This is a test sentence for embedding generation"
            });
            if (!Array.isArray(embedding) || embedding.length === 0) {
              throw new AkashChatError("Failed to generate embeddings");
            }
            logger.info("Generated embedding with length:", embedding.length);
          }
        }
      ]
    }
  ]
};
var index_default = akashChatPlugin;
export {
  akashChatPlugin,
  index_default as default
};
//# sourceMappingURL=index.js.map