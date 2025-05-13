import { getConfig } from "../environment";
import { AkashChatOptions } from "../types";

export const generateTextAction = {
  name: "generateText",
  description: "Generate text using Akash Chat API",
  parameters: {
    prompt: {
      type: "string",
      description: "The prompt to generate text from",
    },
    model: {
      type: "string",
      description: "The model to use for generation. Can be 'small', 'medium', or 'large'",
      optional: true,
    },
    temperature: {
      type: "number",
      description: "The temperature for text generation (0.0 to 1.0)",
      optional: true,
    },
    maxTokens: {
      type: "number",
      description: "Maximum number of tokens to generate",
      optional: true,
    },
    systemPrompt: {
      type: "string",
      description: "System prompt to provide context",
      optional: true,
    },
  },
  handler: async (runtime: any, params: any) => {
    const config = getConfig(runtime);
    const modelSize = params.model || "medium";
    
    let modelName: string;
    switch(modelSize.toLowerCase()) {
      case "small":
        modelName = config.AKASH_CHAT_SMALL_MODEL;
        break;
      case "large":
        modelName = config.AKASH_CHAT_LARGE_MODEL;
        break;
      case "medium":
      default:
        modelName = config.AKASH_CHAT_MEDIUM_MODEL;
        break;
    }
    
    const options: AkashChatOptions = {
      modelName,
      temperature: params.temperature,
      maxTokens: params.maxTokens,
      systemPrompt: params.systemPrompt,
    };
    
    try {
      // This would normally call the actual implementation from index.ts
      // For this example, we're just describing the integration
      const result = await runtime.plugins.akashChat.generateText(params.prompt, options);
      return result;
    } catch (error) {
      console.error("Error generating text:", error);
      throw error;
    }
  },
}; 