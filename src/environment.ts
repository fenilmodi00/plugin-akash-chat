import { z } from "zod";

export const AkashChatConfigSchema = z.object({
  API_KEY: z.string().min(1, "API_KEY is required for Akash Chat API"),
  AKASH_CHAT_SMALL_MODEL: z.string().optional().default("Meta-Llama-3-1-8B-Instruct-FP8"),
  AKASH_CHAT_MEDIUM_MODEL: z.string().optional().default("Meta-Llama-3-2-3B-Instruct"),
  AKASH_CHAT_LARGE_MODEL: z.string().optional().default("Meta-Llama-3-3-70B-Instruct"),
  AKASHCHAT_EMBEDDING_MODEL: z.string().optional().default("BAAI-bge-large-en-v1-5"),
  AKASHCHAT_EMBEDDING_DIMENSIONS: z.number().optional().default(1024),
});

export type AkashChatConfig = z.infer<typeof AkashChatConfigSchema>;

export function getConfig(runtime: any): AkashChatConfig {
  const config: Record<string, any> = {
    API_KEY: runtime.getSetting("API_KEY"),
    AKASH_CHAT_SMALL_MODEL: runtime.getSetting("AKASH_CHAT_SMALL_MODEL"),
    AKASH_CHAT_MEDIUM_MODEL: runtime.getSetting("AKASH_CHAT_MEDIUM_MODEL"),
    AKASH_CHAT_LARGE_MODEL: runtime.getSetting("AKASH_CHAT_LARGE_MODEL"),
    AKASHCHAT_EMBEDDING_MODEL: runtime.getSetting("AKASHCHAT_EMBEDDING_MODEL"),
    AKASHCHAT_EMBEDDING_DIMENSIONS: parseInt(runtime.getSetting("AKASHCHAT_EMBEDDING_DIMENSIONS") || "1024", 10),
  };

  return AkashChatConfigSchema.parse(config);
} 