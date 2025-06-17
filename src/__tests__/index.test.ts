import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ModelType, EventType, logger } from '@elizaos/core';
import type { IAgentRuntime } from '@elizaos/core';
import akashchatPlugin from '../index';

// Mock undici fetch
vi.mock('undici', () => ({
  fetch: vi.fn(),
}));

// Import the mocked fetch
import { fetch } from 'undici';

// Get the mocked fetch for type safety
const mockFetch = vi.mocked(fetch);

// Create a mock runtime
const createMockRuntime = (): IAgentRuntime => {
  const mockRuntime = {
    getSetting: vi.fn(),
    character: {
      system: 'You are a helpful AI assistant.',
    },
    emitEvent: vi.fn(),
    useModel: vi.fn(),
  } as unknown as IAgentRuntime;

  // Setup default settings
  (mockRuntime.getSetting as any).mockImplementation((key: string) => {
    const settings: Record<string, string> = {
      'AKASH_CHAT_API_KEY': 'test-api-key',
      'AKASH_CHAT_BASE_URL': 'https://chatapi.akash.network/api/v1',
      'AKASH_CHAT_SMALL_MODEL': 'Meta-Llama-3-1-8B-Instruct-FP8',
      'AKASH_CHAT_LARGE_MODEL': 'Meta-Llama-3-3-70B-Instruct',
      'AKASH_CHAT_EMBEDDING_MODEL': 'BAAI-bge-large-en-v1-5',
      'AKASH_CHAT_EMBEDDING_DIMENSIONS': '1024',
    };
    return settings[key];
  });

  return mockRuntime;
};

describe('Akash Chat Plugin', () => {
  let mockRuntime: IAgentRuntime;

  beforeEach(() => {
    mockRuntime = createMockRuntime();
    vi.clearAllMocks();
  });

  describe('Plugin Configuration', () => {
    it('should have correct plugin metadata', () => {
      expect(akashchatPlugin.name).toBe('akashchat');
      expect(akashchatPlugin.description).toBe('Akash Chat API plugin for language model capabilities via Akash Network');
      expect(akashchatPlugin.config).toBeDefined();
    });

    it('should have all required config options', () => {
      const config = akashchatPlugin.config;
      expect(config).toHaveProperty('AKASH_CHAT_API_KEY');
      expect(config).toHaveProperty('AKASH_CHAT_BASE_URL');
      expect(config).toHaveProperty('AKASH_CHAT_SMALL_MODEL');
      expect(config).toHaveProperty('AKASH_CHAT_LARGE_MODEL');
      expect(config).toHaveProperty('AKASH_CHAT_EMBEDDING_MODEL');
      expect(config).toHaveProperty('AKASH_CHAT_EMBEDDING_DIMENSIONS');
    });
  });

  describe('Plugin Initialization', () => {
    it('should warn when API key is missing', async () => {
      const loggerSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {});
      const runtimeWithoutKey = createMockRuntime();
      (runtimeWithoutKey.getSetting as any).mockReturnValue(undefined);

      await akashchatPlugin.init!({}, runtimeWithoutKey);

      expect(loggerSpy).toHaveBeenCalledWith(
        expect.stringContaining('Akash Chat functionality will be limited')
      );
      loggerSpy.mockRestore();
    });

    it('should validate API key when present', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [{ id: 'test-model' }] }),
      } as Response);

      await akashchatPlugin.init!({}, mockRuntime);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://chatapi.akash.network/api/v1/models',
        expect.objectContaining({
          headers: { 
            Authorization: 'Bearer test-api-key',
            'Content-Type': 'application/json'
          },
        })
      );
    });
  });

  describe('Model Handlers', () => {
    describe('TEXT_EMBEDDING', () => {
      it('should return test vector for null input', async () => {
        const handler = akashchatPlugin.models![ModelType.TEXT_EMBEDDING];
        const result = await handler(mockRuntime, null);

        expect(Array.isArray(result)).toBe(true);
        expect(result).toHaveLength(1024);
        expect(result[0]).toBe(0.1);
      });

      it('should handle string input', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: [{ embedding: new Array(1024).fill(0.5) }],
          }),
        } as Response);

        const handler = akashchatPlugin.models![ModelType.TEXT_EMBEDDING];
        const result = await handler(mockRuntime, 'test text');

        expect(mockFetch).toHaveBeenCalledWith(
          'https://chatapi.akash.network/api/v1/embeddings',
          expect.objectContaining({
            method: 'POST',
            headers: {
              Authorization: 'Bearer test-api-key',
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model: 'BAAI-bge-large-en-v1-5',
              input: 'test text',
            }),
          })
        );

        expect(Array.isArray(result)).toBe(true);
        expect(result).toHaveLength(1024);
      });

      it('should handle object input with text property', async () => {
        const mockFetch = vi.mocked(fetch);
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: [{ embedding: new Array(1024).fill(0.5) }],
          }),
        } as Response);

        const handler = akashchatPlugin.models![ModelType.TEXT_EMBEDDING];
        const result = await handler(mockRuntime, { text: 'test text' });

        expect(Array.isArray(result)).toBe(true);
        expect(result).toHaveLength(1024);
      });

      it('should handle API errors gracefully', async () => {
        const mockFetch = vi.mocked(fetch);
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 401,
          statusText: 'Unauthorized',
        } as Response);

        const handler = akashchatPlugin.models![ModelType.TEXT_EMBEDDING];
        const result = await handler(mockRuntime, 'test text');

        expect(Array.isArray(result)).toBe(true);
        expect(result).toHaveLength(1024);
        expect(result[0]).toBe(0.4); // Error indicator
      });
    });

    describe('TEXT_SMALL', () => {
      it('should be defined and callable', () => {
        const handler = akashchatPlugin.models![ModelType.TEXT_SMALL];
        expect(handler).toBeDefined();
        expect(typeof handler).toBe('function');
      });
    });

    describe('TEXT_LARGE', () => {
      it('should be defined and callable', () => {
        const handler = akashchatPlugin.models![ModelType.TEXT_LARGE];
        expect(handler).toBeDefined();
        expect(typeof handler).toBe('function');
      });
    });

    describe('TEXT_TOKENIZER_ENCODE', () => {
      it('should be defined and callable', () => {
        const handler = akashchatPlugin.models![ModelType.TEXT_TOKENIZER_ENCODE];
        expect(handler).toBeDefined();
        expect(typeof handler).toBe('function');
      });

      it('should return array of tokens', async () => {
        const handler = akashchatPlugin.models![ModelType.TEXT_TOKENIZER_ENCODE];
        const result = await handler(mockRuntime, { 
          prompt: 'Hello world',
          modelType: ModelType.TEXT_SMALL 
        });

        expect(Array.isArray(result)).toBe(true);
      });
    });

    describe('TEXT_TOKENIZER_DECODE', () => {
      it('should be defined and callable', () => {
        const handler = akashchatPlugin.models![ModelType.TEXT_TOKENIZER_DECODE];
        expect(handler).toBeDefined();
        expect(typeof handler).toBe('function');
      });

      it('should return decoded string', async () => {
        const handler = akashchatPlugin.models![ModelType.TEXT_TOKENIZER_DECODE];
        const result = await handler(mockRuntime, { 
          tokens: [1, 2, 3],
          modelType: ModelType.TEXT_SMALL 
        });

        expect(typeof result).toBe('string');
      });
    });

    describe('OBJECT_SMALL', () => {
      it('should be defined and callable', () => {
        const handler = akashchatPlugin.models![ModelType.OBJECT_SMALL];
        expect(handler).toBeDefined();
        expect(typeof handler).toBe('function');
      });
    });

    describe('OBJECT_LARGE', () => {
      it('should be defined and callable', () => {
        const handler = akashchatPlugin.models![ModelType.OBJECT_LARGE];
        expect(handler).toBeDefined();
        expect(typeof handler).toBe('function');
      });
    });
  });

  describe('Tests Configuration', () => {
    it('should have test suite defined', () => {
      expect(akashchatPlugin.tests).toBeDefined();
      expect(Array.isArray(akashchatPlugin.tests)).toBe(true);
      expect(akashchatPlugin.tests).toHaveLength(1);
    });

    it('should have all required test cases', () => {
      const testSuite = akashchatPlugin.tests![0];
      expect(testSuite.name).toBe('akashchat_plugin_tests');
      expect(testSuite.tests).toHaveLength(6);

      const testNames = testSuite.tests.map(test => test.name);
      expect(testNames).toContain('akashchat_test_url_and_api_key_validation');
      expect(testNames).toContain('akashchat_test_text_embedding');
      expect(testNames).toContain('akashchat_test_text_large');
      expect(testNames).toContain('akashchat_test_text_small');
      expect(testNames).toContain('akashchat_test_text_tokenizer_encode');
      expect(testNames).toContain('akashchat_test_text_tokenizer_decode');
    });
  });

  describe('Integration Tests', () => {
    it('should run API validation test', async () => {
      const mockFetch = vi.mocked(fetch);
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [{ id: 'test-model' }] }),
      } as Response);

      const testSuite = akashchatPlugin.tests![0];
      const validationTest = testSuite.tests.find(test => 
        test.name === 'akashchat_test_url_and_api_key_validation'
      );

      expect(validationTest).toBeDefined();
      await expect(validationTest!.fn(mockRuntime)).resolves.not.toThrow();
    });

    it('should run embedding test with mock runtime', async () => {
      const testSuite = akashchatPlugin.tests![0];
      const embeddingTest = testSuite.tests.find(test => 
        test.name === 'akashchat_test_text_embedding'
      );

      // Mock the useModel method to return a valid embedding
      (mockRuntime.useModel as any).mockResolvedValueOnce(new Array(1024).fill(0.5));

      expect(embeddingTest).toBeDefined();
      await expect(embeddingTest!.fn(mockRuntime)).resolves.not.toThrow();
    });
  });
});
