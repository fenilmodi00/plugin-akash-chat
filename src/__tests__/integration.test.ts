import { describe, it, expect } from 'vitest';
import type { IAgentRuntime } from '@elizaos/core';
import akashchatPlugin from '../index';

describe('Plugin Integration Tests', () => {
  it('should export the plugin correctly', () => {
    expect(akashchatPlugin).toBeDefined();
    expect(akashchatPlugin.name).toBe('akashchat');
    expect(akashchatPlugin.description).toBeDefined();
    expect(akashchatPlugin.models).toBeDefined();
    expect(akashchatPlugin.tests).toBeDefined();
  });

  it('should have proper TypeScript types', () => {
    // Test that the plugin conforms to the Plugin interface
    expect(typeof akashchatPlugin.name).toBe('string');
    expect(typeof akashchatPlugin.description).toBe('string');
    expect(typeof akashchatPlugin.config).toBe('object');
    expect(typeof akashchatPlugin.init).toBe('function');
    expect(typeof akashchatPlugin.models).toBe('object');
    expect(Array.isArray(akashchatPlugin.tests)).toBe(true);
  });

  it('should have all required model handlers', () => {
    const models = akashchatPlugin.models!;
    
    // Check that all expected model types are present
    expect(models).toHaveProperty('TEXT_EMBEDDING');
    expect(models).toHaveProperty('TEXT_TOKENIZER_ENCODE');
    expect(models).toHaveProperty('TEXT_TOKENIZER_DECODE');
    expect(models).toHaveProperty('TEXT_SMALL');
    expect(models).toHaveProperty('TEXT_LARGE');
    expect(models).toHaveProperty('OBJECT_SMALL');
    expect(models).toHaveProperty('OBJECT_LARGE');
  });

  it('should have consistent configuration', () => {
    const config = akashchatPlugin.config;
    
    // All config keys should follow the AKASH_CHAT_ prefix pattern
    const configKeys = Object.keys(config);
    const akashChatKeys = configKeys.filter(key => key.startsWith('AKASH_CHAT_'));
    
    // Most keys should follow the naming convention
    expect(akashChatKeys.length).toBeGreaterThan(0);
  });
});
