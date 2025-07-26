/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  GoogleGenAI,
} from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { Config } from '../config/config.js';
import { getEffectiveModel } from './modelCheck.js';
import { UserTierId } from '../code_assist/types.js';
import * as https from 'https';
import * as http from 'http';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  userTier?: UserTierId;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  CLOUD_SHELL = 'cloud-shell',
}

export type ContentGeneratorConfig = {
  model: string;
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType | undefined;
  proxy?: string | undefined;
};

export function createContentGeneratorConfig(
  config: Config,
  authType: AuthType | undefined,
): ContentGeneratorConfig {
  const geminiApiKey = process.env.GEMINI_API_KEY || undefined;
  const googleApiKey = process.env.GOOGLE_API_KEY || undefined;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT || undefined;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION || undefined;

  // Use runtime model from config if available; otherwise, fall back to parameter or default
  const effectiveModel = config.getModel() || DEFAULT_GEMINI_MODEL;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    model: effectiveModel,
    authType,
    proxy: config?.getProxy(),
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.CLOUD_SHELL
  ) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;
    getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
      contentGeneratorConfig.proxy,
    );

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;

    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

// Custom HTTP agent for logging requests
class LoggingHttpsAgent extends https.Agent {
  constructor(options?: https.AgentOptions) {
    super(options);
  }
}

// Override fetch to log requests
const originalFetch = globalThis.fetch;
globalThis.fetch = async function(input: any, init?: any) {
  const url = typeof input === 'string' ? input : input.url;
  console.log('\n🚀 FETCH REQUEST:', {
    method: init?.method || 'GET',
    url: url,
    headers: init?.headers,
    timestamp: new Date().toISOString()
  });
  
  if (init?.body) {
    console.log('📤 REQUEST BODY:', init.body.toString());
  }
  
  const startTime = Date.now();
  const response = await originalFetch.call(this, input, init);
  const endTime = Date.now();
  
  console.log('📥 FETCH RESPONSE:', {
    status: response.status,
    statusText: response.statusText,
    headers: Object.fromEntries(response.headers.entries()),
    duration: `${endTime - startTime}ms`,
    timestamp: new Date().toISOString()
  });
  
  return response;
};

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const version = process.env.CLI_VERSION || process.version;
  const httpOptions = {
    headers: {
      'User-Agent': `GeminiCLI/${version} (${process.platform}; ${process.arch})`,
    },
    agent: new LoggingHttpsAgent(),
  };
  if (
    config.authType === AuthType.LOGIN_WITH_GOOGLE ||
    config.authType === AuthType.CLOUD_SHELL
  ) {
    return createCodeAssistContentGenerator(
      httpOptions,
      config.authType,
      gcConfig,
      sessionId,
    );
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    console.log('🔧 Creating GoogleGenAI with config:', {
      authType: config.authType,
      apiKey: config.apiKey ? `${config.apiKey.substring(0, 10)}...` : 'undefined',
      vertexai: config.vertexai,
      httpOptions: httpOptions,
      model: config.model
    });
    
    // Intercept fetch calls to log HTTP requests with full details
    const originalFetch = global.fetch;
    global.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = input.toString();
      const method = init?.method || 'GET';
      const headers = init?.headers || {};
      const body = init?.body;
      
      console.log('🌐 HTTP Request Details:');
      console.log('  URL:', url);
      console.log('  Method:', method);
      console.log('  Headers:', JSON.stringify(headers, null, 2));
      
      if (body) {
        console.log('  Body length:', typeof body === 'string' ? body.length : 'non-string');
        if (typeof body === 'string' && body.length < 2000) {
          console.log('  Body preview:', body.substring(0, 500) + (body.length > 500 ? '...' : ''));
        }
      }
      
      // Parse URL to show query params if any
      try {
        const urlObj = new URL(url);
        if (urlObj.search) {
          console.log('  Query params:', urlObj.search);
        }
        console.log('  Hostname:', urlObj.hostname);
        console.log('  Path:', urlObj.pathname);
      } catch (e) {
        console.log('  URL parsing failed:', e);
      }
      
      console.log('  ---');
      
      return originalFetch(input, init);
    };
    
    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
    });

    console.log('✅ GoogleGenAI created successfully');
    return googleGenAI.models;
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
  );
}
