import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { LLMProvider, LLMProviderOptions, LLMMessage, LLMStreamCallback } from './llmProvider';
import { Logger } from '../services/logger';

interface GeminiGenerateContentResponse {
    candidates?: Array<{
        content?: {
            parts?: Array<{ text?: string }>;
        };
    }>;
    promptFeedback?: {
        blockReason?: string;
    };
    error?: {
        message?: string;
    };
}

export class GeminiApiProvider extends LLMProvider {
    private workingDirectory = '';

    constructor(outputChannel: vscode.OutputChannel) {
        super(outputChannel);
        this.initializationPromise = this.initialize();
    }

    async initialize(): Promise<void> {
        if (this.isInitialized) {
            return;
        }

        try {
            Logger.info('Starting Gemini API provider initialization...');

            await this.setupWorkingDirectory();

            const apiKey = this.getApiKey();
            if (!apiKey) {
                Logger.warn('No Gemini API key found in configuration');
                throw new Error('Missing Gemini API key');
            }

            process.env.GOOGLE_GENERATIVE_AI_API_KEY = apiKey;

            this.isInitialized = true;
            Logger.info('Gemini API provider initialized successfully');
        } catch (error) {
            Logger.error(`Failed to initialize Gemini API provider: ${error}`);
            this.initializationPromise = null;
            this.isInitialized = false;
            throw error;
        }
    }

    private async setupWorkingDirectory(): Promise<void> {
        try {
            const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

            if (workspaceRoot) {
                const superdesignDir = path.join(workspaceRoot, '.superdesign');
                if (!fs.existsSync(superdesignDir)) {
                    fs.mkdirSync(superdesignDir, { recursive: true });
                    Logger.info(`Created .superdesign directory: ${superdesignDir}`);
                }
                this.workingDirectory = superdesignDir;
            } else {
                Logger.warn('No workspace root found, using temporary directory');
                const tempDir = path.join(os.tmpdir(), 'superdesign-gemini');
                if (!fs.existsSync(tempDir)) {
                    fs.mkdirSync(tempDir, { recursive: true });
                    Logger.info(`Created temporary directory: ${tempDir}`);
                }
                this.workingDirectory = tempDir;
            }
        } catch (error) {
            Logger.error(`Failed to setup Gemini working directory: ${error}`);
            this.workingDirectory = process.cwd();
        }
    }

    private getApiKey(): string | undefined {
        const config = vscode.workspace.getConfiguration('superdesign');
        const apiKey = config.get<string>('geminiApiKey');
        return apiKey?.trim() ? apiKey.trim() : undefined;
    }

    private getModelId(): string {
        const config = vscode.workspace.getConfiguration('superdesign');
        const model = config.get<string>('geminiModel');
        return model && model.trim().length > 0 ? model.trim() : 'gemini-1.5-pro-latest';
    }

    async query(
        prompt: string,
        options?: Partial<LLMProviderOptions>,
        abortController?: AbortController,
        onMessage?: LLMStreamCallback
    ): Promise<LLMMessage[]> {
        await this.ensureInitialized();

        const apiKey = this.getApiKey();
        if (!apiKey) {
            throw new Error('Gemini API key is not configured. Please run "Configure Gemini API Key" command.');
        }

        const modelId = this.getModelId();
        const systemPrompt = options?.customSystemPrompt;
        const resolvedPrompt = prompt ?? '';

        const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${apiKey}`;
        const body: Record<string, any> = {
            contents: [
                {
                    role: 'user',
                    parts: [
                        {
                            text: resolvedPrompt
                        }
                    ]
                }
            ]
        };

        if (systemPrompt && systemPrompt.trim().length > 0) {
            body.systemInstruction = {
                role: 'system',
                parts: [
                    {
                        text: systemPrompt
                    }
                ]
            };
        }

        if (options?.maxTurns) {
            body.generationConfig = {
                candidateCount: 1,
                responseLogprobs: false,
            };
        }

        const messages: LLMMessage[] = [];

        try {
            Logger.info(`Sending request to Gemini API using model ${modelId}`);

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body),
                signal: abortController?.signal
            });

            if (!response.ok) {
                const errorText = await response.text();
                Logger.error(`Gemini API request failed with status ${response.status}: ${errorText}`);
                throw new Error(`Gemini API error (${response.status}): ${errorText}`);
            }

            const data = (await response.json()) as GeminiGenerateContentResponse;

            if (data.promptFeedback?.blockReason) {
                throw new Error(`Gemini blocked the prompt: ${data.promptFeedback.blockReason}`);
            }

            const parts = data.candidates?.[0]?.content?.parts ?? [];
            const text = parts
                .map((part) => part?.text)
                .filter((partText): partText is string => typeof partText === 'string' && partText.trim().length > 0)
                .join('\n')
                .trim();

            const llmMessage: LLMMessage = {
                type: 'assistant',
                role: 'assistant',
                message: text,
                content: text,
                text
            };

            messages.push(llmMessage);

            if (text && onMessage) {
                onMessage(llmMessage);
            }

            Logger.info('Gemini API query completed successfully');
            return messages;
        } catch (error) {
            if (abortController?.signal.aborted) {
                Logger.warn('Gemini API request aborted by user');
                throw new Error('Gemini request was cancelled');
            }

            const errorMessage = error instanceof Error ? error.message : String(error);
            Logger.error(`Gemini API query failed: ${errorMessage}`);

            if (!this.isAuthError(errorMessage)) {
                vscode.window.showErrorMessage(`Gemini API query failed: ${errorMessage}`);
            }

            throw error;
        }
    }

    isReady(): boolean {
        return this.isInitialized;
    }

    async waitForInitialization(): Promise<boolean> {
        try {
            await this.ensureInitialized();
            return true;
        } catch (error) {
            Logger.error(`Gemini API provider initialization failed: ${error}`);
            return false;
        }
    }

    getWorkingDirectory(): string {
        return this.workingDirectory;
    }

    hasValidConfiguration(): boolean {
        return Boolean(this.getApiKey());
    }

    async refreshConfiguration(): Promise<boolean> {
        const apiKey = this.getApiKey();
        if (!apiKey) {
            Logger.warn('Gemini API key refresh failed: key not found');
            return false;
        }

        process.env.GOOGLE_GENERATIVE_AI_API_KEY = apiKey;
        Logger.info('Gemini API key refreshed from settings');

        if (!this.isInitialized) {
            await this.initialize();
        }

        return true;
    }

    isAuthError(errorMessage: string): boolean {
        const normalized = errorMessage.toLowerCase();
        return (
            normalized.includes('api key') ||
            normalized.includes('unauthorized') ||
            normalized.includes('permission') ||
            normalized.includes('forbidden')
        );
    }

    getProviderName(): string {
        return 'Gemini API';
    }

    getProviderType(): 'api' | 'binary' {
        return 'api';
    }
}
