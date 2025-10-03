import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { LLMProvider, LLMProviderOptions, LLMMessage, LLMStreamCallback } from './llmProvider';
import { Logger } from '../services/logger';

interface MistralChatCompletionResponse {
    id?: string;
    choices?: Array<{
        index?: number;
        message?: {
            role?: string;
            content?: string;
        };
        finish_reason?: string;
    }>;
    error?: {
        message?: string;
        type?: string;
    };
}

export class MistralApiProvider extends LLMProvider {
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
            Logger.info('Starting Mistral API provider initialization...');

            await this.setupWorkingDirectory();

            const apiKey = this.getApiKey();
            if (!apiKey) {
                Logger.warn('No Mistral API key found in configuration');
                throw new Error('Missing Mistral API key');
            }

            process.env.MISTRAL_API_KEY = apiKey;

            this.isInitialized = true;
            Logger.info('Mistral API provider initialized successfully');
        } catch (error) {
            Logger.error(`Failed to initialize Mistral API provider: ${error}`);
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
                const tempDir = path.join(os.tmpdir(), 'superdesign-mistral');
                if (!fs.existsSync(tempDir)) {
                    fs.mkdirSync(tempDir, { recursive: true });
                    Logger.info(`Created temporary directory: ${tempDir}`);
                }
                this.workingDirectory = tempDir;
            }
        } catch (error) {
            Logger.error(`Failed to setup Mistral working directory: ${error}`);
            this.workingDirectory = process.cwd();
        }
    }

    private getApiKey(): string | undefined {
        const config = vscode.workspace.getConfiguration('superdesign');
        const apiKey = config.get<string>('mistralApiKey');
        return apiKey?.trim() ? apiKey.trim() : undefined;
    }

    private getModelId(): string {
        const config = vscode.workspace.getConfiguration('superdesign');
        const model = config.get<string>('mistralModel');
        return model && model.trim().length > 0 ? model.trim() : 'mistral-large-latest';
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
            throw new Error('Mistral API key is not configured. Please run "Configure Mistral API Key" command.');
        }

        const modelId = this.getModelId();
        const systemPrompt = options?.customSystemPrompt;
        const resolvedPrompt = prompt ?? '';

        const url = 'https://api.mistral.ai/v1/chat/completions';
        const messagesPayload: Array<{ role: string; content: string }> = [];

        if (systemPrompt && systemPrompt.trim().length > 0) {
            messagesPayload.push({ role: 'system', content: systemPrompt });
        }

        messagesPayload.push({ role: 'user', content: resolvedPrompt });

        const body = {
            model: modelId,
            messages: messagesPayload,
            stream: false,
            temperature: 0.7
        };

        const messages: LLMMessage[] = [];

        try {
            Logger.info(`Sending request to Mistral API using model ${modelId}`);

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${apiKey}`
                },
                body: JSON.stringify(body),
                signal: abortController?.signal
            });

            if (!response.ok) {
                const errorText = await response.text();
                Logger.error(`Mistral API request failed with status ${response.status}: ${errorText}`);
                throw new Error(`Mistral API error (${response.status}): ${errorText}`);
            }

            const data = (await response.json()) as MistralChatCompletionResponse;
            const text = data.choices?.[0]?.message?.content?.trim() ?? '';

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

            Logger.info('Mistral API query completed successfully');
            return messages;
        } catch (error) {
            if (abortController?.signal.aborted) {
                Logger.warn('Mistral API request aborted by user');
                throw new Error('Mistral request was cancelled');
            }

            const errorMessage = error instanceof Error ? error.message : String(error);
            Logger.error(`Mistral API query failed: ${errorMessage}`);

            if (!this.isAuthError(errorMessage)) {
                vscode.window.showErrorMessage(`Mistral API query failed: ${errorMessage}`);
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
            Logger.error(`Mistral API provider initialization failed: ${error}`);
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
            Logger.warn('Mistral API key refresh failed: key not found');
            return false;
        }

        process.env.MISTRAL_API_KEY = apiKey;
        Logger.info('Mistral API key refreshed from settings');

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
            normalized.includes('forbidden') ||
            normalized.includes('invalid token') ||
            normalized.includes('invalid_api_key')
        );
    }

    getProviderName(): string {
        return 'Mistral API';
    }

    getProviderType(): 'api' | 'binary' {
        return 'api';
    }
}
