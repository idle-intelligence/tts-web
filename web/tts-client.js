export class TtsClient {
    constructor(options = {}) {
        this.modelType = options.modelType || 'pocket-tts';

        this.onStatus = options.onStatus || (() => {});
        this.onError = options.onError || console.error;
        this.onChunk = options.onChunk || (() => {});
        this.onDone = options.onDone || (() => {});
        this.onGenStart = options.onGenStart || (() => {});
        this.onVoiceLoaded = options.onVoiceLoaded || (() => {});
        this.onProgress = options.onProgress || (() => {});

        this.baseUrl = (options.baseUrl || '').replace(/\/+$/, '');
        this.wasmBaseUrl = options.wasmBaseUrl || null;
        this.workerUrl = options.workerUrl || (this.baseUrl + (this.modelType === 'tada' ? '/tada-worker.js' : '/worker.js'));
        this.modelUrl = options.modelUrl || null;
        this.voiceUrl = options.voiceUrl || null;
        this.tokenizerUrl = options.tokenizerUrl || null;

        this.worker = null;
        this.sampleRate = 24000;
        this._ready = false;
        this._pendingResolve = null;
        this._pendingReject = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            this.worker = new Worker(this.workerUrl, { type: 'module' });
            this.worker.onmessage = (e) => this._handleMessage(e);
            this.worker.onerror = (err) => {
                const msg = err.message || err.filename
                    ? `Worker error: ${err.message} (${err.filename}:${err.lineno}:${err.colno})`
                    : 'Worker failed to load — check browser console';
                console.error('[tts-client] worker error event:', err);
                this.onError(new Error(msg));
                if (this._pendingReject) {
                    this._pendingReject(new Error(msg));
                    this._pendingReject = null;
                    this._pendingResolve = null;
                }
            };
            this._pendingResolve = resolve;
            this._pendingReject = reject;

            if (this.modelType === 'tada') {
                const msg = { type: 'load', baseUrl: this.baseUrl || location.origin };
                if (this.wasmBaseUrl) msg.wasmBaseUrl = this.wasmBaseUrl;
                this.worker.postMessage(msg);
            } else {
                const config = { baseUrl: this.baseUrl };
                if (this.modelUrl) config.modelUrl = this.modelUrl;
                if (this.voiceUrl) config.voiceUrl = this.voiceUrl;
                if (this.tokenizerUrl) config.tokenizerUrl = this.tokenizerUrl;
                this.worker.postMessage({ type: 'load', config });
            }
        });
    }

    loadVoice(name) {
        if (!this._ready) throw new Error('Not initialized');
        this.worker.postMessage({ type: 'load_voice', name });
    }

    setVoice(name, file) {
        if (!this._ready) throw new Error('Not initialized');
        this.worker.postMessage({ type: 'setVoice', name, file: file || name, baseUrl: this.baseUrl || location.origin });
    }

    generate(text, temperature = 0.7, options = {}) {
        if (!this._ready) throw new Error('Not initialized');
        if (this.modelType === 'tada') {
            this.worker.postMessage({
                type: 'generate',
                text,
                temperature,
                noiseTemp: options.noiseTemp,
                numFlowSteps: options.numFlowSteps,
                cfgScale: options.cfgScale,
            });
        } else {
            this.worker.postMessage({ type: 'generate', text, temperature });
        }
    }

    cancel() {
        if (this.worker && this.modelType === 'tada') {
            this.worker.postMessage({ type: 'cancel' });
        }
    }

    isReady() { return this._ready; }

    destroy() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        this._ready = false;
    }

    _handleMessage(e) {
        const { type, ...data } = e.data;
        switch (type) {
            case 'status':
                if (data.ready) this._ready = true;
                this.onStatus(data.text, data.ready || false, data.progress);
                break;
            case 'loaded':
                if (data.sampleRate) this.sampleRate = data.sampleRate;
                this._ready = true;
                if (this._pendingResolve) {
                    this._pendingResolve();
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
                break;
            case 'voice_loaded':
                this.onVoiceLoaded(data.name, data.voiceIndex);
                break;
            case 'voiceLoaded':
                this.onVoiceLoaded(data.name);
                break;
            case 'gen_start':
                this.onGenStart(data.numTokens);
                break;
            case 'chunk':
                this.onChunk(data.data, data.step);
                break;
            case 'progress':
                this.onProgress(data.step, data.totalTokens, data.isEos, data.tokenId);
                break;
            case 'audio':
                // TADA: full audio delivered at once — call onChunk with the complete buffer
                this.sampleRate = data.sampleRate || this.sampleRate;
                this.onChunk(data.samples, -1);
                break;
            case 'done':
                this.onDone(data.totalSteps);
                break;
            case 'error':
                this.onError(new Error(data.message || data.error));
                if (this._pendingReject) {
                    this._pendingReject(new Error(data.message || data.error));
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
                break;
        }
    }
}
