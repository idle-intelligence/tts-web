export class TtsClient {
    constructor(options = {}) {
        this.onStatus = options.onStatus || (() => {});
        this.onError = options.onError || console.error;
        this.onChunk = options.onChunk || (() => {});
        this.onDone = options.onDone || (() => {});
        this.onGenStart = options.onGenStart || (() => {});
        this.onVoiceLoaded = options.onVoiceLoaded || (() => {});

        this.baseUrl = (options.baseUrl || '').replace(/\/+$/, '');
        this.workerUrl = options.workerUrl || (this.baseUrl + '/worker.js');
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

            const config = { baseUrl: this.baseUrl };
            if (this.modelUrl) config.modelUrl = this.modelUrl;
            if (this.voiceUrl) config.voiceUrl = this.voiceUrl;
            if (this.tokenizerUrl) config.tokenizerUrl = this.tokenizerUrl;
            this.worker.postMessage({ type: 'load', config });
        });
    }

    loadVoice(name) {
        if (!this._ready) throw new Error('Not initialized');
        this.worker.postMessage({ type: 'load_voice', name });
    }

    generate(text, temperature = 0.7) {
        if (!this._ready) throw new Error('Not initialized');
        this.worker.postMessage({ type: 'generate', text, temperature });
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
                this.onStatus(data.text, data.ready || false, data.progress);
                break;
            case 'loaded':
                this.sampleRate = data.sampleRate;
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
            case 'gen_start':
                this.onGenStart(data.numTokens);
                break;
            case 'chunk':
                this.onChunk(data.data, data.step);
                break;
            case 'done':
                this.onDone(data.totalSteps);
                break;
            case 'error':
                this.onError(new Error(data.message));
                if (this._pendingReject) {
                    this._pendingReject(new Error(data.message));
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
                break;
        }
    }
}
