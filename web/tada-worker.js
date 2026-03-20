// tada-worker.js — TADA-1B TTS Web Worker
// Uses TadaEngine with shard-based loading (Burn/wgpu LLM + candle VibeVoice)

const CACHE_NAME = 'tada-model-v2';

let wasm = null;
let engine = null;
let voiceBytes = null;
let voiceText = null;

// Cached fetch with progress reporting
async function cachedFetch(url, label) {
    const cache = await caches.open(CACHE_NAME);
    let resp = await cache.match(url);
    if (resp) {
        postMessage({type: 'status', text: `${label}: cached`, ready: false});
        return new Uint8Array(await resp.arrayBuffer());
    }

    postMessage({type: 'status', text: `${label}: downloading...`, ready: false});
    resp = await fetch(url);
    const reader = resp.body.getReader();
    const contentLength = +resp.headers.get('Content-Length') || 0;
    const chunks = [];
    let received = 0;

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (contentLength) {
            postMessage({
                type: 'status',
                text: `${label}: ${(received/1024/1024).toFixed(1)}MB / ${(contentLength/1024/1024).toFixed(1)}MB`,
                ready: false,
                progress: received / contentLength,
            });
        }
    }

    const data = new Uint8Array(received);
    let pos = 0;
    for (const chunk of chunks) {
        data.set(chunk, pos);
        pos += chunk.length;
    }

    const cacheResp = new Response(data, {headers: {'Content-Type': 'application/octet-stream'}});
    await cache.put(url, cacheResp);

    return data;
}

async function handleLoad(baseUrl, wasmBaseUrl) {
    try {
        const wasmBase = wasmBaseUrl || baseUrl;
        postMessage({type: 'status', text: 'Loading WASM module...', ready: false});
        wasm = await import(wasmBase + '/tada_wasm.js');
        await wasm.default(wasmBase + '/tada_wasm_bg.wasm');

        // Initialize WebGPU device
        postMessage({type: 'status', text: 'Initializing WebGPU...', ready: false});
        await wasm.initWgpuDevice();

        // Create engine
        engine = new wasm.TadaEngine();

        // Download model and tokenizer
        const [modelData, tokenizerData] = await Promise.all([
            cachedFetch(baseUrl + '/tada-1b-q4_0.gguf', 'Model'),
            cachedFetch(baseUrl + '/tokenizer.json', 'Tokenizer'),
        ]);

        // Feed model data as a single shard
        postMessage({type: 'status', text: 'Loading model weights...', ready: false});
        engine.appendModelShard(modelData);

        // Load model (parses GGUF, initializes Burn LLM + candle VibeVoice)
        postMessage({type: 'status', text: 'Initializing model...', ready: false});
        engine.loadModel(tokenizerData);

        // GPU warmup: pre-compile shader pipelines to avoid stall on first gen
        postMessage({type: 'status', text: 'Reticulating splines...', ready: false});
        await engine.warmup();

        // Load default voice prompt (ljspeech_long: 32 acoustic tokens)
        postMessage({type: 'status', text: 'Loading voice prompt...', ready: false});
        try {
            const [voiceStBytes, voiceJsonResp] = await Promise.all([
                fetch(baseUrl + '/voices/ljspeech_long.safetensors').then(r => r.arrayBuffer()),
                fetch(baseUrl + '/voices/ljspeech_long.json').then(r => r.json()),
            ]);
            voiceBytes = new Uint8Array(voiceStBytes);
            voiceText = voiceJsonResp.text;
        } catch (voiceErr) {
            console.warn('[tada-worker] voice prompt load failed (zero-shot fallback):', voiceErr);
            voiceBytes = null;
            voiceText = null;
        }

        postMessage({type: 'status', text: 'Ready', ready: true});
        postMessage({type: 'loaded'});
    } catch (e) {
        console.error('[tada-worker] load error:', e);
        postMessage({type: 'error', error: e.message || String(e)});
    }
}

async function handleGenerate(text, temperature, noiseTemp, numFlowSteps, cfgScale) {
    if (!engine) {
        postMessage({type: 'error', error: 'Model not loaded'});
        return;
    }

    try {
        // Tokenize — use voice text if available for correct alignment
        let tokenIds, prefixLen, transitionSteps;
        if (voiceBytes && voiceText) {
            const result = engine.tokenizeWithVoice(text, voiceText);
            tokenIds = result.tokenIds;
            prefixLen = result.prefixLen;
            transitionSteps = 5; // ljspeech_long has 32 tokens, trim 5, keep 27
        } else {
            tokenIds = engine.tokenize(text);
            prefixLen = 0;
            transitionSteps = 0;
        }

        const numTokens = tokenIds.length;
        // Content tokens = total minus prefix minus voice text tokens minus EOT padding
        const voiceTokenCount = voiceBytes ? (numTokens - prefixLen - transitionSteps) : 0; // approximate
        const contentTokens = numTokens - prefixLen - voiceTokenCount;
        console.log(`[tada-worker] tokens=${numTokens}, prefixLen=${prefixLen}, voiceTokens≈${voiceTokenCount}, contentTokens≈${contentTokens}`);
        postMessage({type: 'gen_start', numTokens: contentTokens});

        // Start generation with voice prompt (or zero-shot if not loaded)
        engine.startGeneration(
            tokenIds,
            temperature,
            noiseTemp || 0.9,
            numFlowSteps || 10,
            cfgScale || 1.0,
            voiceBytes || null,
            prefixLen,
            transitionSteps,
        );

        // Generation loop
        let step = 0;
        while (true) {
            const result = await engine.generationStep();
            if (result === null || result === undefined) break;

            const info = typeof result === 'string' ? JSON.parse(result) : result;
            step = info.step || step + 1;

            // Only show progress for content tokens (skip prompt phase)
            const promptTokens = info.total_tokens - contentTokens;
            if (info.step >= promptTokens) {
                const contentStep = info.step - promptTokens;
                postMessage({
                    type: 'progress',
                    step: contentStep,
                    totalTokens: contentTokens,
                    isEos: info.is_eos,
                    tokenId: info.token_id,
                });
            }

            if (info.is_done) break;
        }

        // Decode audio
        postMessage({type: 'progress', step: -1, totalTokens: 0, isEos: false, tokenId: 0, decoding: true});
        const pcm = engine.decodeAudio();

        if (pcm && pcm.length > 0) {
            postMessage({type: 'audio', samples: pcm, sampleRate: 24000});
        }

        postMessage({type: 'done', totalSteps: step + 1});
    } catch (e) {
        postMessage({type: 'error', error: e.message || String(e)});
    }
}

async function handleSetVoice(name, file, baseUrl) {
    try {
        postMessage({type: 'status', text: `Loading voice: ${name}...`, ready: false});
        const voiceUrl = baseUrl + '/voices/' + file + '.safetensors';
        const jsonUrl = baseUrl + '/voices/' + file + '.json';
        const [voiceStBytes, voiceJsonResp] = await Promise.all([
            fetch(voiceUrl).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); }),
            fetch(jsonUrl).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
        ]);
        voiceBytes = new Uint8Array(voiceStBytes);
        voiceText = voiceJsonResp.text;
        postMessage({type: 'status', text: 'Ready', ready: true});
        postMessage({type: 'voiceLoaded', name});
    } catch (e) {
        console.error('[tada-worker] setVoice error:', e);
        postMessage({type: 'error', error: e.message || String(e)});
    }
}

self.onmessage = async (e) => {
    const msg = e.data;
    switch (msg.type) {
        case 'load':
            await handleLoad(msg.baseUrl, msg.wasmBaseUrl);
            break;
        case 'generate':
            await handleGenerate(msg.text, msg.temperature, msg.noiseTemp, msg.numFlowSteps, msg.cfgScale);
            break;
        case 'setVoice':
            await handleSetVoice(msg.name, msg.file || msg.name, msg.baseUrl);
            break;
        case 'cancel':
            if (engine) engine.cancelGeneration();
            break;
    }
};
