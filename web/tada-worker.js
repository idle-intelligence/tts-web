// tada-worker.js — TADA-1B TTS Web Worker

const CACHE_NAME = 'tada-model-v1';

let model = null;  // TadaModel instance

// Cached fetch with progress reporting
async function cachedFetch(url, label) {
    const cache = await caches.open(CACHE_NAME);
    let resp = await cache.match(url);
    if (resp) {
        postMessage({type: 'status', text: `${label}: cached`, ready: false});
        return await resp.arrayBuffer();
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
            postMessage({type: 'status', text: `${label}: ${(received/1024/1024).toFixed(1)}MB / ${(contentLength/1024/1024).toFixed(1)}MB`, ready: false, progress: received/contentLength});
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

    return data.buffer;
}

async function handleLoad(baseUrl, wasmBaseUrl) {
    try {
        const wasmBase = wasmBaseUrl || baseUrl;
        postMessage({type: 'status', text: 'Loading WASM module...', ready: false});
        const wasm = await import(wasmBase + '/tada_wasm.js');
        await wasm.default({ module_or_path: wasmBase + '/tada_wasm_bg.wasm' });

        const [modelData, tokenizerData] = await Promise.all([
            cachedFetch(baseUrl + '/tada-1b-q4_0.gguf', 'Model'),
            cachedFetch(baseUrl + '/tokenizer.json', 'Tokenizer'),
        ]);

        postMessage({type: 'status', text: 'Initializing model...', ready: false});
        console.log('[tada-worker] model data:', modelData.byteLength, 'bytes, tokenizer:', tokenizerData.byteLength, 'bytes');
        model = new wasm.TadaModel(
            new Uint8Array(modelData),
            new Uint8Array(tokenizerData),
        );

        postMessage({type: 'status', text: 'Ready', ready: true});
        postMessage({type: 'loaded'});
    } catch (e) {
        console.error('[tada-worker] load error:', e);
        postMessage({type: 'error', error: e.message || String(e)});
    }
}

async function handleGenerate(text, temperature, noiseTemp, numFlowSteps) {
    if (!model) {
        postMessage({type: 'error', error: 'Model not loaded'});
        return;
    }

    try {
        // Tokenize
        const tokenIds = model.tokenize(text);
        const numTokens = tokenIds.length;
        postMessage({type: 'gen_start', numTokens});

        // Start generation
        model.start_generation(tokenIds, temperature, noiseTemp || 0.9, numFlowSteps || 10);

        // Generation loop
        let step = 0;
        while (true) {
            const result = model.generation_step();
            if (result === null || result === undefined) break;

            const info = typeof result === 'string' ? JSON.parse(result) : result;
            step = info.step || step + 1;

            postMessage({
                type: 'progress',
                step: info.step,
                totalTokens: info.total_tokens,
                isEos: info.is_eos,
                tokenId: info.token_id,
            });

            if (info.is_done) break;
        }

        // Decode audio
        postMessage({type: 'status', text: 'Decoding audio...', ready: true});
        const pcm = model.decode_audio();

        if (pcm && pcm.length > 0) {
            postMessage({type: 'audio', samples: pcm, sampleRate: model.sample_rate()});
        }

        postMessage({type: 'done', totalSteps: step});
    } catch (e) {
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
            await handleGenerate(msg.text, msg.temperature, msg.noiseTemp, msg.numFlowSteps);
            break;
        case 'cancel':
            if (model) model.cancel_generation();
            break;
    }
};
