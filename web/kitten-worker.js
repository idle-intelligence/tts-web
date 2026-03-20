// kitten-worker.js — KittenTTS Web Worker

const CACHE_NAME = 'kitten-model-v1';

let wasm = null;
let engine = null;

async function cachedFetch(url, label) {
    const cache = await caches.open(CACHE_NAME);
    let resp = await cache.match(url);
    if (resp) {
        postMessage({ type: 'status', text: `${label}: cached`, ready: false });
        return new Uint8Array(await resp.arrayBuffer());
    }

    postMessage({ type: 'status', text: `${label}: downloading...`, ready: false });
    resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);

    const reader = resp.body.getReader();
    const contentLength = +resp.headers.get('Content-Length') || 0;
    const chunks = [];
    let received = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (contentLength) {
            postMessage({
                type: 'status',
                text: `${label}: ${(received / 1024 / 1024).toFixed(1)} MB / ${(contentLength / 1024 / 1024).toFixed(1)} MB`,
                ready: false,
                progress: received / contentLength,
            });
        }
    }

    const data = new Uint8Array(received);
    let pos = 0;
    for (const chunk of chunks) { data.set(chunk, pos); pos += chunk.length; }

    await cache.put(url, new Response(data, { headers: { 'Content-Type': 'application/octet-stream' } }));
    return data;
}

async function handleLoad(modelUrl, voicesUrl, wasmBaseUrl) {
    try {
        postMessage({ type: 'status', text: 'Loading WASM module...', ready: false });
        wasm = await import(wasmBaseUrl + '/kitten_wasm.js');
        await wasm.default({ module_or_path: wasmBaseUrl + '/kitten_wasm_bg.wasm' });

        engine = new wasm.KittenEngine();

        const [modelData, voicesData] = await Promise.all([
            cachedFetch(modelUrl, 'Model'),
            cachedFetch(voicesUrl, 'Voices'),
        ]);

        postMessage({ type: 'status', text: 'Loading model weights...', ready: false });
        engine.loadModel(modelData);

        postMessage({ type: 'status', text: 'Loading voices...', ready: false });
        engine.loadVoices(voicesData);

        const voiceNames = engine.getVoiceNames();
        postMessage({ type: 'status', text: 'Ready', ready: true });
        postMessage({ type: 'loaded', sampleRate: 24000, voiceNames });
    } catch (e) {
        console.error('[kitten-worker] load error:', e);
        postMessage({ type: 'error', error: e.message || String(e) });
    }
}

async function handleGenerate(ipa, voiceIdx, speed, textLen) {
    if (!engine) {
        postMessage({ type: 'error', error: 'Model not loaded' });
        return;
    }
    if (!engine.isReady()) {
        postMessage({ type: 'error', error: 'Model not ready' });
        return;
    }

    try {
        postMessage({ type: 'status', text: 'Synthesizing...', ready: true });
        const samples = engine.synthesizeFromIpa(
            ipa,
            voiceIdx >>> 0,
            speed,
            textLen >>> 0,
        );
        postMessage({ type: 'audio', samples, sampleRate: 24000 });
        postMessage({ type: 'done', totalSteps: 1 });
    } catch (e) {
        postMessage({ type: 'error', error: e.message || String(e) });
    }
}

self.onmessage = async (e) => {
    const msg = e.data;
    switch (msg.type) {
        case 'load':
            await handleLoad(msg.modelUrl, msg.voicesUrl, msg.wasmBaseUrl);
            break;
        case 'generate':
            await handleGenerate(msg.ipa, msg.voiceIdx, msg.speed, msg.textLen);
            break;
    }
};
