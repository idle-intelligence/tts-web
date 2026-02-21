const HF_BASE = 'https://huggingface.co/idle-intelligence/pocket-tts-int8/resolve/main';
const VOICE_BASE = 'https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main';

let model = null;
let tokenizer = null;
let voiceIndices = {};   // name → voice_index
let activeVoiceIndex = -1;

function post(type, data = {}, transferables = []) {
    self.postMessage({ type, ...data }, transferables);
}

// ---- Fetch with Cache API + progress ----
const CACHE_NAME = 'tts-model-v1';

async function cachedFetch(url, label) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(url);
    if (cached) {
        post('status', { text: `${label} (cached)` });
        return await cached.arrayBuffer();
    }
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
    const contentLength = parseInt(resp.headers.get('Content-Length') || '0', 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        post('status', { text: label, progress: { loaded, total: contentLength } });
    }
    const buf = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { buf.set(chunk, offset); offset += chunk.byteLength; }
    try {
        await cache.put(url, new Response(buf.buffer, { headers: { 'Content-Type': 'application/octet-stream' } }));
    } catch (e) { console.warn('[worker] cache error:', e); }
    return buf.buffer;
}

// ---- Minimal protobuf decoder for sentencepiece .model files ----
function decodeSentencepieceModel(buffer) {
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    let pos = 0;

    function readVarint() {
        let result = 0, shift = 0;
        while (pos < buffer.length) {
            const b = buffer[pos++];
            result |= (b & 0x7f) << shift;
            shift += 7;
            if ((b & 0x80) === 0) return result;
        }
        return result;
    }

    function readBytes(n) {
        const data = buffer.slice(pos, pos + n);
        pos += n;
        return data;
    }

    function decodePiece(data) {
        let pPos = 0, piece = '', score = 0, type = 1;
        const pView = new DataView(data.buffer, data.byteOffset, data.byteLength);
        while (pPos < data.length) {
            const key = readVarIntFrom(data, pPos);
            pPos = key.pos;
            const fieldNum = key.val >>> 3;
            const wireType = key.val & 0x7;
            if (fieldNum === 1 && wireType === 2) {
                const len = readVarIntFrom(data, pPos);
                pPos = len.pos;
                piece = new TextDecoder().decode(data.slice(pPos, pPos + len.val));
                pPos += len.val;
            } else if (fieldNum === 2 && wireType === 5) {
                score = pView.getFloat32(pPos, true);
                pPos += 4;
            } else if (fieldNum === 3 && wireType === 0) {
                const v = readVarIntFrom(data, pPos);
                type = v.val;
                pPos = v.pos;
            } else {
                if (wireType === 0) { const v = readVarIntFrom(data, pPos); pPos = v.pos; }
                else if (wireType === 1) { pPos += 8; }
                else if (wireType === 2) { const len = readVarIntFrom(data, pPos); pPos = len.pos + len.val; }
                else if (wireType === 5) { pPos += 4; }
                else break;
            }
        }
        return { piece, score, type };
    }

    function readVarIntFrom(buf, p) {
        let result = 0, shift = 0;
        while (p < buf.length) {
            const b = buf[p++];
            result |= (b & 0x7f) << shift;
            shift += 7;
            if ((b & 0x80) === 0) return { val: result, pos: p };
        }
        return { val: result, pos: p };
    }

    const pieces = [];
    while (pos < buffer.length) {
        const key = readVarint();
        const fieldNum = key >>> 3;
        const wireType = key & 0x7;
        if (fieldNum === 1 && wireType === 2) {
            const len = readVarint();
            const data = readBytes(len);
            const p = decodePiece(data);
            pieces.push(p);
        } else {
            if (wireType === 0) { readVarint(); }
            else if (wireType === 1) { pos += 8; }
            else if (wireType === 2) { const len = readVarint(); pos += len; }
            else if (wireType === 5) { pos += 4; }
            else break;
        }
    }
    return pieces;
}

// ---- Unigram tokenizer (Viterbi) ----
class UnigramTokenizer {
    constructor(pieces) {
        this.pieces = pieces;
        this.vocab = new Map();
        this.unkId = 0;
        for (let i = 0; i < pieces.length; i++) {
            const p = pieces[i];
            if (p.type === 2) this.unkId = i;
            if (p.type === 1 || p.type === 4) {
                this.vocab.set(p.piece, { id: i, score: p.score });
            }
            if (p.type === 6) {
                this.vocab.set(p.piece, { id: i, score: p.score });
            }
        }
    }

    encode(text) {
        const normalized = '\u2581' + text.replace(/ /g, '\u2581');
        return this._viterbi(normalized);
    }

    _viterbi(text) {
        const n = text.length;
        const best = new Array(n + 1);
        best[0] = { score: 0, len: 0, id: -1 };
        for (let i = 1; i <= n; i++) {
            best[i] = { score: -Infinity, len: 0, id: -1 };
        }

        for (let i = 0; i < n; i++) {
            if (best[i].score === -Infinity) continue;
            for (let len = 1; len <= n - i && len <= 64; len++) {
                const sub = text.substring(i, i + len);
                const entry = this.vocab.get(sub);
                if (entry) {
                    const newScore = best[i].score + entry.score;
                    if (newScore > best[i + len].score) {
                        best[i + len] = { score: newScore, len: len, id: entry.id };
                    }
                }
            }
            if (best[i + 1].score === -Infinity) {
                const ch = text.charCodeAt(i);
                const byteStr = `<0x${ch.toString(16).toUpperCase().padStart(2, '0')}>`;
                const byteEntry = this.vocab.get(byteStr);
                const fallbackId = byteEntry ? byteEntry.id : this.unkId;
                const fallbackScore = byteEntry ? byteEntry.score : -100;
                best[i + 1] = { score: best[i].score + fallbackScore, len: 1, id: fallbackId };
            }
        }

        const ids = [];
        let p = n;
        while (p > 0) {
            ids.push(best[p].id);
            p -= best[p].len;
        }
        ids.reverse();
        return new Uint32Array(ids);
    }
}

// ---- Handlers ----
async function handleLoad(config) {
    const base = (config.baseUrl || '').replace(/\/+$/, '');

    // 1. Import WASM
    post('status', { text: 'Loading WASM module...' });
    const wasmJsUrl = base ? (base + '/pkg/tts_wasm.js') : new URL('../pkg/tts_wasm.js', import.meta.url).href;
    const wasmBgUrl = base ? (base + '/pkg/tts_wasm_bg.wasm') : new URL('../pkg/tts_wasm_bg.wasm', import.meta.url).href;
    const wasmModule = await import(wasmJsUrl);
    await wasmModule.default(wasmBgUrl);

    // 2. Download and load tokenizer
    const tokUrl = config.tokenizerUrl || `${HF_BASE}/tokenizer.model`;
    const tokBuf = await cachedFetch(tokUrl, 'Downloading tokenizer');
    const pieces = decodeSentencepieceModel(new Uint8Array(tokBuf));
    tokenizer = new UnigramTokenizer(pieces);
    post('status', { text: `Tokenizer loaded (${pieces.length} pieces)` });

    // 3. Download and init model
    const modelUrl = config.modelUrl || `${HF_BASE}/model.safetensors`;
    const modelBuf = await cachedFetch(modelUrl, 'Downloading model');
    post('status', { text: 'Initializing model...' });
    model = new wasmModule.Model(new Uint8Array(modelBuf));

    // 4. Ready (voice loaded separately)
    const sampleRate = model.sample_rate();
    post('status', { text: 'Select a voice', ready: true });
    post('loaded', { sampleRate });
}

async function handleLoadVoice(name) {
    if (name in voiceIndices) {
        activeVoiceIndex = voiceIndices[name];
        post('voice_loaded', { name, voiceIndex: activeVoiceIndex });
        return;
    }

    const url = `${VOICE_BASE}/embeddings_v2/${name}.safetensors`;
    const voiceBuf = await cachedFetch(url, `Downloading voice: ${name}`);
    post('status', { text: `Loading voice: ${name}...` });
    const voiceIndex = model.add_voice(new Uint8Array(voiceBuf));
    voiceIndices[name] = voiceIndex;
    activeVoiceIndex = voiceIndex;
    post('status', { text: 'Ready', ready: true });
    post('voice_loaded', { name, voiceIndex });
}

async function handleGenerate(text, temperature) {
    const [processedText, framesAfterEos] = model.prepare_text(text);
    const tokenIds = tokenizer.encode(processedText);

    post('gen_start', { numTokens: tokenIds.length });

    model.start_generation(activeVoiceIndex, tokenIds, framesAfterEos, temperature);

    let step = 0;
    while (true) {
        const chunk = model.generation_step();
        if (!chunk) break;
        // Debug: log PCM stats for each step
        let min = Infinity, max = -Infinity, sum = 0;
        for (let i = 0; i < chunk.length; i++) {
            const v = chunk[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        console.log(`[gen] step=${step} len=${chunk.length} min=${min.toFixed(6)} max=${max.toFixed(6)} mean=${(sum/chunk.length).toFixed(6)}`);
        post('chunk', { data: chunk, step }, [chunk.buffer]);
        step++;
    }

    post('done', { totalSteps: step });
}

self.onmessage = async (e) => {
    const { type, ...data } = e.data;
    try {
        if (type === 'load') {
            await handleLoad(data.config || {});
        } else if (type === 'load_voice') {
            await handleLoadVoice(data.name);
        } else if (type === 'generate') {
            await handleGenerate(data.text, data.temperature || 0.7);
        }
    } catch (err) {
        post('error', { message: err.message || String(err) });
        console.error(err);
    }
};
