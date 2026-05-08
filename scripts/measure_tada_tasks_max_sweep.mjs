#!/usr/bin/env node
/**
 * TADA tasks_max sweep script.
 * For each tasks_max value in {256, 512, 1024, 2048}:
 *   1. Edit crates/tada-wasm/src/lib.rs to set tasks_max=N
 *   2. Rebuild WASM (wasm-pack)
 *   3. Restart dev server
 *   4. Run headless measurement
 *   5. Append row to docs/tada/results.md
 *
 * After sweep, restores tasks_max=512 in source.
 */
import { chromium } from '/Users/tc/node_modules/playwright/index.mjs';
import { readFileSync, existsSync, appendFileSync, writeFileSync, truncateSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';
import { execSync, spawn } from 'node:child_process';

const __dir = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dir, '..');
const URL = 'http://localhost:8081';
const TIMINGS_FILE = '/tmp/wasm-timings.jsonl';
const LIB_RS = join(REPO_ROOT, 'crates/tada-wasm/src/lib.rs');

const CACHE_PATCH = `
(function() {
  const noop = async () => undefined;
  const stubCache = { match: async () => undefined, put: noop, delete: noop, keys: async () => [], add: noop, addAll: noop };
  const _origOpen = caches.open.bind(caches);
  caches.open = async (name) => {
    if (name && name.startsWith('tada-')) return stubCache;
    return _origOpen(name);
  };
})();
`;

function setTasksMax(n) {
    const src = readFileSync(LIB_RS, 'utf8');
    const updated = src.replace(/tasks_max: \d+/, `tasks_max: ${n}`);
    if (!updated.includes(`tasks_max: ${n}`)) {
        throw new Error(`Failed to set tasks_max=${n} in lib.rs`);
    }
    writeFileSync(LIB_RS, updated);
    // Verify
    const check = readFileSync(LIB_RS, 'utf8').match(/tasks_max: (\d+)/);
    console.log(`[sweep] tasks_max in source: ${check ? check[1] : 'NOT FOUND'}`);
}

function buildWasm() {
    console.log('[sweep] Building WASM...');
    const start = Date.now();
    execSync(
        'wasm-pack build crates/tada-wasm --target web --release -- --features wasm --no-default-features',
        { cwd: REPO_ROOT, stdio: 'inherit', timeout: 600000 }
    );
    const elapsed = ((Date.now() - start) / 1000).toFixed(0);
    console.log(`[sweep] WASM build done in ${elapsed}s`);
}

function restartServer() {
    console.log('[sweep] Restarting dev server...');
    try {
        execSync('pkill -f "web/serve.mjs"', { stdio: 'ignore' });
    } catch (_) { /* process may not exist */ }
    // Give it a moment to die
    execSync('sleep 2');
    const child = spawn('node', ['web/serve.mjs'], {
        cwd: REPO_ROOT,
        detached: true,
        stdio: 'ignore',
    });
    child.unref();
    // Give it a moment to start
    execSync('sleep 3');
    console.log('[sweep] Dev server restarted');
}

async function runMeasurement(tasksMax, wavCaptureName) {
    console.log(`\n=== Measure: tasks_max=${tasksMax}, capture=${wavCaptureName} ===`);

    // Clear timings file
    try { truncateSync(TIMINGS_FILE, 0); } catch (_) {}
    const initialLines = 0;

    const browser = await chromium.launch({
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--enable-unsafe-webgpu',
            '--use-angle=metal',
            '--enable-features=Vulkan',
            '--enable-webgpu-developer-features',
        ],
    });

    const ctx = await browser.newContext();

    await ctx.route('**/tada-worker.js', async (route) => {
        const response = await route.fetch();
        const body = await response.text();
        console.log(`[script] worker JS intercepted (${body.length} bytes); injecting cache stub`);
        route.fulfill({ response, body: CACHE_PATCH + body });
    });

    const page = await ctx.newPage();

    await page.addInitScript(({ captureName }) => {
        const origPostMessage = Worker.prototype.postMessage;
        Worker.prototype.postMessage = function(msg, ...rest) {
            if (msg && msg.type === 'generate') {
                msg = { ...msg, wavCaptureName: captureName };
            }
            return origPostMessage.call(this, msg, ...rest);
        };
    }, { captureName: wavCaptureName });

    page.on('console', m => {
        const t = m.text();
        if (m.type() === 'error' || t.includes('[tada') || t.includes('Ready') || t.includes('timing')) {
            console.log(`[browser/${m.type()}]`, t.slice(0, 300));
        }
    });
    page.on('pageerror', e => console.error('[pageerror]', e.message));

    console.log('[script] loading page...');
    await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

    await page.click('input[name="model"][value="tada"]');
    await sleep(500);

    const webGPUStatus = await page.evaluate(async () => {
        if (!navigator.gpu) return 'no_gpu_object';
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return 'adapter_null';
            const device = await adapter.requestDevice();
            return device ? 'ok' : 'device_null';
        } catch (e) {
            return 'error:' + e.message;
        }
    });
    console.log(`[script] WebGPU status: ${webGPUStatus}`);

    if (webGPUStatus !== 'ok') {
        console.error(`[script] WebGPU not functional (${webGPUStatus}) — aborting`);
        await browser.close();
        return null;
    }

    await page.evaluate((cfg) => {
        const slider = document.getElementById('cfgSlider');
        if (slider) { slider.value = String(cfg); }
    }, 1.6);

    console.log('[script] waiting for TADA model ready (up to 300s)...');
    try {
        await page.waitForFunction(
            () => document.getElementById('statusText')?.textContent?.includes('Ready'),
            { timeout: 300000 }
        );
    } catch (_) {
        console.log('[script] statusText timeout — checking voice buttons...');
    }

    await page.waitForFunction(() => {
        const btn = document.querySelector('#tadaMatrixGrid .voice-btn[data-speaker="ex01"]');
        return btn && !btn.disabled;
    }, { timeout: 90000 });

    await page.evaluate((cfg) => {
        const slider = document.getElementById('cfgSlider');
        if (slider) { slider.value = String(cfg); }
    }, 1.6);

    console.log(`[script] TADA ready, tasks_max=${tasksMax}, cfg=1.6, clicking ex01...`);
    await page.click('#tadaMatrixGrid .voice-btn[data-speaker="ex01"]');
    console.log('[script] ex01 clicked, waiting for inference...');

    const deadline = Date.now() + 300000;
    let newLines = 0;
    while (Date.now() < deadline) {
        const content = existsSync(TIMINGS_FILE)
            ? readFileSync(TIMINGS_FILE, 'utf8').split('\n').filter(Boolean)
            : [];
        newLines = content.length - initialLines;
        if (newLines > 0) {
            console.log(`[script] timing line landed (total ${content.length})`);
            break;
        }
        await sleep(2000);
    }

    if (newLines > 0) {
        const audioPath = '/tmp/' + wavCaptureName;
        const audioDeadline = Date.now() + 30000;
        while (Date.now() < audioDeadline) {
            if (existsSync(audioPath)) {
                const sz = readFileSync(audioPath).length;
                console.log(`[script] audio file ready: ${audioPath} (${sz} bytes)`);
                break;
            }
            await sleep(1000);
        }
    }

    await browser.close();

    if (newLines <= 0) {
        console.error('[script] timed out — inference may have failed');
        return null;
    }

    const finalLines = readFileSync(TIMINGS_FILE, 'utf8').split('\n').filter(Boolean);
    const lastLine = finalLines[finalLines.length - 1];
    console.log('[result] last timing:', lastLine);
    return JSON.parse(lastLine);
}

// ---- Main ----

const VALUES = [256, 512, 1024, 2048];
const results = {};
const failures = [];

console.log('=== TADA tasks_max sweep ===');
console.log(`Values: ${VALUES.join(', ')}`);

for (const n of VALUES) {
    console.log(`\n====== tasks_max=${n} ======`);

    // 1. Edit source
    try {
        setTasksMax(n);
    } catch (e) {
        console.error(`[sweep] Failed to set tasks_max=${n}: ${e.message}`);
        failures.push({ n, reason: 'source edit failed: ' + e.message });
        continue;
    }

    // 2. Build
    try {
        buildWasm();
    } catch (e) {
        const msg = String(e.message || e).slice(0, 500);
        console.error(`[sweep] Build failed for tasks_max=${n}: ${msg}`);
        failures.push({ n, reason: 'build failed: ' + msg });
        continue;
    }

    // 3. Restart server
    restartServer();

    // 4. Measure
    const wavName = `tada-tmax-${n}-fox.wav`;
    const timing = await runMeasurement(n, wavName);
    if (!timing) {
        console.error(`[sweep] Measurement failed for tasks_max=${n}`);
        failures.push({ n, reason: 'measurement failed (no timing line)' });
        continue;
    }

    results[n] = timing;

    // 5. Append row to results.md
    const gen_s = (timing.gen_ms / 1000).toFixed(2);
    const dec_s = (timing.decode_ms / 1000).toFixed(2);
    const aud_s = (timing.audio_duration_ms / 1000).toFixed(2);
    const rtf = timing.rtf.toFixed(2);
    const row = `| wasm-tmax-${n} | burn+candle | wgpu+cpu | Var-C simd=on cfg=1.6 tasks_max=${n} | 1.3G | fox | — | ${gen_s} | ${dec_s} | ${aud_s} | ${rtf}x | tada-tmax-${n}-fox.wav |`;
    console.log(`\n[sweep] Row: ${row}`);
}

// ---- Restore tasks_max=512 ----
console.log('\n[sweep] Restoring tasks_max=512 in source...');
setTasksMax(512);
console.log('[sweep] Source restored.');

// ---- Summary ----
console.log('\n=== Sweep Results ===');
const measured = Object.keys(results).map(Number).sort((a, b) => a - b);
for (const n of measured) {
    const t = results[n];
    console.log(`tasks_max=${n}: gen=${(t.gen_ms/1000).toFixed(2)}s decode=${(t.decode_ms/1000).toFixed(2)}s audio=${(t.audio_duration_ms/1000).toFixed(2)}s RTF=${t.rtf.toFixed(2)}x`);
}
if (failures.length) {
    console.log('\nFailures:');
    for (const f of failures) console.log(`  tasks_max=${f.n}: ${f.reason}`);
}

const minEntry = measured.reduce((best, n) => {
    const r = results[n].rtf;
    return (!best || r < results[best].rtf) ? n : best;
}, null);
if (minEntry) {
    console.log(`\nMinimum RTF: ${results[minEntry].rtf.toFixed(2)}x at tasks_max=${minEntry}`);
}

// ---- Append rows to results.md ----
const resultsPath = join(REPO_ROOT, 'docs/tada/results.md');
let resultsBlock = '\n\n## tasks-max-sweep\n\n';
resultsBlock += '| ID | Engine | Device | Model | Size | Text | Load(s) | Gen(s) | Decode(s) | Audio(s) | RTF | File |\n';
resultsBlock += '|----|--------|--------|-------|------|------|---------|--------|-----------|----------|-----|------|\n';
for (const n of measured) {
    const t = results[n];
    const gen_s = (t.gen_ms / 1000).toFixed(2);
    const dec_s = (t.decode_ms / 1000).toFixed(2);
    const aud_s = (t.audio_duration_ms / 1000).toFixed(2);
    const rtf = t.rtf.toFixed(2);
    resultsBlock += `| wasm-tmax-${n} | burn+candle | wgpu+cpu | Var-C simd=on cfg=1.6 tasks_max=${n} | 1.3G | fox | — | ${gen_s} | ${dec_s} | ${aud_s} | ${rtf}x | tada-tmax-${n}-fox.wav |\n`;
}

appendFileSync(resultsPath, resultsBlock);
console.log('\n[sweep] Appended rows to docs/tada/results.md');

// ---- Lab-notebook entry ----
const notebookPath = join(REPO_ROOT, 'docs/tada/lab-notebook.md');
const today = new Date().toISOString().slice(0, 10);
let sha = 'unknown';
try { sha = execSync('git -C ' + REPO_ROOT + ' rev-parse --short HEAD').toString().trim(); } catch {}

const lines = measured.map(n => {
    const t = results[n];
    return `- wasm-tmax-${n}: gen ${(t.gen_ms/1000).toFixed(2)}s, decode ${(t.decode_ms/1000).toFixed(2)}s, audio ${(t.audio_duration_ms/1000).toFixed(2)}s, RTF ${t.rtf.toFixed(2)}x → tada-tmax-${n}-fox.wav`;
}).join('\n');

const baselineRtf = results[512] ? results[512].rtf : null;
const speedupLine = (minEntry && baselineRtf)
    ? `- Speedup vs production (tmax=512): ${(baselineRtf / results[minEntry].rtf).toFixed(2)}x = ${(((baselineRtf - results[minEntry].rtf) / baselineRtf) * 100).toFixed(1)}% improvement at tmax=${minEntry}`
    : '- Speedup vs production: N/A (tmax=512 not measured or is the winner)';

const failureLines = failures.length
    ? `- Build failures: ${failures.map(f => `tmax=${f.n} (${f.reason})`).join('; ')}`
    : '- Build failures: none';

const entry = `
## ${today} — wasm-tasks-max-sweep — ${sha}
- Method: 4 WASM rebuilds with tasks_max ∈ {256, 512, 1024, 2048}; same Playwright session per measurement; cfg=1.6 (production default); fox phrase
${lines}
${failureLines}
- Minimum RTF: ${minEntry ? results[minEntry].rtf.toFixed(2) + 'x' : 'N/A'} at tasks_max=${minEntry || 'N/A'}
${speedupLine}
- Source restored to tasks_max=512 ✓
`;

appendFileSync(notebookPath, entry);
console.log('[sweep] Appended lab-notebook entry');

process.exit(measured.length >= 3 ? 0 : 1);
