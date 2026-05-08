#!/usr/bin/env node
/**
 * Capture a Chrome DevTools trace of TADA inference for GPU profiling.
 * Uses CDP Tracing API to capture GPU + user timing + devtools timeline.
 * Output: /tmp/wasm-perf-fresh.json
 *
 * Usage: node scripts/measure_tada_trace.mjs
 */
import { chromium } from '/Users/tc/node_modules/playwright/index.mjs';
import { writeFileSync, existsSync, readFileSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const __dir = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dir, '..');
const URL = 'http://localhost:8081';
const TIMINGS_FILE = '/tmp/wasm-timings.jsonl';
const TRACE_OUTPUT = '/tmp/wasm-perf-fresh.json';

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

const initialLines = existsSync(TIMINGS_FILE)
    ? readFileSync(TIMINGS_FILE, 'utf8').split('\n').filter(Boolean).length
    : 0;
console.log(`[script] initial timings lines: ${initialLines}`);

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
const client = await page.context().newCDPSession(page);

page.on('console', m => {
    const t = m.text();
    if (m.type() === 'error' || t.includes('[tada') || t.includes('Ready') || t.includes('timing') || t.includes('step')) {
        console.log(`[browser/${m.type()}]`, t.slice(0, 300));
    }
});
page.on('pageerror', e => console.error('[pageerror]', e.message));

console.log('[script] loading page...');
await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

// Switch to TADA tab
await page.click('input[name="model"][value="tada"]');
await sleep(500);

// Probe WebGPU
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
    process.exit(1);
}

// Wait for TADA model ready
console.log('[script] waiting for TADA model ready (up to 300s)...');
try {
    await page.waitForFunction(
        () => document.getElementById('statusText')?.textContent?.includes('Ready'),
        { timeout: 300000 }
    );
} catch (_) {
    console.log('[script] statusText timeout — checking voice buttons...');
}

// Wait for ex01 button enabled
await page.waitForFunction(() => {
    const btn = document.querySelector('#tadaMatrixGrid .voice-btn[data-speaker="ex01"]');
    return btn && !btn.disabled;
}, { timeout: 90000 });

console.log('[script] TADA ready — starting CDP trace...');

// Start CDP tracing with GPU + user timing categories
const traceEvents = [];
client.on('Tracing.dataCollected', (e) => {
    if (Array.isArray(e.value)) {
        traceEvents.push(...e.value);
    }
});

// Try broad category set to capture GPU work
await client.send('Tracing.start', {
    categories: [
        'devtools.timeline',
        'disabled-by-default-devtools.timeline',
        'disabled-by-default-devtools.timeline.frame',
        'disabled-by-default-devtools.timeline.layers',
        'disabled-by-default-v8.cpu_profiler',
        'blink.user_timing',
        'gpu',
        'disabled-by-default-gpu',
        'disabled-by-default-gpu.device',
    ].join(','),
    options: 'sampling-frequency=1000',
    transferMode: 'ReportEvents',
    streamFormat: 'json',
});

console.log('[script] clicking ex01 for inference...');
await page.click('#tadaMatrixGrid .voice-btn[data-speaker="ex01"]');
console.log('[script] ex01 clicked, waiting for inference...');

// Wait for new timing line (up to 600s for model load + inference)
const deadline = Date.now() + 600000;
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
    await sleep(3000);
}

// Brief pause to let GPU events flush
await sleep(2000);

console.log('[script] stopping trace...');
const traceComplete = new Promise(r => client.once('Tracing.tracingComplete', r));
await client.send('Tracing.end');
await traceComplete;

console.log(`[script] trace collected: ${traceEvents.length} events`);

// Write trace to file
const traceData = { traceEvents };
writeFileSync(TRACE_OUTPUT, JSON.stringify(traceData));

const traceSize = readFileSync(TRACE_OUTPUT).length;
console.log(`[script] trace written: ${TRACE_OUTPUT} (${(traceSize / 1024 / 1024).toFixed(1)} MB)`);

await browser.close();

if (newLines <= 0) {
    console.error('[script] warning — timing file had no new lines; inference may have failed');
    process.exit(1);
}

const finalLines = readFileSync(TIMINGS_FILE, 'utf8').split('\n').filter(Boolean);
const lastLine = finalLines[finalLines.length - 1];
console.log('[result] last timing:', lastLine);
console.log('[done] trace path:', TRACE_OUTPUT);
process.exit(0);
