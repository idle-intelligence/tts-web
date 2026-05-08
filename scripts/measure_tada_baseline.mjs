#!/usr/bin/env node
/**
 * Focused TADA-only baseline measurement script.
 * Navigates to localhost:8081, switches to TADA tab, clicks ex01,
 * waits for inference to complete, confirms /tmp/wasm-timings.jsonl got a row.
 *
 * Does NOT test Pocket TTS or KittenTTS — minimizes memory pressure.
 * Uses the same WebGPU launch flags + Cache stub as test_demo_e2e.mjs.
 */
import { chromium } from '/Users/tc/node_modules/playwright/index.mjs';
import { readFileSync, existsSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';

const URL = 'http://localhost:8081';
const TIMINGS_FILE = '/tmp/wasm-timings.jsonl';

// Snapshot pre-existing timings count so we can wait for a NEW one
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

// Inject Cache stub for tada-worker.js (headless Chromium fails to cache 1.3GB GGUF)
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
await ctx.route('**/tada-worker.js', async (route) => {
  const response = await route.fetch();
  const body = await response.text();
  console.log(`[script] worker JS intercepted (${body.length} bytes); injecting cache stub`);
  route.fulfill({ response, body: CACHE_PATCH + body });
});

const page = await ctx.newPage();
page.on('console', m => {
  const t = m.text();
  if (m.type() === 'error' || t.includes('[tada') || t.includes('Ready') || t.includes('token') || t.includes('timing')) {
    console.log(`[browser/${m.type()}]`, t.slice(0, 300));
  }
});
page.on('pageerror', e => console.error('[pageerror]', e.message));

console.log('[script] loading page...');
await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

// Switch to TADA tab
console.log('[script] switching to TADA tab...');
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
  console.error(`[script] WebGPU not functional (${webGPUStatus}) — cannot run TADA inference in headless. Exiting.`);
  await browser.close();
  process.exit(1);
}

// Wait for TADA model ready (statusText includes "Ready")
console.log('[script] waiting for TADA model ready (up to 300s for 1.3GB model download)...');
try {
  await page.waitForFunction(
    () => document.getElementById('statusText')?.textContent?.includes('Ready'),
    { timeout: 300000 }
  );
} catch (e) {
  // Some builds may use different status text; try also waiting for voice buttons enabled
  console.log('[script] statusText "Ready" timeout — checking voice buttons...');
}

// Ensure ex01 button is present and enabled (wait up to 90s from here)
console.log('[script] waiting for ex01 voice button to be enabled...');
await page.waitForFunction(() => {
  const btn = document.querySelector('#tadaMatrixGrid .voice-btn[data-speaker="ex01"]');
  return btn && !btn.disabled;
}, { timeout: 90000 });

console.log('[script] TADA model ready, clicking ex01...');

// Click ex01 speaker button — auto-generates with DEMO_TEXT, fires /timings on completion
await page.click('#tadaMatrixGrid .voice-btn[data-speaker="ex01"]');
console.log('[script] ex01 clicked, waiting for inference to complete...');

// Wait for /tmp/wasm-timings.jsonl to gain a NEW line (poll up to 300s for full inference)
const deadline = Date.now() + 300000;
let newLines = 0;
while (Date.now() < deadline) {
  const content = existsSync(TIMINGS_FILE)
    ? readFileSync(TIMINGS_FILE, 'utf8').split('\n').filter(Boolean)
    : [];
  newLines = content.length - initialLines;
  if (newLines > 0) {
    console.log(`[script] new timing line landed (${newLines} added, total ${content.length})`);
    break;
  }
  await sleep(2000);
}

await browser.close();

if (newLines <= 0) {
  console.error('[script] timed out waiting for timing line — inference may have failed or hung');
  process.exit(1);
}

// Print the latest line for parsing
const finalLines = readFileSync(TIMINGS_FILE, 'utf8').split('\n').filter(Boolean);
console.log('[result] last line:', finalLines[finalLines.length - 1]);
process.exit(0);
