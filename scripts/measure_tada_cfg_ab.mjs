#!/usr/bin/env node
/**
 * TADA CFG A/B measurement script.
 * Runs two headless inference sessions:
 *   Run A: cfg_scale=1.6 → /tmp/tada-cfg16-fox.wav
 *   Run B: cfg_scale=1.0 → /tmp/tada-cfg10-fox.wav
 *
 * Appends rows to docs/tada/results.md and lab-notebook entry.
 */
import { chromium } from '/Users/tc/node_modules/playwright/index.mjs';
import { readFileSync, existsSync, appendFileSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const __dir = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dir, '..');
const URL = 'http://localhost:8081';
const TIMINGS_FILE = '/tmp/wasm-timings.jsonl';

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

async function runMeasurement(cfgScale, wavCaptureName) {
    console.log(`\n=== Run: cfg_scale=${cfgScale}, capture=${wavCaptureName} ===`);

    // Snapshot timings count before run
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

    // Inject page-side patch: intercept Worker.postMessage to inject wavCaptureName into generate messages
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
        return null;
    }

    // Set cfg slider value BEFORE clicking (it's read at click time)
    await page.evaluate((cfg) => {
        const slider = document.getElementById('cfgSlider');
        if (slider) { slider.value = String(cfg); }
    }, cfgScale);

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

    // Re-set cfg slider (may have been changed during load)
    await page.evaluate((cfg) => {
        const slider = document.getElementById('cfgSlider');
        if (slider) { slider.value = String(cfg); }
    }, cfgScale);

    console.log(`[script] TADA ready, cfg_scale=${cfgScale}, clicking ex01...`);
    await page.click('#tadaMatrixGrid .voice-btn[data-speaker="ex01"]');
    console.log('[script] ex01 clicked, waiting for inference...');

    // Wait for new timing line (up to 300s)
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

    // Also wait for audio file if timing landed
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

console.log('=== TADA CFG A/B measurement ===');
console.log('Run A: cfg_scale=1.6 (production default)');
console.log('Run B: cfg_scale=1.0 (CFG disabled)');

const timingA = await runMeasurement(1.6, 'tada-cfg16-fox.wav');
if (!timingA) { console.error('Run A failed'); process.exit(1); }

const timingB = await runMeasurement(1.0, 'tada-cfg10-fox.wav');
if (!timingB) { console.error('Run B failed'); process.exit(1); }

// ---- Compute metrics ----
const genA_s = (timingA.gen_ms / 1000).toFixed(2);
const decA_s = (timingA.decode_ms / 1000).toFixed(2);
const audA_s = (timingA.audio_duration_ms / 1000).toFixed(2);
const rtfA = timingA.rtf.toFixed(2);

const genB_s = (timingB.gen_ms / 1000).toFixed(2);
const decB_s = (timingB.decode_ms / 1000).toFixed(2);
const audB_s = (timingB.audio_duration_ms / 1000).toFixed(2);
const rtfB = timingB.rtf.toFixed(2);

const speedupRatio = (timingB.rtf / timingA.rtf).toFixed(3);
const pctChange = (((timingB.rtf - timingA.rtf) / timingA.rtf) * 100).toFixed(1);
const cfgHypothesis = timingB.rtf < timingA.rtf
    ? `CONFIRMED — CFG-off is ${Math.abs(pctChange)}% faster`
    : `REFUTED — CFG-off is ${Math.abs(pctChange)}% slower (gap is something else)`;

console.log('\n=== Results ===');
console.log(`cfg=1.6: gen=${genA_s}s  decode=${decA_s}s  audio=${audA_s}s  RTF=${rtfA}x`);
console.log(`cfg=1.0: gen=${genB_s}s  decode=${decB_s}s  audio=${audB_s}s  RTF=${rtfB}x`);
console.log(`speedup ratio (cfg10/cfg16): ${speedupRatio}  (${pctChange}%)`);
console.log(`CFG hypothesis: ${cfgHypothesis}`);

// ---- Audio file sizes ----
const sizeA = existsSync('/tmp/tada-cfg16-fox.wav') ? readFileSync('/tmp/tada-cfg16-fox.wav').length : 0;
const sizeB = existsSync('/tmp/tada-cfg10-fox.wav') ? readFileSync('/tmp/tada-cfg10-fox.wav').length : 0;
console.log(`audio files: cfg16=${sizeA} bytes, cfg10=${sizeB} bytes`);

// ---- Append rows to results.md ----
const resultsPath = join(REPO_ROOT, 'docs/tada/results.md');
const rowA = `| wasm-cfg16 | burn+candle | wgpu+cpu | Var-C simd=on cfg=1.6 tasks_max=512 | 1.3G | fox | — | ${genA_s} | ${decA_s} | ${audA_s} | ${rtfA}x | tada-cfg16-fox.wav |`;
const rowB = `| wasm-cfg10 | burn+candle | wgpu+cpu | Var-C simd=on cfg=1.0 tasks_max=512 | 1.3G | fox | — | ${genB_s} | ${decB_s} | ${audB_s} | ${rtfB}x | tada-cfg10-fox.wav |`;

appendFileSync(resultsPath, '\n' + rowA + '\n' + rowB + '\n');
console.log('\n[script] Appended rows to docs/tada/results.md');

// ---- Lab-notebook entry ----
const notebookPath = join(REPO_ROOT, 'docs/tada/lab-notebook.md');
const today = new Date().toISOString().slice(0, 10);
// Get commit sha
const { execSync } = await import('node:child_process');
let sha = 'unknown';
try { sha = execSync('git -C ' + REPO_ROOT + ' rev-parse --short HEAD').toString().trim(); } catch {}

const entry = `
## ${today} — wasm-cfg-ab — ${sha}
- Method: Playwright headless TADA-only, fresh session each (cfg slider set before ex01 click)
- Run A (cfg=1.6): gen ${genA_s}s, decode ${decA_s}s, audio ${audA_s}s, RTF ${rtfA}x → tada-cfg16-fox.wav (${sizeA} bytes)
- Run B (cfg=1.0): gen ${genB_s}s, decode ${decB_s}s, audio ${audB_s}s, RTF ${rtfB}x → tada-cfg10-fox.wav (${sizeB} bytes)
- Speedup: cfg10_rtf / cfg16_rtf = ${speedupRatio} (${pctChange}% change)
- CFG hypothesis: ${cfgHypothesis}
- QUALITY: UNVERIFIED — pending user audition. Both files in /tmp/. User listens before any commit lands cfg=1.0 as production default.
`;

appendFileSync(notebookPath, entry);
console.log('[script] Appended lab-notebook entry');

process.exit(0);
