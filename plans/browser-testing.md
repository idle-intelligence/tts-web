# Autonomous Browser Testing for TADA WASM

## Rules
- **NEVER open or automate the user's personal browser.**
- Use **Playwright's bundled headless Chromium** for all browser testing.
- The user should not need to interact with any browser window.

## Setup

```bash
# Install Playwright (if not already)
npm install -D playwright
npx playwright install chromium

# Start dev server (background)
node web/serve.mjs &
# Serves on http://localhost:8081
```

## What to Test

### 1. WASM Load + Model Init
- Navigate to `http://localhost:8081`
- Select "TADA-1B" model radio button
- Wait for status to show "Ready" (model download + WASM init + GPU warmup)
- Verify no console errors

### 2. Voice Selection
- Click a speaker button (e.g. "3 👩")
- Click a style button (e.g. "Neutral")
- Verify voice loads (status returns to "Ready")
- Verify text input is populated with a demo phrase

### 3. Audio Generation
- Click "Generate" button
- Wait for progress to show "Decoding audio..."
- Wait for `<audio>` element to appear with a valid blob URL
- Verify audio duration > 0

### 4. Parameter Controls
- Verify sliders exist: Noise, CFG, Steps
- Change slider values before generating
- Verify generation completes with different params

## Playwright Example

```javascript
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // Enable WebGPU (Chromium flag)
  // Note: headless Chromium may not support WebGPU —
  // use headful mode with Xvfb if needed, or test with --headed

  page.on('console', msg => console.log(`[browser] ${msg.text()}`));
  page.on('pageerror', err => console.error(`[browser error] ${err}`));

  await page.goto('http://localhost:8081');

  // Select TADA model
  await page.click('input[value="tada"]');

  // Wait for model to load (can take 30-60s on first run)
  await page.waitForFunction(
    () => document.querySelector('#statusText')?.textContent === 'Ready',
    { timeout: 120000 }
  );
  console.log('Model loaded');

  // Click a voice
  await page.click('[data-speaker="ex03"]');
  await page.waitForTimeout(1000);

  // Click Generate
  await page.click('#generateBtn');

  // Wait for audio output (generation can take 20-60s)
  await page.waitForSelector('audio[src]', { timeout: 120000 });
  console.log('Audio generated');

  // Check audio element
  const audioDur = await page.evaluate(() => {
    const audio = document.querySelector('audio');
    return audio ? audio.duration : 0;
  });
  console.log(`Audio duration: ${audioDur}s`);

  await browser.close();
})();
```

## WebGPU in Headless Chrome

WebGPU may not work in fully headless mode. Options:
1. Use `chromium.launch({ headless: false })` with virtual display (Xvfb on Linux)
2. Use `--headless=new` flag (newer Chrome headless supports more GPU features)
3. On macOS, headful mode with `DISPLAY` not needed — just use `headless: false`
4. For CI: skip WebGPU tests, only test WASM loading + JS logic

## File Paths

- Dev server: `web/serve.mjs` (port 8081)
- Frontend: `web/index.html`
- TADA worker: `web/tada-worker.js`
- TTS client: `web/tts-client.js`
- WASM pkg: `crates/tada-wasm/pkg/`
- Voice prompts: `voices/` and `voices/matrix/`
- Model symlink: `tada-1b-q4_0.gguf` → actual GGUF file

## Build Before Testing

```bash
# Build WASM (required after Rust changes)
wasm-pack build crates/tada-wasm --target web --release -- --features wasm --no-default-features

# JS/HTML changes don't need rebuild — just refresh
```

## What NOT to Do
- Don't open `http://localhost:8081` in the user's browser
- Don't use `open` or `xdg-open` commands
- Don't use browser automation tools that control the user's browser (Selenium with default profile, etc.)
- Don't run tests that require user interaction
