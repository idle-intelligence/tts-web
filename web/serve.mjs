#!/usr/bin/env node
/**
 * Dev server for TTS browser demo.
 *
 * Serves the web app and WASM pkg.
 *
 * Usage: node web/serve.mjs [--port 8081]
 */

import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(fileURLToPath(import.meta.url), "../..");
const PORT = parseInt(process.argv.find((_, i, a) => a[i - 1] === "--port") ?? "8081");

const MIME = {
    ".html": "text/html",
    ".js":   "text/javascript",
    ".mjs":  "text/javascript",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".wav":  "audio/wav",
    ".css":  "text/css",
    ".bin":  "application/octet-stream",
    ".safetensors": "application/octet-stream",
    ".model": "application/octet-stream",
    ".gguf":  "application/octet-stream",
};

const server = createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const pathname = decodeURIComponent(url.pathname);

    // CORS headers
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "credentialless");

    // Route to file
    let filePath;
    if (pathname === "/" || pathname === "/index.html") {
        filePath = join(ROOT, "web/index.html");
    } else if (pathname.startsWith("/tada-pkg/")) {
        // TADA WASM build output from crates/tada-wasm/pkg/
        filePath = join(ROOT, "crates/tada-wasm/pkg", pathname.slice("/tada-pkg/".length));
    } else if (pathname.startsWith("/pkg/")) {
        // WASM build output from crates/tts-wasm/pkg/
        filePath = join(ROOT, "crates/tts-wasm", pathname);
    } else if (pathname.startsWith("/voices/")) {
        // Voice prompts from repo root voices/
        filePath = join(ROOT, pathname);
    } else if (pathname.endsWith(".gguf") || pathname === "/tokenizer.model" || pathname === "/tokenizer.json") {
        // Model files served from repo root
        filePath = join(ROOT, pathname);
    } else {
        filePath = join(ROOT, "web", pathname);
    }

    if (!existsSync(filePath) || !statSync(filePath).isFile()) {
        res.writeHead(404);
        res.end("Not found: " + pathname);
        return;
    }

    const ext = extname(filePath);
    const mime = MIME[ext] ?? "application/octet-stream";
    const stat = statSync(filePath);

    // Support range requests for large files
    const range = req.headers.range;
    if (range && stat.size > 1_000_000) {
        const match = range.match(/bytes=(\d+)-(\d*)/);
        if (match) {
            const start = parseInt(match[1]);
            const end = match[2] ? parseInt(match[2]) : stat.size - 1;
            res.writeHead(206, {
                "Content-Type": mime,
                "Content-Range": `bytes ${start}-${end}/${stat.size}`,
                "Content-Length": end - start + 1,
                "Accept-Ranges": "bytes",
            });
            createReadStream(filePath, { start, end }).pipe(res);
            return;
        }
    }

    res.writeHead(200, {
        "Content-Type": mime,
        "Content-Length": stat.size,
        "Accept-Ranges": "bytes",
    });
    createReadStream(filePath).pipe(res);
});

server.listen(PORT, "0.0.0.0", () => {
    console.log(`\nTTS dev server running:`);
    console.log(`  Local:   http://localhost:${PORT}`);
    console.log(`\nModel weights fetched from HuggingFace at runtime.\n`);
});
