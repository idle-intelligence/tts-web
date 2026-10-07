/// End-to-end TTS generation example.
///
/// Select a language and voice by name — the GGUF model, tokenizer and voice
/// embedding are downloaded and cached automatically:
///
///   cargo run --example tts_generate -p tts-core --release -- \
///     --language italian --voice giovanni \
///     --text "Ciao, questo e un test." \
///     --output /tmp/test_tts.wav
///
/// `--model`/`--tokenizer`/`--voice` still take a local file path, overriding
/// the download for offline use:
///
///   cargo run --example tts_generate -p tts-core --release -- \
///     --model /path/to/pocket-tts-q8_0.gguf \
///     --tokenizer /path/to/tokenizer.model \
///     --text "Hello, this is a test of the text to speech system." \
///     --voice /path/to/alba.safetensors \
///     --output /tmp/test_tts.wav \
///     [--temperature 0.7]
///
/// `--list` prints the available languages and voices and exits.
///
/// `--model` is a GGUF checkpoint (loaded via `GgufTensors::from_bytes`), not a
/// safetensors file. `--tokenizer` is the matching SentencePiece `tokenizer.model`.
///
/// Writes a Float32 PCM WAV at 24000 Hz (mono).
use candle_core::{Device, Result as CResult, Tensor};
use candle_nn::VarBuilder;
use mimi_rs::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use tts_core::flow_lm::{FlowLMState, Rng};
use tts_core::generation::EosGate;
use tts_core::text_config::TextConfig;
use tts_core::tokenizer::Tokenizer;
use tts_core::tts_model::{TTSState, prepare_text_prompt};

// ---------------------------------------------------------------------------
// Model catalog — languages and voices the HF repos offer, mirroring the
// choices on the web demo (web/index.html's "pocket-multilingual" tab).
// ---------------------------------------------------------------------------

/// Full language folder names, as laid out by both
/// `idle-intelligence/pocket-tts-gguf` and the upstream Kyutai voice repo.
pub const LANGUAGES: &[&str] = &[
    "english", "french", "german", "spanish", "portuguese", "italian",
];

/// Voice names. The same 26 names exist under every language's `embeddings/`
/// folder — voice and language are orthogonal, so this is one shared list.
pub const VOICES: &[&str] = &[
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette", "eponine",
    "estelle", "eve", "fantine", "george", "giovanni", "jane", "javert", "jean", "juergen",
    "lola", "marius", "mary", "michael", "paul", "peter_yearsley", "rafael", "stuart_bell", "vera",
];

/// The voice the web demo pre-selects for each language.
pub fn default_voice_for_language(language: &str) -> &'static str {
    match language {
        "french" => "estelle",
        "german" => "juergen",
        "spanish" => "lola",
        "portuguese" => "rafael",
        "italian" => "giovanni",
        _ => "alba", // english
    }
}

const GGUF_REPO_BASE: &str = "https://huggingface.co/idle-intelligence/pocket-tts-gguf/resolve/main";
/// Voice embeddings are a KV cache tied to the exact weights that produced
/// them, so they stay pinned to the revision our GGUFs were quantized from
/// (also the one pocket-tts 3.3.0 pins) — never `main`.
const VOICE_REPO_BASE: &str =
    "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/4e1e0a3e611c51c0b4ed8174fc10f32a54644303";

pub fn model_url(language: &str) -> String {
    format!("{GGUF_REPO_BASE}/languages/{language}/pocket-tts-q8_0.gguf")
}

pub fn tokenizer_url(language: &str) -> String {
    format!("{GGUF_REPO_BASE}/languages/{language}/tokenizer.model")
}

pub fn voice_url(language: &str, voice: &str) -> String {
    format!("{VOICE_REPO_BASE}/languages/{language}/embeddings/{voice}.safetensors")
}

fn print_list() {
    println!("models: pocket-tts (kitten_generate -p kitten-core has KittenTTS)");
    println!("languages: {}", LANGUAGES.join(", "));
    println!("voices (shared across all languages): {}", VOICES.join(", "));
    println!("default voice per language:");
    for lang in LANGUAGES {
        println!("  {lang}: {}", default_voice_for_language(lang));
    }
}

// ---------------------------------------------------------------------------
// CLI arg parsing (manual, no external deps)
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub struct Args {
    /// Local GGUF path override. `None` means: download+cache by `language`.
    pub model_path: Option<String>,
    /// Local tokenizer path override. `None` means: download+cache by `language`.
    pub tokenizer_path: Option<String>,
    pub text: String,
    /// Either a local safetensors path (contains '/' or '\\', or ends in
    /// `.safetensors`) or a voice name to download+cache. `None` means: use
    /// the default voice for `language`.
    pub voice_path: Option<String>,
    pub output_path: String,
    pub temperature: f32,
    pub language: String,
    /// When set, `--model` is read as an unquantized safetensors checkpoint
    /// (via VarBuilder) instead of a Q8_0 GGUF, for numerical parity work
    /// against the official PyTorch weights.
    pub safetensors: bool,
    pub list: bool,
}

/// A bare voice argument is a path override when it looks like one (has a
/// path separator or the `.safetensors` extension); otherwise it's a name to
/// resolve against `VOICES` and download.
fn looks_like_path(value: &str) -> bool {
    value.contains('/') || value.contains('\\') || value.ends_with(".safetensors")
}

/// Parse CLI args. `args` is the full argv, i.e. `args[0]` is the program name
/// (matching `std::env::args()`), so this can be exercised directly in tests.
pub fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut model_path = None;
    let mut tokenizer_path = None;
    let mut text = None;
    let mut voice_path = None;
    let mut output_path = None;
    let mut temperature = 0.7f32;
    let mut language = "english".to_string();
    let mut safetensors = false;
    let mut list = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args.get(i).ok_or("--model requires a value")?.clone());
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = Some(args.get(i).ok_or("--tokenizer requires a value")?.clone());
            }
            "--text" => {
                i += 1;
                text = Some(args.get(i).ok_or("--text requires a value")?.clone());
            }
            "--voice" => {
                i += 1;
                voice_path = Some(args.get(i).ok_or("--voice requires a value")?.clone());
            }
            "--output" => {
                i += 1;
                output_path = Some(args.get(i).ok_or("--output requires a value")?.clone());
            }
            "--temperature" => {
                i += 1;
                let raw = args.get(i).ok_or("--temperature requires a value")?;
                temperature = raw
                    .parse()
                    .map_err(|_| format!("--temperature must be a float, got {raw:?}"))?;
            }
            "--language" => {
                i += 1;
                language = args.get(i).ok_or("--language requires a value")?.clone();
            }
            "--safetensors" => {
                safetensors = true;
            }
            "--list" => {
                list = true;
            }
            other => {
                return Err(format!("Unknown arg: {other}"));
            }
        }
        i += 1;
    }

    if list {
        return Ok(Args {
            model_path,
            tokenizer_path,
            text: text.unwrap_or_default(),
            voice_path,
            output_path: output_path.unwrap_or_else(|| "/tmp/test_tts.wav".to_string()),
            temperature,
            language,
            safetensors,
            list,
        });
    }

    if !LANGUAGES.contains(&language.as_str()) {
        return Err(format!(
            "unknown language {language:?}; valid languages: {}",
            LANGUAGES.join(", ")
        ));
    }

    if let Some(ref v) = voice_path
        && !looks_like_path(v)
        && !VOICES.contains(&v.as_str())
    {
        return Err(format!(
            "unknown voice {v:?}; valid voices: {}",
            VOICES.join(", ")
        ));
    }

    Ok(Args {
        model_path,
        tokenizer_path,
        text: text.ok_or("--text is required")?,
        voice_path,
        output_path: output_path.unwrap_or_else(|| "/tmp/test_tts.wav".to_string()),
        temperature,
        language,
        safetensors,
        list,
    })
}

// ---------------------------------------------------------------------------
// Download + cache
// ---------------------------------------------------------------------------

/// `~/.cache/tts-web` by default, overridable for tests via `TTS_WEB_CACHE_DIR`.
fn cache_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("TTS_WEB_CACHE_DIR") {
        return std::path::PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home).join(".cache").join("tts-web")
}

/// Download `url` to `cache_path` if not already cached, verifying the
/// downloaded size against the server's `Content-Length`, then return the
/// file's bytes (from cache or freshly downloaded).
fn cached_download(url: &str, cache_path: &std::path::Path) -> Result<Vec<u8>, String> {
    if cache_path.exists() {
        eprintln!("  cached: {}", cache_path.display());
        return std::fs::read(cache_path).map_err(|e| format!("failed to read cache {}: {e}", cache_path.display()));
    }

    eprintln!("  downloading {url}...");
    use std::io::Read;
    let agent = ureq::AgentBuilder::new().build();
    let resp = agent
        .get(url)
        .set("User-Agent", "tts-web/0.1 (+https://github.com/idle-intelligence/tts-web)")
        .call()
        .map_err(|e| format!("download failed for {url}: {e}"))?;

    let expected_len: Option<usize> = resp
        .header("Content-Length")
        .and_then(|s| s.parse().ok());

    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("failed reading response body for {url}: {e}"))?;

    if let Some(expected) = expected_len
        && bytes.len() != expected
    {
        return Err(format!(
            "size mismatch downloading {url}: expected {expected} bytes, got {}",
            bytes.len()
        ));
    }
    if bytes.is_empty() {
        return Err(format!("downloaded 0 bytes from {url}"));
    }

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("failed to create cache dir {}: {e}", parent.display()))?;
    }
    let tmp_path = cache_path.with_extension("tmp");
    std::fs::write(&tmp_path, &bytes).map_err(|e| format!("failed to write cache file: {e}"))?;
    std::fs::rename(&tmp_path, cache_path).map_err(|e| format!("failed to finalize cache file: {e}"))?;
    eprintln!("  cached to {} ({} bytes)", cache_path.display(), bytes.len());

    Ok(bytes)
}

fn resolve_model_bytes(args: &Args) -> Result<Vec<u8>, String> {
    if let Some(ref path) = args.model_path {
        return std::fs::read(path).map_err(|e| format!("failed to read model {path}: {e}"));
    }
    let url = model_url(&args.language);
    let cache_path = cache_dir().join("pocket-tts").join(&args.language).join("pocket-tts-q8_0.gguf");
    cached_download(&url, &cache_path)
}

fn resolve_tokenizer_bytes(args: &Args) -> Result<Vec<u8>, String> {
    if let Some(ref path) = args.tokenizer_path {
        return std::fs::read(path).map_err(|e| format!("failed to read tokenizer {path}: {e}"));
    }
    let url = tokenizer_url(&args.language);
    let cache_path = cache_dir().join("pocket-tts").join(&args.language).join("tokenizer.model");
    cached_download(&url, &cache_path)
}

/// Returns `None` when no voice is wanted at all — never the case today since
/// a default is always picked, but kept for symmetry with the file-based path.
fn resolve_voice_bytes(args: &Args) -> Result<Vec<u8>, String> {
    let raw = args.voice_path.clone().unwrap_or_else(|| default_voice_for_language(&args.language).to_string());
    if looks_like_path(&raw) {
        return std::fs::read(&raw).map_err(|e| format!("failed to read voice {raw}: {e}"));
    }
    let url = voice_url(&args.language, &raw);
    let cache_path = cache_dir().join("pocket-tts").join(&args.language).join("voices").join(format!("{raw}.safetensors"));
    cached_download(&url, &cache_path)
}

// ---------------------------------------------------------------------------
// RNG using rand + rand_distr (already in tts-core deps)
// ---------------------------------------------------------------------------

struct SimpleRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl SimpleRng {
    fn new(temperature: f32) -> Self {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        Self { inner: rng, distr }
    }
}

impl Rng for SimpleRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

// ---------------------------------------------------------------------------
// WAV writer (Float32 PCM, format type 3, mono, 24000 Hz)
// ---------------------------------------------------------------------------

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    let num_samples = samples.len() as u32;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size = num_samples * 4; // float32 = 4 bytes each
    let chunk_size = 36 + data_size;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // fmt chunk (18 bytes for float32 PCM: standard 16 + 2-byte cbSize = 0)
    f.write_all(b"fmt ")?;
    f.write_all(&18u32.to_le_bytes())?;    // chunk size (18 for float32 format)
    f.write_all(&3u16.to_le_bytes())?;     // audio format: IEEE_FLOAT = 3
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;
    f.write_all(&0u16.to_le_bytes())?;     // cbSize = 0

    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor statistics helper for logging
// ---------------------------------------------------------------------------

fn log_tensor_stats(label: &str, data: &[f32]) {
    if data.is_empty() {
        eprintln!("[{label}] empty");
        return;
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();
    eprintln!(
        "[{label}] len={} min={:.4} max={:.4} mean={:.4} nan={nan_count} inf={inf_count}",
        data.len(), min, max, mean
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Result of running generation end to end, without writing a WAV file —
/// used by `run()` and exercised directly by the multi-language network test.
pub struct GenResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub total_steps: usize,
    /// Whether the model emitted an EOS token before generation stopped
    /// (as opposed to running out of `max_frames`).
    pub eos_reached: bool,
}

/// Download/load everything `args` points at and run generation to
/// completion. Shared by `run()` (which also writes a WAV) and the network
/// test (which checks EOS + finite samples per language).
pub fn generate(args: &Args) -> CResult<GenResult> {
    eprintln!("=== TTS Generate ===");
    eprintln!("language: {}", args.language);
    eprintln!("model: {:?} (download+cache if not set)", args.model_path);
    eprintln!("tokenizer: {:?} (download+cache if not set)", args.tokenizer_path);
    eprintln!("voice: {:?}", args.voice_path);
    eprintln!("output: {}", args.output_path);
    eprintln!("temperature: {}", args.temperature);

    if args.safetensors && args.model_path.is_none() {
        return Err(candle_core::Error::Msg(
            "--safetensors requires --model (an explicit safetensors path)".to_string(),
        ));
    }

    // --- Load model ---
    eprintln!("\n[1] Loading model ({})...", if args.safetensors { "safetensors" } else { "GGUF" });
    let (model, cfg) = if args.safetensors {
        let tensors = candle_core::safetensors::load(args.model_path.as_ref().unwrap(), &Device::Cpu)?;
        eprintln!("  read {} tensors from safetensors", tensors.len());
        let cfg = tts_core::config::TTSConfig::v202601_for_safetensors_keys(
            tensors.keys(),
            args.temperature,
        )?;
        let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F32, &Device::Cpu);
        let model = tts_core::tts_model::TTSModel::load(vb, &cfg)?;
        (model, cfg)
    } else {
        let model_bytes = resolve_model_bytes(args).map_err(candle_core::Error::Msg)?;
        eprintln!("  read {} MB", model_bytes.len() / (1024 * 1024));
        let mut gguf = mimi_rs::gguf_loader::GgufTensors::from_bytes(&model_bytes, &Device::Cpu)?;
        let cfg = tts_core::config::TTSConfig::v202601_for_gguf(&gguf, args.temperature)?;
        let model = tts_core::tts_model::TTSModel::load_gguf(&mut gguf, &cfg)?;
        (model, cfg)
    };

    let sample_rate = model.sample_rate() as u32;
    eprintln!("  model loaded OK (sample_rate={})", sample_rate);
    eprintln!(
        "  ldim={} dim={} num_layers={}",
        cfg.flow_lm.ldim, cfg.flow_lm.d_model, cfg.flow_lm.num_layers
    );

    // --- Load voice ---
    let voice_state = {
        eprintln!("\n[2] Loading voice...");
        let voice_bytes = resolve_voice_bytes(args).map_err(candle_core::Error::Msg)?;
        eprintln!("  read {} KB", voice_bytes.len() / 1024);

        let tensors = candle_core::safetensors::load_buffer(&voice_bytes, &Device::Cpu)?;
        eprintln!("  loaded {} tensors from voice file", tensors.len());

        // Check format: KV cache (per-layer) or audio_prompt (single tensor)
        if tensors.contains_key("audio_prompt") {
            // audio_prompt format: feed through backbone to build KV cache
            let audio_prompt = tensors.get("audio_prompt").unwrap();
            eprintln!("  audio_prompt shape: {:?}", audio_prompt.shape());
            let mut state = model.init_flow_lm_state();
            model.prompt_text(&mut state, &[])?; // no text tokens, but we need to init
            // Feed audio_prompt as backbone input (it's already in embedding space)
            // For now, use prompt_text with empty tokens + backbone directly
            eprintln!("  WARN: audio_prompt voice format not yet supported for backbone injection");
            eprintln!("  Using empty state instead");
            model.init_flow_lm_state()
        } else {
            // KV cache format
            // Derive layer count from the loaded model config rather than hardcoding it,
            // so a checkpoint with a different depth doesn't silently get truncated.
            let num_layers = cfg.flow_lm.num_layers;
            let mut layer_states = Vec::with_capacity(num_layers);

            for i in 0..num_layers {
                let cache_name = format!("transformer.layers.{i}.self_attn/cache");
                let cache = tensors
                    .get(&cache_name)
                    .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor: {cache_name}")))?;

                let k = cache.narrow(0, 0, 1)?.squeeze(0)?;
                let v = cache.narrow(0, 1, 1)?.squeeze(0)?;
                let seq_len = k.dim(1)?;

                if i == 0 {
                    eprintln!("  voice seq_len from layer 0: {seq_len}");
                    eprintln!("  k shape: {:?}", k.shape());
                }

                layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState::with_kv(
                    k.contiguous()?,
                    v.contiguous()?,
                    seq_len,
                )));
            }

            // Loudly fail if the voice file has more layer caches than the model config
            // expects, instead of silently ignoring the extras (e.g. a future 24-layer
            // checkpoint's voice file loaded against a 6-layer config).
            let extra_cache_name = format!("transformer.layers.{num_layers}.self_attn/cache");
            if tensors.contains_key(&extra_cache_name) {
                return Err(candle_core::Error::Msg(format!(
                    "voice file has a layer-{num_layers} cache ({extra_cache_name}) but the \
                     model config only has {num_layers} layers; refusing to silently truncate"
                )));
            }

            TTSState {
                flow_lm_state: FlowLMState {
                    transformer_state: StreamingTransformerState { layer_states },
                },
            }
        }
    };
    eprintln!("  voice state ready");

    // --- Prepare text and token IDs ---
    eprintln!("\n[3] Preparing text...");
    let raw_text = &args.text;
    let text_config = TextConfig::for_language(&args.language);
    let (prepared_text, frames_after_eos) =
        prepare_text_prompt(raw_text, cfg.model_recommended_frames_after_eos, &text_config);
    eprintln!("  raw: {raw_text:?}");
    eprintln!("  prepared: {prepared_text:?}");
    eprintln!("  frames_after_eos: {frames_after_eos}");

    eprintln!("  loading tokenizer...");
    let tokenizer_bytes = resolve_tokenizer_bytes(args).map_err(candle_core::Error::Msg)?;
    let tokenizer = Tokenizer::from_model_bytes(&tokenizer_bytes)?;

    let token_ids: Vec<u32> = tokenizer.encode(&prepared_text);
    eprintln!("  token_ids ({} tokens): {:?}", token_ids.len(), token_ids);

    // --- Run prompt_text ---
    eprintln!("\n[4] Running prompt_text...");
    let mut tts_state = voice_state.clone();
    model.prompt_text(&mut tts_state, &token_ids)?;
    let seq_len = tts_state.flow_lm_state.transformer_state.current_seq_len();
    eprintln!("  prompt_text done, seq_len={seq_len}");

    // --- Init mimi state and RNG ---
    let mut mimi_state = model.init_mimi_state(1, &Device::Cpu)?;
    let mut rng = SimpleRng::new(args.temperature);

    // --- BOS latent: NaN tensor [1, 1, ldim] ---
    let ldim = cfg.flow_lm.ldim;
    let nan_data: Vec<f32> = vec![f32::NAN; ldim];
    let mut prev_latent = Tensor::from_vec(nan_data, (1usize, 1usize, ldim), &Device::Cpu)?;

    // --- Generation loop ---
    // Use same max_frames formula as wasm binding
    let max_frames = ((token_ids.len() as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    eprintln!("\n[5] Generating audio ({max_frames} max frames, frames_after_eos={frames_after_eos})...");

    let mut audio_chunks: Vec<f32> = Vec::new();
    let mut eos_gate = EosGate::new(frames_after_eos);
    let mut total_steps = 0usize;
    let mut eos_reached = false;

    for step in 0..max_frames {
        let (next_latent, is_eos) =
            model.generate_step(&mut tts_state, &prev_latent, &mut rng)?;

        // Check-then-append order (matches tts_model.py:874-892): decide
        // whether this step's latent is produced *before* decoding/appending
        // its audio, not after.
        if !eos_gate.accept(step, is_eos) {
            eprintln!("  EOS countdown reached 0 at step {step}, stopping");
            break;
        }
        if is_eos {
            eos_reached = true;
        }

        let audio_chunk = model.decode_latent(&next_latent, &mut mimi_state)?;
        let pcm = audio_chunk.flatten_all()?.to_vec1::<f32>()?;

        let pcm_min = pcm.iter().cloned().fold(f32::INFINITY, f32::min);
        let pcm_max = pcm.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pcm_mean = pcm.iter().sum::<f32>() / pcm.len() as f32;
        let nan_count = pcm.iter().filter(|x| x.is_nan()).count();
        let inf_count = pcm.iter().filter(|x| x.is_infinite()).count();

        eprintln!(
            "step {:3}: pcm_len={} is_eos={} min={:.4} max={:.4} mean={:.4} nan={nan_count} inf={inf_count}",
            step, pcm.len(), is_eos, pcm_min, pcm_max, pcm_mean
        );
        if is_eos {
            eprintln!("  EOS detected at step {step}");
        }

        audio_chunks.extend_from_slice(&pcm);
        total_steps = step + 1;

        prev_latent = next_latent;
    }

    let total_seconds = audio_chunks.len() as f64 / sample_rate as f64;
    eprintln!("\n[6] Generation complete:");
    eprintln!("  total_steps: {total_steps}");
    eprintln!("  total_samples: {}", audio_chunks.len());
    eprintln!("  total_duration: {:.2}s", total_seconds);
    log_tensor_stats("final_audio", &audio_chunks);

    Ok(GenResult {
        samples: audio_chunks,
        sample_rate,
        total_steps,
        eos_reached,
    })
}

fn run() -> CResult<()> {
    let argv: Vec<String> = std::env::args().collect();
    let args = parse_args(&argv).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    if args.list {
        print_list();
        return Ok(());
    }

    let result = generate(&args)?;

    eprintln!("\n[7] Writing WAV to {}...", args.output_path);
    write_wav(&args.output_path, &result.samples, result.sample_rate)
        .map_err(|e| candle_core::Error::Msg(format!("failed to write WAV: {e}")))?;
    eprintln!(
        "  wrote {} samples ({:.2}s) at {}Hz",
        result.samples.len(),
        result.samples.len() as f64 / result.sample_rate as f64,
        result.sample_rate
    );
    eprintln!("\nDone! Audio saved to {}", args.output_path);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
