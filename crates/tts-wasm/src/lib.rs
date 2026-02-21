use candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::VarBuilder;
use mimi_rs::mimi::MimiState;
use mimi_rs::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use tts_core::flow_lm::{FlowLMState, Rng};
use tts_core::tts_model::TTSState;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

// ---- WasmRng ----

struct WasmRng {
    inner: Box<rand::rngs::StdRng>,
    distr: rand_distr::Normal<f32>,
}

impl WasmRng {
    fn new(temperature: f32) -> Self {
        use rand::SeedableRng;
        let std = temperature.sqrt();
        let distr = rand_distr::Normal::new(0f32, std).unwrap();
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        Self { inner: Box::new(rng), distr }
    }
}

impl Rng for WasmRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

// ---- GenState ----

struct GenState {
    tts_state: TTSState,
    mimi_state: MimiState,
    prev_latent: Tensor,
    rng: WasmRng,
    max_frames: usize,
    frames_after_eos: usize,
    eos_countdown: Option<usize>,
    step: usize,
}

// ---- Model ----

#[wasm_bindgen]
pub struct Model {
    inner: tts_core::tts_model::TTSModel,
    cfg: tts_core::config::TTSConfig,
    gen_state: Option<GenState>,
    voice_states: Vec<TTSState>,
}

impl Model {
    fn new_(model_weights: &[u8]) -> CResult<Model> {
        let dequantized_bytes = mimi_rs::dequantize::dequantize_and_remap(model_weights);
        let vb = VarBuilder::from_buffered_safetensors(dequantized_bytes, DType::F32, &Device::Cpu)?;
        let cfg = tts_core::config::TTSConfig::v202601(0.7);
        let inner = tts_core::tts_model::TTSModel::load(vb, &cfg)?;
        console_log!("[Model::new] model loaded");

        Ok(Model { inner, cfg, gen_state: None, voice_states: Vec::new() })
    }

    fn add_voice_(&mut self, voice_bytes: &[u8]) -> CResult<usize> {
        let tensors = candle_core::safetensors::load_buffer(voice_bytes, &Device::Cpu)?;
        let num_layers = 6usize;
        let mut layer_states = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let cache_name = format!("transformer.layers.{i}.self_attn/cache");
            let cache = tensors
                .get(&cache_name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing tensor: {cache_name}")))?;

            let k = cache.narrow(0, 0, 1)?.squeeze(0)?;
            let v = cache.narrow(0, 1, 1)?.squeeze(0)?;
            let seq_len = k.dim(1)?;

            layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState::with_kv(
                k.contiguous()?,
                v.contiguous()?,
                seq_len,
            )));
        }

        let tts_state = TTSState {
            flow_lm_state: FlowLMState {
                transformer_state: StreamingTransformerState { layer_states },
            },
        };
        let idx = self.voice_states.len();
        self.voice_states.push(tts_state);
        console_log!("[add_voice] voice {} loaded", idx);
        Ok(idx)
    }

    fn start_generation_(
        &mut self,
        voice_index: usize,
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> CResult<()> {
        if voice_index >= self.voice_states.len() {
            return Err(candle_core::Error::Msg(format!(
                "invalid voice index: {voice_index}"
            )));
        }

        let mut tts_state = self.voice_states[voice_index].clone();
        let max_frames = ((token_ids.len() as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;

        self.inner.prompt_text(&mut tts_state, token_ids)?;

        let mimi_state = self.inner.init_mimi_state(1, &Device::Cpu)?;
        let rng = WasmRng::new(temperature);

        let ldim = self.cfg.flow_lm.ldim;
        let nan_data: Vec<f32> = vec![f32::NAN; ldim];
        let prev_latent = Tensor::from_vec(nan_data, (1, 1, ldim), &Device::Cpu)?;

        self.gen_state = Some(GenState {
            tts_state,
            mimi_state,
            prev_latent,
            rng,
            max_frames,
            frames_after_eos,
            eos_countdown: None,
            step: 0,
        });
        Ok(())
    }

    fn generation_step_(&mut self) -> CResult<Option<js_sys::Float32Array>> {
        let mut state = match self.gen_state.take() {
            Some(s) => s,
            None => return Ok(None),
        };

        if state.step >= state.max_frames {
            return Ok(None);
        }

        let (next_latent, is_eos) =
            self.inner.generate_step(&mut state.tts_state, &state.prev_latent, &mut state.rng)?;

        let audio_chunk =
            self.inner.decode_latent(&next_latent, &mut state.mimi_state)?;

        if is_eos && state.eos_countdown.is_none() {
            state.eos_countdown = Some(state.frames_after_eos);
        }

        let done = if let Some(ref mut countdown) = state.eos_countdown {
            if *countdown == 0 {
                true
            } else {
                *countdown -= 1;
                false
            }
        } else {
            false
        };

        state.prev_latent = next_latent;
        state.step += 1;

        let pcm = audio_chunk.flatten_all()?.to_vec1::<f32>()?;
        let result = js_sys::Float32Array::from(pcm.as_slice());

        if !done {
            self.gen_state = Some(state);
        }

        Ok(Some(result))
    }
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(model_weights: &[u8]) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        Self::new_(model_weights).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn add_voice(&mut self, voice_bytes: &[u8]) -> Result<usize, JsError> {
        self.add_voice_(voice_bytes).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn prepare_text(&self, text: &str) -> js_sys::Array {
        let (processed, frames_after_eos) = tts_core::tts_model::prepare_text_prompt(text);
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from_str(&processed));
        arr.push(&JsValue::from_f64(frames_after_eos as f64));
        arr
    }

    pub fn start_generation(
        &mut self,
        voice_index: usize,
        token_ids: &[u32],
        frames_after_eos: usize,
        temperature: f32,
    ) -> Result<(), JsError> {
        self.start_generation_(voice_index, token_ids, frames_after_eos, temperature)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn generation_step(&mut self) -> Result<Option<js_sys::Float32Array>, JsError> {
        self.generation_step_().map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}
