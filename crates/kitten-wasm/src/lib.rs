#[cfg(feature = "wasm")]
pub mod web {
    use candle_core::{DType, Device, IndexOp, Tensor};
    use kitten_core::{config::KittenConfig, kitten_model::KittenModel, phoneme_map::map_phonemes_to_ids, text_preprocess};
    use wasm_bindgen::prelude::*;

    fn wasm_log(msg: &str) {
        #[cfg(target_family = "wasm")]
        web_sys::console::log_1(&msg.into());
        #[cfg(not(target_family = "wasm"))]
        let _ = msg;
    }

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
    }

    #[wasm_bindgen]
    pub struct KittenEngine {
        model: Option<KittenModel>,
        voices: Option<Tensor>,  // [n_voices, 400, 256]
        voice_names: Vec<String>,
    }

    #[wasm_bindgen]
    impl KittenEngine {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            console_error_panic_hook::set_once();
            Self { model: None, voices: None, voice_names: Vec::new() }
        }

        /// Load model weights from safetensors bytes.
        #[wasm_bindgen(js_name = loadModel)]
        pub fn load_model(&mut self, data: &[u8]) -> Result<(), JsError> {
            wasm_log("[kitten] Loading model...");
            let cfg = KittenConfig::nano();
            let model = KittenModel::load(data, &cfg, &Device::Cpu)
                .map_err(|e| JsError::new(&e.to_string()))?;
            self.model = Some(model);
            wasm_log("[kitten] Model loaded");
            Ok(())
        }

        /// Load voice embeddings from safetensors bytes.
        /// Each tensor in the file is named by voice (e.g., "bella", "jasper")
        /// and has shape [400, 256] F32.
        #[wasm_bindgen(js_name = loadVoices)]
        pub fn load_voices(&mut self, data: &[u8]) -> Result<(), JsError> {
            let st = safetensors::SafeTensors::deserialize(data)
                .map_err(|e| JsError::new(&e.to_string()))?;

            let mut voice_tensors = Vec::new();
            let mut names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
            names.sort(); // deterministic order

            for name in &names {
                let view = st.tensor(name)
                    .map_err(|e| JsError::new(&e.to_string()))?;
                let raw = view.data();
                let shape = view.shape();
                if shape.len() != 2 {
                    return Err(JsError::new(&format!("voice '{}': expected 2D tensor, got {}D", name, shape.len())));
                }
                let t = Tensor::from_raw_buffer(raw, DType::F32, &[shape[0], shape[1]], &Device::Cpu)
                    .map_err(|e| JsError::new(&e.to_string()))?;
                voice_tensors.push(t);
            }

            self.voices = Some(
                Tensor::stack(&voice_tensors, 0)
                    .map_err(|e| JsError::new(&e.to_string()))?,
            ); // [n_voices, 400, 256]
            self.voice_names = names;
            wasm_log(&format!("[kitten] Loaded {} voices", self.voice_names.len()));
            Ok(())
        }

        /// Get available voice names as a comma-separated string.
        #[wasm_bindgen(js_name = getVoiceNames)]
        pub fn get_voice_names(&self) -> String {
            self.voice_names.join(",")
        }

        /// Synthesize from an IPA string.
        ///
        /// - `voice_idx`: index into the loaded voice list
        /// - `speed`: speaking rate multiplier (1.0 = normal)
        /// - `text_len`: used to index the style vector (clamped to 0..=399)
        ///
        /// Returns a Float32Array of PCM samples at 24 kHz.
        #[wasm_bindgen(js_name = synthesizeFromIpa)]
        pub fn synthesize_from_ipa(
            &self,
            ipa: &str,
            voice_idx: u32,
            speed: f32,
            text_len: u32,
        ) -> Result<js_sys::Float32Array, JsError> {
            let model = self.model.as_ref()
                .ok_or_else(|| JsError::new("model not loaded"))?;
            let voices = self.voices.as_ref()
                .ok_or_else(|| JsError::new("voices not loaded"))?;

            let n_voices = voices.dim(0).map_err(|e| JsError::new(&e.to_string()))?;
            let vi = voice_idx as usize;
            if vi >= n_voices {
                return Err(JsError::new(&format!("voice_idx {vi} out of range (n_voices={n_voices})")));
            }

            // Style vector: [1, 256] slice at text_len position
            let idx = (text_len as usize).min(399);
            let style = voices.i((vi, idx, ..))
                .map_err(|e| JsError::new(&e.to_string()))?  // [256]
                .unsqueeze(0)
                .map_err(|e| JsError::new(&e.to_string()))?; // [1, 256]

            let ids = map_phonemes_to_ids(ipa);

            // --- debug ---
            wasm_log(&format!("[kitten] IPA received: {:?}", ipa));
            wasm_log(&format!("[kitten] phoneme IDs (first 10): {:?}", &ids[..ids.len().min(10)]));
            {
                let style_data = style.to_vec2::<f32>()
                    .map_err(|e| JsError::new(&e.to_string()))?;
                let flat: &[f32] = &style_data[0];
                let min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                wasm_log(&format!(
                    "[kitten] style shape=[1,{}] min={:.4} max={:.4} first3=[{:.4},{:.4},{:.4}]",
                    flat.len(), min, max,
                    flat.get(0).copied().unwrap_or(0.0),
                    flat.get(1).copied().unwrap_or(0.0),
                    flat.get(2).copied().unwrap_or(0.0),
                ));
            }
            // --- end debug ---

            let samples = model.synthesize(&ids, &style, speed)
                .map_err(|e| JsError::new(&e.to_string()))?;

            // --- debug ---
            {
                let n = samples.len();
                let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                wasm_log(&format!(
                    "[kitten] output samples={} min={:.4} max={:.4} first5={:?}",
                    n, min, max,
                    &samples[..n.min(5)],
                ));
            }
            // --- end debug ---

            Ok(js_sys::Float32Array::from(samples.as_slice()))
        }

        /// Map an IPA string to phoneme IDs (for debugging).
        #[wasm_bindgen(js_name = mapIpaToIds)]
        pub fn map_ipa_to_ids(ipa: &str) -> Vec<i32> {
            map_phonemes_to_ids(ipa)
        }

        /// Preprocess raw text (expand currencies, units, etc.) before phonemization.
        #[wasm_bindgen(js_name = preprocessText)]
        pub fn preprocess_text(&self, text: &str) -> String {
            text_preprocess::preprocess_text(text)
        }

        /// Sample rate (24000 Hz).
        #[wasm_bindgen(js_name = sampleRate)]
        pub fn sample_rate() -> u32 {
            24000
        }

        /// Check if the model is loaded and ready.
        #[wasm_bindgen(js_name = isReady)]
        pub fn is_ready(&self) -> bool {
            self.model.is_some()
        }
    }
}
