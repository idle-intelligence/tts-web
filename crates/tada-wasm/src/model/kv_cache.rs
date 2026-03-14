//! Ring-buffer KV cache for autoregressive decoding.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::Tensor;

/// Ring-buffer KV cache for one transformer layer.
///
/// Writes new K/V entries at `write_pos` via `slice_assign` (O(1) per step),
/// wrapping around when the buffer is full. RoPE is applied before caching.
pub struct KVCache {
    k: Option<Tensor<Wgpu, 4>>,
    v: Option<Tensor<Wgpu, 4>>,
    offset: usize,
    len: usize,
    write_pos: usize,
    max_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: WgpuDevice,
}

impl KVCache {
    pub fn new(max_len: usize, n_kv_heads: usize, head_dim: usize, device: &WgpuDevice) -> Self {
        Self {
            k: None,
            v: None,
            offset: 0,
            len: 0,
            write_pos: 0,
            max_len,
            n_kv_heads,
            head_dim,
            device: device.clone(),
        }
    }

    /// Update cache with new K, V and return all valid entries.
    ///
    /// K, V shape: [batch, n_kv_heads, 1, head_dim]
    ///
    /// Uses simple concatenation (like candle) instead of ring buffer to avoid
    /// potential GPU tensor aliasing issues with slice_assign.
    pub fn update(
        &mut self,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let (k_full, v_full) = match (self.k.take(), self.v.take()) {
            (Some(prev_k), Some(prev_v)) => {
                // Concatenate along sequence dimension (dim 2)
                let k_full = Tensor::cat(vec![prev_k, k], 2);
                let v_full = Tensor::cat(vec![prev_v, v], 2);
                (k_full, v_full)
            }
            _ => (k, v),
        };

        self.offset += 1;
        self.len += 1;

        self.k = Some(k_full.clone());
        self.v = Some(v_full.clone());

        (k_full, v_full)
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn seq_len(&self) -> usize {
        self.len
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.offset = 0;
        self.len = 0;
        self.write_pos = 0;
    }

    pub fn reset_keep_buffers(&mut self) {
        if let Some(ref k) = self.k {
            let shape = k.dims();
            self.k = Some(Tensor::zeros(shape, &self.device));
        }
        if let Some(ref v) = self.v {
            let shape = v.dims();
            self.v = Some(Tensor::zeros(shape, &self.device));
        }
        self.offset = 0;
        self.len = 0;
        self.write_pos = 0;
    }
}

/// Collection of KV caches for all transformer layers.
pub struct LayerCaches {
    caches: Vec<KVCache>,
}

impl LayerCaches {
    pub fn new(
        n_layers: usize,
        max_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &WgpuDevice,
    ) -> Self {
        Self {
            caches: (0..n_layers)
                .map(|_| KVCache::new(max_len, n_kv_heads, head_dim, device))
                .collect(),
        }
    }

    pub fn get_mut(&mut self, layer: usize) -> Option<&mut KVCache> {
        self.caches.get_mut(layer)
    }

    pub fn seq_len(&self) -> usize {
        self.caches.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn reset_keep_buffers(&mut self) {
        for cache in &mut self.caches {
            cache.reset_keep_buffers();
        }
    }
}
