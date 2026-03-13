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
    pub fn update(
        &mut self,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let [b, _h, _seq, _d] = k.dims();

        if self.k.is_none() {
            self.k = Some(Tensor::zeros(
                [b, self.n_kv_heads, self.max_len, self.head_dim],
                &self.device,
            ));
            self.v = Some(Tensor::zeros(
                [b, self.n_kv_heads, self.max_len, self.head_dim],
                &self.device,
            ));
        }

        let k_buf = self.k.take().unwrap();
        let v_buf = self.v.take().unwrap();
        let [b, h, _, hd] = k_buf.dims();
        let pos = self.write_pos;

        let k_buf = k_buf.slice_assign([0..b, 0..h, pos..pos + 1, 0..hd], k);
        let v_buf = v_buf.slice_assign([0..b, 0..h, pos..pos + 1, 0..hd], v);

        self.write_pos = (self.write_pos + 1) % self.max_len;
        self.offset += 1;
        self.len = (self.len + 1).min(self.max_len);

        let result = if self.len < self.max_len {
            let k_view = k_buf.clone().slice([0..b, 0..h, 0..self.len, 0..hd]);
            let v_view = v_buf.clone().slice([0..b, 0..h, 0..self.len, 0..hd]);
            (k_view, v_view)
        } else {
            (k_buf.clone(), v_buf.clone())
        };

        self.k = Some(k_buf);
        self.v = Some(v_buf);

        result
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
