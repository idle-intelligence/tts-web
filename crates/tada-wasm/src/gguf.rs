//! Q4 GGUF weight loader and WGSL dequantization shaders.
//!
//! Pipeline: GGUF file -> parse header/tensors -> store Q4 blocks as raw bytes
//! on GPU -> dequantize via WGSL compute shader -> matmul.
//!
//! Adapted from stt-web's gguf.rs. Model-specific loading is done externally;
//! this module provides the generic GGUF parser, Q4 GPU primitives, and kernel.

use anyhow::{bail, ensure, Context, Result};
use burn::backend::wgpu::{
    into_contiguous, AutoCompiler, CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate,
    WgpuDevice, WgpuRuntime,
};
use burn::backend::Wgpu;
use burn::module::{Param, ParamId};
use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
use byteorder::{LittleEndian, ReadBytesExt};
use cubecl::prelude::KernelId;
use cubecl::server::{Bindings, CubeCount, Handle};
use cubecl::{CubeTask, Runtime};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const ALIGNMENT: u64 = 32;

// Naive kernel workgroup sizes (16x16 = 256, the WebGPU limit)
const NAIVE_WG_X: u32 = 16;
const NAIVE_WG_Y: u32 = 16;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert IEEE 754 half-precision (f16) bits to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormalized
            let mut e = 1u32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f32_exp = 127u32.wrapping_sub(15 + e - 1);
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13))
    } else {
        // Normalized
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

/// Reverse GGUF dimension order to get PyTorch convention.
///
/// GGUF stores dimensions in reversed order (row-major innermost first),
/// while PyTorch uses `[out_features, in_features]` convention.
pub fn reverse_gguf_dims(gguf_dims: &[u64]) -> Vec<usize> {
    gguf_dims.iter().rev().map(|&d| d as usize).collect()
}

// ---------------------------------------------------------------------------
// GGUF String / Value helpers
// ---------------------------------------------------------------------------

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).context("Invalid UTF-8 in GGUF string")
}

fn skip_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<()> {
    match value_type {
        0 => { reader.read_u8()?; }
        1 => { reader.read_i8()?; }
        2 => { reader.seek(SeekFrom::Current(2))?; }
        3 => { reader.seek(SeekFrom::Current(2))?; }
        4 => { reader.seek(SeekFrom::Current(4))?; }
        5 => { reader.seek(SeekFrom::Current(4))?; }
        6 => { reader.seek(SeekFrom::Current(4))?; }
        7 => { reader.read_u8()?; }
        8 => { let _ = read_gguf_string(reader)?; }
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            for _ in 0..count {
                skip_gguf_value(reader, elem_type)?;
            }
        }
        10 => { reader.seek(SeekFrom::Current(8))?; }
        11 => { reader.seek(SeekFrom::Current(8))?; }
        12 => { reader.seek(SeekFrom::Current(8))?; }
        other => bail!("Unknown GGUF metadata value type: {other}"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GgmlDtype
// ---------------------------------------------------------------------------

/// GGML data type codes used in GGUF tensor descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDtype {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4_K,
}

impl GgmlDtype {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4_K),
            other => bail!("Unsupported GGML dtype code: {other}"),
        }
    }

    pub fn byte_size(&self, num_elements: u64) -> u64 {
        match self {
            Self::F32 => num_elements * 4,
            Self::F16 => num_elements * 2,
            Self::Q4_0 => {
                let num_blocks = num_elements / 32;
                num_blocks * 18
            }
            Self::Q8_0 => {
                let num_blocks = num_elements / 32;
                num_blocks * 34
            }
            Self::Q4_K => {
                let num_blocks = num_elements / 256;
                num_blocks * 144
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTensorInfo
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    dimensions: Vec<u64>,
    dtype: GgmlDtype,
    offset: u64,
}

impl GgufTensorInfo {
    pub fn shape(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn dtype(&self) -> GgmlDtype {
        self.dtype
    }

    pub fn num_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn byte_size(&self) -> u64 {
        self.dtype.byte_size(self.num_elements())
    }
}

// ---------------------------------------------------------------------------
// GgufReader
// ---------------------------------------------------------------------------

/// A reader for GGUF v2/v3 files.
pub struct GgufReader<R: Read + Seek> {
    reader: R,
    version: u32,
    tensor_count: u64,
    tensors: HashMap<String, GgufTensorInfo>,
    data_section_offset: u64,
}

impl<R: Read + Seek> GgufReader<R> {
    /// Parse a GGUF file from the given reader.
    pub fn open(mut reader: R) -> Result<Self> {
        let magic = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF magic")?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})");
        }

        let version = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF version")?;
        if version != 2 && version != 3 {
            bail!("Unsupported GGUF version: {version} (expected 2 or 3)");
        }

        let tensor_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read tensor count")?;
        let metadata_kv_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read metadata KV count")?;

        // Skip metadata key-value pairs
        for i in 0..metadata_kv_count {
            let _key = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read metadata key {i}"))?;
            let value_type = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read metadata value type {i}"))?;
            skip_gguf_value(&mut reader, value_type)
                .with_context(|| format!("Failed to skip metadata value {i}"))?;
        }

        // Parse tensor index
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read tensor name {i}"))?;
            let ndims = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read ndims for tensor {i}"))?;
            let mut dimensions = Vec::with_capacity(ndims as usize);
            for d in 0..ndims {
                dimensions.push(
                    reader
                        .read_u64::<LittleEndian>()
                        .with_context(|| format!("Failed to read dim {d} for tensor {i}"))?,
                );
            }
            let dtype = GgmlDtype::from_u32(
                reader
                    .read_u32::<LittleEndian>()
                    .with_context(|| format!("Failed to read dtype for tensor {i}"))?,
            )?;
            let offset = reader
                .read_u64::<LittleEndian>()
                .with_context(|| format!("Failed to read offset for tensor {i}"))?;

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    dimensions,
                    dtype,
                    offset,
                },
            );
        }

        let current_pos = reader.stream_position()?;
        let data_section_offset = align_up(current_pos, ALIGNMENT);

        Ok(Self {
            reader,
            version,
            tensor_count,
            tensors,
            data_section_offset,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// Iterate over all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Read raw tensor data bytes from the file.
    pub fn tensor_data(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("Tensor '{name}' not found in GGUF"))?
            .clone();
        let byte_size = info.byte_size() as usize;
        let abs_offset = self.data_section_offset + info.offset;
        self.reader.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; byte_size];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// ShardedCursor -- Read + Seek over multiple buffers
// ---------------------------------------------------------------------------

/// A cursor that provides `Read + Seek` over multiple contiguous byte buffers.
///
/// Each shard is kept as a separate `Vec<u8>` to stay under the WASM32
/// `isize::MAX` (~2 GB) per-allocation limit while supporting total sizes > 2 GB.
pub struct ShardedCursor {
    shards: Vec<Vec<u8>>,
    ends: Vec<u64>,
    pos: u64,
    total_len: u64,
}

impl ShardedCursor {
    pub fn new(shards: Vec<Vec<u8>>) -> Self {
        let mut ends = Vec::with_capacity(shards.len());
        let mut total: u64 = 0;
        for s in &shards {
            total += s.len() as u64;
            ends.push(total);
        }
        Self {
            shards,
            ends,
            pos: 0,
            total_len: total,
        }
    }

    fn shard_for_offset(&self, offset: u64) -> Option<(usize, usize)> {
        if offset >= self.total_len {
            return None;
        }
        let shard_idx = self.ends.partition_point(|&end| end <= offset);
        let shard_start = if shard_idx > 0 {
            self.ends[shard_idx - 1]
        } else {
            0
        };
        Some((shard_idx, (offset - shard_start) as usize))
    }
}

impl Read for ShardedCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.total_len {
            return Ok(0);
        }
        let (shard_idx, local_offset) = self.shard_for_offset(self.pos).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "no shard found for offset {} (total_len={})",
                    self.pos, self.total_len
                ),
            )
        })?;
        let shard = &self.shards[shard_idx];
        let available = shard.len() - local_offset;
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&shard[local_offset..local_offset + to_read]);
        self.pos += to_read as u64;
        Ok(to_read)
    }
}

impl Seek for ShardedCursor {
    fn seek(&mut self, style: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match style {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.total_len as i64 + offset,
            SeekFrom::Current(offset) => self.pos as i64 + offset,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

// ---------------------------------------------------------------------------
// Q4Tensor -- GPU buffer of Q4_0 blocks
// ---------------------------------------------------------------------------

/// A Q4_0 quantized weight tensor living on GPU.
///
/// The buffer contains raw Q4_0 blocks (18 bytes per block of 32 elements).
/// The WGSL shader interprets the buffer as `array<u32>`.
pub struct Q4Tensor {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
}

impl Q4Tensor {
    /// Upload raw Q4_0 bytes to a GPU storage buffer.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`.
    /// `raw_bytes` must contain exactly `(N * K / 32) * 18` bytes.
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = WgpuRuntime::client(device);

        // Pad to 4-byte alignment for array<u32> access in the WGSL shader.
        let padded = if !raw_bytes.len().is_multiple_of(4) {
            let pad = 4 - (raw_bytes.len() % 4);
            let mut buf = raw_bytes.to_vec();
            buf.resize(raw_bytes.len() + pad, 0);
            buf
        } else {
            raw_bytes.to_vec()
        };
        let handle = client.create_from_slice(&padded);

        Ok(Self {
            handle,
            shape,
            num_blocks,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

// ---------------------------------------------------------------------------
// Q4Linear
// ---------------------------------------------------------------------------

/// A linear layer with Q4_0 quantized weights.
///
/// Stores weights as `[out_features, in_features]` in Q4_0 format and an
/// optional f32 bias. Forward: `x @ weights^T + bias` via fused dequant+matmul.
pub struct Q4Linear {
    weights: Q4Tensor,
    bias: Option<Tensor<Wgpu, 1>>,
}

impl Q4Linear {
    pub fn new(weights: Q4Tensor, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weights, bias }
    }

    /// Forward pass: `x @ weights^T + bias`.
    ///
    /// `x` shape: `[B, M, K]` where `K = in_features`.
    /// Returns shape: `[B, M, N]` where `N = out_features`.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let out = q4_matmul(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}

// ---------------------------------------------------------------------------
// Q4 matmul kernel dispatch
// ---------------------------------------------------------------------------

struct Q4MatmulNaiveKernel {
    workgroup_size_x: u32,
    workgroup_size_y: u32,
}

impl KernelSource for Q4MatmulNaiveKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_naive.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.workgroup_size_x * 1000 + self.workgroup_size_y)
    }
}

/// Fused Q4_0 dequant+matmul on GPU.
///
/// Computes `output[B, M, N] = input[B, M, K] x weights[N, K]^T`.
pub fn q4_matmul(input: Tensor<Wgpu, 3>, weights: &Q4Tensor) -> Tensor<Wgpu, 3> {
    let cube_input: CubeTensor<WgpuRuntime> = input.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    assert_eq!(cube_input.shape.num_dims(), 3, "Input must be 3D [B, M, K]");
    let b = cube_input.shape.dims[0];
    let m = cube_input.shape.dims[1];
    let k = cube_input.shape.dims[2];
    let [n, wk] = weights.shape();
    assert_eq!(
        k, wk,
        "K dimension mismatch: input has {k}, weights have {wk}"
    );

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();
    let blocks_per_row = k / 32;

    let output_handle = client.empty(b * m * n * 4);

    let info: [u32; 5] = [
        b as u32,
        m as u32,
        k as u32,
        n as u32,
        blocks_per_row as u32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(weights.handle.clone().binding())
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(
        Q4MatmulNaiveKernel {
            workgroup_size_x: NAIVE_WG_X,
            workgroup_size_y: NAIVE_WG_Y,
        },
        CubeDim::new_2d(NAIVE_WG_X, NAIVE_WG_Y),
    );
    let wg_x = n.div_ceil(NAIVE_WG_X as usize) as u32;
    let wg_y = (b * m).div_ceil(NAIVE_WG_Y as usize) as u32;
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_x, wg_y),
            bindings,
        )
        .expect("Q4 naive matmul kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, m, n]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

// ---------------------------------------------------------------------------
// EmbeddingStore -- Q4 embeddings for token lookups
// ---------------------------------------------------------------------------

/// Q4 embedding table stored as CPU bytes for efficient row lookups.
///
/// Dequantizes individual rows on-the-fly (avoids materializing the full
/// embedding table as f32, which could be hundreds of MB).
pub struct EmbeddingStore {
    cpu_bytes: Vec<u8>,
    vocab_size: usize,
    dim: usize,
}

impl EmbeddingStore {
    /// Create from raw Q4 bytes.
    pub fn new(cpu_bytes: Vec<u8>, vocab_size: usize, dim: usize) -> Self {
        Self {
            cpu_bytes,
            vocab_size,
            dim,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dequantize a single row into an existing CPU buffer (for accumulation).
    ///
    /// Adds the dequantized embedding to `out_buf` (which must be `dim` f32s).
    /// This avoids creating a GPU tensor per embedding lookup.
    pub fn embed_id_add_cpu(&self, id: u32, out_buf: &mut [f32]) {
        assert_eq!(out_buf.len(), self.dim);
        let blocks_per_row = self.dim / 32;
        let bytes_per_row = blocks_per_row * 18;
        let row_offset = (id as usize) * bytes_per_row;
        let row_bytes = &self.cpu_bytes[row_offset..row_offset + bytes_per_row];

        for block in 0..blocks_per_row {
            let bo = block * 18;
            let d = f16_to_f32(u16::from_le_bytes([row_bytes[bo], row_bytes[bo + 1]]));
            let base = block * 32;
            for j in 0..16 {
                let byte = row_bytes[bo + 2 + j];
                out_buf[base + j] += ((byte & 0x0F) as f32 - 8.0) * d;
                out_buf[base + j + 16] += (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
            }
        }
    }

    /// Dequantize a single row into a fresh buffer (returns owned Vec).
    pub fn embed_id(&self, id: u32) -> Vec<f32> {
        let mut buf = vec![0.0f32; self.dim];
        self.embed_id_add_cpu(id, &mut buf);
        buf
    }

    /// Dequantize a single row and upload to GPU as a [1, dim] tensor.
    pub fn embed_id_to_gpu(&self, id: u32, device: &WgpuDevice) -> Tensor<Wgpu, 2> {
        let data = self.embed_id(id);
        Tensor::<Wgpu, 2>::from_data(
            TensorData::new(data, [1, self.dim]),
            device,
        )
    }

    /// Dequantize the entire embedding table to f32 and upload to GPU.
    ///
    /// Returns a [vocab_size, dim] f32 tensor on GPU. Used for tied lm_head
    /// where we need the full table resident on GPU for matmul.
    pub fn to_gpu_f32(&self, device: &WgpuDevice) -> Tensor<Wgpu, 2> {
        let blocks_per_row = self.dim / 32;
        let bytes_per_row = blocks_per_row * 18;
        let mut data = vec![0.0f32; self.vocab_size * self.dim];

        for row in 0..self.vocab_size {
            let row_offset = row * bytes_per_row;
            let row_bytes = &self.cpu_bytes[row_offset..row_offset + bytes_per_row];
            let out_offset = row * self.dim;

            for block in 0..blocks_per_row {
                let bo = block * 18;
                let d = f16_to_f32(u16::from_le_bytes([row_bytes[bo], row_bytes[bo + 1]]));
                let base = out_offset + block * 32;
                for j in 0..16 {
                    let byte = row_bytes[bo + 2 + j];
                    data[base + j] = ((byte & 0x0F) as f32 - 8.0) * d;
                    data[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
                }
            }
        }

        Tensor::<Wgpu, 2>::from_data(
            TensorData::new(data, [self.vocab_size, self.dim]),
            device,
        )
    }
}

// ---------------------------------------------------------------------------
// RmsNormLayer — wrapper for GGUF weight loading
// ---------------------------------------------------------------------------

/// RMSNorm layer wrapping burn::nn::RmsNorm for GGUF weight loading.
pub struct RmsNormLayer {
    pub inner: burn::nn::RmsNorm<Wgpu>,
}

impl RmsNormLayer {
    pub fn forward<const D: usize>(&self, x: Tensor<Wgpu, D>) -> Tensor<Wgpu, D> {
        self.inner.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Helper: load f32 tensor from GGUF
// ---------------------------------------------------------------------------

/// Load an f32/f16 tensor from GGUF as a 1D f32 vector (for norms, biases, etc.).
pub fn load_f32_tensor<R: Read + Seek>(
    reader: &mut GgufReader<R>,
    name: &str,
) -> Result<Vec<f32>> {
    let info = reader
        .tensor_info(name)
        .with_context(|| format!("Tensor '{name}' not found"))?
        .clone();
    let bytes = reader.tensor_data(name)?;
    let data: Vec<f32> = match info.dtype() {
        GgmlDtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        GgmlDtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        other => bail!("Cannot load {:?} tensor '{name}' as f32", other),
    };
    Ok(data)
}

/// Dequantize a Q4_0 byte buffer to f32.
///
/// Input: raw Q4_0 blocks (18 bytes each: 2-byte f16 scale + 16 bytes of 32 nibbles).
/// Output: `num_elements` f32 values.
pub fn dequantize_q4_to_f32(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    assert_eq!(bytes.len(), num_blocks * 18, "Q4_0 byte count mismatch");
    let mut out = vec![0.0f32; num_elements];
    for block in 0..num_blocks {
        let bo = block * 18;
        let d = f16_to_f32(u16::from_le_bytes([bytes[bo], bytes[bo + 1]]));
        let base = block * 32;
        for j in 0..16 {
            let byte = bytes[bo + 2 + j];
            out[base + j] = ((byte & 0x0F) as f32 - 8.0) * d;
            out[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
        }
    }
    out
}

/// Dequantize a Q8_0 byte buffer to f32.
///
/// Q8_0 block format: 34 bytes per block of 32 values.
///   - 2 bytes: f16 scale
///   - 32 bytes: int8 quantized values
/// Output: `num_elements` f32 values.
pub fn dequantize_q8_to_f32(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    assert_eq!(bytes.len(), num_blocks * 34, "Q8_0 byte count mismatch");
    let mut out = vec![0.0f32; num_elements];
    for block in 0..num_blocks {
        let bo = block * 34;
        let d = f16_to_f32(u16::from_le_bytes([bytes[bo], bytes[bo + 1]]));
        let base = block * 32;
        for j in 0..32 {
            let q = bytes[bo + 2 + j] as i8;
            out[base + j] = q as f32 * d;
        }
    }
    out
}

/// Load an F32 linear weight from GGUF, handling F32, F16, Q4_0, and Q8_0 tensors.
///
/// Quantized tensors are dequantized to F32 at load time.
/// Returns (f32_data, [out_features, in_features]).
pub fn load_f32_weight_any<R: Read + Seek>(
    reader: &mut GgufReader<R>,
    name: &str,
) -> Result<(Vec<f32>, [usize; 2])> {
    let info = reader
        .tensor_info(name)
        .with_context(|| format!("Tensor '{name}' not found"))?
        .clone();
    let shape = reverse_gguf_dims(info.shape());
    let bytes = reader.tensor_data(name)?;
    let data = match info.dtype() {
        GgmlDtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        GgmlDtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        GgmlDtype::Q4_0 => {
            let num_elements = info.num_elements() as usize;
            dequantize_q4_to_f32(&bytes, num_elements)
        }
        GgmlDtype::Q8_0 => {
            let num_elements = info.num_elements() as usize;
            dequantize_q8_to_f32(&bytes, num_elements)
        }
        other => bail!("Cannot load {:?} tensor '{name}' as f32 weight", other),
    };
    Ok((data, [shape[0], shape[1]]))
}

// ---------------------------------------------------------------------------
// Q8Tensor -- GPU buffer of Q8_0 blocks
// ---------------------------------------------------------------------------

/// A Q8_0 quantized weight tensor living on GPU.
///
/// The buffer contains raw Q8_0 blocks (34 bytes per block of 32 elements).
/// The WGSL shader interprets the buffer as `array<u32>`.
pub struct Q8Tensor {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
}

impl Q8Tensor {
    /// Upload raw Q8_0 bytes to a GPU storage buffer.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`.
    /// `raw_bytes` must contain exactly `(N * K / 32) * 34` bytes.
    pub fn from_q8_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q8_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 34;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q8_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = WgpuRuntime::client(device);

        // Pad to 4-byte alignment for array<u32> access in the WGSL shader.
        let padded = if !raw_bytes.len().is_multiple_of(4) {
            let pad = 4 - (raw_bytes.len() % 4);
            let mut buf = raw_bytes.to_vec();
            buf.resize(raw_bytes.len() + pad, 0);
            buf
        } else {
            raw_bytes.to_vec()
        };
        let handle = client.create_from_slice(&padded);

        Ok(Self {
            handle,
            shape,
            num_blocks,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

// ---------------------------------------------------------------------------
// Q8Linear
// ---------------------------------------------------------------------------

/// A linear layer with Q8_0 quantized weights.
///
/// Stores weights as `[out_features, in_features]` in Q8_0 format and an
/// optional f32 bias. Forward: `x @ weights^T + bias` via fused dequant+matmul.
pub struct Q8Linear {
    weights: Q8Tensor,
    bias: Option<Tensor<Wgpu, 1>>,
}

impl Q8Linear {
    pub fn new(weights: Q8Tensor, bias: Option<Tensor<Wgpu, 1>>) -> Self {
        Self { weights, bias }
    }

    /// Forward pass: `x @ weights^T + bias`.
    ///
    /// `x` shape: `[B, M, K]` where `K = in_features`.
    /// Returns shape: `[B, M, N]` where `N = out_features`.
    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let out = q8_matmul(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}

// ---------------------------------------------------------------------------
// Q8 matmul kernel dispatch
// ---------------------------------------------------------------------------

struct Q8MatmulNaiveKernel {
    workgroup_size_x: u32,
    workgroup_size_y: u32,
}

impl KernelSource for Q8MatmulNaiveKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wgsl/shader_q8_0.wgsl"))
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.workgroup_size_x * 1000 + self.workgroup_size_y)
    }
}

/// Fused Q8_0 dequant+matmul on GPU.
///
/// Computes `output[B, M, N] = input[B, M, K] x weights[N, K]^T`.
pub fn q8_matmul(input: Tensor<Wgpu, 3>, weights: &Q8Tensor) -> Tensor<Wgpu, 3> {
    let cube_input: CubeTensor<WgpuRuntime> = input.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    assert_eq!(cube_input.shape.num_dims(), 3, "Input must be 3D [B, M, K]");
    let b = cube_input.shape.dims[0];
    let m = cube_input.shape.dims[1];
    let k = cube_input.shape.dims[2];
    let [n, wk] = weights.shape();
    assert_eq!(
        k, wk,
        "K dimension mismatch: input has {k}, weights have {wk}"
    );

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();
    let blocks_per_row = k / 32;

    let output_handle = client.empty(b * m * n * 4);

    let info: [u32; 5] = [
        b as u32,
        m as u32,
        k as u32,
        n as u32,
        blocks_per_row as u32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    let bindings = Bindings::new()
        .with_buffer(weights.handle.clone().binding())
        .with_buffer(cube_input.handle.clone().binding())
        .with_buffer(output_handle.clone().binding())
        .with_buffer(info_handle.binding());

    let kernel = SourceKernel::new(
        Q8MatmulNaiveKernel {
            workgroup_size_x: NAIVE_WG_X,
            workgroup_size_y: NAIVE_WG_Y,
        },
        CubeDim::new_2d(NAIVE_WG_X, NAIVE_WG_Y),
    );
    let wg_x = n.div_ceil(NAIVE_WG_X as usize) as u32;
    let wg_y = (b * m).div_ceil(NAIVE_WG_Y as usize) as u32;
    client
        .launch(
            Box::new(kernel) as Box<dyn CubeTask<AutoCompiler>>,
            CubeCount::new_2d(wg_x, wg_y),
            bindings,
        )
        .expect("Q8 naive matmul kernel launch failed");

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, m, n]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

/// Load a Q4_0 linear layer from GGUF (no bias).
pub fn load_q4_linear<R: Read + Seek>(
    reader: &mut GgufReader<R>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Q4Linear> {
    let info = reader
        .tensor_info(name)
        .with_context(|| format!("Tensor '{name}' not found"))?
        .clone();

    if info.dtype() != GgmlDtype::Q4_0 {
        bail!("Expected Q4_0 for '{name}', got {:?}", info.dtype());
    }

    let shape = reverse_gguf_dims(info.shape());
    let bytes = reader.tensor_data(name)?;
    let q4 = Q4Tensor::from_q4_bytes(&bytes, [shape[0], shape[1]], device)?;
    Ok(Q4Linear::new(q4, None))
}

/// Load a Q8_0 linear layer from GGUF (no bias).
///
/// Falls back to F32Linear (via `load_f32_weight_any`) when the tensor is F16/F32.
pub fn load_q8_linear<R: Read + Seek>(
    reader: &mut GgufReader<R>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Q8Linear> {
    let info = reader
        .tensor_info(name)
        .with_context(|| format!("Tensor '{name}' not found"))?
        .clone();

    if info.dtype() != GgmlDtype::Q8_0 {
        bail!("Expected Q8_0 for '{name}', got {:?}", info.dtype());
    }

    let shape = reverse_gguf_dims(info.shape());
    let bytes = reader.tensor_data(name)?;
    let q8 = Q8Tensor::from_q8_bytes(&bytes, [shape[0], shape[1]], device)?;
    Ok(Q8Linear::new(q8, None))
}

/// Load a linear layer from GGUF as Q8Linear if Q8_0, otherwise dequant to F32Linear.
///
/// Returns `Ok((Some(q8), None))` for Q8_0 or `Ok((None, Some(f32)))` for other dtypes.
pub fn load_q8_or_f32_linear<R: Read + Seek>(
    reader: &mut GgufReader<R>,
    name: &str,
    device: &WgpuDevice,
) -> Result<VVLinearLoaded> {
    use crate::model::vibevoice::VVLinear;
    use crate::model::F32Linear;
    use burn::tensor::TensorData;

    let info = reader
        .tensor_info(name)
        .with_context(|| format!("Tensor '{name}' not found"))?
        .clone();

    if info.dtype() == GgmlDtype::Q8_0 {
        let shape = reverse_gguf_dims(info.shape());
        let bytes = reader.tensor_data(name)?;
        let q8 = Q8Tensor::from_q8_bytes(&bytes, [shape[0], shape[1]], device)?;
        Ok(VVLinear::Q8(Q8Linear::new(q8, None)))
    } else {
        let (data, shape) = load_f32_weight_any(reader, name)?;
        let w = Tensor::<Wgpu, 2>::from_data(TensorData::new(data, [shape[0], shape[1]]), device);
        Ok(VVLinear::F32(F32Linear::new(w, None)))
    }
}

pub type VVLinearLoaded = crate::model::vibevoice::VVLinear;

/// Load an RMS norm weight from GGUF and create a Burn RmsNorm layer.
pub fn load_rms_norm<R: Read + Seek>(
    reader: &mut GgufReader<R>,
    name: &str,
    eps: f64,
    device: &WgpuDevice,
) -> Result<burn::nn::RmsNorm<Wgpu>> {
    let data = load_f32_tensor(reader, name)?;
    let num_elements = data.len();
    let weight: Tensor<Wgpu, 1> =
        Tensor::from_data(TensorData::new(data, [num_elements]), device);
    Ok(burn::nn::RmsNorm {
        gamma: Param::initialized(ParamId::new(), weight),
        epsilon: eps,
    })
}
