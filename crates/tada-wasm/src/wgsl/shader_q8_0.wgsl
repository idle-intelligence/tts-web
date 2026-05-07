// Q8_0 Dequantization + Matrix Multiplication Compute Shader
//
// One-thread-per-output-element kernel, no shared memory.
// Uses block-level iteration with sequential accumulation.
// Compatible with WebGPU's 256 workgroup invocation limit (16x16=256).
//
// Computes: output[B, M, N] = input[B, M, K] x weights[N, K]^T
//
// Q8_0 block format: 34 bytes per block of 32 values.
//   Bytes 0-1: f16 scale
//   Bytes 2-33: 32 x int8 quantized values (signed -128..127)
//
// Dequant: value[i] = int8[i] * scale

@group(0) @binding(0) var<storage, read_write> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<u32>;

fn read_u32_unaligned(byte_offset: u32) -> u32 {
    let word = byte_offset >> 2u;
    let shift = (byte_offset & 3u) << 3u;
    if (shift == 0u) {
        return weights[word];
    }
    return (weights[word] >> shift) | (weights[word + 1u] << (32u - shift));
}

fn read_f16_scale(block_byte_offset: u32) -> f32 {
    let bits = read_u32_unaligned(block_byte_offset) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = info[0];
    let M = info[1];
    let K = info[2];
    let N = info[3];
    let blocks_per_row = info[4];

    let n = gid.x;
    let bm = gid.y;
    let m = bm % M;
    let b = bm / M;

    if (n >= N || b >= B) {
        return;
    }

    var acc: f32 = 0.0;
    let input_base = b * M * K + m * K;

    // Block-level iteration: read scale once per 32 elements
    for (var blk: u32 = 0u; blk < blocks_per_row; blk = blk + 1u) {
        let global_block = n * blocks_per_row + blk;
        // Q8_0: 34 bytes per block (2 scale + 32 int8)
        let block_byte = global_block * 34u;
        let scale = read_f16_scale(block_byte);
        let k_base = blk * 32u;

        // Read 32 int8 values as 8 x u32 words (4 bytes each)
        let data_start = block_byte + 2u;
        let k_off = input_base + k_base;

        for (var wi: u32 = 0u; wi < 8u; wi = wi + 1u) {
            let packed = read_u32_unaligned(data_start + wi * 4u);

            // Extract 4 bytes and sign-extend from int8
            let b0 = i32(packed & 0xFFu);
            let b1 = i32((packed >> 8u) & 0xFFu);
            let b2 = i32((packed >> 16u) & 0xFFu);
            let b3 = i32((packed >> 24u) & 0xFFu);

            let s0 = select(b0, b0 - 256, b0 > 127);
            let s1 = select(b1, b1 - 256, b1 > 127);
            let s2 = select(b2, b2 - 256, b2 > 127);
            let s3 = select(b3, b3 - 256, b3 > 127);

            let base_i = wi * 4u;

            // Sequential accumulation to match CPU precision
            acc = acc + f32(s0) * scale * input[k_off + base_i];
            acc = acc + f32(s1) * scale * input[k_off + base_i + 1u];
            acc = acc + f32(s2) * scale * input[k_off + base_i + 2u];
            acc = acc + f32(s3) * scale * input[k_off + base_i + 3u];
        }
    }

    output[b * M * N + m * N + n] = acc;
}
