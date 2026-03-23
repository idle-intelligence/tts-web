#!/Users/tc/Code/idle-intelligence/tts-web/scripts/venv/bin/python3
"""Dump per-step intermediate values from TADA-1B generation into a .npz file.

For each step where step >= shift_acoustic, saves:
  hidden_state_step_{i}      — LLM hidden state (condition vector) passed to flow matching
  flow_input_step_{i}        — noisy input to flow matching (initial noise sample, full dim)
  flow_output_step_{i}       — full 528-dim output after flow ODE solve
  acoustic_step_{i}          — 512-dim acoustic features extracted from flow output
  time_bits_step_{i}         — 16-dim gray-code raw float values (before thresholding)
  time_before_step_{i}       — decoded time_before integer (from flow output)
  time_after_step_{i}        — decoded time_after integer (from flow output)
  input_token_step_{i}       — input token ID at this step
  time_before_input_step_{i} — time_before value fed AS INPUT to this step
  time_after_input_step_{i}  — time_after value fed AS INPUT to this step

Usage:
    scripts/venv/bin/python3 scripts/save_generation_debug.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
REFS_TADA = os.path.join(REPO_ROOT, "refs", "tada")
if REFS_TADA not in sys.path:
    sys.path.insert(0, REFS_TADA)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEXT = "Time is money, who can afford to pay attention?"
OUTPUT_NPZ = "/tmp/python_debug_time.npz"
MODEL_DIR = "/Users/tc/Code/idle-intelligence/hf/tada-1b"
VOICE_NAME = "ljspeech"
TEMPERATURE = 0.9    # noise_temperature
NOISE_TEMP = 0.6     # text_temperature
FLOW_STEPS = 10
SEED = 42

# ---------------------------------------------------------------------------
# Tokenizer path patch must happen before any tada imports
# ---------------------------------------------------------------------------
def _ensure_llama_tokenizer_local():
    local_path = "/Users/tc/Code/idle-intelligence/hf/Llama-3.2-1B"
    if os.path.exists(os.path.join(local_path, "tokenizer.json")):
        return local_path
    tmp_path = "/tmp/llama-3.2-1b-tokenizer"
    if not os.path.exists(os.path.join(tmp_path, "tokenizer.json")):
        print("[debug] Downloading Llama-3.2-1B tokenizer ...", file=sys.stderr)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        tok.save_pretrained(tmp_path)
    return tmp_path

_LLAMA_TOKENIZER_PATH = _ensure_llama_tokenizer_local()
import tada.modules.aligner as _aligner_mod
_aligner_mod.AlignerConfig.tokenizer_name = _LLAMA_TOKENIZER_PATH  # type: ignore[assignment]

import torch
import numpy as np
import json

from tada.modules.encoder import EncoderOutput
import tada.modules.tada as _tada_mod
from tada.modules.tada import TadaForCausalLM, InferenceOptions
from tada.utils.gray_code import decode_gray_code_to_time

# ---------------------------------------------------------------------------
# Shim transformers API changes (same as tada_reference_generate.py)
# ---------------------------------------------------------------------------
from transformers import GenerationMixin
from transformers.cache_utils import DynamicCache

_orig_prepare_generation_config = GenerationMixin._prepare_generation_config

def _shim_prepare_generation_config(self, generation_config, *args, **kwargs):
    return _orig_prepare_generation_config(self, generation_config, **kwargs)

GenerationMixin._prepare_generation_config = _shim_prepare_generation_config  # type: ignore[method-assign]

def _shim_prepare_cache(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, *args, **kwargs):
    if "past_key_values" not in model_kwargs or model_kwargs.get("past_key_values") is None:
        model_kwargs["past_key_values"] = DynamicCache()

GenerationMixin._prepare_cache_for_generation = _shim_prepare_cache  # type: ignore[method-assign]

# Patch acoustic_features dtype (bfloat16 model needs float cast)
_orig_forward_one_step = _tada_mod.TadaForCausalLM.forward_one_step

def _patched_forward_one_step(self, *args, acoustic_features=None, **kwargs):
    if acoustic_features is not None:
        acoustic_features = acoustic_features.to(dtype=next(self.parameters()).dtype)
    return _orig_forward_one_step(self, *args, acoustic_features=acoustic_features, **kwargs)

_tada_mod.TadaForCausalLM.forward_one_step = _patched_forward_one_step  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# Module-level accumulators — populated during generation, read after
# ---------------------------------------------------------------------------
_debug_flow_records: list[dict] = []     # [{_idx, flow_input, hidden_state, flow_output}, ...]
_debug_input_capture: dict = {}          # {acoustic_step_idx: {token_id, tb_in, ta_in}}
_debug_acoustic_step_idx: list = [0]     # mutable counter shared by closures


def _make_capture_hooks():
    """Return patched _solve_flow_matching and forward_one_step functions
    that write into the module-level accumulators while forwarding to the originals."""

    _orig_solve = _tada_mod.TadaForCausalLM._solve_flow_matching
    _orig_fwd   = _tada_mod.TadaForCausalLM.forward_one_step

    def _capture_solve(self_c, speech, cond, neg_cond, *a, **kw):
        idx = _debug_acoustic_step_idx[0]
        flow_input_np = speech.detach().cpu().float().numpy().copy()
        hidden_np = cond.detach().cpu().float().numpy().copy()

        result = _orig_solve(self_c, speech, cond, neg_cond, *a, **kw)

        flow_output_np = result.detach().cpu().float().numpy().copy()
        _debug_flow_records.append({
            "_idx": idx,
            "flow_input": flow_input_np,
            "hidden_state": hidden_np,
            "flow_output": flow_output_np,
        })
        _debug_acoustic_step_idx[0] += 1
        return result

    def _capture_fwd(self_c, *args_fwd, **kw_fwd):
        # forward_one_step(input_ids, acoustic_features, acoustic_masks,
        #                  time_len_before, time_len_after, ...)
        # May be called with positional or keyword args; resolve both.
        positional_names = ["input_ids", "acoustic_features", "acoustic_masks",
                            "time_len_before", "time_len_after"]
        resolved = {}
        for idx_p, name in enumerate(positional_names):
            if idx_p < len(args_fwd):
                resolved[name] = args_fwd[idx_p]
            elif name in kw_fwd:
                resolved[name] = kw_fwd[name]

        ids_t = resolved.get("input_ids")
        tb_t  = resolved.get("time_len_before")
        ta_t  = resolved.get("time_len_after")

        tok_id = int(ids_t[0, 0].item()) if ids_t is not None and ids_t.numel() > 0 else -1
        tb_in  = int(tb_t[0, 0].item())  if tb_t  is not None and tb_t.numel()  > 0 else 0
        ta_in  = int(ta_t[0, 0].item())  if ta_t  is not None and ta_t.numel()  > 0 else 0

        # Store keyed by the current acoustic step index
        key = _debug_acoustic_step_idx[0]
        _debug_input_capture[key] = {"token_id": tok_id, "tb_in": tb_in, "ta_in": ta_in}
        return _orig_fwd(self_c, *args_fwd, **kw_fwd)

    return _capture_solve, _capture_fwd, _orig_solve, _orig_fwd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[debug] Device: {device}", file=sys.stderr)
    print(f"[debug] Text: {TEXT!r}", file=sys.stderr)
    print(f"[debug] Output: {OUTPUT_NPZ}", file=sys.stderr)
    print(f"[debug] noise_temperature={TEMPERATURE}, text_temperature={NOISE_TEMP}, flow_steps={FLOW_STEPS}, seed={SEED}", file=sys.stderr)

    # --- Load model ---
    print("[debug] Loading TadaForCausalLM ...", file=sys.stderr)
    model = TadaForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=False,
    )
    model.eval()
    model.to(device)

    shift_acoustic = model.config.shift_acoustic
    num_time_bits = model.num_time_bits
    time_dim = model.time_dim
    acoustic_dim = model.config.acoustic_dim

    print(f"[debug] shift_acoustic={shift_acoustic}, num_time_bits={num_time_bits}, time_dim={time_dim}", file=sys.stderr)
    print(f"[debug] acoustic_dim={acoustic_dim}, total_flow_dim={acoustic_dim + time_dim}", file=sys.stderr)

    # --- Load voice prompt ---
    voice_path = os.path.join(REPO_ROOT, "voices", VOICE_NAME + ".safetensors")
    meta_path = os.path.splitext(voice_path)[0] + ".json"
    print(f"[debug] Loading voice from {voice_path}", file=sys.stderr)

    from safetensors.torch import load_file as st_load
    tensors = st_load(voice_path, device=device)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        voice_text = meta.get("text", "")
        sample_rate = meta.get("sample_rate", 24000)
    else:
        voice_text = ""
        sample_rate = 24000

    token_values    = tensors["token_values"].unsqueeze(0).to(device)
    token_positions = tensors["token_positions"].unsqueeze(0).to(device)
    token_masks     = tensors["token_masks"].unsqueeze(0).to(device)
    audio_len       = tensors["audio_len"].unsqueeze(0).to(device)

    from tada.modules.aligner import AlignerConfig
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(AlignerConfig.tokenizer_name)
    text_token_ids_voice = tokenizer.encode(voice_text, add_special_tokens=False)
    text_tokens_len = torch.tensor([len(text_token_ids_voice)], dtype=torch.long, device=device)
    text_tokens = torch.tensor([text_token_ids_voice], dtype=torch.long, device=device)

    prompt = EncoderOutput(
        audio=torch.zeros(1, int(audio_len.item()), device=device),
        audio_len=audio_len,
        text=[voice_text],
        text_tokens=text_tokens,
        text_tokens_len=text_tokens_len,
        token_positions=token_positions,
        token_values=token_values,
        token_masks=token_masks,
        sample_rate=sample_rate,
    )

    opts = InferenceOptions(
        noise_temperature=TEMPERATURE,
        text_temperature=NOISE_TEMP,
        num_flow_matching_steps=FLOW_STEPS,
    )

    # --- Install capture hooks ---
    _debug_flow_records.clear()
    _debug_input_capture.clear()
    _debug_acoustic_step_idx[0] = 0

    _capture_solve, _capture_fwd, _orig_solve, _orig_fwd = _make_capture_hooks()
    _tada_mod.TadaForCausalLM._solve_flow_matching = _capture_solve  # type: ignore[method-assign]
    _tada_mod.TadaForCausalLM.forward_one_step = _capture_fwd        # type: ignore[method-assign]

    print("[debug] Generating ...", file=sys.stderr)
    try:
        with torch.no_grad():
            output = model.generate(
                prompt=prompt,
                text=TEXT,
                num_transition_steps=5,
                num_extra_steps=0,
                inference_options=opts,
            )
    finally:
        _tada_mod.TadaForCausalLM._solve_flow_matching = _orig_solve
        _tada_mod.TadaForCausalLM.forward_one_step = _orig_fwd

    debug_records = _debug_flow_records
    input_capture = _debug_input_capture

    print(f"[debug] Generation complete. Captured {len(debug_records)} acoustic steps.", file=sys.stderr)

    # --- Build npz ---
    npz_data = {}

    for rec in debug_records:
        i = rec["_idx"]
        fo = rec["flow_output"]   # could be (1, total_dim) or (total_dim,)
        fi = rec["flow_input"]    # same
        hs = rec["hidden_state"]  # (1, 1, hidden_dim) or (1, hidden_dim) or (hidden_dim,)

        # Flatten to 1D / (hidden_dim,)
        fo = fo.reshape(-1)
        fi = fi.reshape(-1)
        if hs.ndim == 3:
            hs = hs[0, 0]
        elif hs.ndim == 2:
            hs = hs[0]
        # hs is now (hidden_dim,)

        acoustic_arr  = fo[:acoustic_dim]    # (512,)
        time_bits_arr = fo[acoustic_dim:]    # (time_dim,) = (16,)

        # Decode gray code bits → integers
        time_bits_t = torch.from_numpy(time_bits_arr.astype(np.float32))
        time_before_int = int(decode_gray_code_to_time(
            time_bits_t[:num_time_bits].unsqueeze(0), num_time_bits).squeeze())
        time_after_int  = int(decode_gray_code_to_time(
            time_bits_t[num_time_bits:].unsqueeze(0), num_time_bits).squeeze())

        # Input token and time inputs from the forward_one_step capture
        cap = input_capture.get(i, {})
        token_id = cap.get("token_id", -1)
        tb_in    = cap.get("tb_in", 0)
        ta_in    = cap.get("ta_in", 0)

        npz_data[f"hidden_state_step_{i}"]      = hs
        npz_data[f"flow_input_step_{i}"]        = fi
        npz_data[f"flow_output_step_{i}"]       = fo
        npz_data[f"acoustic_step_{i}"]          = acoustic_arr
        npz_data[f"time_bits_step_{i}"]         = time_bits_arr
        npz_data[f"time_before_step_{i}"]       = np.array(time_before_int, dtype=np.int64)
        npz_data[f"time_after_step_{i}"]        = np.array(time_after_int,  dtype=np.int64)
        npz_data[f"input_token_step_{i}"]       = np.array(token_id,        dtype=np.int64)
        npz_data[f"time_before_input_step_{i}"] = np.array(tb_in,           dtype=np.int64)
        npz_data[f"time_after_input_step_{i}"]  = np.array(ta_in,           dtype=np.int64)

    np.savez(OUTPUT_NPZ, **npz_data)
    print(f"\n[debug] Saved {len(npz_data)} arrays to {OUTPUT_NPZ}", file=sys.stderr)

    # --- Report ---
    num_steps_captured = len(debug_records)
    print(f"\n[debug] Captured {num_steps_captured} acoustic steps (step >= shift_acoustic={shift_acoustic})", file=sys.stderr)

    if num_steps_captured > 0:
        print("\nArray shapes for step 0:", file=sys.stderr)
        for key_prefix in [
            "hidden_state_step", "flow_input_step", "flow_output_step",
            "acoustic_step", "time_bits_step", "time_before_step",
            "time_after_step", "input_token_step",
            "time_before_input_step", "time_after_input_step",
        ]:
            k = f"{key_prefix}_0"
            if k in npz_data:
                arr = npz_data[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}", file=sys.stderr)

        print(f"\nAll {num_steps_captured} steps:", file=sys.stderr)
        print(f"  {'i':>3}  {'tok_id':>8}  {'tb_in':>6}  {'ta_in':>6}  {'tb_out':>6}  {'ta_out':>6}", file=sys.stderr)
        for i in range(num_steps_captured):
            tok    = int(npz_data.get(f"input_token_step_{i}",       np.array(-1)))
            tb_in_v= int(npz_data.get(f"time_before_input_step_{i}", np.array(0)))
            ta_in_v= int(npz_data.get(f"time_after_input_step_{i}",  np.array(0)))
            tb_out = int(npz_data.get(f"time_before_step_{i}",       np.array(0)))
            ta_out = int(npz_data.get(f"time_after_step_{i}",        np.array(0)))
            print(f"  {i:>3}  {tok:>8}  {tb_in_v:>6}  {ta_in_v:>6}  {tb_out:>6}  {ta_out:>6}", file=sys.stderr)

    # Audio output info
    if output.audio and output.audio[0] is not None:
        wav = output.audio[0]
        duration = wav.shape[-1] / 24000
        print(f"\n[debug] Audio duration: {duration:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
