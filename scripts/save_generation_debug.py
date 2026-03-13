#!/usr/bin/env python3
"""Save per-step generation intermediates from Python TADA for comparison with Rust.

Captures hidden states, acoustic vectors, time predictions, and top logits
at each step of the generation loop.
"""
import math
import os
import struct
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
REFS_TADA = os.path.join(REPO_ROOT, "refs", "tada")
if REFS_TADA not in sys.path:
    sys.path.insert(0, REFS_TADA)

HF_MODEL_PATH = "/Users/tc/Code/idle-intelligence/hf/tada-1b"
TEXT = "The quick brown fox jumps over the lazy dog."
OUTPUT_PATH = os.path.join(REPO_ROOT, "samples", "python_generation_debug.bin")
FLOW_STEPS = 10
SEED = 42
NUM_EXTRA_STEPS = 50
NUM_TOP_LOGITS = 5

HIDDEN_DIM = 2048
ACOUSTIC_DIM = 512


def main():
    import torch
    torch.manual_seed(SEED)

    # --- Monkey-patches (same as test_gguf_zeroshot.py) ---
    def _ensure_llama_tokenizer_local():
        local_path = "/tmp/llama-3.2-1b-tokenizer"
        if not os.path.exists(os.path.join(local_path, "tokenizer.json")):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
            tok.save_pretrained(local_path)
        return local_path

    _LLAMA_TOKENIZER_PATH = _ensure_llama_tokenizer_local()
    import tada.modules.aligner as _aligner_mod
    _aligner_mod.AlignerConfig.tokenizer_name = _LLAMA_TOKENIZER_PATH

    from transformers import GenerationMixin
    from transformers.cache_utils import DynamicCache

    _orig_prepare = GenerationMixin._prepare_generation_config
    def _shim_prepare(self, generation_config, *args, **kwargs):
        return _orig_prepare(self, generation_config, **kwargs)
    GenerationMixin._prepare_generation_config = _shim_prepare

    _orig_cache = GenerationMixin._prepare_cache_for_generation
    def _shim_cache(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, *args, **kwargs):
        if "past_key_values" not in model_kwargs or model_kwargs.get("past_key_values") is None:
            model_kwargs["past_key_values"] = DynamicCache()
    GenerationMixin._prepare_cache_for_generation = _shim_cache

    import tada.modules.tada as _tada_mod
    _orig_fos = _tada_mod.TadaForCausalLM.forward_one_step
    def _patched_fos(self, *args, acoustic_features=None, **kwargs):
        if acoustic_features is not None:
            model_dtype = next(self.parameters()).dtype
            acoustic_features = acoustic_features.to(dtype=model_dtype)
        return _orig_fos(self, *args, acoustic_features=acoustic_features, **kwargs)
    _tada_mod.TadaForCausalLM.forward_one_step = _patched_fos

    from tada.modules.tada import TadaForCausalLM, InferenceOptions
    from tada.modules.encoder import EncoderOutput

    # Patch from_pretrained to skip encoder/decoder loading (not needed for zero-shot debug)
    _orig_from_pretrained = TadaForCausalLM.from_pretrained.__func__
    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Skip encoder/decoder — call grandparent from_pretrained directly
        from transformers import PreTrainedModel
        self = PreTrainedModel.from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
        return self
    TadaForCausalLM.from_pretrained = _patched_from_pretrained

    # --- Load model ---
    print(f"Loading model from {HF_MODEL_PATH}...")
    model = TadaForCausalLM.from_pretrained(HF_MODEL_PATH, torch_dtype=torch.float32)

    # Remove weight_norm (decoder may have them if loaded)
    removed = 0
    for _, module in model.named_modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
            removed += 1
    if removed:
        print(f"Removed {removed} weight_norm parametrizations")
    model.eval()

    # --- Storage for captured data ---
    step_data = []  # list of dicts per step

    # --- Monkey-patch _generate to capture intermediates ---
    # We wrap the critical section: after forward_one_step and after flow matching
    _orig_generate = _tada_mod.TadaForCausalLM._generate

    def _instrumented_generate(self, input_ids, input_lengths, prompt_acoustic_features=None,
                                prompt_acoustic_masks=None, prompt_time_len_before=None,
                                prompt_time_len_after=None, num_steps=1024, log_time=True,
                                inference_options=InferenceOptions(), use_text_in_prompt=False,
                                verbose=False, return_logits=False, **kwargs):
        """Instrumented _generate that captures per-step data."""
        from tada.utils.gray_code import decode_gray_code_to_time

        start_header_id = self.tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if not use_text_in_prompt:
            prompt_token_len = prompt_acoustic_features.shape[1] if prompt_acoustic_features is not None else 0
            pad_id = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
            bos_id = self.tokenizer.bos_token_id
            eos_id = self.tokenizer.eos_token_id
            keep_mask = torch.zeros(input_ids.shape[0], prompt_token_len, dtype=torch.bool, device=input_ids.device)
            for b in range(input_ids.shape[0]):
                in_header = False
                for t in range(prompt_token_len):
                    token = input_ids[b, t].item()
                    if token == start_header_id:
                        in_header = True
                        keep_mask[b, t] = True
                    elif token == end_header_id:
                        in_header = False
                        keep_mask[b, t] = True
                    elif in_header:
                        keep_mask[b, t] = True
                    elif token in (eot_id, bos_id, eos_id):
                        keep_mask[b, t] = True
            input_ids = input_ids.clone()
            input_ids[:, :prompt_token_len][~keep_mask] = pad_id

        opts = inference_options
        acoustic_features = torch.zeros(input_ids.shape[0], 1, self.config.acoustic_dim, device=input_ids.device)
        acoustic_masks = torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)
        time_len_before = torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)
        time_len_after = torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)

        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, True)
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, 1, num_steps)
        model_kwargs["cache_position"] = torch.arange(1, device=input_ids.device, dtype=torch.long)

        all_acoustic_features = []
        all_time_before = []
        all_logits = []
        all_output_token_ids = []
        llm_time = []
        diffusion_time = []
        acoustic_feat_type = "none"
        time_len_type = "none"

        prompt_len = input_ids.shape[1]
        step_start = 0
        shift_acoustic = self.config.shift_acoustic
        neg_cond = torch.zeros(
            input_ids.shape[0], self.config.hidden_size, device=input_ids.device, dtype=input_ids.dtype
        )
        pad_token_id = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

        # No prefill in zero-shot mode (prompt_acoustic_features is empty)
        prefill_len = 0

        B = input_ids.shape[0]
        step_logs = []
        last_time_before = None

        for step in range(step_start, num_steps):
            input_slice = input_ids[:, step : step + 1] if step < input_ids.shape[1] else input_ids[:, -1:]
            token_id = input_slice[0, 0].item()

            # forward_one_step
            model_inputs = self.prepare_inputs_for_generation(input_slice, **model_kwargs)
            outputs = self.forward_one_step(
                **model_inputs,
                acoustic_features=acoustic_features,
                acoustic_masks=acoustic_masks,
                time_len_before=time_len_before,
                time_len_after=time_len_after,
                compute_logits=True,
                **kwargs,
            )

            assert outputs.hidden_states is not None
            hidden_states = outputs.hidden_states[-1][:B]
            logits = outputs.logits[:B]

            # Capture hidden state
            hidden_vec = hidden_states[0, 0].detach().float().cpu()  # [hidden_dim]

            # Top-k logits
            raw_logits = logits[0, 0].detach().float().cpu()  # [vocab_size]
            top_vals, top_idxs = torch.topk(raw_logits, NUM_TOP_LOGITS)

            # Flow matching (only for steps >= shift_acoustic)
            has_acoustic = False
            acoustic_vec = None
            tb_val = 0
            ta_val = 0

            cond = hidden_states

            if step >= shift_acoustic:
                # Run flow matching
                speech = torch.randn(cond.shape[0], self.config.acoustic_dim).to(cond) * opts.noise_temperature
                speech = torch.cat(
                    [speech, torch.randn(cond.shape[0], self.time_dim).to(cond) * opts.noise_temperature], dim=-1
                )
                speech = self._solve_flow_matching(
                    speech=speech,
                    cond=cond,
                    neg_cond=neg_cond.unsqueeze(0) if neg_cond.dim() == 1 else (neg_cond.unsqueeze(1) if neg_cond.dim() == 2 and neg_cond.shape != cond.shape else neg_cond),
                    num_steps=opts.num_flow_matching_steps,
                    acoustic_cfg_scale=opts.acoustic_cfg_scale,
                    duration_cfg_scale=opts.duration_cfg_scale,
                    cfg_schedule=opts.cfg_schedule,
                    time_schedule=opts.time_schedule,
                )

                time_len_gray_code = speech[..., -self.time_dim:]
                predicted_time_len_before = decode_gray_code_to_time(
                    time_len_gray_code[..., :self.num_time_bits], self.num_time_bits
                ).unsqueeze(0)
                predicted_time_len_after = decode_gray_code_to_time(
                    time_len_gray_code[..., self.num_time_bits:], self.num_time_bits
                ).unsqueeze(0)

                has_acoustic = True

                if (prompt_acoustic_features is not None
                    and step - shift_acoustic < prompt_acoustic_features.shape[1]):
                    acoustic_features_step = prompt_acoustic_features[:, step - shift_acoustic].unsqueeze(1)
                    acoustic_masks_step = prompt_acoustic_masks[:, step - shift_acoustic].unsqueeze(1)
                else:
                    acoustic_features_step = speech.unsqueeze(0)
                    acoustic_masks_step = torch.ones(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)
                    acoustic_features_step = (
                        acoustic_features_step[..., :-self.time_dim] if self.time_dim > 0 else acoustic_features_step
                    )
                all_acoustic_features.append(acoustic_features_step)

                # Denormalize for saving: acoustic_features * std + mean
                acoustic_raw = speech[0, :self.config.acoustic_dim].detach().float().cpu()
                acoustic_denorm = acoustic_raw * self.config.acoustic_std + self.config.acoustic_mean
                acoustic_vec = acoustic_denorm

                if (prompt_time_len_before is not None
                    and prompt_time_len_after is not None
                    and step - shift_acoustic < prompt_time_len_before.shape[1] - 1):
                    time_len_before_step = prompt_time_len_before[:, step - shift_acoustic + 1].unsqueeze(1)
                    time_len_after_step = prompt_time_len_after[:, step - shift_acoustic + 1].unsqueeze(1)
                else:
                    time_len_before_step = predicted_time_len_before
                    time_len_after_step = predicted_time_len_after

                all_time_before.append(time_len_before_step)
                last_time_before = time_len_before_step
                tb_val = time_len_before_step[0, 0].item()
                ta_val = time_len_after_step[0, 0].item()

                # Update for next step
                acoustic_features = acoustic_features_step
                acoustic_masks = acoustic_masks_step
                time_len_before = time_len_before_step
                time_len_after = time_len_after_step

            # Token sampling
            if step >= input_ids.shape[1] - 1:
                token_logits = logits[:, -1, :].clone()
                token_logits[:, pad_token_id] = float("-inf")

                if opts.text_do_sample:
                    if opts.text_repetition_penalty != 1.0:
                        score = torch.gather(token_logits, 1, input_ids)
                        score = torch.where(
                            score < 0,
                            score * opts.text_repetition_penalty,
                            score / opts.text_repetition_penalty,
                        )
                        token_logits = token_logits.scatter(1, input_ids, score)

                    token_logits = token_logits / opts.text_temperature

                    if opts.text_top_k > 0:
                        top_k = min(opts.text_top_k, token_logits.size(-1))
                        indices_to_remove = token_logits < torch.topk(token_logits, top_k, dim=-1).values[..., -1:]
                        token_logits = token_logits.masked_fill(indices_to_remove, float("-inf"))

                    if 0.0 < opts.text_top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = (
                            cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= opts.text_top_p
                        )
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        token_logits = token_logits.masked_fill(indices_to_remove, float("-inf"))

                    probs = torch.softmax(token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = token_logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token.long()], dim=1)
                all_output_token_ids.append(next_token)
            else:
                all_output_token_ids.append(input_ids[:, step + 1].unsqueeze(1))

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

            # Save step data
            step_info = {
                "step": step,
                "token_id": token_id,
                "hidden": hidden_vec,  # [hidden_dim]
                "has_acoustic": has_acoustic,
                "acoustic": acoustic_vec,  # [acoustic_dim] or None
                "time_before": tb_val,
                "time_after": ta_val,
                "top_logit_indices": top_idxs.tolist(),
                "top_logit_values": top_vals.tolist(),
            }
            step_data.append(step_info)

        # Append trailing time_before
        if last_time_before is not None:
            all_time_before.append(last_time_before)

        from tada.modules.tada import SyncTokGenerationOutput
        return SyncTokGenerationOutput(
            text_token_ids=torch.cat(all_output_token_ids, dim=1) if len(all_output_token_ids) > 0 else None,
            acoustic_features=torch.cat([f if f.ndim == 3 else f.unsqueeze(1) for f in all_acoustic_features], dim=1),
            time_before=torch.cat([f if f.ndim == 2 else f.unsqueeze(1) for f in all_time_before], dim=1),
            llm_time=torch.tensor(llm_time).mean() if llm_time else torch.tensor(0.0),
            diffusion_time=torch.tensor(diffusion_time).mean() if diffusion_time else torch.tensor(0.0),
            logits=torch.cat(all_logits, dim=1) if all_logits else None,
            step_logs=step_logs,
        )

    # Apply the instrumented _generate
    _tada_mod.TadaForCausalLM._generate = _instrumented_generate

    # --- Run generation ---
    prompt = EncoderOutput.empty("cpu")
    opts = InferenceOptions(
        noise_temperature=0.9,
        text_temperature=0.6,
        num_flow_matching_steps=FLOW_STEPS,
    )

    print(f"Generating zero-shot: {TEXT!r}")
    print(f"  seed={SEED}, flow_steps={FLOW_STEPS}, num_extra_steps={NUM_EXTRA_STEPS}")
    print(f"  noise_temp={opts.noise_temperature}, text_temp={opts.text_temperature}")
    print(f"  acoustic_cfg_scale={opts.acoustic_cfg_scale}, cfg_schedule={opts.cfg_schedule}")
    print(f"  time_schedule={opts.time_schedule}")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            prompt=prompt,
            text=TEXT,
            num_transition_steps=0,
            num_extra_steps=NUM_EXTRA_STEPS,
            inference_options=opts,
        )
    elapsed = time.time() - t0
    print(f"Generation: {elapsed:.1f}s, {len(step_data)} steps captured")

    # --- Print summary table ---
    print()
    print(f"{'step':>4} | {'tok_id':>6} | {'token':>20} | {'hid_mean':>10} | {'hid_std':>10} | {'ac_mean':>10} | {'ac_std':>10} | {'tb':>4} | {'ta':>4}")
    print("-" * 110)
    tokenizer = model.tokenizer
    for sd in step_data:
        h = sd["hidden"]
        hm = h.mean().item()
        hs = h.std().item()
        tok_str = tokenizer.convert_ids_to_tokens([sd["token_id"]])[0]
        if len(tok_str) > 20:
            tok_str = tok_str[:17] + "..."
        if sd["has_acoustic"]:
            a = sd["acoustic"]
            am = a.mean().item()
            astd = a.std().item()
            print(f"{sd['step']:4d} | {sd['token_id']:6d} | {tok_str:>20s} | {hm:10.6f} | {hs:10.6f} | {am:10.4f} | {astd:10.4f} | {sd['time_before']:4d} | {sd['time_after']:4d}")
        else:
            print(f"{sd['step']:4d} | {sd['token_id']:6d} | {tok_str:>20s} | {hm:10.6f} | {hs:10.6f} | {'n/a':>10s} | {'n/a':>10s} | {'n/a':>4s} | {'n/a':>4s}")

    # --- Save binary ---
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    num_steps = len(step_data)

    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<III", num_steps, HIDDEN_DIM, ACOUSTIC_DIM))

        for sd in step_data:
            f.write(struct.pack("<I", sd["token_id"]))

            # Hidden state
            h = sd["hidden"]
            assert h.shape[0] == HIDDEN_DIM, f"hidden dim mismatch: {h.shape[0]} vs {HIDDEN_DIM}"
            for val in h.numpy():
                f.write(struct.pack("<f", float(val)))

            # Acoustic
            has_ac = 1 if sd["has_acoustic"] else 0
            f.write(struct.pack("<I", has_ac))
            if sd["has_acoustic"]:
                a = sd["acoustic"]
                assert a.shape[0] == ACOUSTIC_DIM, f"acoustic dim mismatch: {a.shape[0]} vs {ACOUSTIC_DIM}"
                for val in a.numpy():
                    f.write(struct.pack("<f", float(val)))
                f.write(struct.pack("<II", int(sd["time_before"]), int(sd["time_after"])))

            # Top logits
            n_top = len(sd["top_logit_indices"])
            f.write(struct.pack("<I", n_top))
            for idx, val in zip(sd["top_logit_indices"], sd["top_logit_values"]):
                f.write(struct.pack("<If", int(idx), float(val)))

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nSaved {num_steps} steps to {OUTPUT_PATH} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
