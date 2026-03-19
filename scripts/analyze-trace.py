#!/usr/bin/env python3
"""Analyze Chrome DevTools performance traces for TADA WASM inference.

Usage: python scripts/analyze-trace.py /path/to/Trace-*.json
"""

import json
import sys

def analyze(path):
    with open(path) as f:
        trace = json.load(f)
    events = trace.get('traceEvents', trace) if isinstance(trace, dict) else trace
    print(f"Loaded {len(events)} events from {path}\n")

    # Find all threads
    tids = {}
    for e in events:
        if isinstance(e, dict) and e.get('name') == 'thread_name':
            tids[e.get('tid')] = e.get('args', {}).get('name', '')

    # Find worker threads (DedicatedWorker)
    worker_tids = [tid for tid, name in tids.items() if 'DedicatedWorker' in name]
    gpu_tid = next((tid for tid, name in tids.items() if 'CrGpuMain' in name), None)

    # Find the TADA worker — it's the one with the most long microtasks
    tada_tid = None
    max_long = 0
    for tid in worker_tids:
        count = sum(1 for e in events if isinstance(e, dict) and e.get('tid') == tid
                    and e.get('name') == 'RunMicrotasks' and e.get('dur', 0) > 50000)
        if count > max_long:
            max_long = count
            tada_tid = tid

    if not tada_tid:
        print("Could not find TADA worker thread")
        return

    print(f"TADA worker: tid={tada_tid} ({max_long} generation steps)\n")

    # Extract generation steps (RunMicrotasks > 50ms on TADA worker)
    steps = [e for e in events if isinstance(e, dict) and e.get('tid') == tada_tid
             and e.get('name') == 'RunMicrotasks' and e.get('dur', 0) > 50000]
    steps.sort(key=lambda e: e['ts'])

    if not steps:
        print("No generation steps found")
        return

    print(f"{'Step':>5}  {'Duration':>10}  {'Gap':>10}  {'Cumulative':>12}")
    print("-" * 45)

    total_dur = 0
    total_gap = 0
    warmup_time = 0
    decode_time = 0
    steady_durs = []

    for i, e in enumerate(steps):
        dur = e['dur'] / 1000
        if i == 0:
            gap = 0
        else:
            gap = (e['ts'] - (steps[i-1]['ts'] + steps[i-1]['dur'])) / 1000

        total_dur += dur
        total_gap += gap
        cum = total_dur + total_gap

        if i == 0:
            warmup_time = dur
        elif i == len(steps) - 1 and dur > 2000:
            decode_time = dur
        elif i >= 3:
            steady_durs.append(dur)

        if i < 5 or i >= len(steps) - 3:
            print(f"{i:5d}  {dur:9.1f}ms  {gap:9.1f}ms  {cum:11.1f}ms")
        elif i == 5:
            print(f"  ... ({len(steps) - 8} more steps) ...")

    print("-" * 45)

    total_span = (steps[-1]['ts'] + steps[-1]['dur'] - steps[0]['ts']) / 1000
    print(f"\nSummary:")
    print(f"  Total steps:       {len(steps)}")
    print(f"  Total span:        {total_span:.0f}ms ({total_span/1000:.1f}s)")
    print(f"  Total compute:     {total_dur:.0f}ms")
    print(f"  Total idle/gaps:   {total_gap:.0f}ms ({100*total_gap/total_span:.0f}%)")
    print(f"  Warmup (step 0):   {warmup_time:.0f}ms")
    if decode_time:
        print(f"  Decode (last step): {decode_time:.0f}ms")

    if steady_durs:
        avg = sum(steady_durs) / len(steady_durs)
        print(f"\nSteady state (steps 3-{len(steps)-2}):")
        print(f"  Avg step:          {avg:.0f}ms")
        print(f"  Min step:          {min(steady_durs):.0f}ms")
        print(f"  Max step:          {max(steady_durs):.0f}ms")
        print(f"  Step count:        {len(steady_durs)}")

    # GPU tasks analysis
    gpu_tasks = [e for e in events if isinstance(e, dict) and 'GPUTask' in str(e.get('name', ''))]
    if gpu_tasks:
        gpu_durs = [e['dur']/1000 for e in gpu_tasks]
        print(f"\nGPU tasks: {len(gpu_tasks)}")
        print(f"  Total:             {sum(gpu_durs):.0f}ms")
        print(f"  Avg:               {sum(gpu_durs)/len(gpu_durs):.1f}ms")
        print(f"  Max:               {max(gpu_durs):.0f}ms")
        big_gpu = [d for d in gpu_durs if d > 10]
        if big_gpu:
            print(f"  >10ms tasks:       {len(big_gpu)} (total {sum(big_gpu):.0f}ms)")

    # Readback stalls (gaps > 500ms in first few steps)
    early_gaps = []
    for i in range(1, min(5, len(steps))):
        gap = (steps[i]['ts'] - (steps[i-1]['ts'] + steps[i-1]['dur'])) / 1000
        if gap > 200:
            early_gaps.append((i, gap))
    if early_gaps:
        print(f"\nReadback stalls (early steps):")
        for step, gap in early_gaps:
            print(f"  Step {step}: {gap:.0f}ms gap")

    # Recommendations
    print(f"\n{'='*45}")
    print("RECOMMENDATIONS:")
    if warmup_time > 500:
        print(f"  [HIGH] Add GPU warmup — saves {warmup_time:.0f}ms on first gen")
    if early_gaps:
        total_stall = sum(g for _, g in early_gaps)
        print(f"  [HIGH] Fix readback stalls — saves {total_stall:.0f}ms")
    if steady_durs and avg > 300:
        print(f"  [HIGH] Reduce per-step time (currently {avg:.0f}ms, native is ~225ms)")
    if len(steps) > 40:
        print(f"  [HIGH] Skip VibeVoice during prompt phase — {len(steps)} steps is too many")
    if decode_time > 3000:
        print(f"  [MED]  Optimize decode ({decode_time:.0f}ms)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze-trace.py <trace.json>")
        sys.exit(1)
    analyze(sys.argv[1])
