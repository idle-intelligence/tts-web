#!/usr/bin/env python3
"""Analyze Chrome DevTools trace for TTS worker performance.

Usage: python scripts/analyze_trace.py <trace.json>
"""
import json
import sys

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    print(f"Loaded {len(events)} events from {sys.argv[1]}\n")

    # Find worker thread by looking for HandlePostMessage with long duration
    thread_durations = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("name") == "HandlePostMessage" and e.get("ph") == "X":
            tid = e.get("tid")
            dur = e.get("dur", 0)
            if tid not in thread_durations:
                thread_durations[tid] = []
            thread_durations[tid].append(dur)

    # Worker thread = the one with the longest HandlePostMessage events
    worker_tid = None
    if thread_durations:
        worker_tid = max(thread_durations, key=lambda t: max(thread_durations[t]))
        print(f"Worker thread: tid={worker_tid}")
    else:
        print("No HandlePostMessage events found")
        sys.exit(1)

    # All HandlePostMessage on worker (= generation calls)
    gen_events = sorted(
        [e for e in events if isinstance(e, dict)
         and e.get("tid") == worker_tid
         and e.get("name") == "HandlePostMessage"
         and e.get("ph") == "X"],
        key=lambda e: e.get("ts", 0),
    )

    print(f"\n{'#':>3}  {'Duration':>10}  {'Timestamp':>12}  {'Gap':>8}")
    print("-" * 45)
    prev_end = None
    for i, e in enumerate(gen_events):
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        gap = f"{(ts - prev_end) / 1e3:.0f}ms" if prev_end else "—"
        prev_end = ts + dur
        print(f"{i:3d}  {dur / 1e3:>8.0f}ms  {ts / 1e6:>10.3f}s  {gap:>8}")

    # GC events on worker
    gc_events = [e for e in events if isinstance(e, dict)
                 and e.get("tid") == worker_tid
                 and "GC" in str(e.get("name", ""))
                 and e.get("dur", 0) > 1000]
    if gc_events:
        gc_events.sort(key=lambda e: e.get("dur", 0), reverse=True)
        print(f"\nGC events on worker (>1ms): {len(gc_events)}")
        for e in gc_events[:10]:
            print(f"  {e['name']} dur={e['dur']/1e3:.1f}ms")
    else:
        print("\nNo significant GC events on worker thread")

    # WASM compilation events
    wasm_events = [e for e in events if isinstance(e, dict)
                   and "wasm" in str(e.get("name", "")).lower()
                   and e.get("ph") == "X"
                   and e.get("dur", 0) > 1000]
    if wasm_events:
        print(f"\nWASM compilation events (>1ms): {len(wasm_events)}")
        for e in sorted(wasm_events, key=lambda e: e["dur"], reverse=True)[:5]:
            print(f"  {e['name']} dur={e['dur']/1e3:.1f}ms")

    # Summary
    if len(gen_events) >= 2:
        durations = [e["dur"] for e in gen_events if e["dur"] > 10000]
        if durations:
            print(f"\nSummary ({len(durations)} generations):")
            print(f"  First: {durations[0]/1e3:.0f}ms")
            print(f"  Last:  {durations[-1]/1e3:.0f}ms")
            print(f"  Min:   {min(durations)/1e3:.0f}ms")
            print(f"  Max:   {max(durations)/1e3:.0f}ms")
            print(f"  Avg:   {sum(durations)/len(durations)/1e3:.0f}ms")
            if durations[-1] > durations[0] * 1.3:
                print(f"  ⚠ Last generation is {durations[-1]/durations[0]:.1f}x slower than first — possible leak")
            else:
                print(f"  ✓ No degradation trend detected")


if __name__ == "__main__":
    main()
