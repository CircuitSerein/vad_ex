"""Reference (Python onnxruntime) per-chunk latency, to compare against the NIF (bench/bench.exs).

Same hot loop as reference_probs.py: one inference per 512-sample chunk. Uses data/speech_01.pcm
when present, else a deterministic synthetic stream.

    test/corpus/.venv/bin/python test/corpus/bench_ref.py
"""
import os
import time

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
MODEL = os.path.join(REPO, "priv", "models", "silero_vad.onnx")
CLIP = os.path.join(ROOT, "data", "speech_01.pcm")
WIN, CTX, SR = 512, 64, 16000
CHUNK_SECS = WIN / SR

if os.path.exists(CLIP):
    print(f"source: {os.path.relpath(CLIP, REPO)}")
    pcm = np.fromfile(CLIP, dtype="<i2").astype(np.float32) / 32768.0
else:
    print("source: synthetic (no corpus clip — run fetch_corpus.sh)")
    i = np.arange(SR * 2)
    pcm = ((i * 2741 % 4001) - 2000).astype(np.float32) / 32768.0

n_chunks = len(pcm) // WIN
loops = max(1, 3000 // max(1, n_chunks))

sess = ort.InferenceSession(MODEL)
out_names = [o.name for o in sess.get_outputs()]


def run(measure):
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros(CTX, dtype=np.float32)
    sr = np.array([SR], dtype=np.int64)
    us = []
    for _ in range(loops):
        for j in range(n_chunks):
            win = pcm[j * WIN:(j + 1) * WIN]
            inp = np.concatenate([ctx, win]).reshape(1, WIN + CTX).astype(np.float32)
            t0 = time.perf_counter()
            outs = sess.run(None, {"input": inp, "state": state, "sr": sr})
            if measure:
                us.append((time.perf_counter() - t0) * 1e6)
            named = dict(zip(out_names, outs))
            state = named["stateN"]
            ctx = win[-CTX:]
    return us


run(False)  # warmup
us = sorted(run(True))
mean = sum(us) / len(us)
p50 = us[len(us) // 2]
p99 = us[min(len(us) - 1, int(0.99 * (len(us) - 1)))]
rtf = CHUNK_SECS * 1e6 / mean
print("=== reference per-chunk latency (Python onnxruntime) ===")
print(f"  chunks      {len(us)}")
print(f"  mean        {mean:.1f} µs/chunk")
print(f"  p50         {p50:.1f} µs")
print(f"  p99         {p99:.1f} µs")
print(f"  max         {us[-1]:.1f} µs")
print(f"  realtime    {rtf:.1f}x")
