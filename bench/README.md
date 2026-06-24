# bench/

```sh
VAD_EX_BUILD=1 mix run bench/bench.exs                  # NIF (Elixir)
test/corpus/.venv/bin/python test/corpus/bench_ref.py   # reference (Python onnxruntime)
```

`bench.exs` reports per-chunk inference latency (mean / p50 / p99 / max), the implied real-time
factor and streams-per-core, and concurrent-`Session` throughput driven through the public
`VadEx.Session` API. It uses a real corpus clip (`test/corpus/data/speech_01.pcm`) when the corpus
has been fetched, otherwise a deterministic synthetic stream. A 512-sample chunk is 32 ms of audio,
so the realtime factor = 32 ms / mean latency and streams-per-core ≈ that factor.

**Reference target:** Silero v5 ONNX single CPU thread ≈ **189 µs per 31.25 ms chunk** (RTS ≈ 165×)
on a Ryzen Threadripper 3960X. The Rust/`ort` NIF should match or beat that.

**Measured (Apple Silicon, 10 schedulers, static ORT 1.24, 512-sample chunks):**
- single stream: ~139 µs/chunk mean, ~160 µs p99 → **~230× realtime** (beats the reference target)
- concurrency: 40 `Session`s, steady state → ~41k chunks/s, **~1326× realtime aggregate**
- Python `onnxruntime` 1.27 baseline on the same clip: ~89 µs/chunk

**Notes:**
- ORT intra-op + inter-op threads are already pinned to 1 in the NIF (`load_model`) — correct for a
  tiny `[1,576]` input where intra-op parallelism is pure overhead. So per-stream cost is minimal.
- ~1.6× slower per-call than Python is the BEAM↔NIF boundary: DirtyCpu scheduler hop + rustler term
  decode + per-call ndarray/Value allocation. Largely inherent; closing it would need chunk batching.
- Concurrent scaling is ~58% of the single-stream-×-cores ideal. `process_chunk` runs on a DirtyCpu
  scheduler, so parallelism is capped at the dirty-CPU scheduler count (≈ cores), not the `Session`
  count; the rest is the real `Session` path overhead (cast + telemetry span + endpointer per chunk).
- The concurrency number is steady-state: per-`Session` `load_model` + Level3 graph optimization is
  warmed up *outside* the timed window.
