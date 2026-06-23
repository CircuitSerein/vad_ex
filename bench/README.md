# bench/

Latency / RTF benchmark harness. Publish the result table in the README.

**Target to beat / match:** Silero v5 ONNX single CPU thread ≈ **189 µs per 31.25 ms chunk**
(RTS ≈ 165×) on a Ryzen Threadripper 3960X. The Elixir NIF path via `ort` should be in the
same range or better (no Python overhead).

**What to measure:**
- per-chunk inference latency (µs), mean + P99, over a 1-hour stream in a `VadEx.Session` loop
- end-to-end throughput: max concurrent `VadEx.Session` processes on one node
- vs Python `silero-vad` baseline on identical audio

See `docs/build-plan.md` §8.
