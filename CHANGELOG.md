# Changelog

All notable changes to `vad_ex` are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [0.1.0] — 2026-06-24

First release: minimum-shippable streaming VAD + endpointing on hex, precompiled (no Rust toolchain
needed by consumers).

### Added — Stages 0–3 implemented & verified on a real build (2026-06-24)
- **Silero v6.2 ONNX inference** through the Rust NIF (`process_chunk`): s16le decode → 64-sample
  context concat → unified `state[2,1,128]` → `ort` run → state threading. Verified live against a
  golden vector (`test/fixtures/golden_v6_2.json`).
- **Model bundling**: `priv/models/silero_vad.onnx` (v6.2, MIT, 2,327,524 B) + `mix vad_ex.fetch_model`
  (pinned tag + SHA-256 verify).
- **Membrane filter**: `VadEx.Membrane.Filter` re-windows the byte stream to exact 512-sample chunks
  (accumulates across arbitrary upstream buffering), passes audio through unchanged, emits the same
  telemetry as `Session`.
- **Tests** (`mix test --include nif`): endpointer unit + NIF golden/reset/error + Session integration
  + Membrane `Testing.Pipeline` (passthrough + odd-buffer accumulation). 12 tests, 0 failures.
- Bench (M-series, single-thread ORT, v6): **~140 µs/chunk, ~229× realtime**.

### Added — Stage 4: precompiled packaging (config + CI authored, locally verified) (2026-06-24)
- **Static ONNX Runtime linking**: `ort` built with `download-binaries` + `tls-rustls`,
  so the NIF is a single self-contained `.so`/`.dll` (no `libonnxruntime` to ship). Verified via
  `otool -L` (no libonnxruntime ref; 23 MB) and 12 tests green with `ORT_DYLIB_PATH` unset.
- **`.github/workflows/release.yml`**: precompiled build matrix (4 lean targets, native runner each,
  no cross) via `philss/rustler-precompiled-action`, uploads artifacts to the GitHub release on tag.
- **`.github/workflows/ci.yml`**: source-build test on Linux + macOS, plus `cargo fmt`/`clippy` lint.

### Added — Stage 5: real-audio validation + benchmark (2026-06-24)
- **Real-audio gate** (`test/real_audio_test.exs`, `@tag :real_audio`): the NIF reproduces reference
  Silero (Python `onnxruntime` 1.27) **within 2e-3** across a real corpus — 20 LibriSpeech speakers
  (CC BY 4.0) + 6 ESC-50 environmental-noise clips — cross-runtime (NIF ORT 1.24 vs Python 1.27),
  plus behavioral asserts (speech drives prob high; sirens/horns/fireworks/engines never read as
  speech). Corpus is fetched on demand (`test/corpus/`, pure-Python decode); audio is gitignored,
  only the derived golden vector + per-clip sha256 are committed.
- **Benchmark harness** (`bench/bench.exs` + `test/corpus/bench_ref.py`): single stream
  **~139 µs/chunk (p99 160 µs) → ~230× realtime**; 40 concurrent `Session`s → ~41k chunks/s,
  **~1326× realtime aggregate** (10 cores). Python `onnxruntime` 1.27 baseline ~89 µs/chunk.

### Changed
- Model pinned to **Silero v6** (was v5) — drop-in identical tensor contract, lower error rates.
- **Linking: `load-dynamic` → static.** Removed the `init_ort_from` and `ensure_initialized` NIFs
  and their call sites — no runtime ORT init needed (the runtime is statically linked).
- Precompiled targets trimmed to a lean 4 (darwin-arm64, linux-gnu x64/arm64, windows-msvc-x64);
  `nif_versions ["2.15"]`. `force_build` is now auto-on for `*-dev` versions, so a dev checkout's
  `mix test` builds the NIF from source without `VAD_EX_BUILD`.
- Dependency corrections: `rustler 0.34 → 0.38`, `ndarray 0.16 → 0.17`, `ort` `+api-24` feature
  (rc.12 targets ONNX Runtime **1.24**, not 1.26). Rust resource API → `Resource` trait + `#[resource_impl]`.

### Fixed
- The endpointer's `ms_to_chunks` helper rounded down (`250ms → 7`); now rounds up (`→ 8`), matching
  the struct default and the unit test.
- `new_stream` now returns `{:ok, stream}` (matches `Session.init`).

### Scaffolded
- Project skeleton: mix project, Rust NIF crate (`native/vad_ex`), module stubs.
- Key design decisions: Silero over TEN-VAD, a Rustler + `ort` NIF over `ortex`, dual
  GenServer / Membrane API. See [`docs/architecture.md`](docs/architecture.md).

[0.1.0]: https://github.com/CircuitSerein/vad_ex/releases/tag/v0.1.0
