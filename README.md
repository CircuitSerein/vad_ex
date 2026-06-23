# vad_ex

**Streaming Voice Activity Detection + endpointing for the BEAM.**

Real-time speech/silence detection over an audio stream, plus utterance endpointing
(when did a turn start/end) — backed by [Silero VAD](https://github.com/snakers4/silero-vad)
running through a Rust ([ONNX Runtime](https://github.com/pykeio/ort)) NIF. Ships as a
**precompiled** binary, so consumers need no Rust toolchain.

> **Status: pre-release.** No functional `0.1.0` on hex yet — the first release (streaming VAD +
> endpointing, precompiled) is in progress. See [`CHANGELOG.md`](CHANGELOG.md) and
> [`docs/architecture.md`](docs/architecture.md).

## Why

As of June 2026 there is **no streaming VAD package on hex.pm**. The Membrane / `ex_webrtc`
transport layer is mature, but the VAD logic layer above it is empty — the only prior art is
a [tutorial gist by Underjord](https://underjord.io/voice-activity-detection-elixir-membrane.html)
(Silero v4 via the now-stale `ortex`, no endpointing, not packaged). `vad_ex` fills that gap:
v5/v6 model, precompiled NIF, real endpointing, telemetry.

## Planned API

```elixir
# Standalone, one supervised process per audio stream
{:ok, vad} = VadEx.Session.start_link(sample_rate: 16_000, threshold: 0.5)
:ok = VadEx.Session.process(vad, pcm_chunk_512_samples_s16le)
# → emits [:vad_ex, :speech_start] / [:vad_ex, :speech_end] telemetry

# Or as a Membrane element, between an Opus decoder and your sink
child(:vad, %VadEx.Membrane.Filter{threshold: 0.5})
```

## License

MIT. The bundled Silero VAD model is MIT (snakers4/silero-vad).
