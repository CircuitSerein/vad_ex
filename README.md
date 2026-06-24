# vad_ex

**Streaming Voice Activity Detection + endpointing for the BEAM.**

Real-time speech/silence detection over an audio stream, plus utterance **endpointing** (when did a
turn start and end) — backed by [Silero VAD](https://github.com/snakers4/silero-vad) v6 running
through a Rust ([ONNX Runtime](https://github.com/pykeio/ort)) NIF. The ONNX Runtime is **statically
linked** into a **precompiled** binary, so consumers need no Rust toolchain, no `libonnxruntime`, and
no system dependencies — just add the dep.

- **One supervised process per stream** (`VadEx.Session`) — 10k concurrent calls = 10k crash-isolated
  GenServers, the BEAM's structural edge for real-time voice at scale.
- **Endpointing built in** — a pure-Elixir state machine turns per-chunk probabilities into
  `speech_start` / `speech_end` events with configurable min-speech, min-silence, and padding.
- **`:telemetry` native** — per-chunk inference spans + speech-boundary events.
- **Optional Membrane filter** — drop `VadEx.Membrane.Filter` into a pipeline; core works without it.

## Why

As of mid-2026 there is **no streaming VAD package on hex.pm**. The Membrane / `ex_webrtc` transport
layer is mature, but the VAD *logic* layer above it is empty — the only prior art is a
[tutorial gist by Underjord](https://underjord.io/voice-activity-detection-elixir-membrane.html)
(Silero v4 via the now-stale `ortex`, no endpointing, not packaged). `vad_ex` fills that gap: current
v6 model, precompiled self-contained NIF, real endpointing, telemetry.

## Install

```elixir
def deps do
  [{:vad_ex, "~> 0.1"}]
end
```

A precompiled NIF is downloaded for your platform (macOS arm64, Linux x86_64/arm64, Windows x86_64).
No Rust toolchain required. To build from source instead, set `VAD_EX_BUILD=1` (needs Rust).

## Usage

```elixir
# One supervised process per audio stream.
{:ok, vad} = VadEx.Session.start_link(threshold: 0.5, min_silence_ms: 500)

# Speech-boundary events arrive via :telemetry — attach a handler.
:telemetry.attach_many(
  "my-vad",
  [[:vad_ex, :speech_start], [:vad_ex, :speech_end]],
  fn
    [:vad_ex, :speech_start], %{ts_ms: ts}, _meta, _ ->
      IO.puts("speech started at #{ts} ms")

    [:vad_ex, :speech_end], %{ts_ms: ts, duration_ms: dur}, _meta, _ ->
      IO.puts("utterance ended at #{ts} ms (#{dur} ms long)")
  end,
  nil
)

# Feed 512-sample (1024-byte) s16le @ 16 kHz chunks. process/2 is async (cast); results via telemetry.
:ok = VadEx.Session.process(vad, pcm_chunk)

# Reset RNN + endpointer state between calls/utterances.
:ok = VadEx.Session.reset(vad)
```

As a Membrane element, between (say) an Opus decoder and your sink:

```elixir
child(:vad, %VadEx.Membrane.Filter{threshold: 0.5})
```

### Session options

| option           | default  | meaning                                            |
|------------------|----------|----------------------------------------------------|
| `:threshold`     | `0.5`    | speech-probability cutoff                          |
| `:min_speech_ms` | `250`    | speech must persist this long to emit `speech_start` |
| `:min_silence_ms`| `500`    | silence this long ends the utterance               |
| `:speech_pad_ms` | `100`    | padding kept around detected speech                |
| `:sample_rate`   | `16_000` | input rate (16 kHz; 8 kHz is on the roadmap)       |
| `:metadata`      | `%{}`    | merged into every telemetry event (e.g. `%{call_id: …}`) |

### Telemetry events

- `[:vad_ex, :chunk, :start | :stop | :exception]` — per-chunk inference span (`:telemetry.span/3`).
- `[:vad_ex, :speech_start]` — measurement `%{ts_ms}`.
- `[:vad_ex, :speech_end]` — measurement `%{ts_ms, duration_ms}`.

## Performance

512-sample chunk = 32 ms of audio. Measured on Apple Silicon (10 schedulers), static ORT 1.24,
via `bench/bench.exs`:

| workload                             | result                                              |
|--------------------------------------|-----------------------------------------------------|
| single stream                        | **139 µs/chunk** (p99 160 µs) → **~230× realtime**  |
| 40 concurrent streams (10 cores)     | **41k chunks/s** → **~1326× realtime aggregate**    |
| reference: Python `onnxruntime` 1.27 | ~89 µs/chunk on the same clip                       |

One stream costs ~0.4% of a core; a single node serves on the order of a thousand concurrent
real-time streams. See [`bench/README.md`](bench/README.md) for methodology and the gap analysis.

## Accuracy

The NIF is validated against reference Silero (Python `onnxruntime`) on a real-audio corpus —
20 LibriSpeech speakers + ESC-50 environmental noise. Per-chunk probabilities match the reference
**within 2e-3** across ~6000 chunks, cross-runtime (NIF ORT 1.24 vs Python 1.27); speech clips drive
the probability high, real noise (sirens, horns, fireworks, engines) never reads as speech. See
[`test/corpus/README.md`](test/corpus/README.md) to reproduce.

## How it works

Silero v6 ONNX (opset 16, unified LSTM `state[2,1,128]`, 64-sample look-back context per 512-sample
window) runs inside a Rustler NIF on a DirtyCpu scheduler. The model session is loaded once per
`Session`; per-stream RNN state lives in a process-owned resource the BEAM reclaims on process death.
Probabilities feed `VadEx.Endpointer`, a pure-Elixir state machine. Full design:
[`docs/architecture.md`](docs/architecture.md).

## Roadmap

- 8 kHz (telephony) input path
- A shared model-session pool for higher concurrent throughput
- Pluggable backends (NeMo Frame-VAD MarbleNet; evaluate TEN-VAD)
- Music robustness corpus + broader target matrix (Intel macOS, musl, windows-gnu)

## License

MIT. The bundled Silero VAD model is MIT (snakers4/silero-vad).
