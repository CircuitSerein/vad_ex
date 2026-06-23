defmodule VadEx do
  @moduledoc """
  Streaming Voice Activity Detection + endpointing for the BEAM.

  `vad_ex` detects speech vs. silence in a real-time audio stream and emits
  utterance boundary events (`speech_start` / `speech_end`). It is backed by the
  Silero VAD ONNX model run through a Rust NIF (ONNX Runtime via `ort`).

  ## Two ways to use it

    * `VadEx.Session` — a supervised GenServer, one per audio stream. Feed it raw
      PCM chunks; it runs inference, drives the endpointing state machine, and
      emits `:telemetry` events. Has no Membrane dependency.

    * `VadEx.Membrane.Filter` — an optional Membrane element that wraps the same
      core, to drop into an existing media pipeline.

  Audio contract for v0.1: **16 kHz, mono, signed 16-bit little-endian PCM**,
  fed in **512-sample (1024-byte) chunks** (the Silero v5 window at 16 kHz).

  See `docs/architecture.md` and `docs/research/` for the full design rationale.
  """

  @typedoc "Raw PCM audio: mono, s16le. One chunk = 512 samples (1024 bytes) at 16 kHz."
  @type pcm_chunk :: binary()

  @typedoc "Speech probability for a single chunk, 0.0..1.0."
  @type probability :: float()
end
