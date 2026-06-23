defmodule VadEx.Endpointer do
  @moduledoc """
  Turns a stream of per-chunk speech probabilities into utterance boundary
  events. Silero only outputs a probability per chunk; the endpointing logic —
  hysteresis, minimum durations, padding — lives here.

  This module is the project's main differentiator over prior art that exposed
  the raw Silero probability with no endpointing.

  ## State machine

      SILENCE  --p>=thr-->  STARTING(accumulate speech)
      STARTING --reached min_speech_ms--> SPEECH      [emit speech_start, padded back]
      STARTING --p<thr before min_speech--> SILENCE   (false start dropped)
      SPEECH   --p<thr-->   TRAILING(accumulate silence)
      TRAILING --p>=thr-->  SPEECH                    (silence broke, continue)
      TRAILING --reached min_silence_ms--> SILENCE    [emit speech_end {duration_ms}]

  ## v0.2 upgrade path

  This is the `SilenceEndpointer` strategy. `VadEx.Endpointer` is also a
  `@behaviour`: v0.2 will add a `TransformerEndpointer` (LiveKit turn-detector /
  Smart-Turn style) that answers "did the human actually finish their turn?"
  semantically rather than by silence alone. v0.1 ships silence-based only.
  """

  # --- Behaviour for pluggable strategies (v0.2: transformer turn-detection) ---
  @callback push(state :: term(), probability :: float(), metadata :: map()) :: term()
  @callback reset(state :: term()) :: term()

  @chunk_ms 32

  defstruct phase: :silence,
            threshold: 0.5,
            min_speech_chunks: 8,
            min_silence_chunks: 16,
            pad_chunks: 3,
            run: 0,
            elapsed_ms: 0,
            speech_started_at: nil

  @doc "Build endpointer state from session opts."
  def new(opts \\ []) do
    %__MODULE__{
      threshold: opts[:threshold] || 0.5,
      min_speech_chunks: ms_to_chunks(opts[:min_speech_ms] || 250),
      min_silence_chunks: ms_to_chunks(opts[:min_silence_ms] || 500),
      pad_chunks: ms_to_chunks(opts[:speech_pad_ms] || 100)
    }
  end

  @doc "Feed one chunk probability; advances the state machine and emits events."
  def push(%__MODULE__{} = s, prob, metadata) do
    s = %{s | elapsed_ms: s.elapsed_ms + @chunk_ms}
    speech? = prob >= s.threshold
    do_transition(s, speech?, metadata)
  end

  @doc "Reset to silence; keep config."
  def reset(%__MODULE__{} = s), do: %{s | phase: :silence, run: 0, speech_started_at: nil}

  # --- Transitions --------------------------------------------------------

  defp do_transition(%{phase: :silence} = s, true, _m),
    do: %{s | phase: :starting, run: 1}

  defp do_transition(%{phase: :silence} = s, false, _m), do: s

  defp do_transition(%{phase: :starting} = s, true, m) do
    run = s.run + 1

    if run >= s.min_speech_chunks do
      started_at = s.elapsed_ms - (run + s.pad_chunks) * @chunk_ms
      VadEx.Telemetry.speech_start(max(started_at, 0), m)
      %{s | phase: :speech, run: 0, speech_started_at: max(started_at, 0)}
    else
      %{s | run: run}
    end
  end

  defp do_transition(%{phase: :starting} = s, false, _m),
    do: %{s | phase: :silence, run: 0}

  defp do_transition(%{phase: :speech} = s, true, _m), do: s

  defp do_transition(%{phase: :speech} = s, false, _m),
    do: %{s | phase: :trailing, run: 1}

  defp do_transition(%{phase: :trailing} = s, true, _m),
    do: %{s | phase: :speech, run: 0}

  defp do_transition(%{phase: :trailing} = s, false, m) do
    run = s.run + 1

    if run >= s.min_silence_chunks do
      ended_at = s.elapsed_ms + s.pad_chunks * @chunk_ms
      duration = ended_at - (s.speech_started_at || ended_at)
      VadEx.Telemetry.speech_end(ended_at, duration, m)
      %{s | phase: :silence, run: 0, speech_started_at: nil}
    else
      %{s | run: run}
    end
  end

  # Round UP: 250ms / 32ms = 7.8 -> 8 chunks. Flooring (div/2) gave 7, which both the struct
  # default (min_speech_chunks: 8) and the unit test assume should be 8.
  defp ms_to_chunks(ms), do: max(div(ms + @chunk_ms - 1, @chunk_ms), 1)
end
