defmodule VadEx.Telemetry do
  @moduledoc """
  Telemetry events, following the Keathley conventions
  (https://keathley.io/blog/telemetry-conventions.html).

  Emitted events:

    * `[:vad_ex, :chunk, :start | :stop | :exception]` — per-chunk inference span.
      `:stop` measurement `:duration` (native time). Metadata: `:speech_detected`,
      plus any session `:metadata`.
    * `[:vad_ex, :speech_start]` — measurement `%{ts_ms: integer}`.
    * `[:vad_ex, :speech_end]` — measurement `%{ts_ms: integer, duration_ms: integer}`.

  An optional `VadEx.PromEx.Plugin` (v0.2) will expose these to PromEx/Prometheus.
  """

  @spec span_chunk(map(), non_neg_integer(), (-> float())) :: float()
  def span_chunk(metadata, sample_bytes, fun) do
    :telemetry.span([:vad_ex, :chunk], Map.put(metadata, :sample_bytes, sample_bytes), fn ->
      prob = fun.()
      {prob, Map.merge(metadata, %{speech_detected: prob >= 0.5, probability: prob})}
    end)
  end

  @spec speech_start(integer(), map()) :: :ok
  def speech_start(ts_ms, metadata) do
    :telemetry.execute([:vad_ex, :speech_start], %{ts_ms: ts_ms}, metadata)
  end

  @spec speech_end(integer(), integer(), map()) :: :ok
  def speech_end(ts_ms, duration_ms, metadata) do
    :telemetry.execute([:vad_ex, :speech_end], %{ts_ms: ts_ms, duration_ms: duration_ms}, metadata)
  end
end
