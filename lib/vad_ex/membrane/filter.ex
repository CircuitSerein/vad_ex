if Code.ensure_loaded?(Membrane.Filter) do
  defmodule VadEx.Membrane.Filter do
    @moduledoc """
    Optional Membrane element wrapping the `vad_ex` core. Compiles only when
    `membrane_core` is available.

    Accepts `%Membrane.RawAudio{sample_format: :s16le, channels: 1, sample_rate: 16_000}`,
    passes every buffer through to `:output` **unchanged**, and runs VAD on the side: it
    re-windows the byte stream into exact 512-sample (1024-byte) chunks (independent of how
    upstream buffers the audio), runs Silero inference per window, and drives the same
    `VadEx.Endpointer` state machine as `VadEx.Session`. It emits the same telemetry events
    (`[:vad_ex, :chunk | :speech_start | :speech_end]`); attach a `:telemetry` handler to react.
    Slot it after a `Membrane.Opus.Decoder` (configured to 16 kHz) and before your sink/ASR.

    Parent notifications on speech boundaries are a v0.2 item (they need the wider Endpointer
    callback — see `docs/research/07-2026-06-23-verification-and-models.md` §5).

    See `docs/research/03-membrane-integration.md`.
    """

    use Membrane.Filter

    alias Membrane.RawAudio
    alias VadEx.{Native, Endpointer, Telemetry}

    # 512 samples * 2 bytes (s16le) = one Silero window @16k.
    @window_bytes 1024

    def_options(
      threshold: [spec: float(), default: 0.5],
      min_speech_ms: [spec: pos_integer(), default: 250],
      min_silence_ms: [spec: pos_integer(), default: 500],
      speech_pad_ms: [spec: pos_integer(), default: 100],
      metadata: [spec: map(), default: %{}],
      model_path: [spec: String.t() | nil, default: nil]
    )

    def_input_pad(:input,
      accepted_format: %RawAudio{sample_format: :s16le, channels: 1, sample_rate: 16_000},
      flow_control: :auto
    )

    def_output_pad(:output,
      accepted_format: %RawAudio{sample_format: :s16le, channels: 1, sample_rate: 16_000},
      flow_control: :auto
    )

    @impl true
    def handle_init(_ctx, opts) do
      model_path = opts.model_path || Path.join(:code.priv_dir(:vad_ex), "models/silero_vad.onnx")
      :ok = Native.ensure_initialized()

      with {:ok, model} <- Native.load_model(model_path),
           {:ok, stream} <- Native.new_stream(model) do
        state = %{
          model: model,
          stream: stream,
          endpointer: Endpointer.new(opts |> Map.from_struct() |> Map.to_list()),
          metadata: Map.new(opts.metadata),
          acc: <<>>
        }

        {[], state}
      else
        {:error, reason} -> raise "vad_ex NIF init failed: #{inspect(reason)}"
      end
    end

    @impl true
    def handle_stream_format(:input, format, _ctx, state),
      do: {[stream_format: {:output, format}], state}

    @impl true
    def handle_buffer(:input, buffer, _ctx, state) do
      # Pass the buffer through untouched; VAD runs on a re-windowed copy of the byte stream.
      {endpointer, acc} =
        consume(state.endpointer, state.acc <> buffer.payload, state.model, state.stream, state.metadata)

      {[buffer: {:output, buffer}], %{state | endpointer: endpointer, acc: acc}}
    end

    # Drain every complete 1024-byte window from the accumulator; keep the remainder.
    defp consume(ep, data, model, stream, meta) when byte_size(data) >= @window_bytes do
      <<window::binary-size(@window_bytes), rest::binary>> = data
      ep = process_window(model, stream, ep, window, meta)
      consume(ep, rest, model, stream, meta)
    end

    defp consume(ep, data, _model, _stream, _meta), do: {ep, data}

    defp process_window(model, stream, ep, window, meta) do
      prob =
        Telemetry.span_chunk(meta, byte_size(window), fn ->
          case Native.process_chunk(model, stream, window) do
            {:ok, p} -> p
            {:error, _} = err -> throw(err)
          end
        end)

      Endpointer.push(ep, prob, meta)
    end
  end
end
