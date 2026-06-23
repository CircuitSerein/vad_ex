defmodule VadEx.Session do
  @moduledoc """
  One supervised VAD stream = one `GenServer`. Feed it raw PCM chunks; it runs
  Silero inference per chunk and drives the `VadEx.Endpointer` state machine,
  emitting `:telemetry` events on speech boundaries.

  This per-stream-process design is deliberate: 10k concurrent streams = 10k
  isolated, crash-isolated processes, the BEAM's structural advantage for
  real-time voice at scale (see `docs/research/04-prior-art.md` and the Type-B
  note in `docs/build-plan.md`).

  ## Options

    * `:model_path` — path to `silero_vad.onnx` (default: bundled in `priv/models`)
    * `:sample_rate` — `16_000` (default) or `8_000`
    * `:threshold` — speech probability cutoff, default `0.5`
    * `:min_speech_ms` — default `250`
    * `:min_silence_ms` — default `500`
    * `:speech_pad_ms` — default `100`
    * `:metadata` — map merged into telemetry metadata (e.g. `%{call_id: ...}`)
  """

  use GenServer

  alias VadEx.{Native, Endpointer, Telemetry}

  @default_model Application.compile_env(:vad_ex, :model_path, nil)

  # --- Public API ---------------------------------------------------------

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []), do: GenServer.start_link(__MODULE__, opts, name: opts[:name])

  @doc "Process one PCM chunk (512 samples s16le @16k). Async; results via telemetry."
  @spec process(GenServer.server(), VadEx.pcm_chunk()) :: :ok
  def process(server, pcm_chunk), do: GenServer.cast(server, {:process, pcm_chunk})

  @doc "Reset RNN + endpointer state between utterances/sessions."
  @spec reset(GenServer.server()) :: :ok
  def reset(server), do: GenServer.call(server, :reset)

  # --- Callbacks ----------------------------------------------------------

  @impl true
  def init(opts) do
    model_path = opts[:model_path] || @default_model || default_model_path()
    :ok = Native.ensure_initialized()

    with {:ok, model} <- Native.load_model(model_path),
         {:ok, stream} <- Native.new_stream(model) do
      state = %{
        model: model,
        stream: stream,
        endpointer: Endpointer.new(opts),
        metadata: Map.new(opts[:metadata] || %{}),
        sample_rate: opts[:sample_rate] || 16_000
      }

      {:ok, state}
    else
      {:error, reason} -> {:stop, {:vad_ex_init_failed, reason}}
    end
  end

  @impl true
  def handle_cast({:process, chunk}, state) do
    prob =
      Telemetry.span_chunk(state.metadata, byte_size(chunk), fn ->
        case Native.process_chunk(state.model, state.stream, chunk) do
          {:ok, p} -> p
          {:error, _} = err -> throw(err)
        end
      end)

    # Endpointer consumes the probability, emits speech_start/speech_end via telemetry.
    endpointer = Endpointer.push(state.endpointer, prob, state.metadata)
    {:noreply, %{state | endpointer: endpointer}}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    :ok = Native.reset_stream(state.stream)
    {:reply, :ok, %{state | endpointer: Endpointer.reset(state.endpointer)}}
  end

  # --- Internal -----------------------------------------------------------

  defp default_model_path do
    Path.join(:code.priv_dir(:vad_ex), "models/silero_vad.onnx")
  end
end
