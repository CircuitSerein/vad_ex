defmodule VadEx.Native do
  @moduledoc """
  Low-level NIF bindings to the Rust ONNX-Runtime VAD core. Not a public API —
  use `VadEx.Session`.

  Distribution: precompiled binaries via `RustlerPrecompiled` (no Rust toolchain
  needed by consumers). Set `VAD_EX_BUILD=1` to force a local build with `rustler`.

  ## Resources

    * `model` — an immutable, shared ONNX `Session` (`ResourceArc<VadSession>`).
    * `stream` — per-stream mutable RNN state `(h, c, context)`
      (`ResourceArc<Mutex<StreamState>>`). Owned by the calling process; the
      BEAM drops it when that process dies.

  See `docs/research/02-onnx-nif-rustler-ort.md` for the full NIF design.
  """

  @version Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :vad_ex,
    crate: "vad_ex",
    base_url: "https://github.com/CircuitSerein/vad_ex/releases/download/v#{@version}",
    force_build: System.get_env("VAD_EX_BUILD") in ["1", "true"],
    version: @version,
    targets: ~w[
      aarch64-apple-darwin
      aarch64-unknown-linux-gnu
      aarch64-unknown-linux-musl
      x86_64-apple-darwin
      x86_64-unknown-linux-gnu
      x86_64-unknown-linux-musl
      x86_64-pc-windows-gnu
      x86_64-pc-windows-msvc
    ],
    nif_versions: ["2.15", "2.16", "2.17"]

  @doc """
  Initialize ort's load-dynamic `libonnxruntime` path (call once before `load_model/1` in
  production, with the bundled `priv/lib` dylib). Optional for local dev when `ORT_DYLIB_PATH`
  is set. Process-global, first-call-wins.
  """
  @spec init_ort_from(dylib_path :: binary()) :: {:ok, :ok} | {:error, term()}
  def init_ort_from(_dylib_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Resolve and initialize the bundled `libonnxruntime` from `priv/lib`, once per node. Called by
  `VadEx.Session` / `VadEx.Membrane.Filter` before loading the model. If no dylib is bundled
  (local dev), this is a no-op and ort falls back to `ORT_DYLIB_PATH` / the system default.
  """
  @spec ensure_initialized() :: :ok
  def ensure_initialized do
    if :persistent_term.get({__MODULE__, :ort_initialized}, false) do
      :ok
    else
      priv = :code.priv_dir(:vad_ex)

      case Path.wildcard(Path.join([priv, "lib", "*onnxruntime*"])) do
        [dylib | _] -> init_ort_from(dylib)
        [] -> :ok
      end

      :persistent_term.put({__MODULE__, :ort_initialized}, true)
      :ok
    end
  end

  @doc "Load the Silero ONNX model from a file path → `{:ok, model_ref}`."
  @spec load_model(path :: binary()) :: {:ok, reference()} | {:error, term()}
  def load_model(_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Allocate fresh per-stream RNN state for `model`."
  @spec new_stream(model :: reference()) :: {:ok, reference()} | {:error, term()}
  def new_stream(_model), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Run inference on one 512-sample (s16le) chunk, advancing the stream's RNN state.
  Returns the speech probability for that chunk.
  """
  @spec process_chunk(model :: reference(), stream :: reference(), audio :: binary()) ::
          {:ok, float()} | {:error, term()}
  def process_chunk(_model, _stream, _audio), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Zero a stream's RNN state (h, c, context) — call between utterances/sessions."
  @spec reset_stream(stream :: reference()) :: :ok
  def reset_stream(_stream), do: :erlang.nif_error(:nif_not_loaded)
end
