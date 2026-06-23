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

  See `docs/architecture.md` for the NIF design.
  """

  @version Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :vad_ex,
    crate: "vad_ex",
    base_url: "https://github.com/CircuitSerein/vad_ex/releases/download/v#{@version}",
    # Build from source for dev checkouts (no published artifacts exist for a "*-dev" version)
    # or when VAD_EX_BUILD is set; released (non-dev) versions pull the precompiled NIF.
    force_build: System.get_env("VAD_EX_BUILD") in ["1", "true"] or String.contains?(@version, "-dev"),
    version: @version,
    # Lean v0.1 matrix. Each target ships ONE self-contained .so — ONNX Runtime is statically
    # linked into the NIF via ort's `download-binaries`, so there is no libonnxruntime to bundle.
    # pyke ships static ORT for all four (🟢). Intel macOS (x86_64-apple-darwin), musl, and
    # windows-gnu are deferred to a later release.
    targets: ~w[
      aarch64-apple-darwin
      aarch64-unknown-linux-gnu
      x86_64-unknown-linux-gnu
      x86_64-pc-windows-msvc
    ],
    # One artifact per target, built targeting NIF 2.15 (forward-compatible with 2.16/2.17).
    nif_versions: ["2.15"]

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
