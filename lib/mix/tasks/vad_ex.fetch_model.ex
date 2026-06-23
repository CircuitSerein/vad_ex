defmodule Mix.Tasks.VadEx.FetchModel do
  @shortdoc "Download + verify the bundled Silero VAD ONNX model into priv/models"
  @moduledoc """
  Fetches the Silero VAD ONNX model into `priv/models/silero_vad.onnx` and verifies its size +
  SHA-256 against the pinned values below.

  Pinned to a **tagged** ref (not `master`) so the bundled weights cannot silently change — a future
  v6.3 could keep the identical byte size with different weights, which would break the golden vector.
  See docs/research/07-2026-06-23-verification-and-models.md.

      mix vad_ex.fetch_model        # download if missing/invalid, then verify
      mix vad_ex.fetch_model --force  # re-download even if a valid file exists
  """
  use Mix.Task

  # Silero VAD v6.2 — MIT (snakers4/silero-vad). Verified live 2026-06-23.
  @model_url "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx"
  @sha256 "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
  @size 2_327_524
  @dest "priv/models/silero_vad.onnx"

  @impl Mix.Task
  def run(args) do
    force? = "--force" in args

    cond do
      not force? and valid?() ->
        Mix.shell().info("[vad_ex] model already present and verified: #{@dest}")

      true ->
        Mix.shell().info("[vad_ex] downloading Silero v6.2 model…")
        download!()
        verify!()
        Mix.shell().info("[vad_ex] OK — #{@dest} (#{@size} bytes, sha256 verified)")
    end
  end

  defp valid? do
    case File.read(@dest) do
      {:ok, bin} -> byte_size(bin) == @size and sha256(bin) == @sha256
      _ -> false
    end
  end

  defp download! do
    {:ok, _} = Application.ensure_all_started(:inets)
    {:ok, _} = Application.ensure_all_started(:ssl)
    File.mkdir_p!(Path.dirname(@dest))

    request = {String.to_charlist(@model_url), []}
    http_opts = [timeout: 60_000, connect_timeout: 30_000, autoredirect: true]

    case :httpc.request(:get, request, http_opts, body_format: :binary) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        File.write!(@dest, body)

      {:ok, {{_, status, _}, _, _}} ->
        Mix.raise("[vad_ex] download failed: HTTP #{status} from #{@model_url}")

      {:error, reason} ->
        Mix.raise("[vad_ex] download failed: #{inspect(reason)}")
    end
  end

  defp verify! do
    bin = File.read!(@dest)
    size = byte_size(bin)
    sha = sha256(bin)

    cond do
      size != @size ->
        Mix.raise("[vad_ex] size mismatch: got #{size}, expected #{@size}")

      sha != @sha256 ->
        Mix.raise("[vad_ex] sha256 mismatch:\n  got      #{sha}\n  expected #{@sha256}")

      true ->
        :ok
    end
  end

  defp sha256(bin), do: :crypto.hash(:sha256, bin) |> Base.encode16(case: :lower)
end
