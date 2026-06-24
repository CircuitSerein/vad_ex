defmodule VadEx.RealAudioTest do
  use ExUnit.Case, async: false

  # The real-audio gate. Runs the NIF over a corpus of genuine recordings (LibriSpeech speech +
  # MUSAN noise/music) and asserts process_chunk reproduces reference Silero (Python onnxruntime)
  # bit-for-bit within tolerance, on every clip — proving the Rust port is faithful across diverse
  # real audio, not just one synthetic clip.
  #
  # Excluded by default (needs the built NIF AND the fetched corpus). To run:
  #     test/corpus/fetch_corpus.sh
  #     test/corpus/.venv/bin/python test/corpus/reference_probs.py   # (re)build the golden
  #     VAD_EX_BUILD=1 mix test --include real_audio
  #
  # The golden vector (test/corpus/golden_real_audio.json) is committed; the raw/decoded audio is
  # gitignored and re-fetched on demand. A per-clip sha256 guard ensures the local PCM is exactly
  # the bytes the golden was computed from.
  @moduletag :real_audio

  alias VadEx.Native

  @corpus Path.join(__DIR__, "corpus")
  @golden_path Path.join(@corpus, "golden_real_audio.json")
  @data Path.join(@corpus, "data")
  @model Path.join(:code.priv_dir(:vad_ex), "models/silero_vad.onnx")

  setup_all do
    unless File.exists?(@golden_path) do
      flunk("missing #{@golden_path} — run test/corpus/fetch_corpus.sh then reference_probs.py")
    end

    golden = @golden_path |> File.read!() |> :json.decode()
    {:ok, model} = Native.load_model(@model)
    {:ok, golden: golden, model: model}
  end

  defp chunks(bin, size \\ 1024),
    do: for(<<c::binary-size(size) <- bin>>, do: c)

  defp run_stream(model, pcm) do
    {:ok, stream} = Native.new_stream(model)

    pcm
    |> chunks()
    |> Enum.map(fn chunk ->
      {:ok, prob} = Native.process_chunk(model, stream, chunk)
      prob
    end)
  end

  defp load_pcm!(%{"name" => name, "sha256" => sha}) do
    path = Path.join(@data, "#{name}.pcm")
    assert File.exists?(path), "#{name}: missing #{path} — run test/corpus/fetch_corpus.sh"
    pcm = File.read!(path)
    got = :crypto.hash(:sha256, pcm) |> Base.encode16(case: :lower)
    assert got == sha, "#{name}: PCM sha256 mismatch — re-fetch the corpus (golden is stale)"
    pcm
  end

  test "NIF reproduces reference Silero probs on every real clip", %{golden: golden, model: model} do
    tol = golden["tolerance"]

    for %{"name" => name, "probs" => expected} = clip <- golden["clips"] do
      got = run_stream(model, load_pcm!(clip))
      assert length(got) == length(expected), "#{name}: chunk count #{length(got)} != #{length(expected)}"

      Enum.zip(got, expected)
      |> Enum.with_index()
      |> Enum.each(fn {{g, e}, i} ->
        assert_in_delta g, e, tol, "#{name} chunk #{i}: got #{g}, expected #{e}"
      end)
    end
  end

  test "behavioral: speech clips fire, noise stays non-speech", %{golden: golden, model: model} do
    b = golden["behavioral"]
    thr = b["speech_threshold"]

    for clip <- golden["clips"] do
      probs = run_stream(model, load_pcm!(clip))
      max = Enum.max(probs)
      frac = Enum.count(probs, &(&1 >= thr)) / length(probs)

      case clip["kind"] do
        "speech" ->
          assert max > b["speech_min_max_prob"],
                 "#{clip["name"]}: speech clip max prob #{max} did not exceed #{b["speech_min_max_prob"]}"

        "noise" ->
          assert frac <= b["nonspeech_max_speech_frac"],
                 "#{clip["name"]}: noise clip read as speech #{Float.round(frac * 100, 1)}% of chunks"

        # music may legitimately contain vocals — recorded in the golden, not hard-asserted.
        _ ->
          :ok
      end
    end
  end
end
