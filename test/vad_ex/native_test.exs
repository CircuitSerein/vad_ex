defmodule VadEx.NativeTest do
  use ExUnit.Case, async: false

  # NIF golden-vector tests. Tagged :nif — run only when the NIF is built from source:
  #
  #     VAD_EX_BUILD=1 mix test --include nif
  #
  # (ORT is statically linked into the NIF, so no ORT_DYLIB_PATH is needed.)
  #
  # The fixtures (test/fixtures/*.pcm) are the EXACT s16le bytes the golden probabilities in
  # golden_v6_2.json were computed from (decode i16/32768), so process_chunk must reproduce them.
  @moduletag :nif

  alias VadEx.Native

  @model Path.join(:code.priv_dir(:vad_ex), "models/silero_vad.onnx")
  @golden "test/fixtures/golden_v6_2.json" |> File.read!() |> :json.decode()

  setup do
    {:ok, model} = Native.load_model(@model)
    {:ok, model: model}
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

  test "golden vector: every case reproduces the recorded probabilities", %{model: model} do
    tol = @golden["tolerance"]

    for %{"name" => name, "pcm" => pcm_path, "probs" => expected} <- @golden["cases"] do
      got = run_stream(model, File.read!(pcm_path))
      assert length(got) == length(expected), "#{name}: chunk count"

      Enum.zip(got, expected)
      |> Enum.with_index()
      |> Enum.each(fn {{g, e}, i} ->
        assert_in_delta g, e, tol, "#{name} chunk #{i}: got #{g}, expected #{e}"
      end)
    end
  end

  test "silence stays below the speech threshold", %{model: model} do
    [silence] = Enum.filter(@golden["cases"], &(&1["name"] == "silence"))
    probs = run_stream(model, File.read!(silence["pcm"]))
    assert Enum.max(probs) < @golden["assert"]["silence_max_prob_below"]
  end

  test "reset makes a re-run bit-identical", %{model: model} do
    pcm = File.read!("test/fixtures/tone220_16k_s16le.pcm")
    {:ok, stream} = Native.new_stream(model)

    run = fn ->
      pcm
      |> chunks()
      |> Enum.map(fn c ->
        {:ok, p} = Native.process_chunk(model, stream, c)
        p
      end)
    end

    first = run.()
    :ok = Native.reset_stream(stream)
    second = run.()
    assert first == second
  end

  test "a wrong-sized chunk returns an error, does not crash", %{model: model} do
    {:ok, stream} = Native.new_stream(model)
    assert {:error, _reason} = Native.process_chunk(model, stream, <<0, 0, 0, 0>>)
  end
end
