defmodule VadEx.Membrane.FilterTest do
  use ExUnit.Case, async: false

  # Drives a real Membrane pipeline: Testing.Source -> VadEx.Membrane.Filter -> Testing.Sink.
  # :nif tagged — run with: VAD_EX_BUILD=1 mix test --include nif
  @moduletag :nif

  import Membrane.ChildrenSpec
  import Membrane.Testing.Assertions

  alias Membrane.{Buffer, RawAudio, Testing}

  @fmt %RawAudio{sample_format: :s16le, channels: 1, sample_rate: 16_000}

  defp attach_telemetry(events) do
    parent = self()
    id = "filter-test-#{System.unique_integer([:positive])}"
    :telemetry.attach_many(id, events, fn n, m, meta, _ -> send(parent, {:tel, n, m, meta}) end, nil)
    on_exit(fn -> :telemetry.detach(id) end)
  end

  defp drain(acc \\ []) do
    receive do
      {:tel, n, m, meta} -> drain([{n, m, meta} | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  # Split into odd-sized pieces (NOT multiples of 1024) so the filter must accumulate across
  # buffer boundaries to form 512-sample windows.
  defp split(<<>>, _n), do: []
  defp split(bin, n) when byte_size(bin) <= n, do: [bin]

  defp split(bin, n) do
    <<head::binary-size(n), rest::binary>> = bin
    [head | split(rest, n)]
  end

  test "passes buffers through unchanged and detects the utterance across odd buffer boundaries" do
    pcm = File.read!("test/fixtures/utterance_16k_s16le.pcm")
    payloads = split(pcm, 777)
    refute rem(777, 1024) == 0

    attach_telemetry([[:vad_ex, :speech_start], [:vad_ex, :speech_end]])

    spec =
      child(:src, %Testing.Source{output: payloads, stream_format: @fmt})
      |> child(:vad, %VadEx.Membrane.Filter{threshold: 0.5})
      |> child(:sink, Testing.Sink)

    pid = Testing.Pipeline.start_link_supervised!(spec: spec)

    # Passthrough: the sink receives each sent payload unchanged, in order.
    for p <- payloads, do: assert_sink_buffer(pid, :sink, %Buffer{payload: ^p})
    assert_end_of_stream(pid, :sink)

    # VAD ran on the re-windowed stream -> at least one balanced speech boundary.
    events = drain()
    starts = for {[:vad_ex, :speech_start], _, _} <- events, do: :s
    ends = for {[:vad_ex, :speech_end], _, _} <- events, do: :e

    assert length(starts) >= 1, "utterance must produce a speech_start"
    assert length(starts) == length(ends), "every speech_start closed by a speech_end"
  end

  test "silence through the pipeline produces no speech boundary" do
    pcm = File.read!("test/fixtures/silence_16k_s16le.pcm")
    attach_telemetry([[:vad_ex, :speech_start]])

    spec =
      child(:src, %Testing.Source{output: split(pcm, 777), stream_format: @fmt})
      |> child(:vad, %VadEx.Membrane.Filter{})
      |> child(:sink, Testing.Sink)

    pid = Testing.Pipeline.start_link_supervised!(spec: spec)
    assert_end_of_stream(pid, :sink)

    assert drain() |> Enum.filter(fn {n, _, _} -> n == [:vad_ex, :speech_start] end) == []
  end
end
