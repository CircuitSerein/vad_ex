defmodule VadEx.SessionTest do
  use ExUnit.Case, async: false

  # End-to-end: PCM fixture -> VadEx.Session (NIF inference + endpointer) -> telemetry events.
  # :nif tagged — run with: VAD_EX_BUILD=1 mix test --include nif
  @moduletag :nif

  alias VadEx.Session

  defp attach(events) do
    parent = self()
    id = "sess-test-#{System.unique_integer([:positive])}"

    :telemetry.attach_many(
      id,
      events,
      fn name, meas, meta, _ -> send(parent, {:tel, name, meas, meta}) end,
      nil
    )

    on_exit(fn -> :telemetry.detach(id) end)
    :ok
  end

  # Feed a fixture chunk-by-chunk, then force a sync barrier so every async cast (and the telemetry
  # it emits) has been processed before we inspect the mailbox.
  defp feed(vad, pcm) do
    for <<chunk::binary-size(1024) <- pcm>>, do: Session.process(vad, chunk)
    _ = :sys.get_state(vad)
    :ok
  end

  defp drain(acc \\ []) do
    receive do
      {:tel, name, meas, meta} -> drain([{name, meas, meta} | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  setup do
    {:ok, vad} = Session.start_link(sample_rate: 16_000, threshold: 0.5)
    on_exit(fn -> if Process.alive?(vad), do: GenServer.stop(vad) end)
    {:ok, vad: vad}
  end

  test "chunk telemetry fires and silence produces no speech boundary", %{vad: vad} do
    attach([[:vad_ex, :chunk, :stop], [:vad_ex, :speech_start], [:vad_ex, :speech_end]])
    feed(vad, File.read!("test/fixtures/silence_16k_s16le.pcm"))
    events = drain()

    chunk_stops = Enum.count(events, fn {n, _, _} -> n == [:vad_ex, :chunk, :stop] end)
    starts = Enum.count(events, fn {n, _, _} -> n == [:vad_ex, :speech_start] end)

    assert chunk_stops == 20, "expected one chunk span per 512-sample window"
    assert starts == 0, "silence must not trigger speech_start"
  end

  test "an utterance emits paired speech_start/speech_end with positive duration", %{vad: vad} do
    attach([[:vad_ex, :speech_start], [:vad_ex, :speech_end]])
    feed(vad, File.read!("test/fixtures/utterance_16k_s16le.pcm"))
    events = drain()

    starts = for {[:vad_ex, :speech_start], m, _} <- events, do: m
    ends = for {[:vad_ex, :speech_end], m, _} <- events, do: m

    assert length(starts) >= 1, "speech region must produce at least one speech_start"
    assert length(starts) == length(ends), "every speech_start must be closed by a speech_end"
    assert Enum.all?(ends, &(&1.duration_ms > 0)), "utterance duration must be positive"
    assert hd(starts).ts_ms <= hd(ends).ts_ms, "start precedes end"
  end

  test "metadata flows into telemetry", %{vad: _vad} do
    {:ok, vad} = Session.start_link(sample_rate: 16_000, metadata: %{call_id: "abc"})
    on_exit(fn -> if Process.alive?(vad), do: GenServer.stop(vad) end)
    attach([[:vad_ex, :speech_start]])
    feed(vad, File.read!("test/fixtures/utterance_16k_s16le.pcm"))
    events = drain()
    assert Enum.any?(events, fn {n, _, meta} -> n == [:vad_ex, :speech_start] and meta[:call_id] == "abc" end)
  end
end
