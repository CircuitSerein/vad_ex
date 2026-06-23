defmodule VadEx.EndpointerTest do
  use ExUnit.Case, async: true

  alias VadEx.Endpointer

  # The endpointer is pure (no NIF), so it is fully unit-testable now, before the
  # Rust core exists. These tests pin the state-machine behaviour.

  setup do
    # 32 ms/chunk. min_speech 250ms ≈ 8 chunks, min_silence 500ms ≈ 16 chunks.
    %{ep: Endpointer.new(threshold: 0.5, min_speech_ms: 250, min_silence_ms: 500, speech_pad_ms: 100)}
  end

  defp feed(ep, probs, metadata \\ %{}),
    do: Enum.reduce(probs, ep, fn p, acc -> Endpointer.push(acc, p, metadata) end)

  test "emits speech_start only after min_speech sustained", %{ep: ep} do
    ref = attach([:vad_ex, :speech_start])
    # 7 speech chunks (< 8) → no start yet
    _ep = feed(ep, List.duplicate(0.9, 7))
    refute_received {^ref, [:vad_ex, :speech_start], _, _}
  end

  test "false start below min_speech returns to silence without event", %{ep: ep} do
    ref = attach([:vad_ex, :speech_start])
    _ep = feed(ep, [0.9, 0.9, 0.9, 0.1])
    refute_received {^ref, [:vad_ex, :speech_start], _, _}
  end

  test "full utterance emits speech_start then speech_end with duration", %{ep: ep} do
    s = attach([:vad_ex, :speech_start])
    e = attach([:vad_ex, :speech_end])

    ep
    |> feed(List.duplicate(0.9, 10))
    |> feed(List.duplicate(0.05, 20))

    assert_received {^s, [:vad_ex, :speech_start], %{ts_ms: _}, _}
    assert_received {^e, [:vad_ex, :speech_end], %{duration_ms: d}, _} when d > 0
  end

  defp attach(event) do
    ref = make_ref()
    parent = self()

    :telemetry.attach(
      "test-#{inspect(ref)}",
      event,
      fn name, meas, meta, _ -> send(parent, {ref, name, meas, meta}) end,
      nil
    )

    on_exit(fn -> :telemetry.detach("test-#{inspect(ref)}") end)
    ref
  end
end
