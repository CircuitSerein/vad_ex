# VAD inference benchmark. Build the NIF first, then:
#
#     VAD_EX_BUILD=1 mix run bench/bench.exs
#
# Reports per-chunk inference latency (mean / p50 / p99 / max), the implied real-time factor and
# streams-per-core, and concurrent-Session throughput driven through the public VadEx.Session API.
# Uses a real corpus clip (test/corpus/data/speech_01.pcm) when present, else a synthetic stream.

alias VadEx.{Native, Session}

chunk_bytes = 1024
chunk_secs = 512 / 16_000

model_path = Path.join(:code.priv_dir(:vad_ex), "models/silero_vad.onnx")

# --- build a long chunk list (repeat a real clip if we have one, else synthesize) ---------------
sample_clip = Path.join([File.cwd!(), "test", "corpus", "data", "speech_01.pcm"])

base_pcm =
  if File.exists?(sample_clip) do
    IO.puts("source: #{Path.relative_to_cwd(sample_clip)}")
    File.read!(sample_clip)
  else
    IO.puts("source: synthetic (no corpus clip found — run test/corpus/fetch_corpus.sh for real audio)")
    # 2 s of low-amplitude noise-ish PCM, deterministic.
    for(i <- 0..(16_000 * 2 - 1), into: <<>>, do: <<rem(i * 2741, 4001) - 2000::little-signed-16>>)
  end

chunks =
  for <<c::binary-size(chunk_bytes) <- base_pcm>>, do: c

defmodule B do
  def percentile(sorted, p) do
    idx = min(length(sorted) - 1, round(p / 100 * (length(sorted) - 1)))
    Enum.at(sorted, idx)
  end

  def stat(us, label, chunk_secs) do
    sorted = Enum.sort(us)
    n = length(sorted)
    mean = Enum.sum(sorted) / n
    p50 = percentile(sorted, 50)
    p99 = percentile(sorted, 99)
    max = List.last(sorted)
    rtf = chunk_secs * 1_000_000 / mean
    IO.puts(label)
    IO.puts("  chunks      #{n}")
    IO.puts("  mean        #{Float.round(mean, 1)} µs/chunk")
    IO.puts("  p50         #{p50} µs")
    IO.puts("  p99         #{p99} µs")
    IO.puts("  max         #{max} µs")
    IO.puts("  realtime    #{Float.round(rtf, 1)}x  (one stream uses #{Float.round(100 / rtf, 2)}% of a core)")
    IO.puts("  streams/core ~#{round(rtf)} concurrent real-time streams")
  end
end

# --- latency: time process_chunk on the hot path -------------------------------------------------
{:ok, model} = Native.load_model(model_path)
{:ok, stream} = Native.new_stream(model)

warm = Enum.take(chunks, 50)
Enum.each(warm, fn c -> {:ok, _} = Native.process_chunk(model, stream, c) end)
:ok = Native.reset_stream(stream)

# repeat the clip until we have a long, statistically meaningful stream
target = 3000
loops = max(1, div(target, max(1, length(chunks))))
long = List.flatten(List.duplicate(chunks, loops))

us =
  Enum.map(long, fn c ->
    t0 = System.monotonic_time(:nanosecond)
    {:ok, _} = Native.process_chunk(model, stream, c)
    (System.monotonic_time(:nanosecond) - t0) / 1000
  end)

IO.puts("\n=== per-chunk latency (single stream) ===")
B.stat(us, "Native.process_chunk", chunk_secs)

# --- concurrency: many Sessions in parallel, reset/1 as a drain barrier ---------------------------
# Each Session loads + Level3-optimizes its own ORT session, so spin them up AND warm them
# (forces model load + first inference) OUTSIDE the timed window — we want steady-state throughput,
# not startup cost. process_chunk runs on a DirtyCpu scheduler, so parallelism is capped at the
# dirty-CPU scheduler count (≈ cores), not the Session count.
cores = System.schedulers_online()
k = cores * 4
per = 1000
clip = Enum.take(Stream.cycle(chunks), per)
warm = Enum.take(clip, 20)

IO.puts("\n=== concurrent throughput (#{k} Sessions, #{per} chunks each, #{cores} schedulers) ===")

sessions =
  Enum.map(1..k, fn _ ->
    {:ok, s} = Session.start_link(model_path: model_path)
    Enum.each(warm, &Session.process(s, &1))
    :ok = Session.reset(s)
    s
  end)

t0 = System.monotonic_time(:millisecond)

sessions
|> Enum.map(fn s ->
  Task.async(fn ->
    Enum.each(clip, &Session.process(s, &1))
    :ok = Session.reset(s)
  end)
end)
|> Task.await_many(:infinity)

wall_ms = System.monotonic_time(:millisecond) - t0
Enum.each(sessions, &GenServer.stop/1)
total = k * per
audio_secs = total * chunk_secs
IO.puts("  wall        #{wall_ms} ms")
IO.puts("  chunks      #{total}  (#{Float.round(audio_secs, 1)} s of audio)")
IO.puts("  throughput  #{round(total / (wall_ms / 1000))} chunks/s")
IO.puts("  aggregate   #{Float.round(audio_secs / (wall_ms / 1000), 1)}x realtime across #{cores} cores")
