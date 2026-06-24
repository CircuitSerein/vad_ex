# Real-audio validation corpus

The real-audio gate (`test/real_audio_test.exs`) and the benchmark (`bench/bench.exs`) run against
genuine recordings. The audio itself is **not committed** — only this README, the Python builders,
and the derived golden vector are tracked. Everything under `.cache/` and `data/` is gitignored and
re-fetched on demand.

## Sources & licenses

| kind   | dataset                                   | host (fast CDN)        | license      |
|--------|-------------------------------------------|------------------------|--------------|
| speech | LibriSpeech `test-clean` (20 speakers)    | `openslr/librispeech_asr` on HF | CC BY 4.0   |
| noise  | ESC-50 exterior/urban sounds (6 clips)    | `DynamicSuperb/…ESC50-ExteriorAndUrbanNoises` on HF | CC BY-NC 3.0 |

OpenSLR's own host is frequently unreachable, so we pull from the Hugging Face CDN mirrors. We
redistribute **no audio** — clips are fetched locally and gitignored, and the committed golden
vector contains only derived probabilities + a per-clip sha256. ESC-50 is non-commercial, which is
fine for a local, non-redistributed test fixture. (MUSAN was the original plan for noise/music but
every OpenSLR mirror was dead; music is deferred — ESC-50 covers the non-speech robustness check.)
Cite the datasets if you publish results: Panayotov et al. 2015 (LibriSpeech); Piczak 2015 (ESC-50).

## Usage

Pure Python, no ffmpeg — `soundfile` decodes the audio embedded in the HF parquet files.

```sh
# one-time Python env (the reference model + the decoders)
python3 -m venv test/corpus/.venv
test/corpus/.venv/bin/pip install onnxruntime numpy pyarrow soundfile certifi

# 1. fetch + decode -> data/*.pcm (16 kHz mono s16le) + data/index.tsv
test/corpus/.venv/bin/python test/corpus/fetch_corpus.py

# 2. build the committed golden vector from reference Silero
test/corpus/.venv/bin/python test/corpus/reference_probs.py

# 3. run the gate: the NIF must reproduce the reference probs within tolerance
VAD_EX_BUILD=1 mix test --include real_audio
```

## Files

- `fetch_corpus.py` — downloads the HF parquet shards, decodes via `soundfile` → `data/*.pcm` + `data/index.tsv`
- `reference_probs.py` — streams each clip through reference Silero → `golden_real_audio.json`
- `bench_ref.py` — reference per-chunk latency (compare with `bench/bench.exs`)
- `golden_real_audio.json` — **committed** derived vector; also the manifest (per-clip source + sha256)
- `.cache/`, `data/` — gitignored raw parquet + decoded PCM
