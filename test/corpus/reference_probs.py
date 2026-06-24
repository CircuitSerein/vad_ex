"""Build golden_real_audio.json from the fetched corpus.

Streams every clip in data/ through reference Silero (Python onnxruntime), replicating the NIF
contract exactly, and records per-chunk probabilities. That JSON is committed and doubles as the
manifest (each clip carries its source member + PCM sha256). The NIF-side regression
(test/real_audio_test.exs) then asserts process_chunk reproduces these probs within `tolerance`.

NIF contract: input f32[1,576]=concat(context[64], window[512]); state f32[2,1,128]; sr i64[1];
outputs -> output (prob), stateN.

Run from the repo root (after fetch_corpus.sh):
    test/corpus/.venv/bin/python test/corpus/reference_probs.py
or with any python that has onnxruntime + numpy.
"""
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
MODEL = os.path.join(REPO, "priv", "models", "silero_vad.onnx")
DATA = os.path.join(ROOT, "data")
INDEX = os.path.join(DATA, "index.tsv")
OUT = os.path.join(ROOT, "golden_real_audio.json")

WIN, CTX, SR = 512, 64, 16000
TOLERANCE = 0.002
SPEECH_THRESHOLD = 0.5
# Behavioral sanity bounds (secondary to the exact-match gate, which is what really proves the port):
SPEECH_MIN_MAX_PROB = 0.9       # a real read-English utterance must drive prob high somewhere
NONSPEECH_MAX_SPEECH_FRAC = 0.5  # pure noise must not read as mostly-speech (music excluded: may sing)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def stream(sess, out_names, pcm):
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros(CTX, dtype=np.float32)
    sr = np.array([SR], dtype=np.int64)
    probs = []
    for i in range(len(pcm) // WIN):
        win = pcm[i * WIN:(i + 1) * WIN]
        inp = np.concatenate([ctx, win]).reshape(1, WIN + CTX).astype(np.float32)
        named = dict(zip(out_names, sess.run(None, {"input": inp, "state": state, "sr": sr})))
        probs.append(round(float(named["output"].reshape(-1)[0]), 6))
        state = named["stateN"]
        ctx = win[-CTX:]
    return probs


def main():
    if not os.path.exists(INDEX):
        sys.exit(f"no {INDEX} — run test/corpus/fetch_corpus.sh first")

    sess = ort.InferenceSession(MODEL)
    out_names = [o.name for o in sess.get_outputs()]

    clips = []
    with open(INDEX) as f:
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]

    for name, kind, source in rows:
        pcm_path = os.path.join(DATA, f"{name}.pcm")
        pcm = np.fromfile(pcm_path, dtype="<i2").astype(np.float32) / 32768.0
        probs = stream(sess, out_names, pcm)
        mx = max(probs) if probs else 0.0
        frac = (sum(p >= SPEECH_THRESHOLD for p in probs) / len(probs)) if probs else 0.0
        clips.append({
            "name": name, "kind": kind, "source": source,
            "sha256": sha256_file(pcm_path), "n_chunks": len(probs),
            "max_prob": round(mx, 6), "speech_frac": round(frac, 4), "probs": probs,
        })
        print(f"  {name:12s} {kind:6s} chunks={len(probs):4d} max={mx:.4f} "
              f"speech_frac={frac:.3f}  <- {source}")

    doc = {
        "model": "silero_vad.onnx (snakers4 v6.2)",
        "model_sha256": sha256_file(MODEL),
        "ref_runtime": f"onnxruntime {ort.__version__}",
        "sample_rate": SR, "window": WIN, "context": CTX, "chunk_bytes": WIN * 2,
        "tolerance": TOLERANCE,
        "behavioral": {
            "speech_threshold": SPEECH_THRESHOLD,
            "speech_min_max_prob": SPEECH_MIN_MAX_PROB,
            "nonspeech_max_speech_frac": NONSPEECH_MAX_SPEECH_FRAC,
        },
        "clips": clips,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    kinds = {}
    for c in clips:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {OUT}: {len(clips)} clips ({kinds})")


if __name__ == "__main__":
    main()
