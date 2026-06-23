# Architecture

> Как устроен `vad_ex`. Обновлять при структурных изменениях.

## Обзор

`vad_ex` принимает поток сырого PCM (16 kHz, mono, s16le, окнами по 512 сэмплов), на каждом
окне гоняет Silero VAD через Rust/ONNX-Runtime NIF и получает вероятность речи. Поверх
вероятностей работает конечный автомат endpointing'а, который выдаёт события границ реплики
(`speech_start` / `speech_end`) через `:telemetry`. Один аудиопоток = один supervised GenServer.

## Компоненты

| Компонент | Ответственность | Стек |
|---|---|---|
| `VadEx.Native` | NIF-биндинги: загрузка модели, per-stream RNN-стейт, инференс окна | Rustler 0.34 + `ort` 2.0.0-rc.12 (ONNX RT 1.26) |
| `VadEx.Session` | GenServer-per-stream: оркестрация инференса + endpointer + телеметрия | Elixir / OTP |
| `VadEx.Endpointer` | Конечный автомат границ реплики (hysteresis, min-durations, padding) | Чистый Elixir (behaviour, pluggable) |
| `VadEx.Telemetry` | События `chunk` / `speech_start` / `speech_end` (Keathley-конвенции) | `:telemetry` |
| `VadEx.Membrane.Filter` | Опц. Membrane-элемент поверх ядра | `membrane_core` 1.3 (optional dep) |
| Rust NIF crate | `VadSession` (immutable session) + `StreamState` (Mutex<h,c,context>) | Rust, `ort` load-dynamic |

## Поток аудио

```
PCM s16le 16k, окна 512 сэмплов (1024 байт)
  → VadEx.Session.process/2  (cast)
  → VadEx.Native.process_chunk/3   [DirtyCpu NIF]
        input = concat(context, window) → ONNX run(h,c,sr) → prob, hn, cn
        (per-stream state в ResourceArc<Mutex<..>>, владеет процесс)
  → VadEx.Endpointer.push/3   (silence→starting→speech→trailing→silence)
  → :telemetry  [:vad_ex, :speech_start | :speech_end]
```

Membrane-путь (опц.): `WebRTC.Source → RTP.Opus.Depayloader → Opus.Decoder(16k) → VadEx.Membrane.Filter → sink`.

## NIF-границы (почему так)

- Инференс — `DirtyCpu` NIF (1–5 мс/окно, нельзя на обычном шедулере).
- Модель (`Session`) immutable и shared между потоками; per-stream меняемый стейт `(h, c, context)`
  за `Mutex` в отдельном `ResourceArc`. BEAM сам освобождает стейт со смертью процесса-владельца.
- `ort` в режиме `load-dynamic`: `libonnxruntime` не вшит в сборку, грузится в рантайме из
  `priv/lib` (бандлится рядом с precompiled `.so`). См. ADR 0002, research 02.

## Внешние зависимости

- ONNX Runtime (через `ort`) — нативная либа, бандлится с NIF.
- Silero VAD ONNX-модель (MIT) — в `priv/models/silero_vad.onnx`.
- Опц.: `membrane_core` для фильтра.
- Секретов/БД/сети нет — это библиотека.

## Что НЕ здесь (scope-границы)

- ASR/Whisper, TTS, диалог-оркестрация, barge-in → это voice-orchestrator (`Larynx`), отдельный проект.
- Transformer turn-detection → v0.2 (хук в `Endpointer` behaviour есть, реализации нет).
