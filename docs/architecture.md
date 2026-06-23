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
| `VadEx.Native` | NIF-биндинги: загрузка модели, per-stream RNN-стейт, инференс окна | Rustler 0.38 + `ort` 2.0.0-rc.12 (ONNX RT 1.24, статически слинкован) |
| `VadEx.Session` | GenServer-per-stream: оркестрация инференса + endpointer + телеметрия | Elixir / OTP |
| `VadEx.Endpointer` | Конечный автомат границ реплики (hysteresis, min-durations, padding) | Чистый Elixir (behaviour, pluggable) |
| `VadEx.Telemetry` | События `chunk` / `speech_start` / `speech_end` (Keathley-конвенции) | `:telemetry` |
| `VadEx.Membrane.Filter` | Опц. Membrane-элемент поверх ядра | `membrane_core` 1.3 (optional dep) |
| Rust NIF crate | `VadSession` (`Mutex<Session>`) + `StreamState` (`Mutex<state,context>`) | Rust, `ort` (ORT внутри `.so`) |

## Поток аудио

```
PCM s16le 16k, окна 512 сэмплов (1024 байт)
  → VadEx.Session.process/2  (cast)
  → VadEx.Native.process_chunk/3   [DirtyCpu NIF]
        input = concat(context, window) → ONNX run(input, state, sr) → prob, stateN
        (per-stream state[2,1,128]+context в ResourceArc<Mutex<..>>, владеет процесс)
  → VadEx.Endpointer.push/3   (silence→starting→speech→trailing→silence)
  → :telemetry  [:vad_ex, :speech_start | :speech_end]
```

Membrane-путь (опц.): `WebRTC.Source → RTP.Opus.Depayloader → Opus.Decoder(16k) → VadEx.Membrane.Filter → sink`.

## NIF-границы (почему так)

- Инференс — `DirtyCpu` NIF (1–5 мс/окно, нельзя на обычном шедулере).
- Модель (`Session`) shared между потоками за `Mutex` (в rc.12 `run` берёт `&mut self`); per-stream
  меняемый стейт — унифицированный `state[2,1,128]` + `context` за `Mutex` в отдельном `ResourceArc`.
  BEAM сам освобождает стейт со смертью процесса-владельца.
- ONNX Runtime **статически слинкован** в NIF (`ort` `download-binaries`): отдельной
  `libonnxruntime` нет — один самодостаточный `.so`/`.dll` на таргет.

## Внешние зависимости

- ONNX Runtime (через `ort`) — статически слинкован в NIF (отдельной либы нет).
- Silero VAD ONNX-модель (MIT) — в `priv/models/silero_vad.onnx`.
- Опц.: `membrane_core` для фильтра.
- Секретов/БД/сети нет — это библиотека.

## Что НЕ здесь (scope-границы)

- ASR/Whisper, TTS, диалог-оркестрация, barge-in → это voice-orchestrator (`Larynx`), отдельный проект.
- Transformer turn-detection → v0.2 (хук в `Endpointer` behaviour есть, реализации нет).
