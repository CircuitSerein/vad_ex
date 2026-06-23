//! vad_ex NIF — Silero VAD streaming inference via ONNX Runtime (`ort`).
//!
//! STAGE 1: real `process_chunk` inference against the **verified** Silero v5/v6 contract — a single
//! unified `state` `[2,1,128]` (NOT the v4 `h`/`c` `[2,1,64]` older scaffolds use), with a 64-sample
//! look-back context prepended to each 512-sample window.
//!
//! Contract (verified live against the shipped v6.2 .onnx, opset 16, producer spox):
//!   Inputs:  input f32 [1, 576] (= 64 context + 512 window @16k) | state f32 [2,1,128] | sr i64 [1]
//!   Outputs: output f32 [1, 1] (speech prob 0..1)               | stateN f32 [2,1,128]
//! See docs/architecture.md.
//!
//! ## Session sharing & `Session::run(&mut self)`
//! ort rc.12's `Session::run` takes `&mut self`, so the immutable-shared-session design is impossible
//! as-is. We share ONE session (one copy of the 2.3 MB model) behind a `Mutex` inside the
//! `ResourceArc<VadSession>`; inference is serialized through a sub-millisecond lock. That caps
//! inference parallelism at 1 regardless of dirty-scheduler count — fine for the v0.1 audience.
//! v0.2 scale path: a pool of N sessions (≈ dirty CPU schedulers) to recover core-level parallelism.
//!
//! ## Linking
//! `ort` is built with `download-binaries` (rc.12 → ONNX Runtime 1.24, api-24): the prebuilt ORT
//! is fetched at build time and **statically linked** into this cdylib. The artifact is therefore
//! self-contained — no `libonnxruntime` to ship alongside it, no dlopen at runtime, no env var.

use ndarray::{Array1, Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use rustler::{Binary, Env, Resource, ResourceArc, Term};
use std::sync::Mutex;

// Silero v5/v6 @16k. v0.1 is 16k-only; the 8k path (WINDOW 256 / CONTEXT 32) is a v0.2 item.
const SR: i64 = 16_000;
const WINDOW: usize = 512; // samples per chunk
const CONTEXT: usize = 64; // look-back samples prepended to each window
const INPUT_LEN: usize = CONTEXT + WINDOW; // 576
const STATE_LEN: usize = 2 * 128; // unified LSTM state [2, batch=1, 128], flat length

fn e2s<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// Immutable-across-streams ONNX session, behind a Mutex (run is `&mut self`). One copy of the
/// model shared by every stream; `process_chunk` locks it only for the inference call.
pub struct VadSession {
    session: Mutex<Session>,
}

#[rustler::resource_impl]
impl Resource for VadSession {}

/// Per-stream mutable RNN state. Owned by the calling process; the BEAM drops it on process death.
pub struct StreamState {
    inner: Mutex<HiddenState>,
}

#[rustler::resource_impl]
impl Resource for StreamState {}

pub struct HiddenState {
    state: Vec<f32>, // unified [2,1,128]; passed as `state`, replaced by `stateN` each call
    context: Vec<f32>, // last CONTEXT samples of the previous chunk (look-back)
}

impl HiddenState {
    fn zeroed() -> Self {
        HiddenState {
            state: vec![0.0; STATE_LEN],
            context: vec![0.0; CONTEXT],
        }
    }
}

// rustler 0.38: NIFs are auto-discovered; resources auto-register via #[resource_impl].
fn load(_env: Env, _info: Term) -> bool {
    true
}

#[rustler::nif(schedule = "DirtyCpu")]
fn load_model(path: String) -> Result<ResourceArc<VadSession>, String> {
    let session = Session::builder()
        .map_err(e2s)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(e2s)?
        .with_intra_threads(1)
        .map_err(e2s)?
        .with_inter_threads(1)
        .map_err(e2s)?
        .commit_from_file(&path)
        .map_err(e2s)?;

    Ok(ResourceArc::new(VadSession {
        session: Mutex::new(session),
    }))
}

// Returns `{:ok, stream}` (Result → tuple) to match VadEx.Session.init's `with {:ok, stream} <- ...`.
#[rustler::nif]
fn new_stream(_model: ResourceArc<VadSession>) -> Result<ResourceArc<StreamState>, String> {
    Ok(ResourceArc::new(StreamState {
        inner: Mutex::new(HiddenState::zeroed()),
    }))
}

/// Run inference on one 512-sample (1024-byte s16le) chunk, advancing the stream's RNN state.
/// Returns the speech probability for that chunk.
#[rustler::nif(schedule = "DirtyCpu")]
fn process_chunk(
    model: ResourceArc<VadSession>,
    stream: ResourceArc<StreamState>,
    audio: Binary,
) -> Result<f32, String> {
    let mut st = stream.inner.lock().map_err(e2s)?;

    // 1. Decode s16le -> f32 [-1,1]; expect exactly WINDOW samples.
    let samples = decode_s16le(&audio);
    if samples.len() != WINDOW {
        return Err(format!("expected {WINDOW} samples, got {}", samples.len()));
    }

    // 2. input = concat(context[CONTEXT], window[WINDOW]) -> [1, 576].
    let mut input_vec = Vec::with_capacity(INPUT_LEN);
    input_vec.extend_from_slice(&st.context);
    input_vec.extend_from_slice(&samples);
    let input_arr = Array2::<f32>::from_shape_vec((1, INPUT_LEN), input_vec).map_err(e2s)?;
    let state_arr = Array3::<f32>::from_shape_vec((2, 1, 128), st.state.clone()).map_err(e2s)?;
    let sr_arr = Array1::<i64>::from_vec(vec![SR]);

    // 3. run(input, state, sr) -> (output, stateN). Extract owned values, then release the lock.
    let prob: f32;
    let new_state: Vec<f32>;
    {
        let input_val = Value::from_array(input_arr).map_err(e2s)?;
        let state_val = Value::from_array(state_arr).map_err(e2s)?;
        let sr_val = Value::from_array(sr_arr).map_err(e2s)?;

        let mut session = model.session.lock().map_err(e2s)?;
        let outputs = session
            .run(ort::inputs![
                "input" => input_val,
                "state" => state_val,
                "sr" => sr_val
            ])
            .map_err(e2s)?;

        let (_, out) = outputs
            .get("output")
            .ok_or_else(|| "missing output tensor".to_string())?
            .try_extract_tensor::<f32>()
            .map_err(e2s)?;
        prob = *out.first().ok_or_else(|| "empty output".to_string())?;

        let (_, sn) = outputs
            .get("stateN")
            .ok_or_else(|| "missing stateN tensor".to_string())?
            .try_extract_tensor::<f32>()
            .map_err(e2s)?;
        if sn.len() != STATE_LEN {
            return Err(format!("stateN len {}, expected {STATE_LEN}", sn.len()));
        }
        new_state = sn.to_vec();
    }

    // 4. Thread state + carry the window tail as next chunk's context.
    st.state = new_state;
    st.context = samples[WINDOW - CONTEXT..].to_vec();

    Ok(prob)
}

#[rustler::nif]
fn reset_stream(stream: ResourceArc<StreamState>) -> rustler::Atom {
    if let Ok(mut s) = stream.inner.lock() {
        *s = HiddenState::zeroed();
    }
    atoms::ok()
}

fn decode_s16le(bin: &Binary) -> Vec<f32> {
    bin.as_slice()
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect()
}

mod atoms {
    rustler::atoms! { ok }
}

rustler::init!("Elixir.VadEx.Native", load = load);
