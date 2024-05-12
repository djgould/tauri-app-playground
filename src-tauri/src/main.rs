// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use hound::{SampleFormat, WavReader};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn parse_wav_file(path: &Path) -> Vec<i16> {
    let reader = WavReader::open(path).expect("failed to read file");

    if reader.spec().channels != 1 {
        panic!("expected mono audio file");
    }
    if reader.spec().sample_format != SampleFormat::Int {
        panic!("expected integer sample format");
    }
    if reader.spec().sample_rate != 16000 {
        panic!("expected 16KHz sample rate");
    }
    if reader.spec().bits_per_sample != 16 {
        panic!("expected 16 bits per sample");
    }

    reader
        .into_samples::<i16>()
        .map(|x| x.expect("sample"))
        .collect::<Vec<_>>()
}

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
async fn transcribe(path: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        use std::path::Path;

        println!("Path: {}", path);
        let audio_path = Path::new("/Users/devingould/tauri-app/src-tauri/src/samples/a13.wav");
        if !audio_path.exists() {
            panic!("audio file doesn't exist");
        }
        let whisper_path = Path::new("/Users/devingould/tauri-app/src-tauri/src/models/ggml-small.en-tdrz.bin");
        if !whisper_path.exists() {
            panic!("whisper file doesn't exist");
        }

        // Assuming parse_wav_file and other functions are correctly defined elsewhere
        let original_samples = parse_wav_file(audio_path);
        let mut samples = vec![0.0f32; original_samples.len()];
        whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)
            .expect("failed to convert samples");

        let ctx = WhisperContext::new_with_params(
            &whisper_path.to_string_lossy(),
            WhisperContextParameters::default(),
        )
        .expect("failed to open model");
        let mut state = ctx.create_state().expect("failed to create state");
        let mut params = FullParams::new(SamplingStrategy::default());
        params.set_initial_prompt("experience");
        params.set_progress_callback_safe(|progress| println!("Progress callback: {}%", progress));

        let st = std::time::Instant::now();
        state.full(params, &samples)
            .expect("failed to transcribe audio");

        let et = std::time::Instant::now();

        let num_segments = state.full_n_segments()
            .expect("failed to get number of segments");
        let mut full_text = String::new();
        for i in 0..num_segments {
            let segment = state.full_get_segment_text(i)
                .expect("failed to get segment");
            full_text.push_str(&segment);
            full_text.push(' '); // Add a space between segments
            let start_timestamp = state.full_get_segment_t0(i)
                .expect("failed to get start timestamp");
            let end_timestamp = state.full_get_segment_t1(i)
                .expect("failed to get end timestamp");
            println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
        }
        println!("Transcription took {}ms", (et - st).as_millis());
        Ok(full_text)
    }).await.map_err(|e| e.to_string())?
}


fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![transcribe])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
