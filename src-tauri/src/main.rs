// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::anyhow;
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::de;
use serde::Serialize;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn parse_wav_file(path: &Path) -> Vec<i16> {
    let reader = WavReader::open(path).expect("failed to read file");

    if reader.spec().channels != 1 {
        panic!("expected mono audio file");
    }
    if reader.spec().sample_format != SampleFormat::Int {
        panic!("expected integer sample format");
    }
    if reader.spec().bits_per_sample != 16 {
        panic!("expected 16 bits per sample");
    }

    reader
        .into_samples::<i16>()
        .map(|x| x.expect("sample"))
        .collect::<Vec<_>>()
}

fn parse_and_resample_wav_file(path: &Path, target_sample_rate: f64) -> Vec<i16> {
    let mut reader = WavReader::open(path).expect("failed to read file");
    let spec = reader.spec();

    if spec.channels != 1 {
        panic!("expected mono audio file");
    }
    if spec.sample_format != SampleFormat::Int {
        panic!("expected integer sample format");
    }
    if spec.bits_per_sample != 16 {
        panic!("expected 16 bits per sample");
    }

    // Original sample rate
    let original_sample_rate = spec.sample_rate as f64;

    // Read all samples
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .map(|s| s.expect("failed to read sample"))
        .collect();

    // Set up resampler if the sample rates are different
    if (original_sample_rate - target_sample_rate).abs() > f64::EPSILON {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.90,
            interpolation: rubato::SincInterpolationType::Cubic,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(
            target_sample_rate / original_sample_rate,
            2.0,
            params,
            samples.len(),
            1, // Channels
        )
        .unwrap();

        // Convert i16 to f32 samples
        let f32_samples: Vec<f32> = samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();

        let waves_in = &[f32_samples];
        // Resample
        let resampled_samples = resampler.process(waves_in, None).unwrap();

        // Convert back to i16
        resampled_samples[0]
            .iter()
            .map(|&s| (s * i16::MAX as f32) as i16)
            .collect()
    } else {
        samples
    }
}

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
async fn transcribe(path: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        use std::path::Path;

        println!("Path: {}", path);
        let audio_path = Path::new("/Users/devingould/tauri-app/src-tauri/output.wav");
        if !audio_path.exists() {
            panic!("audio file doesn't exist");
        }
        let whisper_path =
            Path::new("/Users/devingould/tauri-app/src-tauri/src/models/ggml-small.en-tdrz.bin");
        if !whisper_path.exists() {
            panic!("whisper file doesn't exist");
        }

        // Assuming parse_wav_file and other functions are correctly defined elsewhere
        let original_samples = parse_and_resample_wav_file(audio_path, 16000.0);
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
        params.set_tdrz_enable(true);

        let st = std::time::Instant::now();
        state
            .full(params, &samples)
            .expect("failed to transcribe audio");

        let et = std::time::Instant::now();

        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");
        let mut full_text = String::new();
        for i in 0..num_segments {
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            full_text.push_str(&segment);
            full_text.push(' '); // Add a space between segments
            if (state.full_get_segment_speaker_turn_next(i)) {
                full_text.push('-');
            }
            let start_timestamp = state
                .full_get_segment_t0(i)
                .expect("failed to get start timestamp");
            let end_timestamp = state
                .full_get_segment_t1(i)
                .expect("failed to get end timestamp");
            println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
        }
        println!("Transcription took {}ms", (et - st).as_millis());
        Ok(full_text)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(Debug, Serialize)]
struct Error {
    message: String,
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error {
            message: err.to_string(),
        }
    }
}

#[tauri::command]
fn record() -> Result<(), Error> {
    println!("recording");
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no output device available");
    println!("{:#?}", device.name());
    let supported_formats_range = device.supported_input_configs().map_err(|e| anyhow!(e))?;

    for format in supported_formats_range {
        println!("{:?}", format);
    }
    let config = device.default_input_config().map_err(|e| anyhow!(e))?;

    // Define WAV file specifications
    let spec = WavSpec {
        channels: config.channels(),
        sample_rate: config.sample_rate().0,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = Arc::new(Mutex::new(
        WavWriter::create("output.wav", spec).map_err(|e| anyhow!(e))?,
    ));

    let writer_clone = writer.clone();
    let stream = device
        .build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut writer = writer_clone.lock().unwrap();
                for &sample in data {
                    let amplitude = (sample * i16::MAX as f32) as i16; // convert f32 audio samples to i16
                    writer
                        .write_sample(amplitude)
                        .expect("Failed to write sample");
                }
            },
            |err| {
                eprintln!("Error: {:?}", err);
            },
            Some(std::time::Duration::from_secs(30)), // Set a timeout of 30 seconds
        )
        .map_err(|e| anyhow!(e))?;

    stream.play().map_err(|e| anyhow!(e))?;

    // Record for a specific duration
    std::thread::sleep(std::time::Duration::from_secs(10));

    // Finalize the WAV file
    drop(stream);
    drop(writer);

    Ok(())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![transcribe, record])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
