import numpy as np
import json
from pydub import AudioSegment
import pyloudnorm as pyln
import librosa

def load_mp3_as_mono(file_path):
    audio = AudioSegment.from_mp3(file_path).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= (2 ** (8 * audio.sample_width - 1))  # normalize to [-1, 1]
    return samples, audio.frame_rate

def check_loudness(audio_data, sample_rate):
    meter = pyln.Meter(sample_rate)
    lufs = meter.integrated_loudness(audio_data)
    lra = meter.loudness_range(audio_data)
    return lufs, lra

def check_interruptions(audio_data, sample_rate):
    S = librosa.stft(audio_data, n_fft=1024)
    power_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    mean_power = np.mean(power_db)
    above_mean = np.sum(power_db > mean_power)
    total_bins = power_db.size
    return above_mean / total_bins

def estimate_srmr_intelligibility(audio_data, sample_rate):
    # Break signal into frames
    hop_length = 512
    frame_length = 1024
    frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length).T

    modulation_energies = []
    for frame in frames:
        # Hilbert envelope
        envelope = np.abs(librosa.effects.hilbert(frame))
        # FFT of envelope gives modulation spectrum
        modulation_spectrum = np.abs(np.fft.rfft(envelope))
        modulation_energy = np.sum(modulation_spectrum[4:20]) / (np.sum(modulation_spectrum) + 1e-6)  # 4–20 Hz band
        modulation_energies.append(modulation_energy)

    intelligibility_score = np.mean(modulation_energies)
    return intelligibility_score

def analyze_audio(file_path, output_json_path='analysis_result.json'):
    audio_data, sr = load_mp3_as_mono(file_path)

    lufs, lra = check_loudness(audio_data, sr)
    interruptions = check_interruptions(audio_data, sr)
    srmr_intel = estimate_srmr_intelligibility(audio_data, sr)

    results = {
        "lufs": {
            "value": round(lufs, 2),
            "acceptable": -27 < lufs < -17
        },
        "lufs_range": {
            "value": round(lra, 2),
            "acceptable": lra < 5
        },
        "srmr_intelligibility_approx": {
            "value": round(srmr_intel, 2),
            "acceptable": srmr_intel > 3.5
        },
        "interruptions": {
            "value": round(interruptions, 3),
            "acceptable": interruptions > 0.2
        }
    }

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results

# Example usage:
# result = analyze_audio("your_audio_file.mp3")
# print(json.dumps(result, indent=4))
