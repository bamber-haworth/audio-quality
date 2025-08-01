import streamlit as st
import numpy as np
import pyloudnorm as pyln
import librosa
from pydub import AudioSegment
import tempfile
import json
from scipy.signal import hilbert

def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def load_mp3_as_mono(file_path):
    audio = AudioSegment.from_mp3(file_path).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= (2 ** (8 * audio.sample_width - 1))
    return samples, audio.frame_rate

def check_loudness(audio_data, sample_rate):
    meter = pyln.Meter(sample_rate)

    # Integrated loudness
    lufs = meter.integrated_loudness(audio_data)

    # Short-term loudness over 3-second windows
    window_length = 3 * sample_rate
    hop_size = sample_rate

    loudness_values = []
    for i in range(0, len(audio_data) - window_length, hop_size):
        segment = audio_data[i:i+window_length]
        loudness_values.append(meter.integrated_loudness(segment))

    # LRA = difference between 10th and 95th percentile loudness
    if loudness_values:
        lra = np.percentile(loudness_values, 95) - np.percentile(loudness_values, 10)
    else:
        lra = 0.0

    return lufs, lra

def check_interruptions(audio_data, sample_rate):
    S = librosa.stft(audio_data, n_fft=1024)
    power_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    mean_power = np.mean(power_db)
    above_mean = np.sum(power_db > mean_power)
    total_bins = power_db.size
    return above_mean / total_bins


def estimate_srmr_intelligibility(audio_data, sample_rate):
    hop_length = 512
    frame_length = 1024
    frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length).T
    modulation_energies = []
    for frame in frames:
        envelope = np.abs(hilbert(frame))
        modulation_spectrum = np.abs(np.fft.rfft(envelope))
        modulation_energy = np.sum(modulation_spectrum[4:20]) / (np.sum(modulation_spectrum) + 1e-6)
        modulation_energies.append(modulation_energy)
    return np.mean(modulation_energies)

def analyze_audio(file_path):
    audio_data, sr = load_mp3_as_mono(file_path)
    lufs, lra = check_loudness(audio_data, sr)
    interruptions = check_interruptions(audio_data, sr)
    srmr_intel = estimate_srmr_intelligibility(audio_data, sr)

    results = {
        "LUFS": {"value": round(lufs, 2), "acceptable": -27 < lufs < -17},
        "LUFS Range": {"value": round(lra, 2), "acceptable": lra < 5},
        "Interruptions": {"value": round(interruptions, 3), "acceptable": interruptions > 0.2},
        "SRMR (Intelligibility)": {"value": round(srmr_intel, 2), "acceptable": srmr_intel > 3.5}
    }
    return results

# --- Streamlit UI ---

st.title("🎧 Audio Quality Evaluator")
st.markdown("Upload your `.mp3` file to check LUFS, intelligibility, and interruptions.")

uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing..."):
        result = analyze_audio(tmp_path)

    st.subheader("📊 Results:")
    for key, data in result.items():
        st.markdown(f"**{key}**: `{data['value']}` {'✅' if data['acceptable'] else '❌'}")

    st.download_button("Download JSON", data=json.dumps(result, indent=4, default=to_serializable), file_name="audio_quality.json")
