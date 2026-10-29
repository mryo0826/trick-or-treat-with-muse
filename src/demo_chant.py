# Chant-polish demo: chorus window of bgm_60 (37.5-53.5s) with the voice
# hooks re-placed and treated like a produced techno chant instead of raw TTS:
# dotted-eighth delay, Haas doubling, +2st sparkle layer, tighter ducking.
import os
import numpy as np
from scipy.io import wavfile

SR = 44100
BASE = os.path.join(os.path.dirname(__file__), "..")
DUR = 16.0
N = int(SR * DUR)

def load_mono(path):
    sr, d = wavfile.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    return d.astype(np.float32) / 32768.0

bgm_full = load_mono(os.path.join(BASE, "out", "bgm_60.wav"))
i0 = int(37.5 * SR)
bgm = bgm_full[i0:i0 + N].copy()

voice = np.zeros(N, dtype=np.float32)
for lid, start in [("l5", 0.4), ("l6", 4.15), ("l11", 7.9)]:
    v = load_mono(os.path.join(BASE, "work", "voice2", f"{lid}.wav"))
    i = int(start * SR)
    j = min(N, i + len(v))
    voice[i:j] += v[:j - i]

# --- chant treatment ---
dly = int(SR * 0.3516)  # dotted eighth at 128 BPM
treated = voice.copy()
treated[dly:] += voice[:-dly] * 0.34
treated[2 * dly:] += voice[:-2 * dly] * 0.18
haas = int(SR * 0.018)
treated[haas:] += voice[:-haas] * 0.30
# sparkle layer: +2 semitones by resampling (shortens), zero-padded back
r = 2 ** (2 / 12)
idx = (np.arange(N) * r).astype(int)
sparkle = np.where(idx < N, voice[np.minimum(idx, N - 1)], 0.0)
treated += sparkle * 0.13
treated = treated / max(1e-9, np.max(np.abs(treated))) * 0.9

win = int(SR * 0.1)
rms = np.sqrt(np.convolve(voice ** 2, np.ones(win) / win, mode="same"))
duck = 1.0 - 0.5 * np.clip(rms * 4, 0, 1)
mix = bgm * 0.85 * duck + treated * 0.95
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.92

out = os.path.join(BASE, "out", "demo_chant.wav")
wavfile.write(out, SR, (mix * 32767).astype(np.int16))
print("wrote", out)
