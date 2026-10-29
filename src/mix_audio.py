# Mix BGM + placed voice lines, and compute the lip-sync envelope.
import json, os, sys
import numpy as np
from scipy.io import wavfile

BASE = os.path.join(os.path.dirname(__file__), "..")
SR = 44100
TOTAL = 31.0  # seconds (BGM 30s + tail)

SLOTS = {"l0": 0.4, "l1": 6.0, "l2": 10.0, "l3": 14.5,
         "l4": 18.0, "l5": 21.0, "l6": 25.0}

def load_mono(path):
    sr, data = wavfile.read(path)
    assert sr == SR, (path, sr)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float64) / 32768.0

bgm = load_mono(os.path.join(BASE, "out", sys.argv[1] if len(sys.argv) > 1 else "bgm_techno.wav"))
n = int(SR * TOTAL)
bgm_t = np.zeros(n); bgm_t[:min(n, len(bgm))] = bgm[:min(n, len(bgm))]

voice = np.zeros(n)
for lid, start in SLOTS.items():
    v = load_mono(os.path.join(BASE, "work", "voice", f"{lid}.wav"))
    i = int(start * SR)
    j = min(n, i + len(v))
    voice[i:j] += v[: j - i]

# ducking: lower BGM gain where the voice is active (0.15s ramps)
env_v = np.minimum(1.0, np.abs(voice) * 3)
k = int(SR * 0.15)
env_s = np.convolve(env_v, np.ones(k) / k, mode="same")
duck = 1.0 - 0.45 * np.clip(env_s, 0, 1)
mix = bgm_t * 0.78 * duck + voice * 1.0
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.9

wavfile.write(os.path.join(BASE, "out", "mix_full.wav"), SR,
              (mix * 32767).astype(np.int16))

PRE = 14.0
pre = mix[: int(SR * PRE)].copy()
fade = int(SR * 0.5)
pre[-fade:] *= np.linspace(1, 0, fade)
wavfile.write(os.path.join(BASE, "out", "mix_preview.wav"), SR,
              (pre * 32767).astype(np.int16))

# lip-sync envelope: RMS per video frame (30fps) from the voice track
fps = 30
frames = int(TOTAL * fps)
hop = SR // fps
rms = np.array([np.sqrt(np.mean(voice[f * hop:(f + 1) * hop] ** 2) + 1e-12)
                for f in range(frames)])
rms = np.convolve(rms, np.ones(3) / 3, mode="same")
openness = np.clip((rms - 0.02) / 0.16, 0, 1) ** 0.8
np.save(os.path.join(BASE, "work", "envelope.npy"), openness)
with open(os.path.join(BASE, "work", "timeline.json"), "w") as f:
    json.dump(SLOTS, f)
print("mix + envelope done; voice RMS max", float(rms.max()))
