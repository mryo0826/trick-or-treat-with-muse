# Music draft for the Halloween song (all-original composition).
# Synthesizes a ~30s BGM: music-box melody + bass + light percussion.
# Key: A minor, Tempo: 120 BPM, 15 bars x 4 beats = 30.0s
import numpy as np
from scipy.io import wavfile
import os

SR = 44100
BPM = 120
BEAT = 60.0 / BPM          # 0.5s
BAR = BEAT * 4             # 2.0s
BARS = 15
DUR = BAR * BARS           # 30.0s

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- note helpers -----------------------------------------------------------
NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
         "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

def midi(name):
    if len(name) >= 2 and name[1] == "#":
        letter, octv = name[:2], int(name[2:])
    else:
        letter, octv = name[:1], int(name[1:])
    return 12 * (octv + 1) + NAMES[letter]

def freq(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)

track = np.zeros(int(SR * (DUR + 1.0)))

def add(buf_t, wave, gain=1.0):
    i = int(buf_t * SR)
    j = min(len(track), i + len(wave))
    if j > i:
        track[i:j] += wave[: j - i] * gain

# ---- instruments ------------------------------------------------------------
def music_box(m, dur=0.48):
    f = freq(m)
    t = np.arange(int(SR * dur)) / SR
    env = np.exp(-t * 5.0)
    w = (np.sin(2 * np.pi * f * t)
         + 0.45 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 8)
         + 0.20 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t * 12)) * env
    return w * 0.5

def bass_note(m, dur=0.46):
    f = freq(m)
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1.0, t * 40) * np.exp(-t * 2.2)
    w = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 3 * t)) * env
    return w * 0.55

def kick(dur=0.22):
    t = np.arange(int(SR * dur)) / SR
    f = 120 * np.exp(-t * 18) + 45
    ph = np.cumsum(2 * np.pi * f / SR)
    return np.sin(ph) * np.exp(-t * 14) * 0.9

def noise_hit(dur, decay, gain):
    t = np.arange(int(SR * dur)) / SR
    rng = np.random.default_rng(7)
    return rng.standard_normal(len(t)) * np.exp(-t * decay) * gain

# ---- composition ------------------------------------------------------------
# chords per bar (root midi for bass, in A minor)
CHORDS = ["A2", "A2", "F2", "E2", "A2", "A2", "F2", "E2",
          "A2", "D3", "A2", "E2", "F2", "E2", "A2"]
FIFTH = {"A2": "E3", "F2": "C3", "E2": "B2", "D3": "A3"}

# melody: 8 eighth-note slots per bar (None = rest). All original.
MELODY = [
    ["A4", "C5", "E5", "A5", "G5", "E5", "C5", "E5"],      # Am
    ["A5", "G5", "E5", "D5", "C5", "D5", "E5", None],      # Am
    ["F5", "A5", "G5", "F5", "E5", "F5", "A5", "G5"],      # F
    ["E5", "G#5", "B5", "G#5", "E5", "D5", "B4", None],    # E
    ["A4", "C5", "E5", "A5", "C6", "A5", "G5", "E5"],      # Am
    ["E5", "D5", "C5", "D5", "E5", "C5", "A4", None],      # Am
    ["F4", "A4", "C5", "F5", "E5", "C5", "A4", "C5"],      # F
    ["B4", "D5", "E5", "G#5", "B5", "G#5", "E5", None],    # E
    ["A5", "G5", "E5", "C5", "D5", "E5", "A4", "C5"],      # Am
    ["D5", "F5", "A5", "F5", "D5", "F5", "A5", "G5"],      # Dm
    ["E5", "C5", "A4", "C5", "E5", "A5", "G5", "E5"],      # Am
    ["G#5", "B5", "G#5", "E5", "D5", "B4", "G#4", "B4"],   # E
    ["A5", "F5", "C5", "F5", "A5", "G5", "F5", "E5"],      # F
    ["D5", "B4", "G#4", "B4", "E5", None, None, None],     # E
    ["A4", None, None, None, "A5", None, None, None],      # Am (final)
]

for bar in range(BARS):
    t0 = bar * BAR
    root = CHORDS[bar]
    # bass: root - root - fifth - root (quarter notes)
    for qi, nm in enumerate([root, root, FIFTH[root], root]):
        add(t0 + qi * BEAT, bass_note(midi(nm)), 0.8)
    # percussion
    add(t0 + 0 * BEAT, kick(), 0.7)
    add(t0 + 2 * BEAT, kick(), 0.7)
    add(t0 + 1 * BEAT, noise_hit(0.12, 40, 0.20))   # snare-ish
    add(t0 + 3 * BEAT, noise_hit(0.12, 40, 0.20))
    for e in range(8):                              # hat ticks
        add(t0 + e * BEAT / 2, noise_hit(0.05, 90, 0.06))
    # melody (music box, eighth notes)
    for e, nm in enumerate(MELODY[bar]):
        if nm:
            add(t0 + e * BEAT / 2, music_box(midi(nm)), 0.9)

# fade out the last second, normalize
n = len(track)
fade = np.ones(n)
fi = int(SR * 1.0)
fade[-fi:] = np.linspace(1.0, 0.0, fi)
track *= fade
track = track / max(1e-9, np.max(np.abs(track))) * 0.82

wav_path = os.path.join(OUT_DIR, "bgm_draft.wav")
wavfile.write(wav_path, SR, (track * 32767).astype(np.int16))
print("wrote", wav_path, f"{len(track)/SR:.1f}s")
