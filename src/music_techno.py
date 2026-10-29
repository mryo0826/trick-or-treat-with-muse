# Techno arrangement of the Halloween song (all-original composition).
# Same original melody/chords as music_draft.py, re-arranged as generic techno:
# four-on-the-floor kick, claps, offbeat hats, driving 16th synth bass,
# detuned saw pad, lead with dotted-eighth delay. 128 BPM, 16 bars = 30.0s.
import numpy as np
from scipy.io import wavfile
import os

SR = 44100
BPM = 128
BEAT = 60.0 / BPM
BAR = BEAT * 4
BARS = 16
DUR = BAR * BARS
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT_DIR, exist_ok=True)

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

N = int(SR * (DUR + 1.2))
kick_times = []

def add_to(buf, t0, wave, gain=1.0):
    i = int(t0 * SR)
    j = min(len(buf), i + len(wave))
    if j > i:
        buf[i:j] += wave[: j - i] * gain

def one_pole(x, a):
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y

def saw(m, dur):
    f = freq(m)
    t = np.arange(int(SR * dur)) / SR
    ph = (f * t) % 1.0
    return 2 * ph - 1

def kick(dur=0.24):
    t = np.arange(int(SR * dur)) / SR
    f = 130 * np.exp(-t * 16) + 44
    ph = np.cumsum(2 * np.pi * f / SR)
    return np.sin(ph) * np.exp(-t * 13)

def clap(dur=0.18):
    t = np.arange(int(SR * dur)) / SR
    rng = np.random.default_rng(11)
    n = rng.standard_normal(len(t))
    return one_pole(n, 0.35) * np.exp(-t * 22) * 0.9

def hat(dur, decay, gain):
    t = np.arange(int(SR * dur)) / SR
    rng = np.random.default_rng(5)
    n = rng.standard_normal(len(t))
    hp = n - one_pole(n, 0.08)
    return hp * np.exp(-t * decay) * gain

# progression (16 bars) and pad chord tones
ROOTS = ["A1", "A1", "F1", "E1", "A1", "A1", "F1", "E1",
         "A1", "D2", "A1", "E1", "F1", "E1", "A1", "A1"]
PAD = {"A": ["A3", "C4", "E4"], "F": ["F3", "A3", "C4"],
       "E": ["E3", "G#3", "B3"], "D": ["D3", "F3", "A3"]}
MELODY = [
    ["A4", "C5", "E5", "A5", "G5", "E5", "C5", "E5"],
    ["A5", "G5", "E5", "D5", "C5", "D5", "E5", None],
    ["F5", "A5", "G5", "F5", "E5", "F5", "A5", "G5"],
    ["E5", "G#5", "B5", "G#5", "E5", "D5", "B4", None],
    ["A4", "C5", "E5", "A5", "C6", "A5", "G5", "E5"],
    ["E5", "D5", "C5", "D5", "E5", "C5", "A4", None],
    ["F4", "A4", "C5", "F5", "E5", "C5", "A4", "C5"],
    ["B4", "D5", "E5", "G#5", "B5", "G#5", "E5", None],
    ["A5", "G5", "E5", "C5", "D5", "E5", "A4", "C5"],
    ["D5", "F5", "A5", "F5", "D5", "F5", "A5", "G5"],
    ["E5", "C5", "A4", "C5", "E5", "A5", "G5", "E5"],
    ["G#5", "B5", "G#5", "E5", "D5", "B4", "G#4", "B4"],
    ["A5", "F5", "C5", "F5", "A5", "G5", "F5", "E5"],
    ["D5", "B4", "G#4", "B4", "E5", None, None, None],
    ["A4", None, None, None, "A5", None, None, None],
    ["A4", None, "E5", None, "A5", None, None, None],
]

drums = np.zeros(N)
bass_bus = np.zeros(N)
pad_bus = np.zeros(N)
lead_bus = np.zeros(N)

SIX = BEAT / 4  # 16th note
for bar in range(BARS):
    t0 = bar * BAR
    root_name = ROOTS[bar]
    root_m = midi(root_name)
    letter = root_name[0]
    # drums: kick 4-floor, clap on 2&4, hats offbeat 8ths + 16th ticks
    for q in range(4):
        add_to(drums, t0 + q * BEAT, kick(), 0.95)
        kick_times.append(t0 + q * BEAT)
    for q in (1, 3):
        add_to(drums, t0 + q * BEAT, clap(), 0.55)
    for e in range(4):
        add_to(drums, t0 + e * BEAT + BEAT / 2, hat(0.09, 46, 0.30))  # offbeat open-ish
    for s16 in range(16):
        add_to(drums, t0 + s16 * SIX, hat(0.03, 120, 0.10))
    # bass: driving 16ths, octave hop on steps 6 and 14
    for s16 in range(16):
        oct_up = 12 if s16 in (6, 14) else 0
        w = saw(root_m + 12 + oct_up, SIX * 0.92)
        env = np.exp(-np.arange(len(w)) / SR * 9)
        add_to(bass_bus, t0 + s16 * SIX, one_pole(w, 0.22) * env, 0.5)
    # pad: sustained detuned saws (skip the final bar for a cleaner ending)
    if bar < BARS - 1:
        for nm in PAD[letter]:
            for det in (-4, 3):  # cents-ish via slight freq offset baked as second osc
                m = midi(nm)
                f = freq(m) * (1 + det / 1000.0)
                t = np.arange(int(SR * BAR)) / SR
                ph = (f * t) % 1.0
                w = (2 * ph - 1) * np.minimum(1, t * 3) * np.minimum(1, (BAR - t) * 3)
                add_to(pad_bus, t0, one_pole(w, 0.05), 0.055)
    # lead: the original melody as a synth lead (eighth notes)
    for e, nm in enumerate(MELODY[bar]):
        if nm:
            w = saw(midi(nm), BEAT * 0.46)
            env = np.exp(-np.arange(len(w)) / SR * 3.2)
            add_to(lead_bus, t0 + e * BEAT / 2, one_pole(w, 0.3) * env, 0.30)

# dotted-eighth delay on the lead
dly = int(SR * BEAT * 0.75)
lead_bus[dly:] += lead_bus[:-dly] * 0.30
lead_bus[2 * dly:] += lead_bus[:-2 * dly] * 0.16

# sidechain-style ducking of bass+pad around each kick
phase = (np.arange(N) / SR) % BEAT
duck = 1.0 - 0.42 * np.exp(-phase * 10)
bass_bus *= duck
pad_bus *= duck

track = drums + bass_bus + pad_bus + lead_bus
fade = int(SR * 1.0)
track[-fade:] *= np.linspace(1, 0, fade)
track = track / max(1e-9, np.max(np.abs(track))) * 0.85

wav_path = os.path.join(OUT_DIR, "bgm_techno.wav")
wavfile.write(wav_path, SR, (track * 32767).astype(np.int16))
print("wrote", wav_path, f"{len(track)/SR:.1f}s")
