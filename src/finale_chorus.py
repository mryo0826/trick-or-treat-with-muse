# Finale chorus: give the last sung line (l12) a "everyone sings
# together" thickness - the lead stem plus two slightly detuned copies
# (+/-7 cents) and one quiet 90 ms-delayed copy, summed and normalised.
# Output replaces l12 in a copied voice dir (work/voice4_minna), so the
# solo mix stays reproducible; mix_final.py then builds the finale mix.
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, grain_shift

BASE = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(BASE, "work", "voice4", "l12.wav")
DST_DIR = os.path.join(BASE, "work", "voice4_minna")

v = load_mono(SRC)
up = grain_shift(v, 2 ** (7 / 1200))
dn = grain_shift(v, 2 ** (-7 / 1200))
n = len(v)

def fit(x):
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))

delayed = np.concatenate([np.zeros(int(0.09 * SR), dtype=np.float32), v])[:n]
mix = v * 0.9 + fit(up) * 0.42 + fit(dn) * 0.42 + delayed * 0.30
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.92
os.makedirs(DST_DIR, exist_ok=True)
save_wav(os.path.join(DST_DIR, "l12.wav"), mix.astype(np.float32))
print("wrote", os.path.join(DST_DIR, "l12.wav"))
