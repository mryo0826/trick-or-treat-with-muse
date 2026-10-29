# Fix l9/l10: particle intonation. The sung stems carry a pitch valley
# across "wo" (l9) inherited from the raw take's speech prosody - the
# comma in the generation text makes Fish reset the phrase before
# "kao wo dashite". Voice texts never contained spaces (checked all gen
# scripts); the comma is the suspect. Round: fresh raw takes with and
# without the comma, standard guides, measured selection on the target
# span (valley depth + garble), then one mix rebuild and two excerpts.
import json, os, subprocess, sys
import msgpack
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, save_wav, f0_track

KEY = os.environ.get("FISH_KEY")
if not KEY:
    sys.exit("FISH_KEY not set")
BASE = os.path.join(os.path.dirname(__file__), "..")
AKARI = "97eda32449cf443bbc4b8782853ca554"
SCALE_PC = {0, 2, 4, 5, 7, 8, 9, 11}
LINES = {
    "l9": ("くろねこひょっこり、かおをだして", "くろねこひょっこりかおをだして", 0.55),
    "l10": ("みんなあつまれ、おかしのいえへ", "みんなあつまれおかしのいえへ", 0.68),
}

def quantize(f):
    if f <= 0:
        return 220.0
    mf = 69 + 12 * np.log2(f / 440.0)
    best, bd = None, 1e9
    for m in range(int(np.floor(mf)) - 2, int(np.floor(mf)) + 4):
        if m % 12 in SCALE_PC and abs(m - mf) < bd:
            best, bd = m, abs(m - mf)
    return 440.0 * 2 ** ((best - 69) / 12.0)

def pitch_correct(x, strength=0.85):
    f0, hop = f0_track(x)
    tgt = np.array([quantize(f) for f in f0])
    ratio_f = np.clip((tgt / np.maximum(f0, 1.0)) ** strength, 0.6, 1.55)
    ratio_f = np.convolve(ratio_f, np.ones(6) / 6, mode="same")
    ratio = np.interp(np.arange(len(x)), np.arange(len(f0)) * hop + hop // 2, ratio_f)
    G, H = 2048, 512
    xp = np.pad(x, G)
    out = np.zeros(len(xp))
    wsum = np.zeros(len(xp))
    win = np.hanning(G)
    for p in range(0, len(xp) - G, H):
        r = float(np.mean(ratio[max(0, p - G):p])) if p >= G else float(ratio[0])
        r = min(max(r, 0.6), 1.55)
        half = int(G * r / 2)
        c = p + G // 2
        src = xp[max(0, c - half):c + half]
        if len(src) < 8:
            continue
        out[p:p + G] += np.interp(np.linspace(0, len(src) - 1, G),
                                  np.arange(len(src)), src) * win
        wsum[p:p + G] += win
    out = out / np.maximum(wsum, 0.2)
    y = out[G:G + len(x)]
    return (y / max(1e-9, np.max(np.abs(y))) * 0.9).astype(np.float32)

def garble(x, frac):
    seg = x[int(len(x) * frac):]
    win, hop = 1024, 256
    vals = []
    for i in range(0, max(1, len(seg) - win), hop):
        f = np.abs(np.fft.rfft(seg[i:i + win] * np.hanning(win))) + 1e-9
        band = f[5:100]
        vals.append(np.exp(np.mean(np.log(band))) / np.mean(band))
    return float(np.mean(vals)) if vals else 1.0

def span_score(x, frac):
    """Penalise a pitch valley + big steps inside the target span."""
    f0, hop = f0_track(x)
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    tail = midi[int(len(midi) * frac):]
    tail = tail[~np.isnan(tail)]
    if len(tail) < 10:
        return 99.0, ""
    med = np.array([np.median(tail[max(0, i - 2):i + 3]) for i in range(len(tail))])
    i_min = int(np.argmin(med))
    depth = 0.0
    if 2 < i_min < len(med) - 3:
        depth = max(0.0, min(np.median(med[max(0, i_min - 6):i_min - 1]),
                             np.median(med[i_min + 2:i_min + 7])) - med[i_min])
    steps = float(np.max(np.abs(np.diff(med))))
    return depth + 0.3 * steps, " ".join(f"{v:.0f}" for v in med[::4])

def gen_raw(text, path):
    payload = json.dumps({"text": text, "reference_id": AKARI, "format": "wav"})
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/json",
         "-H", "model: s2.1-pro-free",
         "-d", payload, "--output", path, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    return r.stdout.strip()

def hack(guide_path, guide_text, target_text, out_path):
    with open(guide_path, "rb") as fh:
        audio = fh.read()
    payload = {"text": target_text, "format": "wav",
               "references": [{"audio": audio, "text": guide_text}]}
    ppath = os.path.join(BASE, "work", "hack", "fix_particles.msgpack")
    with open(ppath, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))
    r = subprocess.run(
        ["curl", "-sS", "https://api.fish.audio/v1/tts",
         "-K", "-",
         "-H", "Content-Type: application/msgpack",
         "-H", "model: s2.1-pro-free",
         "--data-binary", f"@{ppath}", "--output", out_path, "-w", "%{http_code}"],
        input=f'header = "Authorization: Bearer {KEY}"\n', capture_output=True, text=True)
    return r.stdout.strip()

for lid, (canon, nocomma, frac) in LINES.items():
    cur = load_mono(os.path.join(BASE, "work", "voice4", f"{lid}.wav"))
    cs, cprof = span_score(cur, frac)
    print(f"{lid} CURRENT: span={cs:.2f} garble={garble(cur, frac):.3f} [{cprof}]")
    results = []
    for tag, text in [("nc", nocomma), ("cm", canon)]:
        rp = os.path.join(BASE, "work", "voice2", f"{lid}_{tag}.wav")
        print(lid, tag, "raw http", gen_raw(text, rp))
        raw = load_mono(rp)
        guide = pitch_correct(raw)
        gp = os.path.join(BASE, "work", "guides_simple", f"{lid}_{tag}.wav")
        save_wav(gp, guide)
        op = os.path.join(BASE, "work", "voice4", f"{lid}_{tag}.wav")
        print(lid, tag, "hack http", hack(gp, text, canon, op))
        x = load_mono(op)
        s, prof = span_score(x, frac)
        g = garble(x, frac)
        print(f"  -> span={s:.2f} garble={g:.3f} [{prof}]")
        results.append((s + 2 * g, op))
    results.sort()
    winner = results[0][1]
    old = os.path.join(BASE, "work", "voice4", f"{lid}.wav")
    os.replace(old, os.path.join(BASE, "work", "voice4", f"{lid}_old2.wav"))
    os.replace(winner, old)
    print(lid, "winner:", os.path.basename(winner))

r = subprocess.run([sys.executable, os.path.join(BASE, "src", "mix_final.py"),
                    "voice4", "mix_release"], capture_output=True, text=True)
print("mix_final:", r.returncode)
mix = load_mono(os.path.join(BASE, "out", "mix_release.wav"))
for name, (a, b) in [("l9", (29.4, 35.4)), ("l10", (33.2, 39.4))]:
    seg = mix[int(a * SR):int(b * SR)]
    w = os.path.join(BASE, "out", f"excerpt_{name}.wav")
    save_wav(w, seg.astype(np.float32))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", w,
                    os.path.join(BASE, "out", f"excerpt_{name}.mp3")], check=True)
    print("wrote", w)
