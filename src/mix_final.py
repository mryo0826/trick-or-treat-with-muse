# Final mix: BGM (bgm_60.wav) + hack-generated sung lines (work/voice3/).
# Lines that overrun their 3.75s slot are time-fitted (stretch + pitch
# restore) so phrases never collide. Writes mix_final.wav plus the 30fps
# mouth envelope and timeline used by src/render.py.
import json, os, sys
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(__file__))
from dsp import SR, load_mono, stretch_to, grain_shift

FPS = 30
DUR = 60.0
BASE = os.path.join(os.path.dirname(__file__), "..")
SLOTS = {"l0": 0.6,
         "l1": 7.9, "l2": 11.65, "l3": 15.4, "l4": 19.15,
         "l7": 22.9, "l8": 26.65, "l9": 30.4, "l10": 34.15,
         "l5": 37.9, "l6": 41.65, "l11": 45.4, "l12": 49.15,
         "l13": 53.5}
TEXT = {
    "l0": "ハッピーハロウィン! ミューズだよ。今夜は歌っちゃうぞ!",
    "l1": "トリック・オア・トリート、おかしを ちょうだい",
    "l2": "くれなきゃ いたずら、しちゃうぞ ほら",
    "l3": "かぼちゃランタン、ゆらゆら ゆれて",
    "l4": "おばけも まじょも、あつまる夜",
    "l5": "ハロウィン ハロウィン、つきの夜のパーティー",
    "l6": "トリック・オア・トリート、みんなで歌おう",
    "l7": "まっくら やみよを ともす ランタン",
    "l8": "コウモリ ぱたぱた そらを まわる",
    "l9": "くろねこ ひょっこり かおを だして",
    "l10": "みんな あつまれ おかしの いえへ",
    "l11": "ハロウィン ハロウィン、おどろう あさまで",
    "l12": "トリック・オア・トリート、えがおで うたおう",
    "l13": "ハッピーハロウィン! また らいねん!",
}
MAX_LINE = 3.6

VOICE_DIR = sys.argv[1] if len(sys.argv) > 1 else "voice4"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "mix_hack"

N = int(SR * DUR)
bgm = load_mono(os.path.join(BASE, "out", "bgm_60.wav"))
bgm = bgm[:N] if len(bgm) >= N else np.pad(bgm, (0, N - len(bgm)))

voice = np.zeros(N, dtype=np.float32)
timeline = []
for lid, start in SLOTS.items():
    v = load_mono(os.path.join(BASE, "work", VOICE_DIR, f"{lid}.wav"))
    if lid not in ("l0", "l13") and len(v) / SR > MAX_LINE:
        L1 = len(v)
        v = stretch_to(v, int(MAX_LINE * SR))
        v = grain_shift(v, len(v) / L1)  # restore pitch after time-fit
        v = v / max(1e-9, np.max(np.abs(v))) * 0.9
        print(lid, "time-fitted", round(L1 / SR, 2), "->", MAX_LINE)
    i = int(start * SR)
    j = min(N, i + len(v))
    voice[i:j] += v[:j - i]
    timeline.append({"id": lid, "start": start,
                     "end": start + (j - i) / SR, "text": TEXT[lid]})
timeline.sort(key=lambda x: x["start"])

win = int(SR * 0.1)
rms = np.sqrt(np.convolve(voice ** 2, np.ones(win) / win, mode="same"))
duck = 1.0 - 0.45 * np.clip(rms * 4, 0, 1)
mix = bgm * 0.8 * duck + voice * 0.95
mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.92

out_wav = os.path.join(BASE, "out", f"{OUT_NAME}.wav")
wavfile.write(out_wav, SR, (mix * 32767).astype(np.int16))

frames = int(DUR * FPS)
env = np.zeros(frames, dtype=np.float32)
step = SR // FPS
for f in range(frames):
    seg = voice[f * step:(f + 1) * step]
    if len(seg):
        env[f] = np.sqrt(np.mean(seg ** 2))
env = env / max(1e-9, env.max())
np.save(os.path.join(BASE, "work", "envelope.npy"), env)
with open(os.path.join(BASE, "work", "timeline.json"), "w") as fh:
    json.dump(timeline, fh, ensure_ascii=False, indent=1)
print("wrote", out_wav)
