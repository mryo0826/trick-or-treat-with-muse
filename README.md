# Trick or Treat with Muse

A 60-second vertical (1080×1920) Halloween song video, co-created by a human
(mryo0826: direction, review, final calls) and Muse (composition,
arrangement, voice production, animation code, rendering). The review loop
was deliberate and iterative — direction documents (`storyboard.md`) were
approved before implementation, and every fix was auditioned as a short
excerpt before it entered the mix. What is in this repository is the
distilled result of that loop.

**Video:** the rendered video (60 s, 1080×1920) is not stored in this
repository — it shows Muse's avatar, which is Meta's imagery, so it is
released as a vertical short on social platforms instead (see `NOTICE`).
Everything needed to re-render it, apart from the capture itself, is here.

## What you are hearing

- An all-original song: lyrics, melody and arrangement were written for this
  project. No existing songs, melodies or characters are used.
- The track is techno-leaning (128 BPM, four-on-the-floor, synth bass/pads),
  synthesized entirely in Python (NumPy) — no samples.
- The vocal is AI-generated with Fish Audio (`s2.1-pro-free`) using the
  generic library voice "Akari" (not tied to any character). Plain TTS only
  *reads*, so the singing feel comes from a workaround: each line is first
  pitch-corrected toward the song's scale, then fed back to the API as
  reference audio (via the MessagePack request format — the JSON format
  rejects inline references), which makes the model imitate a sung
  performance. The result honestly sounds a bit like an early Vocaloid —
  that is the character of this piece, not a defect we hid.
- Individual lines were re-rolled when review caught problems the meters
  missed (a pitch spike in `l3`, a truncated take in `l10` that was caught
  by ear and rejected). The last sung line is doubled with detuned/delayed
  copies of itself so the finale feels like everyone singing together.
- The spoken intro/outro lines are the same voice, unprocessed.

## What you are seeing

- Real captured footage of Muse's avatar (in its official Witch costume),
  keyed off its plain studio backdrop onto a drawn Halloween wallpaper and
  set into a rounded "stage window" that dances in code: beat bounce, sway,
  tilt and squash, with amplitude-based lip-sync driven by the vocal
  envelope. The mouth position is measured per frame from the footage
  (eye-pair detection + blob tracking); frames where the head turns or
  blinks are left untouched.
- The world illustrates the lyrics (`storyboard.md` is the staging sheet):
  jack-o'-lanterns light up and sway, ghosts float in and stay, a little
  witch flies across the moon, bats switch from crossing to circling, a
  black cat peeks out from behind the pumpkins, a candy house appears and
  everyone gathers to it, and dawn breaks at the end — they danced until
  morning. Each element also keeps its "instrument part" (pumpkins = kick,
  bats = hi-hat, moon = pad, title = lead vocal).
- Subtitles are kanji with hiragana ruby, set in a glossy pill with 3D
  lettering. Typefaces: Mochiy Pop One (title) and M PLUS Rounded 1c
  (lyrics), both SIL Open Font License 1.1 — see `assets/fonts/`.
- Everything is drawn frame by frame with Pillow and encoded with ffmpeg.

## Pipeline (in `src/`)

1. `gen_voice2.py` — plain TTS takes of every line (Fish Audio API).
2. `gen_hack_all.py` — pitch-correct each take into a guide, regenerate via
   reference audio (the "singing" step). `fix_l11.py`, `fix_l3.py`,
   `fix_l3b.py` and `fix_l9l10.py` are the same flow for single re-done
   lines, with the measurements that motivated each redo.
3. `music_60.py` — the 60-second techno backing track.
4. `finale_chorus.py` — the detuned/doubled finale vocal.
5. `mix_final.py` — mix voice + BGM with ducking; emits the lip-sync
   envelope (`work/envelope.npy`) and `work/timeline.json`.
6. `render.py` — render frames and mux the final MP4.

Other scripts (`music_draft.py`, `demo_*.py`, `guide_sing.py`,
`hack_test.py`, `mix60.py`, …) are the experiment log — kept because the
trial and error is part of the work.

**Not included:** the captured avatar footage (`assets/muse_witch.mp4` in
the working tree). It is Meta's imagery, so it is not redistributed here;
`render.py` expects it via the `CAPTURE_FILE` environment variable.

Requirements: Python 3 with NumPy/SciPy/Pillow/msgpack, ffmpeg, and a Fish
Audio API key in the `FISH_KEY` environment variable.

## Rights & disclosure

- Lyrics, melody, arrangement and code: original to this project, MIT
  License — but see `NOTICE` for the exact scope. The avatar imagery in
  the video is Meta's intellectual property and is not covered by the MIT
  License.
- Vocals: AI-generated (Fish Audio). This disclosure is intentional —
  please keep it when sharing.
- Fonts: SIL Open Font License 1.1 (texts in `assets/fonts/`).

Made for Halloween 2026. 🎃
