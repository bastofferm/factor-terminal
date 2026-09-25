"""Cut the guided tour into a narrated film with a music bed.

Needs the app running (start.bat) and four dev-only packages, deliberately kept
out of requirements.txt because nothing at runtime uses them:

    pip install playwright pillow edge-tts imageio-ffmpeg
    playwright install chromium
    python -m scripts.make_film

The order matters and is the whole trick: the narration is synthesised *first*,
each line is measured, and the recording then holds every page for exactly as
long as its line takes to say. Recording first and writing to fit would mean
either a voice that runs past its picture or silence waiting for it.

    narration  ->  durations  ->  screen recording  ->  mix  ->  mp4

The voice is Microsoft's neural TTS through `edge-tts`, which is a network call:
the narration text below is sent to their endpoint and the audio comes back. It
is marketing copy about a public repository, but it does leave the machine, and
anyone re-cutting this with different words should know that.

The music is generated here rather than licensed. It is four chords and a pulse
built out of sine waves, which is honest about what it is: a bed that stays out
of the way of a voice. If you hold a licence to something better, `--music`
takes a wav or mp3 and skips the synthesis entirely.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import subprocess
import sys
import wave
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

OUT = Path("screenshots")
WORK = OUT / "_film"

SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class Cut:
    """One version of the film: a voice, a script, a bed, and a name.

    Two exist. The house cut is what the README links: measured, factual, the
    register the rest of the project is written in. The floor cut is the same
    product sold the way a thirty-year sales desk would sell it — louder, faster,
    and with a pulse under it instead of a pad.

    What does not change between them is what the software is claimed to do. The
    swagger is in the delivery, not in the capability: the floor cut never says
    the model predicts a return, because the model's own landing page says it
    forecasts risk and not return, and a promo that contradicted the product
    would be the one thing in this repository that was not checkable.
    """

    name: str
    voice: str
    rate: str
    pitch: str
    dest: Path
    title: str
    title_line: str
    end_title: str
    end_line: str
    narration: list[tuple[str | None, str, str]]


# path, label, and what is said over it. Written for the ear rather than the
# eye: shorter sentences than the on-screen captions, and one idea each.
HOUSE: list[tuple[str | None, str, str]] = [
    (None, "open",
     "This is Factor Terminal. A daily multi-asset factor model you can actually "
     "audit. Forty factors, nine blocks, and twenty-two years of history."),
    ("/", "Overview",
     "It opens on everything at a glance. How far the data goes back, what it has "
     "lived through, and the six stages that turn a raw price into a risk number."),
    ("/factors", "Factor Explorer",
     "Every factor gets its own page. The series, the rolling risk, the "
     "distribution, and a full stationarity battery. And when you want to know "
     "exactly how one was built, the formula is right there, rendered from the "
     "rule the pipeline actually runs."),
    ("/raw", "Raw Explorer",
     "Before the model touches anything, you can inspect the raw series "
     "underneath. This is where a bad data print gets told apart from a real "
     "market crash."),
    ("/matrix", "Covariance & PCA",
     "The covariance page shows how the factors move together, block by block, "
     "and budgets risk across them through a security the factors never saw."),
    ("/loadings", "Loadings Lab",
     "The Loadings Lab gives you rolling betas for any security. Change the "
     "window, the estimator, the panel, and watch whether those loadings hold "
     "still."),
    ("/risk", "Risk Lens",
     "And this is the point of the whole exercise. Predicted risk against what "
     "actually happened, with the backtest attached. The model is allowed to be "
     "wrong here, in public."),
    ("/ops", "Operations",
     "One button refreshes the entire chain. Every stage tells you what it reads, "
     "what it writes, and what happens if it fails."),
    ("/health", "Data Health",
     "Because every number you have just seen is only as good as the data behind "
     "it."),
    (None, "close",
     "Factor Terminal. Open the repository, double-click start, and watch it run "
     "on your own machine."),
]

# The same eight pages, pitched. Short sentences, hard stops, and the features
# named as if they were weapons — which is how the register works. Every claim
# about what the software does is still one the software does: forty factors,
# the stored formula, the raw panel, the block budget, the backtest, the refresh
# chain. What is oversold is the telling.
FLOOR: list[tuple[str | None, str, str]] = [
    (None, "open",
     "Alright, listen up. Thirty years I have been doing this. Thirty. And this "
     "one is different. Factor Terminal. Forty factors, nine blocks, twenty-two "
     "years. Every number traceable."),
    ("/", "Overview",
     "You open it up, and boom. The whole model, one screen. How far back it "
     "goes. What it has lived through. Six stages, raw price to risk number. No "
     "black box."),
    ("/factors", "Factor Explorer",
     "Forty factors, every one opens up. Series. Rolling risk. Distribution. Full "
     "stationarity battery. And the formula, rendered from the rule the pipeline "
     "actually runs. How many shops show you that?"),
    ("/raw", "Raw Explorer",
     "Before the model touches anything, you get the raw series underneath. That "
     "is how you tell a bad print from a real crash."),
    ("/matrix", "Covariance & PCA",
     "How every factor moves with every other one, block by block. Plus a risk "
     "budget, measured through a security the factors never saw. No circular "
     "logic."),
    ("/loadings", "Loadings Lab",
     "Rolling betas, any security. Change the window, the estimator, the panel. "
     "If those loadings will not hold still, you find out here. Not later."),
    ("/risk", "Risk Lens",
     "And here is where the other guys go quiet. Predicted risk against what "
     "actually happened. Bias statistic. Mincer-Zarnowitz. Value at risk "
     "coverage. Kupiec. Christoffersen. This model tells you when it is wrong."),
    ("/ops", "Operations",
     "One button, the whole chain refreshes. Every stage says what it reads, what "
     "it writes, and what happens if it fails."),
    ("/health", "Data Health",
     "And every input is watched. Because your numbers are only as good as what "
     "sits underneath them."),
    (None, "close",
     "Factor Terminal. Forty factors. Twenty-two years. Zero black boxes. Clone "
     "it, run it, and check every number yourself."),
]

CUTS: dict[str, Cut] = {
    "house": Cut(
        name="house",
        voice="en-US-AvaNeural",
        # A measured read. The default pace is brisk for something a viewer is
        # also reading charts under.
        rate="-8%", pitch="+0Hz",
        dest=OUT / "factor-terminal-film.mp4",
        title="Factor Terminal",
        title_line="A daily multi-asset factor model, built so that every number "
                   "on screen can be traced back to a series you can open.",
        end_title="Run it yourself",
        end_line="The model write-up, every factor as a formula, and a case study "
                 "working one security end to end are all in the repository.",
        narration=HOUSE,
    ),
    "floor": Cut(
        name="floor",
        # The most authoritative male voice in the set, dropped and pushed.
        # Measured on one line: the flat voice reads at 147 Hz and 137 words a
        # minute, these settings at 130 Hz and 164 — deep enough to carry and
        # fast enough to sound like someone who has said it a thousand times.
        voice="en-US-ChristopherNeural",
        rate="+20%", pitch="-18Hz",
        dest=OUT / "factor-terminal-film-floor.mp4",
        title="Forty factors. Zero black boxes.",
        title_line="Twenty-two years of daily history, and every number on the "
                   "screen traceable to a series you can open yourself.",
        end_title="Check it yourself",
        end_line="That is the whole pitch. Clone it, run it, and audit every "
                 "number in it.",
        narration=FLOOR,
    ),
}


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list[str]) -> None:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise SystemExit(f"ffmpeg failed: {' '.join(args[:6])} ...")


# --- 1. the voice ---------------------------------------------------------

async def _say(text: str, dest: Path, cut: Cut) -> None:
    import edge_tts
    await edge_tts.Communicate(text, cut.voice, rate=cut.rate,
                               pitch=cut.pitch).save(str(dest))


def synthesise(cut: Cut) -> list[tuple[Path, float]]:
    """One wav per line, with its exact duration.

    Measured by decoding rather than by asking for metadata: the duration is
    what the recording is about to be timed against, and an mp3 frame count
    rounded to the nearest tenth would drift across ten cuts.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Path, float]] = []

    for i, (_, label, text) in enumerate(cut.narration):
        mp3 = WORK / f"vo-{i:02d}.mp3"
        wav = WORK / f"vo-{i:02d}.wav"
        print(f"  voicing {i:02d} {label} ...", flush=True)
        asyncio.run(_say(text, mp3, cut))
        run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(mp3),
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav)])
        with wave.open(str(wav)) as w:
            seconds = w.getnframes() / w.getframerate()
        out.append((wav, seconds))
        print(f"    {seconds:5.2f}s")
    return out


# --- 2. the picture -------------------------------------------------------

def record(clips: list[tuple[Path, float]], cut: Cut) -> tuple[Path, list[float]]:
    """Record the tour, and report when each line should start.

    The offsets are *measured* rather than predicted. Working them out from the
    fixed waits in this function plus however long a beat took plus whatever
    `networkidle` decided would be a running sum of six estimates, and a
    hundred-second film would end with the voice a page behind the picture.
    A clock started with the recording costs nothing and cannot drift.

    Each page is held for whichever is longer: the line being spoken over it,
    or the thing happening on screen. A beat that outruns its line leaves a
    beat of silence, which is pacing; a line that outruns its beat holds the
    page, which is also pacing.
    """
    import shutil
    import time

    from playwright.sync_api import sync_playwright
    from scripts import record_tour as rt

    WORK.mkdir(parents=True, exist_ok=True)
    raw = WORK / "raw"
    if raw.exists():
        shutil.rmtree(raw)

    dest = WORK / "picture.webm"
    offsets: list[float] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": rt.WIDTH, "height": rt.HEIGHT},
            device_scale_factor=1,
            record_video_dir=str(raw),
            record_video_size={"width": rt.WIDTH, "height": rt.HEIGHT},
        )
        page = ctx.new_page()
        t0 = time.monotonic()          # the recording's own zero

        def now() -> float:
            return time.monotonic() - t0

        stops = [n for n in cut.narration if n[0] is not None]
        stop_i = 0

        for (path, label, _), (_, seconds) in zip(cut.narration, clips):
            if path is None and label == "open":
                print(f"  title card ({seconds:.1f}s) ...", flush=True)
                page.goto(f"{rt.BASE}/", wait_until="networkidle", timeout=60_000)
                page.wait_for_timeout(900)
                facts = rt.read_facts(page)
                page.evaluate(rt.card_js(
                    f"<h1>{cut.title}</h1><p>{cut.title_line}</p>"
                    f"<div class='stats'>{facts}</div>"))
                page.wait_for_timeout(500)      # let the card fade up first
                offsets.append(now())
                page.wait_for_timeout(int(seconds * 1000))
                page.evaluate(rt.fade_out_js())
                page.wait_for_timeout(700)

            elif path is None:
                print(f"  end card ({seconds:.1f}s) ...", flush=True)
                page.goto(f"{rt.BASE}/", wait_until="networkidle", timeout=60_000)
                page.wait_for_timeout(800)
                page.evaluate(rt.card_js(
                    f"<h1>{cut.end_title}</h1><p>{cut.end_line}</p>"
                    "<p style='margin-top:26px'><code>start.bat</code></p>"))
                page.wait_for_timeout(500)
                offsets.append(now())
                page.wait_for_timeout(int(seconds * 1000) + 1400)

            else:
                stop_i += 1
                print(f"  {stop_i}/{len(stops)} {label} ({seconds:.1f}s) ...",
                      flush=True)
                page.goto(f"{rt.BASE}{path}", wait_until="networkidle",
                          timeout=60_000)
                rt.settle(page, path)
                page.evaluate(rt.overlay_js(
                    f"{stop_i:02d} / {len(stops):02d} · {label}",
                    rt.STOPS[stop_i - 1][2], rt.STOPS[stop_i - 1][3],
                    round(stop_i / len(stops) * 100)))
                page.wait_for_timeout(450)      # the caption slides up
                offsets.append(now())

                spent = rt.timed_beat(page, path)
                if seconds - spent > 0:
                    page.wait_for_timeout(int((seconds - spent) * 1000))

                page.evaluate(rt.fade_out_js())
                page.wait_for_timeout(600)

        video = page.video
        ctx.close()
        browser.close()
        if video is None:
            raise SystemExit("no video was recorded")
        dest.unlink(missing_ok=True)
        shutil.move(video.path(), dest)
        shutil.rmtree(raw, ignore_errors=True)

    return dest, offsets


# --- 3. the bed -----------------------------------------------------------

def compose_calm(seconds: float) -> Path:
    """A four-chord bed, synthesised.

    Deliberately simple: a pad, a plucked arpeggio an octave up, and a sub. The
    job of a bed under a voice is to carry a mood and then get out of the way,
    which is mostly a matter of what it does *not* do — no percussion, no
    melody in the vocal range, nothing that resolves hard enough to pull an ear
    off the words.

    D major at 72 BPM, I - V/vii - vi - IV. The progression lifts without ever
    arriving, which is what keeps a ninety-second bed from feeling like it has
    finished three times before the film has.
    """
    sr = SAMPLE_RATE
    bpm = 72.0
    bar = 60.0 / bpm * 4                      # four beats to the bar
    n = int(seconds * sr)
    t = np.arange(n) / sr

    def hz(semitones_from_a4: float) -> float:
        return 440.0 * 2 ** (semitones_from_a4 / 12)

    # D3 A3 D4 F#4 | C#3 A3 C#4 F4 ... written as semitone offsets from A4.
    chords = [
        [-19, -12, -7, -2, 5],     # D  major
        [-21, -14, -9, -2, 5],     # A/C#
        [-23, -12, -7, -2, 2],     # B  minor
        [-26, -14, -7, -2, 2],     # G  major
    ]

    def adsr(length: int, a: float, d: float, s: float, r: float) -> np.ndarray:
        e = np.ones(length)
        ai, di, ri = int(a * sr), int(d * sr), int(r * sr)
        ai, di, ri = min(ai, length), min(di, length), min(ri, length)
        if ai:
            e[:ai] = np.linspace(0, 1, ai)
        if di and ai + di <= length:
            e[ai:ai + di] = np.linspace(1, s, di)
            e[ai + di:] = s
        if ri:
            e[-ri:] *= np.linspace(1, 0, ri)
        return e

    def voice_at(f: float, start: float, length: float, gain: float,
                 harmonics: tuple[float, ...], env: tuple[float, float, float, float],
                 detune: float = 0.0) -> None:
        i0 = int(start * sr)
        ln = int(length * sr)
        if i0 >= n:
            return
        ln = min(ln, n - i0)
        if ln <= 0:
            return
        tt = np.arange(ln) / sr
        sig = np.zeros(ln)
        for k, amp in enumerate(harmonics, start=1):
            sig += amp * np.sin(2 * np.pi * f * k * tt)
            if detune:
                sig += amp * 0.6 * np.sin(2 * np.pi * f * k * (1 + detune) * tt)
        sig *= adsr(ln, *env) * gain
        bed[i0:i0 + ln] += sig

    bed = np.zeros(n)
    bars = int(math.ceil(seconds / bar)) + 1

    for b in range(bars):
        start = b * bar
        ch = chords[b % len(chords)]

        # Pad: the chord, slow in, slow out, quiet.
        for st in ch[1:]:
            voice_at(hz(st), start, bar * 1.05, 0.055,
                     (1.0, 0.32, 0.12, 0.05), (1.1, 0.7, 0.72, 1.0), detune=0.0016)

        # Sub: the root, felt rather than heard.
        voice_at(hz(ch[0]), start, bar * 0.98, 0.10, (1.0, 0.08),
                 (0.35, 0.5, 0.65, 0.6))

        # Pluck: an arpeggio an octave up, eighth notes, soft attack so it
        # shimmers rather than ticks.
        notes = [ch[2], ch[3], ch[4], ch[3]]
        for j in range(8):
            st = notes[j % len(notes)] + (12 if j >= 4 else 0)
            voice_at(hz(st), start + j * bar / 8, bar / 8 * 2.2, 0.030,
                     (1.0, 0.22, 0.06), (0.012, 0.28, 0.16, 0.5))

    bed = bed[:n]

    # A cheap Schroeder-ish tail: three short feedback delays. Enough to stop it
    # sounding like it was played inside a shoebox, cheap enough to stay honest.
    wet = bed.copy()
    for delay_ms, feedback in ((37, 0.32), (61, 0.26), (89, 0.20)):
        d = int(delay_ms / 1000 * sr)
        tail = np.zeros(n)
        tail[d:] = wet[:-d] * feedback
        wet = wet + tail
    bed = 0.72 * bed + 0.28 * wet

    # Gentle low-pass, so nothing up where the voice lives competes with it.
    k = 24
    bed = np.convolve(bed, np.hanning(k) / np.hanning(k).sum(), mode="same")

    # Fade in and out, and a shallow dip under the whole thing.
    fi, fo = int(2.6 * sr), int(3.4 * sr)
    bed[:fi] *= np.linspace(0, 1, fi)
    bed[-fo:] *= np.linspace(1, 0, fo)

    peak = np.max(np.abs(bed)) or 1.0
    bed = bed / peak * 0.72

    dest = WORK / "music.wav"
    with wave.open(str(dest), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((bed * 32767).astype("<i2").tobytes())
    print(f"  composed {seconds:.0f}s of bed -> {dest.name}")
    return dest


def compose_pulse(seconds: float) -> Path:
    """A faster bed with a pulse in it, for the floor cut.

    Same rules as the calm one — nothing percussive enough to fight a consonant,
    no melody up where the words are — but at 126 rather than 72, and driven by
    a repeating eighth-note figure instead of a pad that hangs. The point is
    forward motion under a voice that is already pushing.

    A minor, i - VI - III - VII. It vamps rather than resolves, which is what a
    ticker sounds like: something continuously happening and never finishing.
    """
    sr = SAMPLE_RATE
    bpm = 126.0
    beat = 60.0 / bpm
    bar = beat * 4
    n = int(seconds * sr)
    bed = np.zeros(n)

    def hz(semitones_from_a4: float) -> float:
        return 440.0 * 2 ** (semitones_from_a4 / 12)

    #        A minor      F major      C major      G major
    roots = [-24, -28, -21, -26]
    triads = [[0, 3, 7], [0, 4, 7], [0, 4, 7], [0, 4, 7]]

    def place(sig: np.ndarray, start: float) -> None:
        i0 = int(start * sr)
        if i0 >= n:
            return
        ln = min(len(sig), n - i0)
        if ln > 0:
            bed[i0:i0 + ln] += sig[:ln]

    def tone(f: float, length: float, gain: float, attack: float, decay: float,
             harmonics: tuple[float, ...] = (1.0, 0.3, 0.1)) -> np.ndarray:
        ln = int(length * sr)
        tt = np.arange(ln) / sr
        sig = np.zeros(ln)
        for k, amp in enumerate(harmonics, start=1):
            sig += amp * np.sin(2 * np.pi * f * k * tt)
        env = np.ones(ln)
        a = min(int(attack * sr), ln)
        if a:
            env[:a] = np.linspace(0, 1, a)
        env[a:] = np.exp(-np.arange(ln - a) / (decay * sr))
        return sig * env * gain

    bars = int(math.ceil(seconds / bar)) + 1
    for b in range(bars):
        t0 = b * bar
        root = roots[b % 4]
        triad = triads[b % 4]

        # Bass on every eighth, accented on the beat. This is the engine.
        for j in range(8):
            gain = 0.115 if j % 2 == 0 else 0.070
            place(tone(hz(root), beat * 0.46, gain, 0.004, 0.09,
                       (1.0, 0.16, 0.04)), t0 + j * beat / 2)

        # A low thump on one and three — a heartbeat, not a kick.
        for j in (0, 2):
            place(tone(hz(root - 12), 0.22, 0.10, 0.006, 0.055, (1.0,)),
                  t0 + j * beat)

        # Chord stabs, short, so they punctuate instead of sustaining.
        for j in (0, 2.5):
            for st in triad:
                place(tone(hz(root + 24 + st), 0.30, 0.030, 0.008, 0.10),
                      t0 + j * beat)

        # The ticker: a high blip every sixteenth, alternating, quiet. Reads as
        # movement without ever being a melody.
        for j in range(16):
            if j % 4 == 2:
                continue
            f = hz(root + 36 + triad[j % len(triad)])
            place(tone(f, 0.055, 0.012 if j % 2 else 0.019, 0.002, 0.018,
                       (1.0, 0.5)), t0 + j * beat / 4)

    bed = bed[:n]

    # Short slap, to give it a room without smearing the pulse.
    for delay_ms, fb in ((23, 0.20), (41, 0.14)):
        d = int(delay_ms / 1000 * sr)
        tail = np.zeros(n)
        tail[d:] = bed[:-d] * fb
        bed = bed + tail

    # Keep it out of the voice's way: the same low-pass, a little gentler.
    k = 14
    bed = np.convolve(bed, np.hanning(k) / np.hanning(k).sum(), mode="same")

    fi, fo = int(1.4 * sr), int(2.6 * sr)
    bed[:fi] *= np.linspace(0, 1, fi)
    bed[-fo:] *= np.linspace(1, 0, fo)

    peak = np.max(np.abs(bed)) or 1.0
    bed = bed / peak * 0.72

    dest = WORK / "music.wav"
    with wave.open(str(dest), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((bed * 32767).astype("<i2").tobytes())
    print(f"  composed {seconds:.0f}s of pulse -> {dest.name}")
    return dest


BEDS = {"house": compose_calm, "floor": compose_pulse}


# --- 4. the mix -----------------------------------------------------------

def build_voice_track(clips: list[tuple[Path, float]], offsets: list[float],
                      total: float) -> Path:
    """Lay each line at the moment its page appeared.

    Built as one delayed-and-summed track rather than a concatenation with
    silences between: concatenating makes every line's position depend on every
    line before it, so one bad measurement moves the rest of the film. Here each
    clip is placed against the recording's own clock and a wrong offset is wrong
    on its own.
    """
    args = [ffmpeg(), "-y", "-loglevel", "error"]
    for wav, _ in clips:
        args += ["-i", str(wav)]

    parts = []
    for i, off in enumerate(offsets):
        ms = max(int(off * 1000), 0)
        parts.append(f"[{i}:a]adelay={ms}|{ms}[v{i}]")
    # Padded out to the full length of the picture rather than stopping on the
    # last word. Without it `-shortest` trims the film to the end of the voice,
    # which cuts the closing card mid-hold and takes the music's fade with it.
    parts.append("".join(f"[v{i}]" for i in range(len(clips)))
                 + f"amix=inputs={len(clips)}:normalize=0:duration=longest,"
                 + f"apad,atrim=0:{total:.3f}[vo]")

    dest = WORK / "voice.wav"
    args += ["-filter_complex", ";".join(parts), "-map", "[vo]",
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(dest)]
    run(args)
    return dest


def mix_and_mux(picture: Path, voice: Path, music: Path, dest: Path) -> Path:
    """Duck the bed under the voice and burn it onto the picture.

    Sidechain compression rather than a fixed level: a bed set quiet enough to
    never trouble the words is a bed nobody hears in the gaps, and one set loud
    enough to be felt fights every sentence. Keyed off the voice it does both,
    dropping away under a line and coming back up between them.
    """
    # The voice is needed twice — once to key the compressor, once in the mix —
    # and a filtergraph label may only be consumed once, so it is split first.
    filt = (
        "[2:a]volume=0.55[bed];"
        "[1:a]volume=1.8,highpass=f=90,asplit=2[key][vox];"
        "[bed][key]sidechaincompress=threshold=0.03:ratio=12:attack=25:"
        "release=450:makeup=1[ducked];"
        "[ducked][vox]amix=inputs=2:normalize=0:duration=first,"
        "alimiter=limit=0.95[a]"
    )
    dest.unlink(missing_ok=True)
    run([ffmpeg(), "-y", "-loglevel", "error",
         "-i", str(picture), "-i", str(voice), "-i", str(music),
         "-filter_complex", filt,
         "-map", "0:v", "-map", "[a]",
         # `medium` rather than `slow`: the source is a screen recording of flat
         # panels and text, which x264 already encodes efficiently, and the
         # slower preset bought nothing visible for several times the wait.
         "-c:v", "libx264", "-preset", "medium", "-crf", "21",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "192k",
         "-shortest", str(dest)])
    return dest


def duration_of(media: Path) -> float:
    """Seconds, read back by decoding rather than from a header."""
    out = subprocess.run(
        [ffmpeg(), "-i", str(media), "-f", "null", "-"],
        capture_output=True, text=True).stderr
    last = [l for l in out.splitlines() if "time=" in l]
    if not last:
        return 0.0
    stamp = last[-1].split("time=")[1].split(" ")[0]
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Narrated film of the app")
    ap.add_argument("--cut", choices=sorted(CUTS), default="house",
                    help="house: the measured read. floor: the sales pitch.")
    ap.add_argument("--voice", default=None,
                    help="override the cut's edge-tts voice")
    ap.add_argument("--music", type=Path, default=None,
                    help="a wav or mp3 to use instead of the synthesised bed")
    ap.add_argument("--keep-work", action="store_true",
                    help="leave the intermediate audio in screenshots/_film")
    args = ap.parse_args()

    cut = CUTS[args.cut]
    if args.voice:
        cut = replace(cut, voice=args.voice)

    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:3100", timeout=5)
    except Exception:
        print("The app is not answering on http://127.0.0.1:3100. "
              "Start it with start.bat first.", file=sys.stderr)
        return 1

    print(f"\ncut: {cut.name}  ({cut.voice}, rate {cut.rate}, pitch {cut.pitch})")

    print("\nnarration")
    clips = synthesise(cut)

    print("\npicture")
    picture, offsets = record(clips, cut)
    total = duration_of(picture)
    print(f"  {total:.1f}s recorded")

    print("\naudio")
    voice = build_voice_track(clips, offsets, total)
    music = args.music if args.music else BEDS[cut.name](total + 0.5)

    print("\nmix")
    dest = mix_and_mux(picture, voice, music, cut.dest)

    if not args.keep_work:
        import shutil
        shutil.rmtree(WORK, ignore_errors=True)

    print(f"\n  {dest} ({dest.stat().st_size // 1024} KB, {total:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
