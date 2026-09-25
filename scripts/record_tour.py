"""Record a guided tour of the running app as a short film.

Needs the app running (start.bat) and the same two dev-only packages the
screenshot script uses, which are deliberately not in requirements.txt:

    pip install playwright pillow
    playwright install chromium
    python -m scripts.record_tour

Playwright records the browsing session itself, so there is no screen capture
and no ffmpeg in the chain — the output is a VP8 webm written when the context
closes. The narration is a lower third injected into each page after it loads,
styled in the app's own palette so the film looks like the product rather than
like a slideshow of it.

Two things are deliberate and easy to get wrong if this is ever rewritten:

  * reduced motion is NOT set, unlike in `capture_screenshots`. A still frame
    wants the entrance animations settled; a film wants to show them, and the
    overview's pipeline walk is half of what that page is for.
  * every interaction is wrapped. A tour that dies because one hover missed is
    worse than a tour with one flat beat, so a failed step is reported and the
    recording carries on.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE = "http://127.0.0.1:3100"
OUT = Path("screenshots")
DEST = OUT / "factor-terminal-tour.webm"

# 16:9 at a size where the terminal's 11px type is still readable when the film
# is played back at half size in a README.
WIDTH, HEIGHT = 1600, 900

# The lower third, in the app's palette. `pointer-events: none` throughout: the
# tour clicks real controls, and an overlay that swallowed a click would leave
# the film showing a button that does not respond.
OVERLAY_CSS = """
#tour { position: fixed; inset: 0; z-index: 2147483000; pointer-events: none;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
#tour .lower {
  position: absolute; left: 34px; bottom: 34px; max-width: 620px;
  background: linear-gradient(152deg, #F3F2EFF2 0%, #E7E5E1F2 100%);
  border: 1px solid #D2CFC8; border-radius: 4px;
  box-shadow: 0 6px 26px rgba(42,47,58,0.17);
  padding: 16px 20px 15px;
  opacity: 0; transform: translateY(10px);
  transition: opacity .45s ease, transform .45s cubic-bezier(.22,.61,.36,1);
}
#tour .lower.on { opacity: 1; transform: none; }
#tour .kicker { font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
                color: #6F7890; font-weight: 600; }
#tour .title  { margin-top: 6px; font-size: 21px; line-height: 1.2;
                font-weight: 600; color: #2F4D73; letter-spacing: -0.01em; }
#tour .body   { margin-top: 7px; font-size: 13px; line-height: 1.5; color: #5A6275; }
#tour .rail   { margin-top: 13px; height: 2px; background: #D2CFC8;
                border-radius: 1px; overflow: hidden; }
#tour .rail > div { height: 100%; background: #2F4D73; width: 0;
                    transition: width .6s ease; }

/* The title and end cards take the whole frame. */
#tour .card {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center;
  background: linear-gradient(152deg, #EDECE9 0%, #E4E2DE 58%, #DBD9D4 100%);
  opacity: 0; transition: opacity .6s ease;
}
#tour .card.on { opacity: 1; }
#tour .card .hues { display: flex; width: 360px; height: 3px; margin-bottom: 26px; }
#tour .card .hues i { flex: 1; }
#tour .card h1 { font-size: 46px; font-weight: 600; color: #2F4D73;
                 letter-spacing: -0.02em; line-height: 1.1; margin: 0; }
#tour .card p  { margin-top: 14px; font-size: 16px; color: #5A6275; max-width: 700px; }
#tour .card .stats { margin-top: 30px; display: flex; gap: 46px; }
#tour .card .stats div { text-align: center; }
#tour .card .stats b { display: block; font-size: 30px; font-weight: 600;
                       color: #2F4D73; font-variant-numeric: tabular-nums; }
#tour .card .stats span { font-size: 11px; letter-spacing: .12em;
                          text-transform: uppercase; color: #6F7890; }
#tour .card code { font-family: Consolas, ui-monospace, monospace; font-size: 15px;
                   background: #FBFAF7; border: 1px solid #D2CFC8;
                   border-radius: 3px; padding: 4px 10px; color: #2F4D73; }
"""

# The nine block hues, so the cards carry the same stripe the masthead does.
HUES = ["#2F4D73", "#6B5B95", "#1F6F5C", "#A14E3A", "#B5892B",
        "#7D6134", "#8C3A5E", "#3E7A8C", "#5B7A3A"]

# path, label, headline, body. The order is the order of the nav.
STOPS: list[tuple[str, str, str, str]] = [
    ("/", "Overview",
     "Start with what it is",
     "Forty daily factors across nine blocks, twenty-two years of history, and "
     "the whole pipeline in six stages. Nothing on this screen is a number the "
     "model invented for it."),
    ("/factors", "Factor Explorer",
     "Every factor, and the arithmetic behind it",
     "One factor at a time: the series, the rolling risk, the distribution, and "
     "the full stationarity battery. Open the profile and the exact formula is "
     "there — rendered from the stored construction rule, so it cannot drift "
     "from the chart beside it."),
    ("/raw", "Raw Explorer",
     "The series before the model touched it",
     "The same battery on the raw panel and on the individual instruments "
     "underneath. This is where a bad print is told apart from a real crash — "
     "and where the Python that builds the factor is printed from the running "
     "source."),
    ("/matrix", "Covariance & PCA",
     "How the factors move together",
     "The correlation matrix with its blocks drawn, its conditioning reported, "
     "and a risk budget across the nine blocks measured through a proxy security "
     "the factors never saw."),
    ("/loadings", "Loadings Lab",
     "Betas that have to survive the window",
     "Rolling factor betas for any security, on either panel, under an "
     "estimation window and estimator you choose. A loading that moves when the "
     "window does is telling you something."),
    ("/risk", "Risk Lens",
     "The point of the exercise",
     "Predicted risk against realised, with the backtest attached: the bias "
     "statistic, the Mincer-Zarnowitz regression, VaR coverage, Kupiec and "
     "Christoffersen. The model is allowed to be wrong here, in public."),
    ("/ops", "Operations",
     "One button, and nothing hidden",
     "The whole refresh chain, stage by stage. Each one opens onto what it "
     "reads, what it writes, what happens if it fails, and the command to run "
     "it on its own."),
    ("/health", "Data Health",
     "Because inputs go stale quietly",
     "Staleness per input measured against the newest observation in the panel "
     "rather than today, so a weekend does not look like a break. Dead "
     "instruments, failed items, and every pipeline run with its scope."),
]


def overlay_js(kicker: str, title: str, body: str, progress: float) -> str:
    """Build (or update) the lower third. Re-injected after every navigation."""
    return f"""
    (() => {{
      let root = document.getElementById('tour');
      if (!root) {{
        const style = document.createElement('style');
        style.textContent = {OVERLAY_CSS!r};
        document.head.appendChild(style);
        root = document.createElement('div');
        root.id = 'tour';
        root.innerHTML = `<div class="lower">
            <div class="kicker"></div><div class="title"></div>
            <div class="body"></div><div class="rail"><div></div></div>
          </div>`;
        document.body.appendChild(root);
      }}
      const l = root.querySelector('.lower');
      l.querySelector('.kicker').textContent = {kicker!r};
      l.querySelector('.title').textContent = {title!r};
      l.querySelector('.body').textContent = {body!r};
      requestAnimationFrame(() => {{
        l.classList.add('on');
        l.querySelector('.rail > div').style.width = {progress!r} + '%';
      }});
    }})();
    """


def card_js(html: str) -> str:
    hues = "".join(f'<i style="background:{h}"></i>' for h in HUES)
    return f"""
    (() => {{
      let root = document.getElementById('tour');
      if (!root) {{
        const style = document.createElement('style');
        style.textContent = {OVERLAY_CSS!r};
        document.head.appendChild(style);
        root = document.createElement('div');
        root.id = 'tour';
        document.body.appendChild(root);
      }}
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = '<div class="hues">{hues}</div>' + {html!r};
      root.appendChild(card);
      requestAnimationFrame(() => card.classList.add('on'));
    }})();
    """


def fade_out_js() -> str:
    return """
    (() => {
      const l = document.querySelector('#tour .lower');
      if (l) l.classList.remove('on');
      document.querySelectorAll('#tour .card').forEach(c => c.classList.remove('on'));
    })();
    """


def run(hold_ms: int) -> int:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(exist_ok=True)
    raw = OUT / "_tour_raw"
    if raw.exists():
        shutil.rmtree(raw)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
            record_video_dir=str(raw),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        page = ctx.new_page()

        # --- title card ---------------------------------------------------
        print("  title ...", flush=True)
        page.goto(f"{BASE}/", wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(1200)
        facts = read_facts(page)
        page.evaluate(card_js(
            "<h1>Factor Terminal</h1>"
            "<p>A daily multi-asset factor model, built so that every number on "
            "screen can be traced back to a series you can open.</p>"
            f"<div class='stats'>{facts}</div>"))
        page.wait_for_timeout(4200)
        page.evaluate(fade_out_js())
        page.wait_for_timeout(700)

        # --- the stops ----------------------------------------------------
        for i, (path, label, title, body) in enumerate(STOPS, 1):
            print(f"  {i}/{len(STOPS)} {label} ...", flush=True)
            page.goto(f"{BASE}{path}", wait_until="networkidle", timeout=60_000)
            settle(page, path)

            page.evaluate(overlay_js(
                f"{i:02d} / {len(STOPS):02d} · {label}", title, body,
                round(i / len(STOPS) * 100)))
            page.wait_for_timeout(hold_ms)

            beat(page, path)

            page.evaluate(fade_out_js())
            page.wait_for_timeout(600)

        # --- end card -----------------------------------------------------
        print("  closing ...", flush=True)
        page.goto(f"{BASE}/", wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(900)
        page.evaluate(card_js(
            "<h1>Run it yourself</h1>"
            "<p>The model write-up, every factor as a formula, and a case study "
            "working one security end to end are all in the repository.</p>"
            "<p style='margin-top:26px'><code>start.bat</code></p>"))
        page.wait_for_timeout(4200)

        video = page.video
        ctx.close()
        browser.close()

        if video is None:
            print("  no video was recorded", file=sys.stderr)
            return 1
        src = Path(video.path())
        DEST.unlink(missing_ok=True)
        shutil.move(str(src), DEST)
        shutil.rmtree(raw, ignore_errors=True)

    print(f"\n  {DEST} ({DEST.stat().st_size // 1024} KB, {WIDTH}x{HEIGHT})")
    return 0


def read_facts(page) -> str:
    """The title card's figures, read off the overview rather than hard-coded.

    A promotional film with stale numbers in it is a promotional film that has
    to be re-cut every time the panel grows, and would be wrong in the meantime.
    """
    try:
        page.wait_for_selector("main .grid .hero-number", timeout=20_000)
        # Keyed off `.hero-number` and its own preceding label rather than child
        # position: Stat wraps its parts in a div, so `:first-child` on the card
        # picks up that wrapper and reads back the whole tile as one string.
        vals = page.evaluate("""
          [...document.querySelectorAll('main .grid .hero-number')]
            .slice(0, 4).map(v => ({
              v: v.textContent.trim(),
              l: v.previousElementSibling?.textContent?.trim() ?? '',
            }))
        """)
        cells = [f"<div><b>{c['v']}</b><span>{c['l']}</span></div>"
                 for c in vals if c["v"]]
        if cells:
            return "".join(cells)
    except Exception as exc:  # pragma: no cover - cosmetic
        print(f"    warning: could not read the overview figures ({exc})",
              file=sys.stderr)
    return ""


def settle(page, path: str) -> None:
    """Wait for the page's own content, not a fixed sleep."""
    charts = path in ("/factors", "/raw", "/matrix", "/loadings", "/risk")
    try:
        page.wait_for_selector(".js-plotly-plot" if charts else "section",
                               timeout=45_000)
        if charts:
            page.wait_for_function(
                "document.querySelectorAll('.js-plotly-plot').length >= 2",
                timeout=30_000)
    except Exception as exc:
        print(f"    warning: {path} never finished drawing ({exc})", file=sys.stderr)
    page.wait_for_timeout(1800)


def beat(page, path: str) -> None:
    """One small piece of theatre per page, so the film is not eight stills.

    Each is optional by construction. A selector that has moved should cost the
    film one flat beat, not the whole recording.
    """
    try:
        if path == "/":
            # Hold a middle stage of the pipeline, which pauses the walk on it.
            page.hover("main ol li:nth-child(4)", timeout=4000)
            page.wait_for_timeout(1600)
            page.mouse.move(WIDTH // 2, HEIGHT - 120)
            page.wait_for_timeout(700)

        elif path == "/factors":
            # The formula behind a factor: the thing this app has that a chart
            # library does not.
            page.click("[aria-label='What is fx_carry?']", timeout=5000)
            page.wait_for_selector("[role='dialog'] .eqn", timeout=8000)
            page.wait_for_timeout(3200)
            page.mouse.wheel(0, 320)
            page.wait_for_timeout(1800)
            page.keyboard.press("Escape")
            page.wait_for_timeout(600)

        elif path in ("/raw", "/loadings", "/risk", "/health"):
            page.mouse.wheel(0, 460)
            page.wait_for_timeout(2000)

        elif path == "/matrix":
            page.mouse.wheel(0, 380)
            page.wait_for_timeout(2200)

        elif path == "/ops":
            # Open a refresh stage onto what it actually does.
            page.click("main button:has-text('Ingest')", timeout=5000)
            page.wait_for_timeout(2400)
    except Exception as exc:
        print(f"    note: skipped the beat on {path} ({exc})", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Record a guided tour of the app")
    ap.add_argument("--hold", type=int, default=4600,
                    help="milliseconds each caption is held before its beat")
    args = ap.parse_args()

    import urllib.request
    try:
        urllib.request.urlopen(BASE, timeout=5)
    except Exception:
        print(f"The app is not answering on {BASE}. Start it with start.bat first.",
              file=sys.stderr)
        return 1

    return run(args.hold)


if __name__ == "__main__":
    raise SystemExit(main())
