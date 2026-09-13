"""Capture each page of the running app and assemble them into an animated GIF.

Needs the app running (start.bat) and two dev-only packages that are deliberately
not in requirements.txt, because nothing at runtime uses them:

    pip install playwright pillow
    playwright install chromium
    python -m scripts.capture_screenshots

Each page is given time to finish fetching and to let Plotly draw — every chart
mounts client-side, so a screenshot taken on load event would catch empty panels.
The script waits for the page's own charts to appear rather than sleeping a fixed
amount, then holds briefly for the entrance animation to settle.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from PIL import Image

BASE = "http://127.0.0.1:3100"
OUT = Path("screenshots")

# Viewport chosen so the two-column layouts actually appear in two columns; the
# grids switch at Tailwind's xl breakpoint of 1280px.
WIDTH, HEIGHT = 1600, 1000

PAGES: list[tuple[str, str, str]] = [
    ("/",          "01-data-health",     "Data Health"),
    ("/factors",   "02-factor-explorer", "Factor Explorer"),
    ("/raw",       "03-raw-explorer",    "Raw Explorer"),
    ("/matrix",    "04-covariance-pca",  "Covariance & PCA"),
    ("/loadings",  "05-loadings-lab",    "Loadings Lab"),
    ("/risk",      "06-risk-lens",       "Risk Lens"),
]

# Seconds each frame is held in the GIF. Long enough to read the panel headings.
FRAME_MS = 3000

# GIF is limited to 256 colours, so a dense financial UI has to be scaled down
# before quantisation or the text turns to mush. Half width keeps the headings
# legible at the size GitHub renders a README image.
GIF_WIDTH = 900

# Fewer than 256 keeps the file small enough to sit in a README without the flat
# panel backgrounds banding, which is where reduced palettes show first.
GIF_COLOURS = 128

# The app's own page background, used to pad shorter frames to the shared canvas.
BACKGROUND = (245, 244, 240)


async def capture(full_page: bool) -> list[Path]:
    from playwright.async_api import async_playwright

    OUT.mkdir(exist_ok=True)
    paths: list[Path] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
            # Charts animate in; a reduced-motion preference settles them faster and
            # removes the entrance transform from the capture.
            reduced_motion="reduce",
        )

        for path, name, label in PAGES:
            print(f"  {label} ...", flush=True)
            await page.goto(f"{BASE}{path}", wait_until="networkidle", timeout=60_000)

            # Wait for the page's own content rather than a fixed sleep. Data Health
            # is the one page with no charts, so it waits for a panel instead.
            selector = "section" if path == "/" else ".js-plotly-plot"
            try:
                await page.wait_for_selector(selector, timeout=45_000)
                if selector != "section":
                    # Let every chart on the page mount, not only the first.
                    await page.wait_for_function(
                        "document.querySelectorAll('.js-plotly-plot').length >= 2",
                        timeout=30_000)
            except Exception as exc:
                print(f"    warning: {label} never finished drawing ({exc})",
                      file=sys.stderr)

            await page.wait_for_timeout(2500)

            dest = OUT / f"{name}.png"
            await page.screenshot(path=str(dest), full_page=full_page)
            paths.append(dest)
            print(f"    {dest} ({dest.stat().st_size // 1024} KB)")

        await browser.close()
    return paths


def build_gif(paths: list[Path], dest: Path) -> None:
    """Assemble the frames, scaled down and quantised to a shared palette.

    GIF needs one canvas for every frame, but the five pages are five different
    lengths. They are padded to the tallest rather than cropped to the shortest —
    cropping would silently cut the bottom off the Factor Explorer, which is the
    page most worth showing. The padding uses the app's own background so the
    shorter pages simply end early instead of sitting on a black band.

    One palette is built from the busiest frame and shared, because a per-frame
    palette shifts colours between frames and makes the whole animation shimmer.
    """
    frames = [Image.open(p).convert("RGB") for p in paths]

    scale = GIF_WIDTH / frames[0].width
    scaled = [f.resize((GIF_WIDTH, round(f.height * scale)), Image.LANCZOS)
              for f in frames]
    canvas_h = max(f.height for f in scaled)

    resized = []
    for f in scaled:
        if f.height == canvas_h:
            resized.append(f)
            continue
        padded = Image.new("RGB", (GIF_WIDTH, canvas_h), BACKGROUND)
        padded.paste(f, (0, 0))
        resized.append(padded)

    # Quantise against the frame with the most distinct colours, so no frame has
    # to borrow a palette that was never built for it.
    busiest = max(resized, key=lambda f: len(f.getcolors(maxcolors=1 << 24) or []))
    palette = busiest.quantize(colors=GIF_COLOURS, method=Image.MEDIANCUT)
    quantised = [f.quantize(palette=palette, dither=Image.FLOYDSTEINBERG)
                 for f in resized]

    quantised[0].save(
        dest,
        save_all=True,
        append_images=quantised[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    print(f"\n  {dest} ({dest.stat().st_size // 1024} KB, "
          f"{len(quantised)} frames, {GIF_WIDTH}x{canvas_h})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Screenshot every page and build a GIF")
    ap.add_argument("--viewport-only", action="store_true",
                    help="capture the visible viewport instead of the whole page")
    args = ap.parse_args()

    import urllib.request
    try:
        urllib.request.urlopen(BASE, timeout=5)
    except Exception:
        print(f"The app is not answering on {BASE}. Start it with start.bat first.",
              file=sys.stderr)
        return 1

    paths = asyncio.run(capture(full_page=not args.viewport_only))
    if not paths:
        return 1
    build_gif(paths, OUT / "factor-terminal.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
