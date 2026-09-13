"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Animate a number from zero to its value on first render.
 *
 * Purely decorative, so it defers to `prefers-reduced-motion` and to any value that
 * is not finite. It also skips the animation when the value changes later — a
 * number counting up again every time a filter moves is distracting rather than
 * lively; only the first appearance is worth the flourish.
 */
export function useCountUp(value: number | null | undefined, durationMs = 620): number | null {
  const [shown, setShown] = useState<number | null>(null);
  const animated = useRef(false);

  useEffect(() => {
    if (value === null || value === undefined || !isFinite(value)) {
      setShown(null);
      return;
    }

    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    if (reduced || animated.current) {
      setShown(value);
      return;
    }

    animated.current = true;
    const target = value;
    const start = performance.now();
    let frame = 0;

    const tick = (now: number) => {
      const t = Math.min((now - start) / durationMs, 1);
      // Ease-out cubic: fast at first, settling gently on the final digits.
      setShown(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) frame = requestAnimationFrame(tick);
      else setShown(target);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, durationMs]);

  return shown;
}
