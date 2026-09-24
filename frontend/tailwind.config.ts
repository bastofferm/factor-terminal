import type { Config } from "tailwindcss";

// A quiet paper palette. Charts carry the colour; the shell stays out of the way.
export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#F5F4F0",
        // The same paper as the page, used as a recessed surface inside a
        // panel: formula boxes and source listings sit on it.
        canvas: "#F5F4F0",
        panel: "#FBFAF7",
        navy: "#2F4D73",
        navy2: "#476D99",
        navy3: "#6B86A8",
        muted: "#6F7890",
        line: "#DDD8CD",
        lineSoft: "#EEECE5",
        pass: "#16A34A",
        warn: "#D97706",
        fail: "#DC2626",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["Consolas", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: { "2xs": "10px" },
      letterSpacing: { label: "0.14em" },
    },
  },
  plugins: [],
} satisfies Config;
