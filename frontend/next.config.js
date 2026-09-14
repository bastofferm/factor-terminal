/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8100";
module.exports = {
  reactStrictMode: true,

  experimental: {
    /**
     * How long a rewrite may take before the proxy gives up. Default is 30s.
     *
     * Estimation is synchronous and can legitimately run longer: a 504-day window
     * rolled daily over twenty years is several thousand regressions, and the
     * request does not return until they are all written. At the default the proxy
     * abandoned it at exactly 30.0s and answered with a bare, non-JSON
     * `Internal Server Error` — so the browser showed "500 Internal Server Error"
     * for a request the API went on to complete successfully. Nothing in either
     * log said "timeout".
     *
     * Five minutes is not an invitation to write slow endpoints; it is the ceiling
     * above which something really is wrong.
     */
    proxyTimeout: 300_000,
  },

  async rewrites() {
    // Same-origin in development, so the browser never deals with CORS.
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};
