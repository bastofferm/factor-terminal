/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8100";
module.exports = {
  reactStrictMode: true,
  async rewrites() {
    // Same-origin in development, so the browser never needs CORS.
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};
