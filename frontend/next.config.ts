import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone', // Enables lean Docker production build
  eslint: {
    // ESLint runs separately in CI; don't block the Docker build
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Type-checking runs separately in CI; don't block the Docker build
    ignoreBuildErrors: true,
  },
  images: {
    domains: ['lh3.googleusercontent.com'],
  },
};

export default nextConfig;
