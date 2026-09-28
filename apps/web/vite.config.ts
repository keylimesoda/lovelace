import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Origin of the llama-server instance the development server proxies to. Validated rather than
// trusted, so a malformed value fails the config instead of silently pointing at the wrong server.
function engineOrigin() {
  const configured = process.env.LOVELACE_BASE_URL ?? 'http://127.0.0.1:8080'
  let url: URL
  try {
    url = new URL(configured)
  } catch {
    throw new Error('LOVELACE_BASE_URL is invalid')
  }
  if (!['http:', 'https:'].includes(url.protocol) || !url.hostname) {
    throw new Error('LOVELACE_BASE_URL must be an HTTP or HTTPS origin')
  }
  if ((url.pathname && url.pathname !== '/') || url.search || url.hash) {
    throw new Error('LOVELACE_BASE_URL must contain only the server origin')
  }
  return url.origin
}

// Every path the server owns that the dashboard or an operator touches. In production the
// dashboard is same-origin behind `llama-server --web-dir`, so these are proxied only to make
// `bun run dev` behave identically without enabling CORS on the server.
const enginePaths = ['/telemetry', '/events', '/metrics', '/slots', '/health', '/v1', '/props']

export default defineConfig(({ command, mode }) => {
  const developmentServer = command === 'serve' && mode === 'development'
  const target = developmentServer ? engineOrigin() : ''

  return {
    plugins: [react(), tailwindcss()],
    server: developmentServer
      ? {
          host: '127.0.0.1',
          port: 5180,
          strictPort: true,
          proxy: Object.fromEntries(
            enginePaths.map((path) => [
              path,
              {
                target,
                changeOrigin: true,
                // /events is an open-ended SSE stream and a generation can outlast any default
                // idle timeout, so neither side may time the connection out.
                timeout: 0,
                proxyTimeout: 0,
              },
            ]),
          ),
        }
      : undefined,
    build: {
      target: 'es2022',
      cssTarget: 'chrome107',
      cssMinify: 'lightningcss',
      sourcemap: false,
      reportCompressedSize: true,
      modulePreload: { polyfill: false },
      // ECharts dominates the bundle and changes only when the dependency does. Isolating it in
      // its own content-hashed chunk means editing a panel does not invalidate a cached copy.
      rolldownOptions: {
        output: {
          codeSplitting: {
            groups: [{ name: 'echarts', test: /node_modules[\\/]echarts|zrender/ }],
          },
        },
      },
      chunkSizeWarningLimit: 700,
    },
  }
})
