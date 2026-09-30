// Full-page PNG of a web page with headless Chromium over the DevTools protocol (no dependencies; Bun's WebSocket).
//   bun bench/screenshot.mjs URL OUT.png [WAIT_MS=8000] [WIDTH=1600]
// Opens URL at WIDTH x 1200, waits WAIT_MS (the dashboard's SSE backlog and a few 1 Hz polls), grows the viewport to the
// tallest scrollable content (page or inner scroll container), waits 1 s more and captures it.
// env CHROMIUM (default: first of chromium, chromium-browser, google-chrome-stable, google-chrome on PATH).
import { spawn, spawnSync } from "node:child_process"
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

const [url, out, waitArg, widthArg] = process.argv.slice(2)
if (!url || !out) {
  console.error("usage: bun bench/screenshot.mjs URL OUT.png [WAIT_MS] [WIDTH]")
  process.exit(2)
}
const waitMs = Number(waitArg || 8000)
const width = Number(widthArg || 1600)
const exe =
  process.env.CHROMIUM ||
  ["chromium", "chromium-browser", "google-chrome-stable", "google-chrome"].find(
    (c) => spawnSync("sh", ["-c", `command -v ${c}`]).status === 0,
  )
if (!exe) {
  console.error("screenshot: no chromium / google-chrome on PATH (set CHROMIUM)")
  process.exit(1)
}
const base = join(homedir(), ".cache/lovelace-bench")
mkdirSync(base, { recursive: true })
const profile = mkdtempSync(join(base, "chromium-shot-"))   // NVMe, never /tmp
const browser = spawn(exe, [
  "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run",
  "--no-default-browser-check", "--hide-scrollbars", "--disable-gpu", `--window-size=${width},1200`, "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const cleanup = async () => {
  const exited = new Promise((r) => (browser.exitCode !== null ? r() : browser.once("exit", r)))
  browser.kill("SIGKILL")
  await exited
  rmSync(profile, { recursive: true, force: true })
}

try {
  const wsUrl = await new Promise((resolve, reject) => {
    let buf = ""
    const t = setTimeout(() => reject(new Error("chromium did not report a DevTools endpoint")), 20000)
    browser.stderr.on("data", (d) => {
      buf += d
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/)
      if (m) { clearTimeout(t); resolve(m[1]) }
    })
    browser.on("exit", (c) => reject(new Error(`chromium exited (${c}): ${buf.slice(-400)}`)))
  })
  const ws = new WebSocket(wsUrl)
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j })
  let id = 0
  const pending = new Map()
  const waiters = []
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data)
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id)
      pending.delete(msg.id)
      msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result)
    } else if (msg.method) {
      for (let i = waiters.length - 1; i >= 0; i--)
        if (waiters[i].method === msg.method) waiters.splice(i, 1)[0].resolve(msg)
    }
  }
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    pending.set(++id, { resolve, reject })
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }))
  })
  const once = (method) => new Promise((resolve) => waiters.push({ method, resolve }))

  const { targetId } = await send("Target.createTarget", { url: "about:blank" })
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true })
  const s = (m, p) => send(m, p, sessionId)
  await s("Page.enable")
  await s("Emulation.setDeviceMetricsOverride", { width, height: 1200, deviceScaleFactor: 1, mobile: false })
  const loaded = once("Page.loadEventFired")
  await s("Page.navigate", { url })
  await Promise.race([loaded, sleep(30000)])
  await sleep(waitMs)
  const { result } = await s("Runtime.evaluate", {
    returnByValue: true,
    expression: `Math.max(document.documentElement.scrollHeight,
      ...[...document.querySelectorAll('*')].map(e => (getComputedStyle(e).overflowY.match(/auto|scroll/) ? e.scrollHeight + e.getBoundingClientRect().top : 0)))`,
  })
  const height = Math.min(20000, Math.max(1200, Math.ceil(result.value || 1200)))
  await s("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false })
  await sleep(1000)
  const shot = await s("Page.captureScreenshot", { format: "png", captureBeyondViewport: true })
  writeFileSync(out, Buffer.from(shot.data, "base64"))
  console.log(`screenshot: ${out} (${width}x${height}, ${exe}, waited ${waitMs} ms)`)
  ws.close()
  await cleanup()
  process.exit(0)
} catch (e) {
  console.error(`screenshot: ${e.message}`)
  await cleanup()
  process.exit(1)
}
