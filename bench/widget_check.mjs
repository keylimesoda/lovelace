// Feed a `lovelace-monitor info` capture through the widget's own Model.js (unmodified) and check that
// every panel label the widget shows is live against lovelace.
//   bun bench/widget_check.mjs INFO_FILE
// The Generation / Prefill rates are the request log's windowed throughput sample from the feed
// (BarWidget.qml -> Model.rateLabel: a rate plus its sample wall time, stale after 30 s).
// env WIDGET_DIR (default: integrations/omarchy of this repo). Prints a markdown table; exit 1 on any failed check.
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const widgetDir = process.env.WIDGET_DIR || join(dirname(fileURLToPath(import.meta.url)), "..", "integrations/omarchy")
const Model = createRequire(import.meta.url)(join(widgetDir, "Model.js"))
const [infoFile] = process.argv.slice(2)
if (!infoFile) {
  console.error("usage: bun bench/widget_check.mjs INFO_FILE")
  process.exit(2)
}
const raw = readFileSync(infoFile, "utf8")
const info = Model.parseInfo(raw)

// the widget's live rates (BarWidget.qml -> Model.rateLabel); "—" = unavailable
const decodeRate = Model.rateLabel(info, "decode_tps")
const prefillRate = Model.rateLabel(info, "prefill_tps")
// a backdated sample must read as unavailable, no matter what the rate was
const stale = Model.parseInfo(raw.replace(/^(rate_epoch_ms=)\d+/m, "$1" + (Date.now() - 40000)))

const labels = [
  ["modelName", Model.modelName(info)],
  ["contextLabel", Model.contextLabel(info)],
  ["kvLabel", Model.kvLabel(info)],
  ["specLabel", Model.specLabel(info)],
  ["gpuLabel", Model.gpuLabel(info)],
  ["prefixHitLabel", Model.prefixHitLabel(info)],
  ["specAcceptLabel", Model.specAcceptLabel(info)],
  ["requestsLabel", Model.requestsLabel(info)],
  ["generatedLabel", Model.generatedLabel(info)],
  ["ctxDeviceLabel", Model.ctxDeviceLabel(info)],
  ["ctxDeviceFraction", String(Model.ctxDeviceFraction(info))],
  ["ctxHostLabel", Model.ctxHostLabel(info)],
  ["ctxStateLabel", Model.ctxStateLabel(info)],
  ["ctxPressureLabel", Model.ctxPressureLabel(info)],
  ["rateLabel (Generation)", decodeRate],
  ["rateLabel (Prefill)", prefillRate],
  ["watts (power_watts)", Model.watts(info.power_watts)],
  ["uptime (epoch)", Model.uptime(info.epoch)],
]

const failures = []
const need = (cond, msg) => cond || failures.push(msg)
need(info.state === "ready", `state=${info.state}, want ready`)
need(info.gpu_name && info.gpu_name !== "GPU", `gpu_name=${JSON.stringify(info.gpu_name)} (NVML block of /telemetry not read)`)
need(Number(info.vram_total) > 0, `vram_total=${JSON.stringify(info.vram_total)}`)
for (const k of ["prompt_tokens", "predicted_tokens", "requests_total", "prefix_hit_tokens", "draft_total"])
  need(info[k] !== undefined && info[k] !== "", `${k} empty (/metrics series missing)`)
need(info.ctx_device_capacity !== undefined && info.ctx_device_capacity !== "",
  "ctx_device_capacity missing (/telemetry cache.kv_capacity_page_groups / kv_capacity)")
need(info.rate_epoch_ms !== undefined && info.rate_epoch_ms !== "",
  "rate_epoch_ms missing (request log's throughput sample not read)")
need(Model.rateLabel(stale, "decode_tps") === "—",
  `backdated sample renders ${JSON.stringify(Model.rateLabel(stale, "decode_tps"))}, want "—"`)
const live = ["prefixHitLabel", "specAcceptLabel", "generatedLabel", "ctxDeviceLabel", "ctxHostLabel"]
for (const [name, value] of labels)
  if (live.includes(name)) need(value !== "—", `${name} renders "—" after traffic`)

const cell = (s) => String(s).replace(/\|/g, "\\|")
const out = [
  `# lovelace monitor widget against lovelace`,
  ``,
  `Model.js: \`${join(widgetDir, "Model.js")}\`; feed: \`${infoFile}\``,
  ``,
  `| label | value |`,
  `|---|---|`,
  ...labels.map(([k, v]) => `| ${k} | ${cell(v)} |`),
  ``,
  `| feed key | value |`,
  `|---|---|`,
  ...Object.entries(info).map(([k, v]) => `| ${k} | ${cell(v)} |`),
  ``,
  `## Checks: ${failures.length ? "FAIL" : "PASS"}`,
  ``,
  ...(failures.length ? failures.map((f) => `- ${f}`) : ["none failed"]),
]
console.log(out.join("\n"))
process.exit(failures.length ? 1 : 0)
