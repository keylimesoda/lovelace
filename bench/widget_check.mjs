// Feed an omarchy-ninfer `ninfer-qwen info` capture through the widget's own Model.js (unmodified) and check that
// every panel label the widget shows is live against lovelace.
//   bun bench/widget_check.mjs INFO_FILE [PREV_INFO_FILE DT_SECONDS]
// PREV_INFO_FILE + DT_SECONDS: an earlier sample, for the Generation / Prefill rates the widget derives from two
// samples (BarWidget.qml: token deltas over >= 2 s of the same container start epoch).
// env WIDGET_DIR (default ~/Work/omarchy-ninfer). Prints a markdown table; exit 1 on any failed check.
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { homedir } from "node:os"
import { join } from "node:path"

const widgetDir = process.env.WIDGET_DIR || join(homedir(), "Work/omarchy-ninfer")
const Model = createRequire(import.meta.url)(join(widgetDir, "Model.js"))
const [infoFile, prevFile, dtArg] = process.argv.slice(2)
if (!infoFile) {
  console.error("usage: bun bench/widget_check.mjs INFO_FILE [PREV_INFO_FILE DT_SECONDS]")
  process.exit(2)
}
const raw = readFileSync(infoFile, "utf8")
const info = Model.parseInfo(raw)

// the widget's own rate derivation (BarWidget.qml), -1 = unavailable
let genTps = -1
let promptTps = -1
if (prevFile) {
  const prev = Model.parseInfo(readFileSync(prevFile, "utf8"))
  const dt = Number(dtArg)
  if (prev.state === "ready" && Number(prev.epoch || 0) === Number(info.epoch || 0) && dt >= 2) {
    genTps = Math.max(0, Number(info.predicted_tokens || 0) - Number(prev.predicted_tokens || 0)) / dt
    promptTps = Math.max(0, Number(info.prompt_tokens || 0) - Number(prev.prompt_tokens || 0)) / dt
  }
}

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
  ["tps (Generation)", Model.tps(genTps)],
  ["tps (Prefill)", Model.tps(promptTps)],
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
const live = ["prefixHitLabel", "specAcceptLabel", "generatedLabel", "ctxDeviceLabel", "ctxHostLabel"]
for (const [name, value] of labels)
  if (live.includes(name)) need(value !== "—", `${name} renders "—" after traffic`)

const cell = (s) => String(s).replace(/\|/g, "\\|")
const out = [
  `# omarchy-ninfer widget against lovelace`,
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
