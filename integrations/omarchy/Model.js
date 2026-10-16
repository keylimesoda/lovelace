// keylimesoda.lovelace — helpers for the lovelace bar widget / panel.
//
// Parses the `lovelace-monitor info` key=value feed and formats values for
// display. All functions are pure so they can be unit-tested outside Qt.

var VALID_STATES = ["off", "starting", "ready", "failed"]

// Parse "key=value\n..." into a plain object. The state is always
// normalized to a valid value (empty/failed feeds read as "off").
function parseInfo(raw) {
  var out = { state: "off" }
  var lines = String(raw || "").split("\n")
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i]
    var eq = line.indexOf("=")
    if (eq <= 0) continue
    out[line.slice(0, eq)] = line.slice(eq + 1)
  }
  if (VALID_STATES.indexOf(out.state) < 0) out.state = "off"
  return out
}

// 1234 -> "1,234"
function group(n) {
  n = Number(n) || 0
  var s = String(Math.round(n))
  var out = ""
  var c = 0
  for (var i = s.length - 1; i >= 0; i--) {
    out = s[i] + out
    if (++c % 3 === 0 && i > 0) out = "," + out
  }
  return out
}

// 1234 -> "1.2k", 1500000 -> "1.5M"
function human(n) {
  n = Number(n) || 0
  if (n >= 1e6) return trim1(n / 1e6) + "M"
  if (n >= 1e3) return trim1(n / 1e3) + "k"
  return String(Math.round(n))
}

function trim1(x) {
  return x.toFixed(1).replace(/\.0$/, "")
}

// tok/s formatting; negative or NaN reads as unavailable.
function tps(x) {
  x = Number(x)
  if (!isFinite(x) || x < 0) return "—"
  if (x >= 100) return String(Math.round(x)) + " tok/s"
  return x.toFixed(1) + " tok/s"
}

// Windowed tok/s from the request log's throughput sample: the feed key holds
// the rate, rate_epoch_ms the sample's wall time. Reads "—" when the sample is
// missing or older than the staleness cap — the engine idles its reporter
// while quiet, so an old sample means there is nothing to show.
var RATE_STALE_MS = 30000
function rateLabel(info, key) {
  if (!info || info[key] === undefined || info[key] === "") return "—"
  var age = Date.now() - Number(info.rate_epoch_ms || 0)
  if (!(age >= 0) || age > RATE_STALE_MS) return "—"
  return tps(Number(info[key]))
}

function watts(x) {
  x = Number(x)
  if (!isFinite(x) || x <= 0) return "—"
  return String(Math.round(x)) + " W"
}

// Container start epoch (seconds) -> "3h 12m" style uptime.
function uptime(epochSec) {
  var e = Number(epochSec) || 0
  if (e <= 0) return "—"
  var s = Math.max(0, Math.floor(Date.now() / 1000 - e))
  var d = Math.floor(s / 86400); s -= d * 86400
  var h = Math.floor(s / 3600); s -= h * 3600
  var m = Math.floor(s / 60); s -= m * 60
  if (d > 0) return d + "d " + h + "h"
  if (h > 0) return h + "h " + m + "m"
  if (m > 0) return m + "m " + s + "s"
  return s + "s"
}

// API model id -> display headline: "qwen3.8-27b" -> "Qwen3.8-27B" (leading
// letter capitalized, parameter-count parts such as "27b" upper-cased).
// Empty when the feed has no model id.
function modelName(info) {
  var id = String((info && info.model_id) || "")
  if (!id) return ""
  var parts = id.split("-")
  for (var i = 0; i < parts.length; i++) {
    if (/^\d+(\.\d+)?[bkmt]$/i.test(parts[i])) parts[i] = parts[i].toUpperCase()
  }
  parts[0] = parts[0].charAt(0).toUpperCase() + parts[0].slice(1)
  return parts.join("-")
}

// Small tag under the headline: runtime and its key settings, e.g.
// "lovelace · MTP 3 · q5_0" (parts the feed lacks are left out).
function runtimeTag(info) {
  if (!info) return ""
  var parts = []
  if (info.runtime) parts.push(info.runtime)
  var spec = String(info.spec || "")
  var draft = Number(info.draft_tokens || 0)
  if (spec) parts.push(spec.toUpperCase() + (draft > 0 ? " " + draft : ""))
  if (info.kv_dtype) parts.push(String(info.kv_dtype))
  return parts.join(" · ")
}

function contextLabel(info) {
  var c = Number(info && info.context_window) || 0
  if (c <= 0) return "—"
  return group(c) + " tok"
}

// "rk4v4-e8" -> "RK4V4 E8"
function kvLabel(info) {
  var k = String((info && info.kv_dtype) || "")
  if (!k) return "—"
  return k.toUpperCase().replace(/-/g, " ")
}

function specLabel(info) {
  var s = String((info && info.spec) || "")
  var d = Number((info && info.draft_tokens) || 0)
  if (!s) return "Off"
  return s.toUpperCase() + (d > 0 ? " · " + d + " draft" : "")
}

// "on"/"off" vision flag from the info feed -> "On"/"Off" (unknown reads off).
function visionLabel(info) {
  return info && info.vision === "on" ? "On" : "Off"
}

// Spec-decode mode from the info feed: "dflash2" | "mtp" (unknown reads mtp).
function specMode(info) {
  return info && info.spec === "dflash2" ? "dflash2" : "mtp"
}

// Context ceilings per (spec, vision) combination on the ninfer engine. Must
// stay in sync with the spec_context() matrix in the bundled lovelace-monitor
// helper — the launch line bakes the matching --max-context into the container.
var CONTEXT_MATRIX = {
  "dflash2:off": 176128,
  "dflash2:on": 139264,
  "mtp:off": 262144,
  "mtp:on": 262144
}

// Ceiling the launch line will use for a given (spec, visionOn) combo.
function comboContext(spec, visionOn) {
  var c = CONTEXT_MATRIX[spec + ":" + (visionOn ? "on" : "off")]
  return c || 0
}

// One-line context impact of flipping a switch, e.g.
// "context 176,128 -> 139,264 tok" (empty string when the context is
// unchanged by the flip, e.g. vision on top of MTP).
function visionImpact(info) {
  var spec = specMode(info)
  var now = !!(info && info.vision === "on")
  var before = comboContext(spec, now)
  var after = comboContext(spec, !now)
  if (before === after) return ""
  var verb = now ? "Disabling frees" : "Enabling costs"
  var tok = before - after
  if (tok < 0) tok = -tok
  return verb + " " + group(tok) + " tok of context"
}

function specImpact(info) {
  var spec = specMode(info)
  var now = spec === "dflash2"
  var visionOn = !!(info && info.vision === "on")
  var before = comboContext(spec, visionOn)
  var after = comboContext(now ? "mtp" : "dflash2", visionOn)
  if (before === after) return ""
  return "context " + group(before) + " -> " + group(after) + " tok"
}

// "RTX 4090 · 21.4/24.0 GB" (feed reports VRAM in bytes)
function gpuLabel(info) {
  if (!info) return "—"
  var name = (info.gpu_name && info.gpu_name !== "GPU") ? info.gpu_name : "GPU"
  var used = Number(info.vram_used) || 0
  var total = Number(info.vram_total) || 0
  if (used > 0 && total > 0)
    return name + " · " + trim1(used / 1073741824) + "/" + trim1(total / 1073741824) + " GB"
  return name
}

// Prefix-cache hit rate: fraction of all prompt tokens served from cache
// (hit tokens accumulate across restarts while prompt tokens reset, so the
// denominator is hit+prompt, not prompt alone).
function prefixHitLabel(info) {
  var hit = Number((info && info.prefix_hit_tokens) || 0)
  var prompt = Number((info && info.prompt_tokens) || 0)
  if (hit + prompt <= 0) return "—"
  return Math.min(100, Math.round(hit / (hit + prompt) * 100)) + "%"
}

function specAcceptLabel(info) {
  var acc = Number((info && info.draft_accepted) || 0)
  var tot = Number((info && info.draft_total) || 0)
  if (tot <= 0) return "—"
  return Math.round(acc / tot * 100) + "% accepted"
}

function requestsLabel(info) {
  var proc = Number((info && info.requests_processing) || 0)
  var def = Number((info && info.requests_deferred) || 0)
  if (proc === 0 && def === 0) return "Idle"
  var s = group(proc) + " in flight"
  if (def > 0) s += " · " + group(def) + " deferred"
  return s
}

function generatedLabel(info) {
  var n = Number((info && info.predicted_tokens) || 0)
  if (n <= 0) return "—"
  return human(n) + " tokens"
}

// Context-cache occupancy from the hybrid runtime's /telemetry: device KV
// tokens served from the paged cache versus configured capacity.
function ctxDeviceLabel(info) {
  var used = Number((info && info.ctx_device_tokens) || 0)
  var cap = Number((info && info.ctx_device_capacity) || 0)
  if (used <= 0 && cap <= 0) return "—"
  if (cap <= 0) return human(used) + " tok"
  return human(used) + " / " + human(cap) + " tok"
}

// Fraction (0..1) of device KV capacity currently occupied; 0 when unknown.
function ctxDeviceFraction(info) {
  var used = Number((info && info.ctx_device_tokens) || 0)
  var cap = Number((info && info.ctx_device_capacity) || 0)
  if (cap <= 0) return 0
  return Math.max(0, Math.min(1, used / cap))
}

// Host KV spill: occupied bytes versus host capacity, in GiB.
function ctxHostLabel(info) {
  var used = Number((info && info.ctx_host_bytes) || 0)
  var cap = Number((info && info.ctx_host_capacity) || 0)
  if (cap <= 0 && used <= 0) return "—"
  return trim1(used / 1073741824) + " / " + trim1(cap / 1073741824) + " GiB"
}

// State-slot occupancy, e.g. "dev 2 · host 8/8".
function ctxStateLabel(info) {
  var dev = Number((info && info.ctx_state_device) || 0)
  var host = Number((info && info.ctx_state_host) || 0)
  var hostCap = Number((info && info.ctx_state_host_capacity) || 0)
  if (dev <= 0 && host <= 0) return "—"
  var s = "dev " + dev
  if (hostCap > 0) s += " · host " + host + "/" + hostCap
  else if (host > 0) s += " · host " + host
  return s
}

// Pressure summary: "Stable" when nothing degraded, otherwise the counts.
function ctxPressureLabel(info) {
  var dropped = Number((info && info.ctx_dropped) || 0)
  var degraded = Number((info && info.ctx_degraded) || 0)
  var evicted = Number((info && info.ctx_evicted) || 0)
  var spill = Number((info && info.ctx_spill) || 0)
  if (dropped + degraded + evicted + spill === 0) return "Stable"
  var parts = []
  if (degraded > 0) parts.push(human(degraded) + " degraded")
  if (dropped > 0) parts.push(human(dropped) + " dropped")
  if (evicted > 0) parts.push(human(evicted) + " evicted")
  if (spill > 0) parts.push(human(spill) + " spilled")
  return parts.join(" · ")
}

if (typeof module !== "undefined") {
  module.exports = {
    parseInfo: parseInfo,
    group: group,
    human: human,
    tps: tps,
    rateLabel: rateLabel,
    watts: watts,
    uptime: uptime,
    modelName: modelName,
    runtimeTag: runtimeTag,
    contextLabel: contextLabel,
    kvLabel: kvLabel,
    specLabel: specLabel,
    visionLabel: visionLabel,
    specMode: specMode,
    comboContext: comboContext,
    visionImpact: visionImpact,
    specImpact: specImpact,
    gpuLabel: gpuLabel,
    prefixHitLabel: prefixHitLabel,
    specAcceptLabel: specAcceptLabel,
    requestsLabel: requestsLabel,
    generatedLabel: generatedLabel,
    ctxDeviceLabel: ctxDeviceLabel,
    ctxDeviceFraction: ctxDeviceFraction,
    ctxHostLabel: ctxHostLabel,
    ctxStateLabel: ctxStateLabel,
    ctxPressureLabel: ctxPressureLabel
  }
}