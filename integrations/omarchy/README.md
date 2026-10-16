# lovelace desktop monitor (Omarchy)

A status-bar widget and drop-down panel for [Omarchy](https://omarchy.org) that monitors and controls the local
lovelace server: start/stop, the model and runtime at a glance, live throughput, GPU and context-cache state.

![lovelace monitor panel](../../docs/img/omarchy-monitor.png)

## What it shows

- **Bar button**: the Qwen mark, tinted by state (theme accent when ready, urgent colour when failed, dimmed when
  off). Left click opens the panel, right click starts or stops the server.
- **Panel**
  - headline: the model, from `/v1/models` (`qwen3.8-27b` → **Qwen3.8-27B**), with a small runtime tag underneath:
    runtime, speculative method and draft tokens, KV cache type (e.g. `lovelace · MTP 3 · q5_0`). The runtime name
    comes from `/telemetry` (lovelace reports a `lovelace` block, including its KV cache types); spec method and
    draft tokens, which the API does not expose, come from the engine container's launch arguments.
  - a power switch with starting/stopping transitions
  - generation / prefill tok/s, windowed over the engine's ~5 s throughput samples (from its request log), and board power
  - model facts: context window, KV cache, spec decode, GPU and VRAM, uptime, endpoint, image
  - throughput: prefix-cache hit rate, draft acceptance, requests
  - context cache: device KV occupancy, host (RAM) spill, sessions, pressure
  - vision toggle (restarts the server with the `vision` profile), and on the ninfer engine a DFlash2/MTP toggle
  - **Dashboard** (opens http://127.0.0.1:8080/) and **Logs** (follows the engine log in a terminal)
- Keyboard: arrows move between the power switch, toggles and buttons; Enter activates; Esc closes.

## How it works

The plugin is a hosted QML bar widget (`BarWidget.qml`, `Model.js`) plus a bash helper, `lovelace-monitor`, that is
its data plane. The widget runs the helper next to its manifest:

- `lovelace-monitor status` every 5 s: `off` / `starting` / `ready` / `failed` from the container state and
  `/v1/models`
- `lovelace-monitor info` every 4 s while the panel is open: `key=value` lines from `/v1/models`, `/telemetry`,
  `/metrics`, the request log's windowed throughput samples (the live tok/s — the `/metrics` counters only advance
  when a request completes), the container's launch arguments, and `nvidia-smi` as a GPU fallback
- `on` / `off` / `toggle` / `restart`, `vision [on|off|toggle]`, `logs`

It drives one of two engines, both serving the API on `127.0.0.1:8080`:

- **lovelace** (default): `deploy/serve.sh` of the checkout the helper lives in (override with `LOVELACE_DIR`),
  containers `lovelace` (engine :8090) and `lovelace-proxy` (API and dashboard :8080). Start runs
  `serve.sh balanced`, or `serve.sh vision` with the vision toggle on; stop runs `serve.sh stop`.
- **ninfer** (fallback): the `ninfer-qwen` container of
  [keylimesoda/ninfer-4090-lin-hybrid](https://github.com/keylimesoda/ninfer-4090-lin-hybrid) (image
  `ninfer-4090:hybrid`, model artifact from `NINFER_MODEL_ARTIFACT`), launched by the helper itself, with the
  DFlash2/MTP toggle.

Pick one with `lovelace-monitor engine lovelace|ninfer` (stored in `~/.config/lovelace-monitor/engine`;
`LOVELACE_MONITOR_ENGINE` overrides it). The helper refuses to start an engine while the other one's container is
running. Settings (engine, vision, spec) live in `~/.config/lovelace-monitor/`.

## Install

Requirements: Omarchy 4 (tested on 4.0.4), Docker with the NVIDIA container toolkit, `curl` and `python3` on the host,
and a lovelace checkout with the `lovelace-serve:latest` image built (see the [Quick start](../../README.md#quick-start)).

`omarchy plugin add` clones a repository whose root is the plugin, so it cannot install this subdirectory. Omarchy does
discover symlinked plugin folders, so install from a checkout with the script here:

```sh
integrations/omarchy/install.sh --enable
```

It validates the plugin (`omarchy plugin validate`), links
`~/.config/omarchy/plugins/keylimesoda.lovelace` → this directory and `~/.local/bin/lovelace-monitor` → the helper,
rescans the shell, and enables the widget in the right bar section. Move it with
`omarchy bar move keylimesoda.lovelace --section left|center|right` (or enable without `--enable` and run
`omarchy plugin enable keylimesoda.lovelace --after <widget-id>`).

- **Update**: `git pull` in the checkout; the shell reloads plugin code by itself.
- **Remove**: `integrations/omarchy/install.sh --uninstall` (runs `omarchy plugin remove keylimesoda.lovelace`,
  which unlinks rather than deletes, and removes the helper links). `~/.config/lovelace-monitor/` is kept.

### Coming from omarchy-ninfer

This widget replaces [keylimesoda/omarchy-ninfer](https://github.com/keylimesoda/omarchy-ninfer) (plugin
`keylimesoda.ninfer`, helper `ninfer-qwen`):

```sh
omarchy plugin remove keylimesoda.ninfer --yes
integrations/omarchy/install.sh --enable --ninfer-qwen-alias
```

The first install copies `~/.config/ninfer-qwen/` to `~/.config/lovelace-monitor/`. `--ninfer-qwen-alias` links
`~/.local/bin/ninfer-qwen` to the new helper for scripts that call the old name (a regular file there is kept as
`ninfer-qwen.bak.<time>`). `NINFER_QWEN_ENGINE` is now `LOVELACE_MONITOR_ENGINE`.

## Development

```sh
omarchy plugin validate integrations/omarchy
/usr/lib/qt6/bin/qmllint -I <dir containing a "qs" link to /usr/share/omarchy/shell> integrations/omarchy/BarWidget.qml
integrations/omarchy/lovelace-monitor info
```

`Model.js` is plain JavaScript (the formatting and parsing of the `info` feed) and also loads under Node for quick
checks.

## License

MIT, as the rest of lovelace ([LICENSE](../../LICENSE)). The Qwen mark (`assets/qwen.png`) is derived from the Qwen
brand artwork, monochromized for a status icon; the mark belongs to its owner (see
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)).
