#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=lovelace-serve:latest

usage() {
    cat <<'USAGE'
Usage:
  deploy/serve.sh PROFILE     Start the engine and capture proxy (balanced, quality, vision)
  deploy/serve.sh stop        Stop both containers
  deploy/serve.sh status      Show engine health and telemetry summary
  deploy/serve.sh logs [engine|proxy]
  deploy/serve.sh --help

CTX overrides the profile context. Defaults: balanced/quality 327680, vision 262144.

Host state lives in ${LOVELACE_STATE:-~/.local/state/lovelace} (log/: request log, lib/captures/: proxy captures).
USAGE
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

container_exists() {
    docker container inspect "$1" >/dev/null 2>&1
}

stop_container() {
    local name=$1
    if container_exists "$name"; then
        docker stop "$name" >/dev/null
        printf 'Stopped %s\n' "$name"
    else
        printf '%s is not running\n' "$name"
    fi
}

start_service() {
    local profile=$1 default_ctx ctx file line label name
    local -a args=() tokens=()
    case "$profile" in
        balanced|quality) default_ctx=327680 ;;
        vision) default_ctx=262144 ;;
        *) fail "unknown profile: $profile (choose balanced, quality, or vision)" ;;
    esac
    ctx=${CTX:-$default_ctx}
    [[ $ctx =~ ^[1-9][0-9]*$ ]] || fail "CTX must be a positive integer"
    file="$ROOT/deploy/profiles/$profile.args"
    [[ -f $file ]] || fail "profile file not found: $file"

    local state=${LOVELACE_STATE:-$HOME/.local/state/lovelace}
    mkdir -p "$state/log" "$state/lib/captures"
    [[ -d ${HOME:?}/models ]] || fail "models directory not found: $HOME/models"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image not found: $IMAGE (build it first)"

    while IFS= read -r line || [[ -n $line ]]; do
        [[ -z $line || $line == \#* ]] && continue
        line=${line//\$\{CTX\}/$ctx}
        read -r -a tokens <<< "$line"
        args+=("${tokens[@]}")
    done < "$file"

    for name in lovelace lovelace-proxy; do
        if container_exists "$name"; then
            label=$(docker inspect --format '{{ index .Config.Labels "com.lovelace.managed" }}' "$name")
            [[ $label == serve ]] || fail "container $name exists but is not managed by this deployment"
            docker rm -f "$name" >/dev/null
        fi
    done

    docker run --detach --init --name lovelace --label com.lovelace.managed=serve \
        --restart unless-stopped --gpus all --network host --user 1000:1000 \
        --mount "type=bind,src=$HOME/models,dst=/models,readonly" \
        --mount "type=bind,src=$state/log,dst=/var/log/lovelace" \
        "$IMAGE" /opt/lovelace/bin/llama-server "${args[@]}"
    if ! docker run --detach --init --name lovelace-proxy --label com.lovelace.managed=serve \
        --restart unless-stopped --network host --user 1000:1000 \
        --mount "type=bind,src=$state/lib,dst=/var/lib/lovelace" \
        "$IMAGE" python3 /opt/lovelace/proxy/capture_proxy.py \
        --listen 127.0.0.1:8080 --upstream http://127.0.0.1:8090 --out /var/lib/lovelace/captures; then
        docker stop lovelace >/dev/null 2>&1 || true
        fail 'could not start capture proxy'
    fi
    printf 'Started profile %s (CTX=%s); engine :8090, capture proxy :8080\n' "$profile" "$ctx"
}

status() {
    local failed=0 telemetry
    printf 'Engine health: '
    if ! curl -fsS http://127.0.0.1:8090/health; then
        printf 'unavailable\n'
        failed=1
    else
        printf '\n'
    fi
    printf 'Telemetry summary:\n'
    if telemetry=$(curl -fsS http://127.0.0.1:8090/telemetry); then
        printf '%s\n' "$telemetry" | python3 -c 'import json,sys; d=json.load(sys.stdin); s=d.get("scheduler",{}); k=d.get("memory",{}).get("kv",{}); t=s.get("kv_budget",{}).get("totals",{}); e=d.get("energy",{}).get("joules") or {}; print("  model:",d.get("model_id","unknown")); print("  uptime seconds:",d.get("uptime_seconds","unknown")); print("  slots busy/total: {}/{}".format(s.get("slots_busy",0),s.get("slots_total",0))); print("  queue waiting:",s.get("queue_deferred",0)); print("  KV cells used/total: {}/{}".format(k.get("cells_used",0),k.get("cells_total",0))); print("  sessions parked/unparked: {}/{}".format(t.get("parks",0),t.get("unparks",0))); print("  board energy joules:",e.get("board","unavailable"))'
    else
        printf '  unavailable\n'
        failed=1
    fi
    printf 'Capture proxy: '
    if docker container inspect lovelace-proxy >/dev/null 2>&1; then
        docker inspect --format '{{.State.Status}}' lovelace-proxy
    else
        printf 'not found\n'
        failed=1
    fi
    return "$failed"
}

case "${1:-}" in
    --help|-h|help) usage ;;
    stop)
        stop_container lovelace-proxy
        stop_container lovelace
        ;;
    status) status ;;
    logs)
        target=${2:-engine}
        case "$target" in
            engine) name=lovelace ;;
            proxy) name=lovelace-proxy ;;
            *) fail 'logs target must be engine or proxy' ;;
        esac
        docker container inspect "$name" >/dev/null 2>&1 || fail "container not found: $name"
        exec docker logs --follow --tail=100 "$name"
        ;;
    balanced|quality|vision) [[ $# -eq 1 ]] || fail 'pass exactly one profile name'; start_service "$1" ;;
    *) usage >&2; exit 2 ;;
esac
